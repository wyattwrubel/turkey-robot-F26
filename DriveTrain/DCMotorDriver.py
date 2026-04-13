import math
import serial


class DC_Driver:
    """
    RoboClaw packet-serial driver for a 4-wheel differential drive.

    Hardware layout -- 4 RoboClaw Solo 60A units, each on its own serial port,
    each controlling a single motor on M1:

      front_left  : port_fl  (default /dev/ttyMotor0)
      front_right : port_fr  (default /dev/ttyMotor1)
      rear_left   : port_rl  (default /dev/ttyMotor2)
      rear_right  : port_rr  (default /dev/ttyMotor3)

    All four units share the same address (0x80) since they are on
    independent serial buses. All communication uses RoboClaw Packet
    Serial mode -- every transmitted packet (including read requests)
    must be terminated with a CRC16-CCITT checksum.

    CLOSED LOOP MODE (cmd 35 -- Drive M1 With Signed Speed):
    Speed is a signed 32-bit int in encoder counts/sec.
    Conversion: counts/sec = (v / (2pi * wheel_radius)) * encoder_CPR_effective
    where encoder_CPR_effective = encoder_CPR * 4  (full quadrature, 4x mode).

    PID gains are tuned and stored directly on each RoboClaw unit via
    BasicMicro Motion Studio. This driver does NOT write PID values at
    startup -- the EEPROM values on each controller are the source of truth.
    Use Motion Studio Auto Tune or manual tuning to set P, I, D, and QPPS
    per unit. Changes survive power cycles automatically.

    SPEED COMMAND ZONES (both sides evaluated together in UpdateMotorSpeeds):

      Both |v| < ol_zero_threshold  (default 0.5 m/s):
          Open-loop duty 0 (cmd 32). Bypasses PID entirely.
          Prevents closed-loop oscillation when robot should be stopped.

      Both |v| < snap_threshold  (default 1.0 m/s):
          Closed-loop cmd 35 at exact speed, no ramp.
          Enforced upstream in MotorDriver._ramp_speeds, documented here
          for reference.

      Either |v| >= snap_threshold:
          Closed-loop cmd 35. Ramp applied upstream by MotorDriver.

    Left-side motors are physically mirrored so their direction is
    negated in software via motor_directions. Right side is correct.

    Speed reads (AverageLinearSpeed) use encoder feedback via cmd 18
    and convert counts/sec back to m/s. The direction sign flip is
    applied on read as well so returned speeds are always in the robot
    frame (positive = forward).
    """

    # -- Serial ports --------------------------------------------------
    port_fl = "/dev/ttyMotor0"
    port_fr = "/dev/ttyMotor1"
    port_rl = "/dev/ttyMotor2"
    port_rr = "/dev/ttyMotor3"

    baudrate = 115200
    timeout  = 0.1

    # -- Physical parameters -------------------------------------------
    max_linear_speed  = 8.7    # m/s -- measured via Motion Studio Auto Tune (71280 counts/sec)
    wheel_radius      = 0.08   # metres

    encoder_CPR           = 1024
    encoder_quadrature_x  = 4
    encoder_CPR_effective = encoder_CPR * encoder_quadrature_x   # 4096

    # -- Speed command zone thresholds ---------------------------------
    #
    # ol_zero_threshold:
    #   If BOTH |v_l| and |v_r| are below this, send open-loop duty 0
    #   (cmd 32) instead of a closed-loop speed command. Prevents PID
    #   integrator wind-up and oscillation at near-zero speeds.
    #   Raise if oscillation persists; lower if dead zone feels too large.
    #
    # snap_threshold:
    #   If BOTH |v_l| and |v_r| are below this (and above ol_zero_threshold),
    #   MotorDriver sends the command instantly without ramping.
    #   DC_Driver receives the value and sends it directly via cmd 35.
    #   Must be > ol_zero_threshold.
    #
    ol_zero_threshold = 0.5   # m/s -- both sides below -> OL duty 0
    snap_threshold    = 1.0   # m/s -- both sides below -> CL instant, no ramp (enforced in MotorDriver)

    # -- Motor direction map -------------------------------------------
    # Left motors are physically mirrored -- negate their speed command
    motor_directions = {
        "fl": -1,
        "fr":  1,
        "rl": -1,
        "rr":  1,
    }

    # -- RoboClaw protocol constants -----------------------------------
    _ADR                = 0x80
    _CMD_M1_DRIVE_SPEED = 35   # closed-loop signed counts/sec
    _CMD_M1_DRIVE_DUTY  = 32   # open-loop signed duty cycle (+-32767)
    _CMD_M1_READ_SPEED  = 18   # read M1 encoder speed
    _CMD_READ_CURRENTS  = 49   # read M1+M2 currents

    # -- Max counts/sec ceiling (safety clamp on command) -------------
    _MAX_COUNTS_PER_SEC = 2_147_483_647   # signed 32-bit max

    def __init__(self):
        self._ser_fl = serial.Serial(port=self.port_fl, baudrate=self.baudrate, timeout=self.timeout)
        self._ser_fr = serial.Serial(port=self.port_fr, baudrate=self.baudrate, timeout=self.timeout)
        self._ser_rl = serial.Serial(port=self.port_rl, baudrate=self.baudrate, timeout=self.timeout)
        self._ser_rr = serial.Serial(port=self.port_rr, baudrate=self.baudrate, timeout=self.timeout)

        # PID gains are stored in each RoboClaw's EEPROM and loaded
        # automatically at power-up. No startup write needed here.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def UpdateMotorSpeeds(self, MotorSpeed: list) -> None:
        """
        Command all four wheels simultaneously.

        Zone logic (both sides evaluated together):

          Both |v| < ol_zero_threshold:
              Open-loop duty 0 via cmd 32. Kills oscillation at near-zero
              speed by bypassing the PID loop entirely.

          Otherwise:
              Closed-loop cmd 35. Ramping and snap decisions are made
              upstream in MotorDriver before this call.

        Args:
            MotorSpeed: [left_speed_m_s, right_speed_m_s]
                        Positive = forward. Clamped to +-max_linear_speed.
        """
        left_v  = float(MotorSpeed[0])
        right_v = float(MotorSpeed[1])

        if abs(left_v) < self.ol_zero_threshold and abs(right_v) < self.ol_zero_threshold:
            self._DutyZero(self._ser_fl)
            self._DutyZero(self._ser_fr)
            self._DutyZero(self._ser_rl)
            self._DutyZero(self._ser_rr)
        else:
            self._DriveMotor(self._ser_fl, left_v,  self.motor_directions["fl"])
            self._DriveMotor(self._ser_fr, right_v, self.motor_directions["fr"])
            self._DriveMotor(self._ser_rl, left_v,  self.motor_directions["rl"])
            self._DriveMotor(self._ser_rr, right_v, self.motor_directions["rr"])

    def AverageLinearSpeed(self):
        """
        Read back encoder speeds and return the average for each side.

        The direction sign flip is applied per motor so that the returned
        values are always in the robot frame (positive = forward).

        Returns:
            (left_speed_m_s, right_speed_m_s)
            Either value is None if any read on that side failed.
        """
        fl = self._ReadMotorSpeed(self._ser_fl, self.motor_directions["fl"])
        fr = self._ReadMotorSpeed(self._ser_fr, self.motor_directions["fr"])
        rl = self._ReadMotorSpeed(self._ser_rl, self.motor_directions["rl"])
        rr = self._ReadMotorSpeed(self._ser_rr, self.motor_directions["rr"])

        left_speed  = (fl + rl) / 2.0 if (fl is not None and rl is not None) else None
        right_speed = (fr + rr) / 2.0 if (fr is not None and rr is not None) else None

        return left_speed, right_speed

    def ReadMotorCurrent(self):
        """
        Read instantaneous motor currents from all four controllers.

        Returns:
            [front_left_A, front_right_A, rear_left_A, rear_right_A]
            Individual values are None if that controller read failed.
        """
        fl = self._ReadCurrent(self._ser_fl)
        fr = self._ReadCurrent(self._ser_fr)
        rl = self._ReadCurrent(self._ser_rl)
        rr = self._ReadCurrent(self._ser_rr)

        return [fl, fr, rl, rr]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _DutyZero(self, ser: serial.Serial) -> None:
        """
        Send open-loop duty cycle 0 to one RoboClaw Solo (cmd 32).

        Bypasses the closed-loop PID entirely and commands zero PWM duty.
        Prevents integrator wind-up and oscillation when both wheel speeds
        are below ol_zero_threshold.

        Packet format: [addr, 32, duty_BE_signed_16, CRC16_BE]
        Duty range: -32767 to +32767. 0 = no drive.
        Motor behaviour at zero duty (freewheel vs brake) is set by the
        Idle Mode setting in Motion Studio.

        Response: [0xFF]  (ACK)
        """
        payload     = (0).to_bytes(2, byteorder='big', signed=True)
        packet      = bytes([self._ADR, self._CMD_M1_DRIVE_DUTY]) + payload
        crc         = self._crc16(packet)
        full_packet = packet + bytes([crc >> 8, crc & 0xFF])

        ser.write(full_packet)
        ack = ser.read(1)
        if ack != b'\xFF':
            print(f"[DC_Driver] NACK (duty zero) on port {ser.port}")

    def _DriveMotor(self, ser: serial.Serial, velocity: float, direction: int) -> None:
        """
        Send a closed-loop M1 velocity command to one RoboClaw Solo (cmd 35).

        Conversion:
            rev_per_sec    = velocity / (2pi * wheel_radius)
            counts_per_sec = rev_per_sec * encoder_CPR_effective

        The direction sign flip accounts for physically mirrored motors.

        Packet format: [addr, 35, speed_BE_signed_32, CRC16_BE]
        Response:      [0xFF]  (ACK)
        """
        velocity = max(-self.max_linear_speed, min(self.max_linear_speed, velocity))

        rev_per_sec    = velocity / (2.0 * math.pi * self.wheel_radius)
        counts_per_sec = int(direction * rev_per_sec * self.encoder_CPR_effective)
        counts_per_sec = max(-self._MAX_COUNTS_PER_SEC, min(self._MAX_COUNTS_PER_SEC, counts_per_sec))

        payload     = counts_per_sec.to_bytes(4, byteorder='big', signed=True)
        packet      = bytes([self._ADR, self._CMD_M1_DRIVE_SPEED]) + payload
        crc         = self._crc16(packet)
        full_packet = packet + bytes([crc >> 8, crc & 0xFF])

        ser.write(full_packet)
        ack = ser.read(1)
        if ack != b'\xFF':
            print(f"[DC_Driver] NACK on port {ser.port}")

    def _ReadMotorSpeed(self, ser: serial.Serial, direction: int):
        """
        Read M1 encoder speed from one RoboClaw Solo over the given port.

        Request packet:  [addr, 18, CRC16_BE]
        Response:        [speed_BE_signed_32 (4), status (1), CRC16_BE (2)]  = 7 bytes

        Returns linear speed in m/s, or None on failure.
        """
        req_bytes = bytes([self._ADR, self._CMD_M1_READ_SPEED])
        crc       = self._crc16(req_bytes)
        full_req  = req_bytes + bytes([crc >> 8, crc & 0xFF])

        ser.reset_input_buffer()
        ser.write(full_req)

        response = ser.read(7)
        if len(response) < 7:
            print(f"[DC_Driver] Short read (speed) on port {ser.port}: got {len(response)} bytes")
            return None

        recv_crc = int.from_bytes(response[5:7], byteorder='big')
        calc_crc = self._crc16(req_bytes + response[:5])
        if recv_crc != calc_crc:
            print(f"[DC_Driver] CRC mismatch (speed) on port {ser.port}")
            return None

        pulses_per_sec = int.from_bytes(response[0:4], byteorder='big', signed=True)
        rev_per_sec    = pulses_per_sec / self.encoder_CPR_effective
        linear_speed   = direction * rev_per_sec * (2.0 * math.pi * self.wheel_radius)

        return linear_speed

    def _ReadCurrent(self, ser: serial.Serial):
        """
        Read M1 current from one RoboClaw Solo over the given port (cmd 49).

        Request packet:  [addr, 49, CRC16_BE]
        Response:        [M1_counts (2), M2_counts (2), CRC16_BE (2)]  = 6 bytes

        Solo only uses M1 -- M2 counts will always be 0, discarded.
        Units: counts / 100.0 = Amps.

        Returns M1 current in Amps, or None on failure.
        """
        req_bytes = bytes([self._ADR, self._CMD_READ_CURRENTS])
        crc       = self._crc16(req_bytes)
        full_req  = req_bytes + bytes([crc >> 8, crc & 0xFF])

        ser.reset_input_buffer()
        ser.write(full_req)

        response = ser.read(6)
        if len(response) < 6:
            print(f"[DC_Driver] Short read (current) on port {ser.port}: got {len(response)} bytes")
            return None

        recv_crc = int.from_bytes(response[4:6], byteorder='big')
        calc_crc = self._crc16(req_bytes + response[:4])
        if recv_crc != calc_crc:
            print(f"[DC_Driver] CRC mismatch (current) on port {ser.port}")
            return None

        m1_counts = int.from_bytes(response[0:2], byteorder='big')
        return m1_counts / 100.0

    # ------------------------------------------------------------------
    # CRC
    # ------------------------------------------------------------------

    @staticmethod
    def _crc16(data: bytes) -> int:
        """
        CRC16-CCITT (poly 0x1021, init 0x0000).
        Covers the address byte, command byte, and all data bytes.
        """
        crc = 0x0000
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ 0x1021) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc