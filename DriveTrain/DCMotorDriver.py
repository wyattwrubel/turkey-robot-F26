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
    Conversion: counts/sec = (v / (2π * wheel_radius)) * encoder_CPR_effective
    where encoder_CPR_effective = encoder_CPR * 4  (full quadrature, 4x mode).

    PID gains are sent to the RoboClaw at startup via cmd 28 (SetM1VelocityPID).
    Tune these via the class-level variables below. QPPS is the maximum expected
    encoder speed in counts/sec -- it acts as the feedforward ceiling.

    ---- TUNING GUIDE ----
    1. Start with P only (I=0, D=0). Increase P until the robot tracks velocity
       commands with acceptable responsiveness without oscillating.
    2. Add I slowly to eliminate steady-state error (robot not reaching target speed).
       Too much I causes wind-up and overshoot.
    3. Add D only if you see overshoot. Encoder noise amplifies D -- keep it small.
    4. QPPS should match your actual max encoder speed at full throttle. If the robot
       stalls or behaves erratically at high speeds, reduce QPPS.
    ----------------------

    Left-side motors are physically mirrored so their direction is
    negated in software via motor_directions. Right side is correct.

    Speed reads (AverageLinearSpeed) use encoder feedback via cmd 18
    and convert counts/sec back to m/s. The direction sign flip is
    applied on read as well so returned speeds are always in the robot
    frame (positive = forward).
    """

    # ── Serial ports ──────────────────────────────────────────────────
    port_fl = "/dev/ttyMotor0"
    port_fr = "/dev/ttyMotor1"
    port_rl = "/dev/ttyMotor2"
    port_rr = "/dev/ttyMotor3"

    baudrate = 115200
    timeout  = 0.1

    # ── Physical parameters ───────────────────────────────────────────
    max_linear_speed = 5.0    # m/s -- used for input clamping only
    wheel_radius     = 0.08   # metres

    encoder_CPR          = 1024
    encoder_quadrature_x = 4
    encoder_CPR_effective = encoder_CPR * encoder_quadrature_x   # 4096

    # ── Velocity PID gains (sent to RoboClaw at startup via cmd 28) ───
    #
    # QPPS: max encoder counts/sec at full speed
    #   = (max_linear_speed / (2π * wheel_radius)) * encoder_CPR_effective
    #   = (5.0 / 0.5027) * 4096 ≈ 40,744 → rounded to 41000
    #
    # P, I, D: RoboClaw encodes these as uint32 = float * 65536 internally.
    #   Pass floats here -- _SetPID() handles the encoding.
    #
    # CONSERVATIVE STARTING VALUES -- tune on the bench:
    #   P: increase first until tracking is responsive but stable
    #   I: add slowly to eliminate steady-state error
    #   D: leave at 0 until P+I are tuned; encoder noise amplifies D
    #
    pid_QPPS = 41000   # counts/sec  ← update if max speed changes
    pid_P    = 1.6     # proportional gain
    pid_I    = 0.15    # integral gain
    pid_D    = 0.0     # derivative gain (start at 0, tune last)

    # ── Motor direction map ───────────────────────────────────────────
    # Left motors are physically mirrored -- negate their speed command
    motor_directions = {
        "fl": -1,
        "fr":  1,
        "rl": -1,
        "rr":  1,
    }

    # ── RoboClaw protocol constants ───────────────────────────────────
    _ADR                  = 0x80
    _CMD_M1_DRIVE_SPEED   = 35    # closed-loop signed counts/sec
    _CMD_M1_SET_PID       = 28    # SetM1VelocityPID
    _CMD_M1_READ_SPEED    = 18    # read M1 encoder speed
    _CMD_READ_CURRENTS    = 49    # read M1+M2 currents

    # ── Max counts/sec ceiling (safety clamp on command) ─────────────
    _MAX_COUNTS_PER_SEC = 2_147_483_647   # signed 32-bit max

    def __init__(self):
        self._ser_fl = serial.Serial(port=self.port_fl, baudrate=self.baudrate, timeout=self.timeout)
        self._ser_fr = serial.Serial(port=self.port_fr, baudrate=self.baudrate, timeout=self.timeout)
        self._ser_rl = serial.Serial(port=self.port_rl, baudrate=self.baudrate, timeout=self.timeout)
        self._ser_rr = serial.Serial(port=self.port_rr, baudrate=self.baudrate, timeout=self.timeout)

        # Push PID gains to all four controllers at startup
        for ser in (self._ser_fl, self._ser_fr, self._ser_rl, self._ser_rr):
            self._SetPID(ser)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def UpdateMotorSpeeds(self, MotorSpeed: list) -> None:
        """
        Command all four wheels simultaneously using closed-loop velocity control.

        Args:
            MotorSpeed: [left_speed_m_s, right_speed_m_s]
                        Positive = forward. Clamped to ±max_linear_speed.
                        Converted to encoder counts/sec for cmd 35.
        """
        left_v, right_v = MotorSpeed[0], MotorSpeed[1]

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

    def _SetPID(self, ser: serial.Serial) -> None:
        """
        Send velocity PID gains to one RoboClaw Solo (cmd 28, SetM1VelocityPID).

        Packet format: [addr, 28, D(4), P(4), I(4), QPPS(4), CRC16(2)]
        Each float gain is encoded as uint32 = round(gain * 65536).
        QPPS is an unsigned 32-bit integer (max counts/sec).

        Response: [0xFF]  (ACK)

        Note: gains are stored in RoboClaw EEPROM only if you issue a
        WriteNVM command after this call. Here we set RAM values only --
        they reset on power cycle. This is intentional so code is always
        the source of truth for gains.
        """
        def float_to_u32(f: float) -> int:
            return max(0, min(0xFFFFFFFF, round(f * 65536)))

        d_enc    = float_to_u32(self.pid_D)
        p_enc    = float_to_u32(self.pid_P)
        i_enc    = float_to_u32(self.pid_I)
        qpps_enc = max(0, min(0xFFFFFFFF, int(self.pid_QPPS)))

        payload = (
            d_enc.to_bytes(4, byteorder='big')
            + p_enc.to_bytes(4, byteorder='big')
            + i_enc.to_bytes(4, byteorder='big')
            + qpps_enc.to_bytes(4, byteorder='big')
        )
        packet      = bytes([self._ADR, self._CMD_M1_SET_PID]) + payload
        crc         = self._crc16(packet)
        full_packet = packet + bytes([crc >> 8, crc & 0xFF])

        ser.write(full_packet)
        ack = ser.read(1)
        if ack != b'\xFF':
            print(f"[DC_Driver] PID set NACK on port {ser.port} "
                  f"(P={self.pid_P}, I={self.pid_I}, D={self.pid_D}, QPPS={self.pid_QPPS})")

    def _DriveMotor(self, ser: serial.Serial, velocity: float, direction: int) -> None:
        """
        Send a closed-loop M1 velocity command to one RoboClaw Solo (cmd 35).

        Conversion:
            rev_per_sec    = velocity / (2π * wheel_radius)
            counts_per_sec = rev_per_sec * encoder_CPR_effective
                           = rev_per_sec * (encoder_CPR * 4)

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

        The 4-byte speed is signed counts/sec in the RoboClaw's quadrature
        count domain. Converted back to m/s and direction-corrected so the
        returned value is always in the robot frame (positive = forward).

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