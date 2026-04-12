import math
import serial


class DC_Driver:
    """
    RoboClaw packet-serial driver for a 4-wheel differential drive.

    Hardware layout -- 4 RoboClaw Solo 60A units, each on its own serial port,
    each controlling a single motor on M1:

      front_left  : port_fl  (default /dev/ttyACM0)
      front_right : port_fr  (default /dev/ttyACM1)
      rear_left   : port_rl  (default /dev/ttyACM2)
      rear_right  : port_rr  (default /dev/ttyACM3)

    All four units share the same address (0x80) since they are on
    independent serial buses. All communication uses RoboClaw Packet
    Serial mode -- every transmitted packet (including read requests)
    must be terminated with a CRC16-CCITT checksum.

    OPEN LOOP MODE:
    Drive command is cmd 32 (Drive M1 With Signed Duty Cycle).
    Duty is a signed 16-bit int: -32767 to +32767.
    Max physical speed (4.0 m/s) is mapped to 0.3 duty fraction,
    i.e. duty = (v / 4.0) * 0.3 * 32767.

    Left-side motors are physically mirrored so their direction is
    negated in software via motor_directions. Right side is correct.

    Speed reads (AverageLinearSpeed) still use the encoder feedback
    via cmd 18 and convert counts/sec back to m/s. The direction
    sign flip is applied on read as well so returned speeds are
    always in the robot frame (positive = forward).

    ---- CLOSED LOOP (original implementation, kept for reference) ----
    # _CMD_M1_DRIVE_SPEED = 35   -- Drive M1 with signed encoder counts/sec
    #
    # def _DriveMotor_ClosedLoop(self, ser, velocity):
    #     if abs(velocity) > self.max_linear_speed:
    #         velocity = math.copysign(self.max_linear_speed, velocity)
    #     rev_per_sec    = velocity / (2.0 * math.pi * self.wheel_radius)
    #     counts_per_sec = int(rev_per_sec * self.encoder_CPR)
    #     counts_per_sec = max(-2_147_483_648, min(2_147_483_647, counts_per_sec))
    #     payload     = counts_per_sec.to_bytes(4, byteorder='big', signed=True)
    #     packet      = bytes([self._ADR, 35]) + payload
    #     crc         = self._crc16(packet)
    #     full_packet = packet + bytes([crc >> 8, crc & 0xFF])
    #     ser.write(full_packet)
    #     ack = ser.read(1)
    #     if ack != b'\xFF':
    #         print(f"[DC_Driver] NACK on port {ser.port}")
    # ------------------------------------------------------------------
    """

    # Serial ports
    port_fl = "/dev/ttyMotor0"
    port_fr = "/dev/ttyMotor1"
    port_rl = "/dev/ttyMotor2"
    port_rr = "/dev/ttyMotor3"

    baudrate = 115200
    timeout  = 0.1

    # Physical speed limit -- used for input clamping and speed read conversion
    max_linear_speed = 5.0   # m/s
    encoder_CPR      = 1024  # quadrature counts per revolution
    wheel_radius     = 0.08  # metres

    # Open-loop scaling:
    # 0.3 duty fraction corresponds to max_linear_speed (4.0 m/s)
    max_duty_fraction = 1
    max_duty_value    = 32767  # signed 16-bit ceiling for cmd 32

    # Left motors are physically mirrored -- negate their duty and speed reads
    motor_directions = {
        "fl": -1,
        "fr":  1,
        "rl": -1,
        "rr":  1,
    }

    _ADR               = 0x80
    _CMD_M1_DUTY       = 32
    _CMD_M1_READ_SPEED = 18
    _CMD_READ_CURRENTS = 49


    def __init__(self):
        self._ser_fl = serial.Serial(port=self.port_fl, baudrate=self.baudrate, timeout=self.timeout)
        self._ser_fr = serial.Serial(port=self.port_fr, baudrate=self.baudrate, timeout=self.timeout)
        self._ser_rl = serial.Serial(port=self.port_rl, baudrate=self.baudrate, timeout=self.timeout)
        self._ser_rr = serial.Serial(port=self.port_rr, baudrate=self.baudrate, timeout=self.timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def UpdateMotorSpeeds(self, MotorSpeed: list) -> None:
        """
        Command all four wheels simultaneously using open-loop duty control.

        Args:
            MotorSpeed: [left_speed_m_s, right_speed_m_s]
                        Positive = forward. Clamped to +/- max_linear_speed.
                        Scaled to duty: 4.0 m/s -> 0.3 duty fraction -> int 9830.
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
        values are always in the robot frame (positive = forward), matching
        the convention used by UpdateMotorSpeeds.

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

    def _DriveMotor(self, ser: serial.Serial, velocity: float, direction: int) -> None:
        """
        Send an open-loop M1 duty command to one RoboClaw Solo.

        Scaling:
            duty_fraction = (velocity / max_linear_speed) * max_duty_fraction
            duty_int      = direction * duty_fraction * max_duty_value

        At max_linear_speed (4.0 m/s):
            duty_fraction = 1.0 * 0.3 = 0.3
            duty_int      = 0.3 * 32767 = 9830

        Packet format: [addr, 32, duty_BE_signed_16, CRC16_BE]
        Response:      [0xFF]  (ACK)
        """
        velocity  = max(-self.max_linear_speed, min(self.max_linear_speed, velocity))

        duty_fraction = (velocity / self.max_linear_speed) * self.max_duty_fraction
        duty_int      = int(direction * duty_fraction * self.max_duty_value)
        duty_int      = max(-self.max_duty_value, min(self.max_duty_value, duty_int))

        payload     = duty_int.to_bytes(2, byteorder='big', signed=True)
        packet      = bytes([self._ADR, self._CMD_M1_DUTY]) + payload
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

        The 4-byte speed value is a signed two's-complement integer.
        direction is applied to the result so the returned speed is always
        in the robot frame (positive = forward), cancelling out the physical
        mirror flip on the left-side motors.

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
        rev_per_sec    = pulses_per_sec / self.encoder_CPR
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