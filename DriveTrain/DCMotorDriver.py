import math
import serial


class DC_Driver:
    """
    RoboClaw packet-serial driver for a 4-wheel differential drive.

    Hardware layout -- 4 RoboClaw units, each on its own serial port,
    each controlling a single motor on M1:

      front_left  : port_fl  (default /tmp/vserial0)
      front_right : port_fr  (default /tmp/vserial1)
      rear_left   : port_rl  (default /tmp/vserial2)
      rear_right  : port_rr  (default /tmp/vserial3)

    All four units share the same address (0x80) since they are on
    independent serial buses. All communication uses RoboClaw Packet
    Serial mode -- every transmitted packet (including read requests)
    must be terminated with a CRC16-CCITT checksum.

    Right-side encoder direction is inverted in Motion Studio firmware,
    so no software sign-flip is needed anywhere in this driver.
    """

    # Serial ports -- one per motor controller TODO update to real ports
    port_fl  = "/dev/ttyACM0"
    port_fr  = "/dev/ttyACM1"
    port_rl  = "/dev/ttyACM2"
    port_rr  = "/dev/ttyACM3"

    baudrate = 115200
    timeout  = 0.05   # seconds -- keep short for real-time control

    # Physical parameters -- TODO update for your platform
    max_linear_speed = 4.0   # m/s (hard clip, both directions)
    encoder_CPR      = 1024  # quadrature counts per revolution
    wheel_radius     = 0.08  # metres

    # Each unit has its own bus so they can all share the same address
    _ADR = 0x80

    # RoboClaw packet-serial command numbers (M1 only)
    _CMD_M1_DRIVE_SPEED = 35
    _CMD_M1_READ_SPEED  = 18
    _CMD_READ_CURRENTS  = 49

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
        Command all four wheels simultaneously.

        Args:
            MotorSpeed: [left_speed_m_s, right_speed_m_s]
                        Positive = forward.
        """
        left_v, right_v = MotorSpeed[0], MotorSpeed[1]

        # Encoder direction is inverted in Motion Studio firmware for the
        # right-side units, so no software negation is needed here.
        self._DriveMotor(self._ser_fl, left_v)
        self._DriveMotor(self._ser_fr, right_v)
        self._DriveMotor(self._ser_rl, left_v)
        self._DriveMotor(self._ser_rr, right_v)

    def AverageLinearSpeed(self):
        """
        Read back encoder speeds and average each side.

        Returns:
            (left_speed_m_s, right_speed_m_s)
            Either value is None if any read on that side failed.
        """
        fl = self._ReadMotorSpeed(self._ser_fl)
        fr = self._ReadMotorSpeed(self._ser_fr)
        rl = self._ReadMotorSpeed(self._ser_rl)
        rr = self._ReadMotorSpeed(self._ser_rr)

        # No negation needed -- encoder direction inversion is handled
        # in Motion Studio firmware for the right-side units.
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

    def _DriveMotor(self, ser: serial.Serial, velocity: float) -> None:
        """
        Send an M1 speed command to one RoboClaw over the given port.

        Packet format: [addr, 35, speed_BE_signed_32, CRC16_BE]
        Response:      [0xFF]  (ACK)
        """
        if abs(velocity) > self.max_linear_speed:
            print(f"[DC_Driver] Speed {velocity:.3f} m/s exceeds limit, clamping.")
            velocity = math.copysign(self.max_linear_speed, velocity)

        # m/s -> encoder counts/sec
        rev_per_sec    = velocity / (2.0 * math.pi * self.wheel_radius)
        counts_per_sec = int(rev_per_sec * self.encoder_CPR)
        counts_per_sec = max(-2_147_483_648, min(2_147_483_647, counts_per_sec))

        payload     = counts_per_sec.to_bytes(4, byteorder='big', signed=True)
        packet      = bytes([self._ADR, self._CMD_M1_DRIVE_SPEED]) + payload
        crc         = self._crc16(packet)
        full_packet = packet + bytes([crc >> 8, crc & 0xFF])

        ser.write(full_packet)

        ack = ser.read(1)
        if ack != b'\xFF':
            print(f"[DC_Driver] NACK on port {ser.port}")

    def _ReadMotorSpeed(self, ser: serial.Serial):
        """
        Read M1 encoder speed from one RoboClaw over the given port.

        Request packet:  [addr, 18, CRC16_BE]
        Response:        [speed_BE_signed_32 (4), status (1), CRC16_BE (2)]  = 7 bytes

        The 4-byte speed value is a signed two's-complement integer — direction
        is already encoded in its sign. The status byte bit 0 indicates the motor
        is decelerating, NOT direction. No sign correction from status is needed.
        """
        req_packet = bytes([self._ADR, self._CMD_M1_READ_SPEED])
        crc        = self._crc16(req_packet)
        full_req   = req_packet + bytes([crc >> 8, crc & 0xFF])

        ser.write(full_req)

        response = ser.read(7)
        if len(response) < 7:
            print(f"[DC_Driver] Short read (speed) on port {ser.port}: got {len(response)} bytes")
            return None

        recv_crc = int.from_bytes(response[5:7], byteorder='big')
        calc_crc = self._crc16(response[:5])
        if recv_crc != calc_crc:
            print(f"[DC_Driver] CRC mismatch (speed) on port {ser.port}")
            return None

        # Speed is a signed 32-bit value -- direction is in the sign, not the status byte
        pulses_per_sec = int.from_bytes(response[0:4], byteorder='big', signed=True)

        rev_per_sec  = pulses_per_sec / self.encoder_CPR
        linear_speed = rev_per_sec * (2.0 * math.pi * self.wheel_radius)

        return linear_speed

    def _ReadCurrent(self, ser: serial.Serial):
        """
        Read M1 current from one RoboClaw over the given port (cmd 49).

        Request packet:  [addr, 49, CRC16_BE]
        Response:        [M1_counts (2), M2_counts (2), CRC16_BE (2)]  = 6 bytes

        The controller always returns both M1 and M2 counts; we take only M1.
        Units: counts / 100.0 = Amps.

        Returns M1 current in Amps, or None on failure.
        """
        req_packet = bytes([self._ADR, self._CMD_READ_CURRENTS])
        crc        = self._crc16(req_packet)
        full_req   = req_packet + bytes([crc >> 8, crc & 0xFF])

        ser.write(full_req)

        response = ser.read(6)
        if len(response) < 6:
            print(f"[DC_Driver] Short read (current) on port {ser.port}: got {len(response)} bytes")
            return None

        recv_crc = int.from_bytes(response[4:6], byteorder='big')
        calc_crc = self._crc16(response[:4])
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