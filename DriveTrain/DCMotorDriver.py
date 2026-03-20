import numpy as np
import serial

class DC_Driver:
    # port = "/dev/ttyACM0"
    port = "/tmp/vserial0"
    baudrate = 115200

    # UPDATE THIS WITH REAL PARAMS TODO
    max_linear_speed = 5 # m/s
    encoder_CPR = 1024 # counts/rev
    wheel_radius = 0.08 # m

    front_left_adr = 0x81
    front_right_adr = 0x82
    rear_left_adr = 0x83
    rear_right_adr = 0x84

    def __init__(self):
        self.DCSerial = serial.Serial(port=self.port, baudrate= self.baudrate, timeout=1)
        
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def UpdateMotorSpeeds(self, MotorSpeed):
        self._DriveMotor(address=self.front_left_adr, velocity=MotorSpeed[0])
        self._DriveMotor(address=self.rear_left_adr, velocity=MotorSpeed[0])
        self._DriveMotor(address=self.front_right_adr, velocity=MotorSpeed[1])
        self._DriveMotor(address=self.rear_right_adr, velocity=MotorSpeed[1])

    def AverageLinearSpeed(self):
        left_vals = [self._ReadMotorSpeed(self.front_left_adr),
                    self._ReadMotorSpeed(self.rear_left_adr)]
        right_vals = [self._ReadMotorSpeed(self.front_right_adr),
                    self._ReadMotorSpeed(self.rear_right_adr)]

        if None in left_vals + right_vals:
            return None, None

        left_speed = sum(left_vals) / 2
        right_speed = sum(right_vals) / 2
        return left_speed, right_speed

    # ------------------------------------------------------------------
    # MotorDriver Interfacing
    # ------------------------------------------------------------------
    def _DriveMotor(self, address: int, velocity: float):
        """Drive M1 with signed linear velocity (m/s)"""
        DRIVE_CMD = 35

        # Clip velocity
        if velocity > self.max_linear_speed or velocity < -self.max_linear_speed:
            print(f"Speed command of {velocity} m/s out of bounds")
            return

        # Convert linear velocity (m/s) → encoder counts/sec
        rev_per_sec = velocity / (2 * np.pi * self.wheel_radius)
        counts_per_sec = int(rev_per_sec * self.encoder_CPR)

        # Clip to signed 32-bit
        counts_per_sec = max(-2_147_483_648, min(2_147_483_647, counts_per_sec))
        
        speed_bytes = counts_per_sec.to_bytes(4, byteorder='big', signed=True)
        packet = bytes([address, DRIVE_CMD]) + speed_bytes
        check_sum = self._crc16_ccitt(packet)
        packet_with_crc = packet + bytes([check_sum >> 8, check_sum & 0xFF])

        self.DCSerial.write(packet_with_crc)

        ack = self.DCSerial.read(1)
        if ack != b'\xFF':
            print(address, " Motor Speed NACK")

    def _ReadMotorSpeed(self, address):
        """ Returns outputted linear speed"""
        READ_SPEED_CMD = 18
        packet = bytes([address, READ_SPEED_CMD])
        self.DCSerial.write(packet)

        # Receive 7 bytes: Speed(4), Status(1), CRC16(2)
        response = self.DCSerial.read(7)

        speed_bytes = response[0:4]
        status = response[4]
        received_crc_bytes = response[5:7]
        crc_calc = self._crc16_ccitt(response[:5])
        crc_bytes = bytes([crc_calc >> 8, crc_calc & 0xFF])
        if received_crc_bytes != crc_bytes:
            print(address, " Invalid Crc Received")
            return None

        pulses_per_sec = int.from_bytes(speed_bytes, byteorder='big', signed=True)

        # Convert to linear speed (m/s)
        rev_per_sec = pulses_per_sec / self.encoder_CPR
        linear_speed = rev_per_sec * (2 * np.pi * self.wheel_radius)

        if status == 1:
            linear_speed = -linear_speed

        return linear_speed

    def ReadMotorCurrent(self, address):
        # TODO: This is new interface, make helper function to read. Returns all 4 currents in Amps


        """ Returns measured motor currents """

        MOTORCURRENT_CMD = 49
        packet = bytes([address, MOTORCURRENT_CMD])

        self.DCSerial.write(packet)

        # Read 6 bytes: M1(2), M2(2), CRC16(2)
        Current_Readings = self.DCSerial.read(6)

        # Validate CRC
        received_crc_bytes = Current_Readings[4:6]
        crc = self._crc16_ccitt(Current_Readings[:4])
        crc_bytes = bytes([crc >> 8, crc & 0xFF])
        if received_crc_bytes != crc_bytes:
            print(address, " Invalid Current Data Received")
            return None

        m1_counts = int.from_bytes(Current_Readings[0:2], byteorder='big')
        m1_amps = m1_counts / 100.0
        return m1_amps
    
        # 16 Bit Checksum
    
    def _crc16_ccitt(packet: bytes) -> int:
        """ Check Sum Protocol Per MotorDriver Manual """
        crc = 0
        for byte in packet:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ 0x1021) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc