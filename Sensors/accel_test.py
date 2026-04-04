import smbus2
import time
import struct

bus = smbus2.SMBus(3)
BNO055_ADDR = 0x28

def init_bno055():
    # Set to CONFIG mode first
    bus.write_byte_data(BNO055_ADDR, 0x3D, 0x00)
    time.sleep(0.05)
    # Set to NDOF fusion mode (gives Euler angles)
    bus.write_byte_data(BNO055_ADDR, 0x3D, 0x0C)
    time.sleep(0.7)  # mode switch needs time to settle

def read_euler():
    # Euler angles at registers 0x1A-0x1F (heading, roll, pitch)
    data = bus.read_i2c_block_data(BNO055_ADDR, 0x1A, 6)
    heading, roll, pitch = struct.unpack('<hhh', bytes(data))
    # Scale factor is 1/16 degrees per LSB
    return heading / 16.0, roll / 16.0, pitch / 16.0

def read_calibration():
    # Calibration status: 0=uncalibrated, 3=fully calibrated
    status = bus.read_byte_data(BNO055_ADDR, 0x35)
    sys  = (status >> 6) & 0x03
    gyro = (status >> 4) & 0x03
    accel= (status >> 2) & 0x03
    mag  = status & 0x03
    return sys, gyro, accel, mag

print("Initializing BNO055...")
init_bno055()
print("Running — Ctrl+C to stop\n")

try:
    while True:
        heading, roll, pitch = read_euler()
        sys, gyro, accel, mag = read_calibration()
        print(f"Heading: {heading:7.2f}°  Roll: {roll:7.2f}°  Pitch: {pitch:7.2f}°  "
              f"| Cal sys:{sys} gyro:{gyro} accel:{accel} mag:{mag}")
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nDone.")