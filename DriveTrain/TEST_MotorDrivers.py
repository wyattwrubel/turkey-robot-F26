import math
import time
import serial

PORTS = {

    "front_left":  "/dev/ttyACM0",
    "front_right": "/dev/ttyACM1",
    "rear_left":   "/dev/ttyACM2",
    "rear_right":  "/dev/ttyACM3",
    # "front_left":  "COM13",
    # "front_right": "COM14",
    # "rear_left":   "COM15",
    # "rear_right":  "COM16",
}

# Left side motors are physically mirrored, so negate them
MOTOR_DIRECTION = {
    "front_left":  -1,
    "front_right":  1,
    "rear_left":   -1,
    "rear_right":   1,
}

BAUDRATE    = 115200
TIMEOUT     = 0.1
ADR         = 0x80
CMD_CURRENT = 49
CMD_M1_DUTY = 32
RUN_TIME    = 3.0

MAX_DUTY_FRACTION = 0.2
MAX_DUTY_VALUE    = 32767
TARGET_DUTY       = int(MAX_DUTY_FRACTION * MAX_DUTY_VALUE)

def crc16(data: bytes) -> int:
    crc = 0x0000
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc

def make_packet(addr, cmd, payload=b""):
    base = bytes([addr, cmd]) + payload
    c = crc16(base)
    return base + bytes([c >> 8, c & 0xFF])

def drive_duty(ser, fraction, direction=1):
    """
    fraction : -1.0 to +1.0, clamped to MAX_DUTY_FRACTION
    direction: 1 or -1, applied before sending to flip motor orientation
    """
    fraction = max(-MAX_DUTY_FRACTION, min(MAX_DUTY_FRACTION, fraction))
    duty = int(direction * fraction * MAX_DUTY_VALUE)
    duty = max(-MAX_DUTY_VALUE, min(MAX_DUTY_VALUE, duty))
    payload = duty.to_bytes(2, "big", signed=True)
    ser.write(make_packet(ADR, CMD_M1_DUTY, payload))
    time.sleep(0.02)
    ack = ser.read(1)
    ser.reset_input_buffer()
    return ack == b"\xff"

def read_current(ser):
    req = bytes([ADR, CMD_CURRENT])
    ser.reset_input_buffer()
    ser.write(make_packet(ADR, CMD_CURRENT))
    time.sleep(0.02)
    r = ser.read(6)
    if len(r) < 6:
        return None
    if crc16(req + r[:4]) != int.from_bytes(r[4:6], "big"):
        return None
    return int.from_bytes(r[:2], "big") / 100.0

for name, port in PORTS.items():
    direction = MOTOR_DIRECTION[name]
    print(f"\n{'='*50}")
    print(f"  Testing: {name.upper()}  ({port})  direction={'NORMAL' if direction == 1 else 'REVERSED'}")
    print(f"{'='*50}")

    try:
        ser = serial.Serial(port=port, baudrate=BAUDRATE, timeout=TIMEOUT)
    except serial.SerialException as e:
        print(f"  ✗ Could not open port: {e}")
        continue

    time.sleep(0.1)
    ser.reset_input_buffer()

    cur = read_current(ser)
    print(f"  [at rest]  current={cur}A")

    ack = drive_duty(ser, MAX_DUTY_FRACTION, direction)
    print(f"  Drive ACK ({int(MAX_DUTY_FRACTION*100)}% duty, value={TARGET_DUTY * direction}): {'✓' if ack else '✗ NACK'}")

    for i in range(int(RUN_TIME / 0.5)):
        time.sleep(0.5)
        cur = read_current(ser)
        print(f"  [t={0.5*(i+1):.1f}s]  current={cur}A")

    drive_duty(ser, 0.0, direction)
    time.sleep(0.3)
    cur = read_current(ser)
    print(f"  [stopped]  current={cur}A")

    ser.close()
    print(f"  Port closed. Waiting 1s before next motor...")
    time.sleep(1.0)

print("\n=== All motors tested ===")