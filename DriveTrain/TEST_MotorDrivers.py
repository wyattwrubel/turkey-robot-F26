import time
import serial

PORTS = {
    "front_left":  "/dev/ttyMotor0",
    "front_right": "/dev/ttyMotor1",
    "rear_left":   "/dev/ttyMotor2",
    "rear_right":  "/dev/ttyMotor3",
}

MOTOR_DIRECTION = {
    "front_left":  -1,
    "front_right":  1,
    "rear_left":   -1,
    "rear_right":   1,
}

BAUDRATE          = 115200
TIMEOUT           = 0.1
ADR               = 0x80
CMD_CURRENT       = 49
CMD_M1_DUTY       = 32
MAX_DUTY_FRACTION = 0.2
MAX_DUTY_VALUE    = 32767

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
    fraction = max(-MAX_DUTY_FRACTION, min(MAX_DUTY_FRACTION, fraction))
    duty = int(direction * fraction * MAX_DUTY_VALUE)
    duty = max(-MAX_DUTY_VALUE, min(MAX_DUTY_VALUE, duty))
    payload = duty.to_bytes(2, "big", signed=True)
    ser.write(make_packet(ADR, CMD_M1_DUTY, payload))
    time.sleep(0.02)
    ack = ser.read(1)
    ser.reset_input_buffer()
    return ack == b"\xff"

mapping = {}  # will store: physical_label -> port

print("\n=== MOTOR MAPPING TEST ===")
print("Each motor will spin forward briefly.")
print("Note which wheel moves and enter its position.\n")

for port_index in range(4):
    port = f"/dev/ttyMotor{port_index}"
    print(f"\n--- Spinning motor on {port} ---")

    try:
        ser = serial.Serial(port=port, baudrate=BAUDRATE, timeout=TIMEOUT)
    except serial.SerialException as e:
        print(f"  ✗ Could not open port: {e}")
        continue

    time.sleep(0.1)
    ser.reset_input_buffer()

    # Spin forward at 20% for 2 seconds (no direction flip yet — raw forward)
    ack = drive_duty(ser, MAX_DUTY_FRACTION, direction=1)
    print(f"  Drive ACK: {'✓' if ack else '✗ NACK'}")
    time.sleep(2.0)
    drive_duty(ser, 0.0)
    ser.close()

    # Ask user which wheel moved
    print("  Which wheel just moved?")
    print("    1) front_left")
    print("    2) front_right")
    print("    3) rear_left")
    print("    4) rear_right")
    print("    s) skip")
    choice = input("  > ").strip().lower()

    label_map = {
        "1": "front_left",
        "2": "front_right",
        "3": "rear_left",
        "4": "rear_right",
    }

    if choice in label_map:
        mapping[label_map[choice]] = port
        print(f"  Mapped {label_map[choice]} -> {port}")
    else:
        print("  Skipped.")

    time.sleep(1.0)

# Print results
print("\n=== MAPPING RESULTS ===")
print("Update your PORTS dict to:\n")
print("PORTS = {")
for label in ["front_left", "front_right", "rear_left", "rear_right"]:
    port = mapping.get(label, "UNKNOWN")
    print(f'    "{label}": "{port}",')
print("}")