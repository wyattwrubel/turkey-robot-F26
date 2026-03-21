"""
test_imu.py — BNO055 live data printer for Raspberry Pi 5

Wiring (JST):
    Red    → Pin 1  (3.3V)
    Black  → Pin 6  (GND)
    Yellow → Pin 3  (SDA)
    Blue   → Pin 5  (SCL)

Run: python3 test_imu.py
"""

import math
import time
import sys


def print_bar(val, max_val=3):
    filled = int((val / max_val) * 10)
    return "█" * filled + "░" * (10 - filled)


try:
    import board
    import busio
    import adafruit_bno055
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install adafruit-circuitpython-bno055 adafruit-blinka")
    sys.exit(1)

print("Connecting to BNO055...")
try:
    i2c    = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_bno055.BNO055_I2C(i2c)
    print("Connected.\n")
except Exception as e:
    print(f"Failed to connect: {e}")
    print("Check wiring and that I2C is enabled (sudo raspi-config)")
    sys.exit(1)

time.sleep(0.5)
print("Reading IMU data — Ctrl+C to stop\n")

def safe_read(fn):
    """Return sensor value or None on any read error."""
    try:
        return fn()
    except Exception:
        return None

try:
    while True:
        euler    = safe_read(lambda: sensor.euler)
        gyro     = safe_read(lambda: sensor.gyro)               # rad/s
        accel    = safe_read(lambda: sensor.linear_acceleration) # m/s² gravity removed
        mag      = safe_read(lambda: sensor.magnetic)            # µT
        gravity  = safe_read(lambda: sensor.gravity)             # m/s²
        quat     = safe_read(lambda: sensor.quaternion)
        temp     = safe_read(lambda: sensor.temperature)         # °C
        cal      = safe_read(lambda: sensor.calibration_status)

        # Heading → radians for UKF
        heading_rad = None
        if euler and euler[0] is not None:
            heading_rad = math.atan2(
                math.sin(math.radians(euler[0])),
                math.cos(math.radians(euler[0]))
            )

        print("\033[H\033[J", end="")  # clear screen

        print("══════════════════════════════════════════════")
        print("  BNO055 Live Data")
        print("══════════════════════════════════════════════")

        # Calibration
        print(f"\n  CALIBRATION")
        if cal:
            s, g, a, m = cal
            print(f"    System  [{print_bar(s)}] {s}/3")
            print(f"    Gyro    [{print_bar(g)}] {g}/3")
            print(f"    Accel   [{print_bar(a)}] {a}/3")
            print(f"    Mag     [{print_bar(m)}] {m}/3")
        else:
            print("    unavailable")

        # Euler / heading
        print(f"\n  EULER ANGLES")
        if euler and euler[0] is not None:
            rad_str = f"  ({heading_rad:+.4f} rad)" if heading_rad is not None else ""
            print(f"    Heading  {euler[0]:>10.3f} deg{rad_str}")
            print(f"    Roll     {euler[1]:>10.3f} deg")
            print(f"    Pitch    {euler[2]:>10.3f} deg")
        else:
            print("    unavailable")

        # Gyroscope
        print(f"\n  GYROSCOPE  (rad/s)")
        if gyro and gyro[0] is not None:
            print(f"    X  {gyro[0]:>10.4f}    Y  {gyro[1]:>10.4f}    Z  {gyro[2]:>10.4f}")
        else:
            print("    unavailable")

        # Linear acceleration
        print(f"\n  LINEAR ACCELERATION  (m/s²)  [gravity removed]")
        if accel and accel[0] is not None:
            print(f"    X  {accel[0]:>10.4f}    Y  {accel[1]:>10.4f}    Z  {accel[2]:>10.4f}")
        else:
            print("    unavailable")

        # Magnetometer
        print(f"\n  MAGNETOMETER  (µT)")
        if mag and mag[0] is not None:
            print(f"    X  {mag[0]:>10.4f}    Y  {mag[1]:>10.4f}    Z  {mag[2]:>10.4f}")
        else:
            print("    unavailable")

        # Gravity vector
        print(f"\n  GRAVITY VECTOR  (m/s²)")
        if gravity and gravity[0] is not None:
            print(f"    X  {gravity[0]:>10.4f}    Y  {gravity[1]:>10.4f}    Z  {gravity[2]:>10.4f}")
        else:
            print("    unavailable")

        # Quaternion
        print(f"\n  QUATERNION")
        if quat and quat[0] is not None:
            print(f"    W  {quat[0]:>10.6f}    X  {quat[1]:>10.6f}")
            print(f"    Y  {quat[2]:>10.6f}    Z  {quat[3]:>10.6f}")
        else:
            print("    unavailable")

        # Temperature
        print(f"\n  TEMPERATURE    {temp} °C" if temp is not None
              else "\n  TEMPERATURE    unavailable")

        print("\n══════════════════════════════════════════════")
        print("  Ctrl+C to stop")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\nStopped.")