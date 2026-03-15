import board
import busio
import adafruit_bno055
import asyncio
import math
import numpy as np

class IMUDriver:
    PUBLISH_RATE_HZ  = 100
    PUBLISH_PERIOD   = 1.0 / PUBLISH_RATE_HZ   # 0.01s

    def __init__(self, AccelPublisher, Simulation=False):
        if(Simulation == False):
            i2c         = busio.I2C(board.SCL, board.SDA)
            self.sensor = adafruit_bno055.BNO055_I2C(i2c)
        self.AccelPublisher = AccelPublisher

    # ------------------------------------------------------------------
    # Sensor reading
    # ------------------------------------------------------------------

    def _get_heading(self):
        """
        Read heading from BNO055 euler angles.
        Returns heading in radians wrapped to (-π, π], or None if not ready.
        """
        euler = self.sensor.euler
        if euler is None or euler[0] is None:
            return None

        heading_deg = euler[0]                  # 0-360 degrees, clockwise from North
        heading_rad = math.radians(heading_deg)
        return math.atan2(                      # wrap to (-π, π]
            math.sin(heading_rad),
            math.cos(heading_rad)
        )

    def is_calibrated(self) -> bool:
        """
        Returns True when BNO055 is fully calibrated.
        Check this before starting mission.
        """
        cal = self.sensor.calibration_status
        if cal is None:
            return False
        sys, gyro, accel, mag = cal
        return sys == 3 and gyro == 3 and mag == 3

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    async def wait_for_calibration(self):
        """
        Block until BNO055 is fully calibrated.
        Call once at mission start — move robot in figure-8 pattern.
        """
        print("Calibrating BNO055 — move robot slowly in figure-8 pattern")
        while not self.is_calibrated():
            cal = self.sensor.calibration_status
            if cal:
                print(f"  sys={cal[0]} gyro={cal[1]} accel={cal[2]} mag={cal[3]}")
            await asyncio.sleep(0.5)
        print("BNO055 fully calibrated — ready")

    # ------------------------------------------------------------------
    # Publisher loop
    # ------------------------------------------------------------------

    async def run(self):
        """
        Continuously reads heading at PUBLISH_RATE_HZ (100 Hz)and
        publishes to all subscribers.
        """
        await self.wait_for_calibration()

        while True:
            heading = self._get_heading()

            if heading is not None:
                pose = np.array([heading])      # (1,) — matches update_accel signature
                await self.AccelPublisher.publish(pose)
            else:
                print("IMUDriver: Sensor not ready, skipping publish")

            await asyncio.sleep(self.PUBLISH_PERIOD)