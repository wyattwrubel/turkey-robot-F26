import smbus2
import struct
import time
import asyncio
import math
import numpy as np

class IMUDriver:
    PUBLISH_RATE_HZ = 100
    PUBLISH_PERIOD  = 1.0 / PUBLISH_RATE_HZ  # 0.01s

    BNO055_ADDR     = 0x28
    REG_OPR_MODE    = 0x3D
    REG_EULER       = 0x1A
    MODE_CONFIG     = 0x00
    MODE_NDOF       = 0x0C

    def __init__(self, AccelPublisher, Simulation=False):
        self.Simulation = Simulation
        if not Simulation:
            self.bus = smbus2.SMBus(3)
            self._init_sensor()
        self.AccelPublisher = AccelPublisher

    def _init_sensor(self):
        self.bus.write_byte_data(self.BNO055_ADDR, self.REG_OPR_MODE, self.MODE_CONFIG)
        time.sleep(0.05)
        self.bus.write_byte_data(self.BNO055_ADDR, self.REG_OPR_MODE, self.MODE_NDOF)
        time.sleep(1.0)

    # ------------------------------------------------------------------
    # Sensor reading
    # ------------------------------------------------------------------

    def _get_heading(self):
        """
        Read heading from BNO055 euler angles.
        Returns heading in radians wrapped to (-π, π], or None on read error.
        """
        try:
            data = self.bus.read_i2c_block_data(self.BNO055_ADDR, self.REG_EULER, 2)
            heading = struct.unpack('<h', bytes(data))[0] / 16.0
        except OSError:
            return None

        heading_rad = math.radians(heading)
        return math.atan2(math.sin(heading_rad), math.cos(heading_rad))

    # ------------------------------------------------------------------
    # Publisher loop
    # ------------------------------------------------------------------

    async def Runner(self):
        """
        Continuously reads heading at PUBLISH_RATE_HZ (100 Hz) and
        publishes to all subscribers.
        """
        if(self.Simulation):
            return

        while True:
            heading = self._get_heading()

            if heading is not None:
                pose = np.array([heading])  # (1,) — matches update_accel signature
                await self.AccelPublisher.publish(pose)
            else:
                print("IMUDriver: Sensor not ready, skipping publish")

            await asyncio.sleep(self.PUBLISH_PERIOD)