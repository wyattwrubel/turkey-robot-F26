import smbus2
import struct
import time
import asyncio
import math
import csv
import os
import numpy as np
from dataclasses import dataclass, fields, astuple
from datetime import datetime


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class IMUReading:
    """All 9 DOF values from the BNO055 plus a UTC timestamp."""
    timestamp:      float   # seconds since epoch (time.time())
    # Euler angles (degrees)
    euler_heading:  float
    euler_roll:     float
    euler_pitch:    float
    # Gyroscope (°/s)
    gyro_x:         float
    gyro_y:         float
    gyro_z:         float
    # Accelerometer (m/s²)
    accel_x:        float
    accel_y:        float
    accel_z:        float
    # Magnetometer (µT)
    mag_x:          float
    mag_y:          float
    mag_z:          float

    # Convenience -------------------------------------------------------

    @property
    def heading_rad(self) -> float:
        """Heading wrapped to (-π, π]."""
        r = math.radians(self.euler_heading)
        return math.atan2(math.sin(r), math.cos(r))

    @property
    def iso_timestamp(self) -> str:
        return datetime.utcfromtimestamp(self.timestamp).strftime("%Y-%m-%dT%H:%M:%S.%f")

    def __str__(self) -> str:
        return (
            
            f"[{self.iso_timestamp}] "
            f"Euler(h={self.euler_heading:7.2f}° r={self.euler_roll:7.2f}° p={self.euler_pitch:7.2f}°) | "
            f"Gyro(x={self.gyro_x:7.2f} y={self.gyro_y:7.2f} z={self.gyro_z:7.2f} °/s) | "
            f"Accel(x={self.accel_x:6.3f} y={self.accel_y:6.3f} z={self.accel_z:6.3f} m/s²) | "
            f"Mag(x={self.mag_x:7.2f} y={self.mag_y:7.2f} z={self.mag_z:7.2f} µT)"
        )


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class DataLogger:
    """
    Writes IMUReading rows to a CSV file.

    The file is created at construction time; the header is written once.
    Rows are flushed every `flush_every` calls (default 10) to balance
    I/O overhead against data safety.

    Parameters
    ----------
    path        : file path; defaults to  imu_log_<ISO-date>.csv
    flush_every : rows between fsync flushes
    echo        : if True, also print each reading to stdout
    """

    CSV_HEADER = [f.name for f in fields(IMUReading)]

    def __init__(
        self,
        path: str | None = None,
        flush_every: int = 10,
        echo: bool = True,
    ):
        if path is None:
            date_str = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            path = f"imu_log_{date_str}.csv"

        self.path        = path
        self.flush_every = flush_every
        self.echo        = echo
        self._row_count  = 0

        self._file = open(self.path, "w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.CSV_HEADER)
        print(f"DataLogger: logging to '{os.path.abspath(self.path)}'")

    # ------------------------------------------------------------------

    def log(self, reading: IMUReading) -> None:
        """Write one reading to CSV (and optionally stdout)."""
        self._writer.writerow(astuple(reading))
        self._row_count += 1

        if self._row_count % self.flush_every == 0:
            self._file.flush()

        # if self.echo:
        #     print(reading)

    def close(self) -> None:
        self._file.flush()
        self._file.close()
        print(f"DataLogger: closed after {self._row_count} rows → '{self.path}'")

    # Context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# IMU driver  (public API unchanged)
# ---------------------------------------------------------------------------

class IMUDriver:
    PUBLISH_RATE_HZ = 100
    PUBLISH_PERIOD  = 1.0 / PUBLISH_RATE_HZ  # 0.01 s

    BNO055_ADDR = 0x28

    # Register map
    REG_OPR_MODE = 0x3D
    REG_EULER    = 0x1A   # 6 bytes: heading, roll, pitch  (all little-endian int16, /16 → degrees)
    REG_GYRO     = 0x14   # 6 bytes: x, y, z               (int16, /16 → °/s)
    REG_ACCEL    = 0x08   # 6 bytes: x, y, z               (int16, /100 → m/s²)
    REG_MAG      = 0x0E   # 6 bytes: x, y, z               (int16, /16 → µT)

    MODE_CONFIG = 0x00
    MODE_NDOF   = 0x0C

    def __init__(self, AccelPublisher, Simulation: bool = False, Log: bool = False):
        self.Simulation = Simulation
        self.logger = DataLogger() if (Log and not Simulation) else None

        if not Simulation:
            self.bus = smbus2.SMBus(3)
            self._init_sensor()

        self.AccelPublisher = AccelPublisher

    # ------------------------------------------------------------------
    # Hardware init  (unchanged)
    # ------------------------------------------------------------------

    def _init_sensor(self):
        self.bus.write_byte_data(self.BNO055_ADDR, self.REG_OPR_MODE, self.MODE_CONFIG)
        time.sleep(0.05)
        self.bus.write_byte_data(self.BNO055_ADDR, self.REG_OPR_MODE, self.MODE_NDOF)
        time.sleep(1.0)

    # ------------------------------------------------------------------
    # Sensor reading helpers
    # ------------------------------------------------------------------

    def _read_xyz(self, reg: int, scale: float) -> tuple[float, float, float] | None:
        """Read 3 consecutive little-endian int16s from *reg* and scale them."""
        try:
            data = self.bus.read_i2c_block_data(self.BNO055_ADDR, reg, 6)
        except OSError:
            return None
        x, y, z = struct.unpack('<hhh', bytes(data))
        return x / scale, y / scale, z / scale

    def _get_heading(self) -> float | None:
        """
        Read heading from BNO055 euler angles.
        Returns heading in radians wrapped to (-π, π], or None on read error.
        (Preserved from original API.)
        """
        try:
            data = self.bus.read_i2c_block_data(self.BNO055_ADDR, self.REG_EULER, 2)
            heading = struct.unpack('<h', bytes(data))[0] / 16.0
        except OSError:
            return None

        heading_rad = math.radians(heading)
        return math.atan2(math.sin(heading_rad), math.cos(heading_rad))

    def _get_9dof(self) -> IMUReading | None:
        """
        Read all 9 DOF values in four I²C transactions and return an IMUReading,
        or None if any transaction fails.
        """
        euler = self._read_xyz(self.REG_EULER,  16.0)   # degrees
        gyro  = self._read_xyz(self.REG_GYRO,   16.0)   # °/s
        accel = self._read_xyz(self.REG_ACCEL, 100.0)   # m/s²
        mag   = self._read_xyz(self.REG_MAG,    16.0)   # µT

        if any(v is None for v in (euler, gyro, accel, mag)):
            return None

        return IMUReading(
            timestamp     = time.time(),
            euler_heading = euler[0],
            euler_roll    = euler[1],
            euler_pitch   = euler[2],
            gyro_x        = gyro[0],
            gyro_y        = gyro[1],
            gyro_z        = gyro[2],
            accel_x       = accel[0],
            accel_y       = accel[1],
            accel_z       = accel[2],
            mag_x         = mag[0],
            mag_y         = mag[1],
            mag_z         = mag[2],
        )

    # ------------------------------------------------------------------
    # Publisher loop  (public API unchanged)
    # ------------------------------------------------------------------

    async def Runner(self):
        """
        Continuously reads all 9 DOF values at PUBLISH_RATE_HZ (100 Hz),
        publishes heading to all subscribers (unchanged behaviour), and
        optionally logs the full reading via the DataLogger.
        """
        if self.Simulation:
            return

        while True:
            reading = self._get_9dof()

            if reading is not None:
                # --- original publish (heading only) ---
                pose = np.array([reading.heading_rad])
                await self.AccelPublisher.publish(-1* pose + 26)

                # --- optional full log ---
                if self.logger is not None:
                    self.logger.log(reading)
            else:
                print("IMUDriver: Sensor not ready, skipping publish")

            await asyncio.sleep(self.PUBLISH_PERIOD)