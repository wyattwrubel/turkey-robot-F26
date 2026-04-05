import numpy as np
from scipy.optimize import least_squares
import serial_asyncio
import asyncio
import math
import pynmea2

class PositioningDriver:

    # ------------------------------------------------------------------
    # UWB Configuration
    # ------------------------------------------------------------------

    # TODO: Fill in anchor positions as [x, y] in meters
    # Frame: origin top-left, X=right, Y=down
    UWB_ANCHORS = [
        [0.0, 0.0],   # A0
        [0.0, 0.0],   # A1
        [0.0, 0.0],   # A2
        [0.0, 0.0],   # A3
    ]

    UWB_PORT     = "/dev/ttyUSBx"   # TODO: Fill in ULA1 serial port
    UWB_BAUDRATE = 115200

    # ------------------------------------------------------------------
    # GPS Configuration
    # ------------------------------------------------------------------

    # TODO: Fill in reference coordinate (lat/lon decimal degrees)
    # This should be the top-left (northernmost + westernmost) corner
    # of your operational area — maps to (0, 0) in your local X/Y frame
    GPS_REFERENCE_LAT = 0.0
    GPS_REFERENCE_LON = 0.0

    GPS_PORT     = "/dev/ttyUSBx"   # TODO: Fill in BU-353N serial port
    GPS_BAUDRATE = 4800

    # ------------------------------------------------------------------

    def __init__(self, PositionPublisher, Mode="UWB", Simulation=False):
        """
        Mode: "UWB" for indoor anchor-based positioning
              "GPS" for outdoor GPS positioning
        """
        self.PositionPublisher = PositionPublisher
        self.Mode              = Mode
        self.Simulation        = Simulation
        self._initial_guess    = np.mean(self.UWB_ANCHORS, axis=0)

    async def connect(self):
        if self.Simulation:
            return

        if self.Mode == "UWB":
            self._reader, _ = await serial_asyncio.open_serial_connection(
                url=self.UWB_PORT, baudrate=self.UWB_BAUDRATE
            )
        elif self.Mode == "GPS":
            self._reader, _ = await serial_asyncio.open_serial_connection(
                url=self.GPS_PORT, baudrate=self.GPS_BAUDRATE
            )

    # ------------------------------------------------------------------
    # UWB
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_uwb_line(line: str):
        """
        Parse ULA1 serial output:
          mc <mask> <range0> <range1> <range2> <range3> ...
        Returns list of valid distances in meters, or None on failure.
        Distances are hex-encoded millimeters. Mask bits indicate validity.
        """
        try:
            parts = line.strip().split()
            if len(parts) < 6 or parts[0] != "mc":
                return None

            mask       = int(parts[1], 16)
            raw_ranges = parts[2:6]
            distances  = []

            for i, raw in enumerate(raw_ranges):
                if not (mask >> i & 1):
                    return None  # at least one anchor invalid — skip frame
                dist_m = int(raw, 16) / 1000.0
                distances.append(dist_m)

            return distances

        except (ValueError, IndexError):
            return None

    def _trilaterate(self, distances):
        """
        Least-squares trilateration from 4 anchor distances.
        Output is in the same frame as UWB_ANCHORS (Y-down, origin top-left).
        """
        anchors = np.array(self.UWB_ANCHORS)

        def residuals(pos):
            return np.sqrt(
                (pos[0] - anchors[:, 0])**2 +
                (pos[1] - anchors[:, 1])**2
            ) - np.array(distances)

        result = least_squares(residuals, self._initial_guess)
        self._initial_guess = result.x  # warm-start next solve
        return result.x                 # [x, y] in meters

    # ------------------------------------------------------------------
    # GPS
    # ------------------------------------------------------------------

    def _latlon_to_xy(self, lat, lon):
        """
        Flat-earth approximation: converts lat/lon to local X/Y in meters.
        Origin is GPS_REFERENCE_LAT/LON (top-left corner of map).
        X = East (right), Y = down (south), matching internal Y-down frame.
        Valid for small areas (< ~1 km).
        """
        R = 6_371_000  # Earth radius in meters
        x =  math.radians(lon - self.GPS_REFERENCE_LON) * R * math.cos(math.radians(self.GPS_REFERENCE_LAT))
        y = -math.radians(lat - self.GPS_REFERENCE_LAT) * R  # negated: south = +Y
        return x, y

    @staticmethod
    def _parse_gps_line(line: str):
        """
        Parse NMEA sentence from BU-353N.
        Returns (lat, lon) in decimal degrees, or None if not a fix sentence.
        """
        try:
            msg = pynmea2.parse(line.strip())
            if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                if msg.latitude and msg.longitude:
                    return msg.latitude, msg.longitude
        except pynmea2.ParseError:
            pass
        return None

    # ------------------------------------------------------------------
    # Runner loops
    # ------------------------------------------------------------------

    async def _run_uwb(self):
        while True:
            try:
                line = await self._reader.readline()
                line = line.decode('utf-8').strip()
                if not line:
                    continue

                distances = self._parse_uwb_line(line)
                if distances is None:
                    continue

                x, y = self._trilaterate(distances)
                pose = np.array([x, y])
                await self.PositionPublisher.publish(pose)

            except Exception as e:
                print(f"PositioningDriver (UWB): {e}")
                await asyncio.sleep(0.1)

    async def _run_gps(self):
        while True:
            try:
                line = await self._reader.readline()
                line = line.decode('utf-8', errors='ignore').strip()
                if not line:
                    continue

                result = self._parse_gps_line(line)
                if result is None:
                    continue

                lat, lon = result
                x, y     = self._latlon_to_xy(lat, lon)
                pose     = np.array([x, y])
                await self.PositionPublisher.publish(pose)

            except Exception as e:
                print(f"PositioningDriver (GPS): {e}")
                await asyncio.sleep(0.1)

    async def Runner(self):
        """Start the positioning loop based on Mode."""
        if self.Simulation:
            return

        await self.connect()

        if self.Mode == "UWB":
            await self._run_uwb()
        elif self.Mode == "GPS":
            await self._run_gps()
        else:
            raise ValueError(f"PositioningDriver: Unknown mode '{self.Mode}'")