import signal
import serial
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

    UWB_ANCHORS = [
        [0.0, 0.0],   # A0
        [0.0, 0.0],   # A1
        [0.0, 0.0],   # A2
        [0.0, 0.0],   # A3
    ]

    UWB_PORT     = "/dev/ttyUSB0"
    UWB_BAUDRATE = 115200

    # ------------------------------------------------------------------
    # GPS Configuration
    # ------------------------------------------------------------------

    GPS_REFERENCE_LAT = 42.2947500000   # (0,0) NW corner
    GPS_REFERENCE_LON = -83.7126851930  # (0,0) NW corner

    GPS_PORT     = "/dev/ttyUSB1"
    GPS_BAUDRATE = 4800

    # ------------------------------------------------------------------
    # Zone bounds (meters) — must match Zone.csv grid dimensions
    # Grid is 50 cols (X) x 82 rows (Y)
    # ------------------------------------------------------------------
    ZONE_X_MIN = 0.0
    ZONE_X_MAX = 50.0
    ZONE_Y_MIN = 0.0
    ZONE_Y_MAX = 82.0

    ZONE_SLACK_M = 5.0

    # ------------------------------------------------------------------

    def __init__(self, PositionPublisher, Mode="UWB", Simulation=False, Verbose=False):
        """
        Mode:    "UWB" for indoor anchor-based positioning
                 "GPS" for outdoor GPS positioning
        Verbose: If True, prints every raw NMEA line, estimated position,
                 and whether it was sent or dropped due to out-of-bounds.
        """
        self.PositionPublisher = PositionPublisher
        self.Mode              = Mode
        self.Simulation        = Simulation
        self.Verbose           = Verbose
        self._initial_guess    = np.mean(self.UWB_ANCHORS, axis=0)
        self._ser              = None

        self._ref_lat = self.GPS_REFERENCE_LAT
        self._ref_lon = self.GPS_REFERENCE_LON

        print(f"[PositioningDriver] Mode={Mode}  Verbose={Verbose}  Simulation={Simulation}")
        print(f"[PositioningDriver] GPS origin (0,0): lat={self._ref_lat}, lon={self._ref_lon}")
        print(f"[PositioningDriver] Zone bounds: X=[{self.ZONE_X_MIN}, {self.ZONE_X_MAX}]m  Y=[{self.ZONE_Y_MIN}, {self.ZONE_Y_MAX}]m")

    async def connect(self):
        if self.Mode == "UWB":
            self._reader, _ = await serial_asyncio.open_serial_connection(
                url=self.UWB_PORT, baudrate=self.UWB_BAUDRATE
            )
            print(f"[PositioningDriver] UWB connected on {self.UWB_PORT}")

        elif self.Mode == "GPS":
            # Plain serial — matches what works in the test script
            self._ser = serial.Serial(self.GPS_PORT, baudrate=self.GPS_BAUDRATE, timeout=2)
            print(f"[PositioningDriver] GPS connected on {self.GPS_PORT} at {self.GPS_BAUDRATE} baud")

    # ------------------------------------------------------------------
    # Zone bounds check
    # ------------------------------------------------------------------

    def _check_bounds(self, x, y):
        x_lo = self.ZONE_X_MIN - self.ZONE_SLACK_M
        x_hi = self.ZONE_X_MAX + self.ZONE_SLACK_M
        y_lo = self.ZONE_Y_MIN - self.ZONE_SLACK_M
        y_hi = self.ZONE_Y_MAX + self.ZONE_SLACK_M

        in_bounds = (x_lo <= x <= x_hi and y_lo <= y <= y_hi)

        if not in_bounds:
            print(
                f"[PositioningDriver] OUT OF BOUNDS — estimated ({x:.2f}, {y:.2f})m  "
                f"zone X=[{self.ZONE_X_MIN},{self.ZONE_X_MAX}] "
                f"Y=[{self.ZONE_Y_MIN},{self.ZONE_Y_MAX}] "
                f"(+{self.ZONE_SLACK_M}m slack) — frame dropped"
            )
            return False

        if self.Verbose:
            print(f"[PositioningDriver] Estimated ({x:.2f}, {y:.2f})m — sending")

        return True

    # ------------------------------------------------------------------
    # UWB
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_uwb_line(line: str):
        try:
            parts = line.strip().split()
            if len(parts) < 6 or parts[0] != "mc":
                return None

            mask       = int(parts[1], 16)
            raw_ranges = parts[2:6]
            distances  = []

            for i, raw in enumerate(raw_ranges):
                if not (mask >> i & 1):
                    return None
                dist_m = int(raw, 16) / 1000.0
                distances.append(dist_m)

            return distances

        except (ValueError, IndexError):
            return None

    def _trilaterate(self, distances):
        anchors = np.array(self.UWB_ANCHORS)

        def residuals(pos):
            return np.sqrt(
                (pos[0] - anchors[:, 0])**2 +
                (pos[1] - anchors[:, 1])**2
            ) - np.array(distances)

        result = least_squares(residuals, self._initial_guess)
        self._initial_guess = result.x
        return result.x

    # ------------------------------------------------------------------
    # GPS
    # ------------------------------------------------------------------

    def _latlon_to_xy(self, lat, lon):
        """
        Flat-earth approximation: converts lat/lon to local X/Y in meters.
        Origin is GPS_REFERENCE_LAT/LON = grid (0, 0) = NW / top-left corner.
        X = East (right), Y = South (down) — matches internal Y-down frame.

        """
        R = 6_371_000
        x =  math.radians(lon - self._ref_lon) * R * math.cos(math.radians(self._ref_lat))
        y = -math.radians(lat - self._ref_lat) * R
        return x, y

    @staticmethod
    def _parse_gps_line(line: str):
        """
        Parse NMEA sentence from BU-353S4 (pl2303, 4800 baud).
        Note: fix quality checks disabled for indoor testing.
        Re-enable before outdoor deployment.
        """
        try:
            msg = pynmea2.parse(line.strip())

            # --- disabled for indoor testing, re-enable for outdoor use ---
            # if hasattr(msg, 'gps_qual'):
            #     if msg.gps_qual == 0:
            #         return None
            # if hasattr(msg, 'status'):
            #     if msg.status != 'A':
            #         return None

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

                if self.Verbose:
                    print(f"[PositioningDriver UWB RAW] {line}")

                distances = self._parse_uwb_line(line)
                if distances is None:
                    continue

                x, y = self._trilaterate(distances)

                if not self._check_bounds(x, y):
                    continue

                pose = np.array([x, y])
                await self.PositionPublisher.publish(pose)
                if self.Verbose:
                    print(f"[PositioningDriver] Sent ({x:.2f}, {y:.2f})m")

            except asyncio.CancelledError:
                print("[PositioningDriver UWB] Shutting down.")
                raise
            except Exception as e:
                print(f"[PositioningDriver UWB] {e}")
                await asyncio.sleep(0.1)

    async def _run_gps(self):
        loop = asyncio.get_running_loop()
        print("[PositioningDriver GPS] Loop started")
        while True:
            try:
                # Run blocking readline in a thread so it doesn't block the event loop
                raw = await loop.run_in_executor(None, self._ser.readline)
                line = raw.decode('ascii', errors='replace').strip()

                if not line:
                    continue

                if self.Verbose:
                    print(f"[PositioningDriver GPS RAW] {line}")

                result = self._parse_gps_line(line)
                if result is None:
                    continue

                lat, lon = result
                x, y     = self._latlon_to_xy(lat, lon)

                if not self._check_bounds(x, y):
                    continue

                pose = np.array([x, y])
                await self.PositionPublisher.publish(pose)
                if self.Verbose:
                    print(f"[PositioningDriver] Sent ({x:.2f}, {y:.2f})m")

            except asyncio.CancelledError:
                print("[PositioningDriver GPS] Shutting down.")
                if self._ser and self._ser.is_open:
                    self._ser.close()
                raise
            except Exception as e:
                print(f"[PositioningDriver GPS] {e}")
                await asyncio.sleep(0.1)

    async def Runner(self):
        """Start the positioning loop based on Mode."""
        if self.Simulation:
            print("[PositioningDriver] Simulation mode — not connecting to hardware.")
            return

        print("[PositioningDriver] Runner() started — connecting to hardware...")
        try:
            await self.connect()
        except Exception as e:
            print(f"[PositioningDriver] connect() failed: {e}")
            return

        if self.Mode == "UWB":
            await self._run_uwb()
        elif self.Mode == "GPS":
            await self._run_gps()
        else:
            raise ValueError(f"PositioningDriver: Unknown mode '{self.Mode}'")