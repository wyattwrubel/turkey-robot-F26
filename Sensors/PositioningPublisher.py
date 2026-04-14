import serial
import numpy as np
import asyncio
import nest_asyncio
import math
import pynmea2


# ---------------------------------------------------------------------------
# PositioningDriver  (GPS only)
# ---------------------------------------------------------------------------

class PositioningDriver:

    # ------------------------------------------------------------------
    # GPS Configuration
    # ------------------------------------------------------------------

    # Derived from observed fix (42.294042, -83.712617) = zone bottom-left ≈ (1, 79)
    # ref = NW corner (0, 0) of the zone grid
    GPS_REFERENCE_LAT = 42.29475348   # NW corner derived from bottom-left fix
    GPS_REFERENCE_LON = -83.71263234

    GPS_PORT     = "/dev/ttyGPS"
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

    def __init__(self, PositionPublisher, Simulation=False, Verbose=False):
        """
        Verbose: If True, prints lat/lon, local X/Y, fix count, and bound status.
        """
        self.PositionPublisher = PositionPublisher
        self.Simulation        = Simulation
        self.Verbose           = Verbose
        self._ser              = None
        self._fix_count        = 0

        self._ref_lat = self.GPS_REFERENCE_LAT
        self._ref_lon = self.GPS_REFERENCE_LON

        print(f"[GPS] Port={self.GPS_PORT}  Baud={self.GPS_BAUDRATE}  "
              f"Simulation={Simulation}")
        print(f"[GPS] Origin: lat={self._ref_lat:.8f}  lon={self._ref_lon:.8f}")
        print(f"[GPS] Zone: X=[{self.ZONE_X_MIN}, {self.ZONE_X_MAX}]m  "
              f"Y=[{self.ZONE_Y_MIN}, {self.ZONE_Y_MAX}]m  "
              f"(±{self.ZONE_SLACK_M}m slack)")

        if not self.Simulation:
            nest_asyncio.apply()
            asyncio.get_event_loop().run_until_complete(self.connect())

    async def connect(self):
        self._ser = serial.Serial(
            self.GPS_PORT, baudrate=self.GPS_BAUDRATE, timeout=2
        )
        print(f"[GPS] Connected on {self.GPS_PORT} at {self.GPS_BAUDRATE} baud")

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
            print(f"[GPS] OUT OF BOUNDS  xy=({x:.2f}, {y:.2f})m  fix #{self._fix_count} dropped")
            return False

        if self.Verbose:
            print(f"[GPS] fix #{self._fix_count:04d}  xy=({x:.2f}, {y:.2f})m  ✓ in bounds")

        return True

    # ------------------------------------------------------------------
    # GPS math
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
        Parse NMEA sentence from BU-353N5 (CP210x, 4800 baud).
        Note: fix quality checks disabled for indoor testing.
        Re-enable before outdoor deployment.
        """
        try:
            msg = pynmea2.parse(line.strip())

            # --- re-enable for outdoor deployment ---
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
    # Runner loop
    # ------------------------------------------------------------------

    async def _run_gps(self):
        loop = asyncio.get_running_loop()
        print("[GPS] Listening for NMEA sentences...")
        while True:
            try:
                raw  = await loop.run_in_executor(None, self._ser.readline)
                line = raw.decode('ascii', errors='replace').strip()

                if not line:
                    continue

                # if self.Verbose:
                #     print(f"[GPS RAW] {line}")

                result = self._parse_gps_line(line)
                if result is None:
                    continue

                lat, lon = result
                x, y     = self._latlon_to_xy(lat, lon)
                self._fix_count += 1

                if self.Verbose:
                    print(f"[GPS] fix #{self._fix_count:04d}  "
                          f"lat={lat:.6f}  lon={lon:.6f}  →  "
                          f"xy=({x:.2f}, {y:.2f})m")

                if not self._check_bounds(x, y):
                    continue

                pose = np.array([x, y])
                await self.PositionPublisher.publish(pose)

            except asyncio.CancelledError:
                print(f"[GPS] Shutting down after {self._fix_count} fixes.")
                if self._ser and self._ser.is_open:
                    self._ser.close()
                raise
            except Exception as e:
                print(f"[GPS] Error: {e}")
                await asyncio.sleep(0.1)

    async def Runner(self):
        """Start the GPS positioning loop."""
        if self.Simulation:
            print("[GPS] Simulation mode — hardware not connected.")
            return

        print("[GPS] Runner() started.")
        try:
            await self.connect()
        except Exception as e:
            print(f"[GPS] connect() failed: {e}")
            return

        await self._run_gps()