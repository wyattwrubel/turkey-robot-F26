import numpy as np
from scipy.optimize import least_squares
import serial_asyncio
import asyncio

class UWB:

    #TODO Update These With Real Placements
    anchors = [
        [12.0, 15.0],
        [50.0, 17.0],
        [0.0,  50.0],
        [50.0, 50.0]
    ]

    #TODO Update These With Real Rpi Ports
    front_port = "/tmp/vserial1"
    rear_port  = "/tmp/vserial2"
    baudrate   = 9600

    def __init__(self, UWBPublisher):
        self._front_position = None
        self._rear_position  = None
        self._front_fresh    = False
        self._rear_fresh     = False
        self._lock           = asyncio.Lock()
        self.UWBPublisher = UWBPublisher
        self._front_last_known = np.mean(self.anchors, axis=0)
        self._rear_last_known  = np.mean(self.anchors, axis=0)

        self.RECEIVER_SEPARATION = 0.9
        self.SEPARATION_TOLERANCE = 0.2

    async def connect(self):
        self._front_reader, _ = await serial_asyncio.open_serial_connection(
            url=self.front_port, baudrate=self.baudrate
        )
        self._rear_reader, _ = await serial_asyncio.open_serial_connection(
            url=self.rear_port, baudrate=self.baudrate
        )

    def _estimate_position(self, distances, initial_guess):
        anchor_positions = np.array(self.anchors)
        distances        = np.array(distances)

        def residuals(pos):
            x, y = pos
            return np.sqrt(
                (x - anchor_positions[:, 0])**2 +
                (y - anchor_positions[:, 1])**2
            ) - distances

        result = least_squares(residuals, initial_guess)
        return result.x

    @staticmethod
    def _parse_distances(line: str):
        try:
            parts = line.strip().split(',')
            if len(parts) != 4:
                print(f"UWBPublisher: Expected 4 distances, got {len(parts)}: {line}")
                return None
            return [float(p) for p in parts]
        except ValueError:
            print(f"UWBPublisher: Error parsing line: {line}")
            return None

    async def _read_front_receiver(self):
        while True:
            try:
                line = await self._front_reader.readline()
                line = line.decode('utf-8').strip()
                if not line:
                    continue

                distances = self._parse_distances(line)
                if distances is None:
                    continue

                x, y = self._estimate_position(distances, self._front_last_known)

                async with self._lock:
                    self._front_position = (x, y)
                    self._front_fresh    = True
                    self._front_last_known = np.array([x, y])

            except Exception as e:
                print(f"UWBPublisher: Front receiver error: {e}")
                await asyncio.sleep(0.1)

    async def _read_rear_receiver(self):
        while True:
            try:
                line = await self._rear_reader.readline()
                line = line.decode('utf-8').strip()
                if not line:
                    continue

                distances = self._parse_distances(line)
                if distances is None:
                    continue

                x, y = self._estimate_position(distances, self._rear_last_known)

                async with self._lock:
                    self._rear_position = (x, y)
                    self._rear_fresh    = True
                    self._rear_last_known = np.array([x, y])

            except Exception as e:
                print(f"UWBPublisher: Rear receiver error: {e}")
                await asyncio.sleep(0.1)

    async def _update_when_ready(self):
        while True:
            await asyncio.sleep(0.01)

            async with self._lock:
                if not (self._front_fresh and self._rear_fresh):
                    continue
                x1, y1 = self._front_position
                x2, y2 = self._rear_position
                self._front_fresh = False
                self._rear_fresh  = False

            dx = x2 - x1
            dy = y2 - y1
            displacement = np.sqrt(dx**2 + dy**2)

            separation_error = abs(displacement - self.RECEIVER_SEPARATION)
            fix_healthy      = separation_error < self.SEPARATION_TOLERANCE

            if fix_healthy:
                # rear confirms front fix is plausible
                pose = np.array([x1, y1])
                await self.UWBPublisher.publish(pose)
            else:
                # rear disagrees with front — something is wrong
                # skip this update entirely rather than publishing bad data
                print(f"UWBPublisher: Fix rejected — displacement {displacement:.3f}m "
                    f"expected {self.RECEIVER_SEPARATION}m")

    async def run(self):
        """Start all UWB coroutines. Expecting data frequency every ~2 Hz"""
        await self.connect()
        await asyncio.gather(
            self._read_front_receiver(),
            self._read_rear_receiver(),
            self._update_when_ready()
        )