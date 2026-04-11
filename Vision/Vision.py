"""
Vision/Vision.py

Drop-in replacement for the original VisionObject.
Hosts two WebSocket servers that the laptop connects to:

    ws://0.0.0.0:8765   /video      — Pi streams raw frames out to all clients;
                                      laptop sends processed frames back on same socket;
                                      Pi forwards received frames to VideoFeed_Vision_pub.

    ws://0.0.0.0:8766   /detections — laptop sends camera-local DynamicObjects JSON;
                                      Pi applies world-frame transform using current
                                      RobotPose and publishes to VisionObject_Vision_pub.

Simulated=True  → no camera opened; laptop_client.py drives everything via WASD sim.
                   VisionObject still hosts both servers so the laptop can connect.
Simulated=False → camera opens immediately and starts streaming.

Constructor signature is unchanged from the original:
    VisionObject(VideoFeed_Vision_pub, VisionObject_Vision_pub,
                 Simulated=False, Verbose=False, ...)
"""

import asyncio
import threading
import time
import math

import cv2
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

# ── Settings — edit these ─────────────────────────────────────────────────────
VIDEO_PORT   = 8765
DETECT_PORT  = 8766
CAMERA_FPS   = 10          # raw frame broadcast rate
JPEG_QUALITY = 80
# ─────────────────────────────────────────────────────────────────────────────


class VisionObject:
    """
    Hosts camera feed and coordinates with laptop for CV processing.
    Applies RobotPose world-frame transform before publishing detections.
    """

    def __init__(
        self,
        VideoFeed_Vision_pub,
        VisionObject_Vision_pub,
        Simulated: bool = False,
        Verbose:   bool = False,
        debug_draw:  bool = False,
        debug_print: bool = False,
    ):
        self._video_pub   = VideoFeed_Vision_pub
        self._detect_pub  = VisionObject_Vision_pub
        self.Verbose      = Verbose
        self.debug_draw   = debug_draw
        self.debug_print  = debug_print
        self.Simulated    = Simulated

        # Latest robot pose [X, Y, theta] — updated via RobotPose_Receiver
        self._robot_pose      = np.array([0.0, 0.0, 0.0])
        self._robot_pose_lock = threading.Lock()

        # Connected /video clients
        self._video_clients: set = set()
        self._video_lock = None   # asyncio.Lock — created inside event loop

        self._latest_raw_frame: bytes | None = None

        # Start the async event loop on its own thread
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="VisionLoop"
        )
        self._loop_thread.start()

    # ------------------------------------------------------------------
    # RobotPose receiver  (Protocol 4: 1x3 numpy [X, Y, theta_rad])
    # ------------------------------------------------------------------

    async def RobotPose_Receiver(self, pose_array):
        try:
            pose = np.asarray(pose_array, dtype=float).flatten()
            if pose.shape[0] != 3:
                print(f"[Vision] RobotPose bad shape: {pose.shape}")
                return
            with self._robot_pose_lock:
                self._robot_pose = pose
        except Exception as e:
            print(f"[Vision] RobotPose_Receiver error: {e}")

    # ------------------------------------------------------------------
    # World-frame transform
    # ------------------------------------------------------------------

    def _to_world(self, x_cam: float, y_cam: float):
        """
        Convert camera-local (x_cam right, y_cam forward) to world frame.

        Map convention:
            origin top-left, +X right, +Y down, +theta CCW, theta=0 → East

        Camera mounted on robot front:
            t = robot_theta + pi/2
            world_x = rx + cos(t)*x_cam + sin(t)*y_cam
            world_y = ry - sin(t)*x_cam + cos(t)*y_cam   (Y inverted)
        """
        with self._robot_pose_lock:
            rx, ry, rtheta = self._robot_pose

        t     = rtheta + (math.pi / 2.0)
        cos_t = math.cos(t)
        sin_t = math.sin(t)

        world_x = rx + cos_t * x_cam + sin_t * y_cam
        world_y = ry - sin_t * x_cam + cos_t * y_cam
        return float(world_x), float(world_y)

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._async_main())

    async def _async_main(self):
        self._video_lock = asyncio.Lock()

        tasks = [
            self._start_detection_server(),
            self._start_video_server(),
        ]
        if not self.Simulated:
            tasks.append(self._camera_loop())

        await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    # Camera loop — only runs when Simulated=False
    # ------------------------------------------------------------------

    async def _camera_loop(self):
        from Vision.UsbCamera import USBCamera
        cam = USBCamera()
        cam.open()
        cam.print_actual_settings()

        period = 1.0 / CAMERA_FPS
        loop   = asyncio.get_event_loop()
        print(f"[Vision] Camera loop running at {CAMERA_FPS} FPS")

        try:
            while True:
                t0 = loop.time()

                ret, frame = await loop.run_in_executor(None, cam.read)
                if not ret or frame is None:
                    await asyncio.sleep(0.1)
                    continue

                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if not ok:
                    continue

                jpeg = buf.tobytes()
                self._latest_raw_frame = jpeg
                await self._broadcast_video(jpeg)

                elapsed = loop.time() - t0
                await asyncio.sleep(max(0.0, period - elapsed))
        finally:
            cam.release()

    # ------------------------------------------------------------------
    # Video WebSocket server  (port 8765)
    # ------------------------------------------------------------------

    async def _start_video_server(self):
        print(f"[Vision] Video WS → ws://0.0.0.0:{VIDEO_PORT}")
        async with websockets.serve(self._video_handler, "0.0.0.0", VIDEO_PORT):
            await asyncio.Future()   # run forever

    async def _video_handler(self, ws: WebSocketServerProtocol, path: str = ""):
        async with self._video_lock:
            self._video_clients.add(ws)

        # Sync new client immediately
        if self._latest_raw_frame:
            try:
                await ws.send(self._latest_raw_frame)
            except Exception:
                pass

        print(f"[Vision] Video client connected: {ws.remote_address}")
        try:
            async for message in ws:
                # Only the laptop sends bytes back (processed frames)
                if isinstance(message, bytes) and self._video_pub is not None:
                    await self._video_pub.publish(message)
        except websockets.ConnectionClosed:
            pass
        finally:
            async with self._video_lock:
                self._video_clients.discard(ws)
            print(f"[Vision] Video client disconnected: {ws.remote_address}")

    async def _broadcast_video(self, jpeg: bytes):
        async with self._video_lock:
            clients = list(self._video_clients)
        if not clients:
            return

        async def _send(ws):
            try:
                await asyncio.wait_for(ws.send(jpeg), timeout=0.3)
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                pass
            except Exception as e:
                print(f"[Vision] Broadcast error: {e}")

        await asyncio.gather(*[_send(ws) for ws in clients])

    # ------------------------------------------------------------------
    # Detection WebSocket server  (port 8766)
    # ------------------------------------------------------------------

    async def _start_detection_server(self):
        print(f"[Vision] Detection WS → ws://0.0.0.0:{DETECT_PORT}")
        async with websockets.serve(self._detection_handler, "0.0.0.0", DETECT_PORT):
            await asyncio.Future()

    async def _detection_handler(self, ws: WebSocketServerProtocol, path: str = ""):
        print(f"[Vision] Detection client connected: {ws.remote_address}")
        try:
            async for message in ws:
                raw = message if isinstance(message, str) else message.decode()
                try:
                    import json
                    payload = json.loads(raw)
                    await self._handle_detection_payload(payload)
                except Exception as e:
                    print(f"[Vision] Detection parse error: {e}")
        except websockets.ConnectionClosed:
            pass
        finally:
            print(f"[Vision] Detection client disconnected: {ws.remote_address}")

    async def _handle_detection_payload(self, payload: dict):
        """
        Receives camera-local payload from laptop.
        Applies world-frame transform to each entry, then publishes.

        Input  (camera-local):  [[x_cam, y_cam, angle_rad], ...]
        Output (world frame):   [[world_x, world_y, angle_rad], ...]
        """
        turkeys_local = payload.get("Turkeys", [])
        humans_local  = payload.get("Humans",  [])
        ghosts_local  = payload.get("Ghosts",  [])

        def _transform(entries):
            out = []
            for entry in entries:
                if len(entry) < 3:
                    continue
                x_cam, y_cam, angle_rad = entry[0], entry[1], entry[2]
                wx, wy = self._to_world(x_cam, y_cam)
                out.append([round(wx, 4), round(wy, 4), round(angle_rad, 6)])
            return out

        world_payload = {
            "Id":      "DynamicObjects",
            "Turkeys": _transform(turkeys_local),
            "Humans":  _transform(humans_local),
            "Ghosts":  _transform(ghosts_local),
        }

        if self.Verbose:
            print(
                f"[Vision] Publishing — "
                f"Turkeys={len(world_payload['Turkeys'])} "
                f"Humans={len(world_payload['Humans'])}"
            )

        if self._detect_pub is not None:
            await self._detect_pub.publish(world_payload)