"""
Vision/Vision.py

Drop-in replacement for the original VisionObject.
Hosts two WebSocket servers that the laptop connects to:

    ws://0.0.0.0:8765   /video      — Pi streams raw frames out to all clients;
                                      laptop sends bounding-box detection JSON back
                                      on the same socket. Pi draws boxes on every
                                      outgoing frame using the last known detections,
                                      clearing only when the laptop explicitly sends
                                      an empty list. No flashing between updates.

    ws://0.0.0.0:8766   /detections — laptop sends camera-local DynamicObjects JSON;
                                      Pi applies world-frame transform using current
                                      RobotPose and publishes to VisionObject_Vision_pub.
                                      Unchanged from original.

No image data travels laptop → Pi. The laptop returns only lightweight JSON:

    Port 8765 payload (BBoxes):
    {
        "Id": "BBoxes",
        "detections": [
            {"class_id": 0, "class_name": "turkey", "confidence": 0.91,
             "bbox": [x1, y1, x2, y2]},
            ...
        ]
    }

    An empty "detections" list means no objects in frame — Pi clears overlays.
    The laptop sends this payload on every processed frame, including empty ones.

    Port 8766 payload: unchanged DynamicObjects format.

Persistent overlay
------------------
_latest_detections holds the last bbox list received from the laptop. The
camera loop annotates every outgoing frame with these detections, so boxes
stay fixed between laptop updates. When the laptop sends an empty list,
_latest_detections is cleared and the clean frame is published.

If the laptop disconnects, _latest_detections is cleared immediately so stale
boxes don't persist on screen indefinitely.

Simulated=True  → no camera opened; laptop_client.py drives everything via WASD sim.
Simulated=False → camera opens immediately and starts streaming.

Constructor signature is unchanged from the original:
    VisionObject(VideoFeed_Vision_pub, VisionObject_Vision_pub,
                 Simulated=False, Verbose=False, ...)

Cross-loop fix
--------------
VisionObject runs its WebSocket servers on its own private event loop (loop B)
spawned on a dedicated thread. The pub/sub subscribers (e.g. UserInterface) live
on uvicorn's event loop (loop A, the one running main()).

Fix: capture loop A at construction time. Then use asyncio.run_coroutine_threadsafe()
to hand publish calls from loop B across to loop A.
"""

import asyncio
import json
import math
import threading

import cv2
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

# ── Settings — edit these ─────────────────────────────────────────────────────
VIDEO_PORT   = 8765
DETECT_PORT  = 8766
CAMERA_FPS   = 10          # raw frame broadcast rate
JPEG_QUALITY = 40

# Annotation appearance (mirrors yolo_detector.annotate on the laptop)
BBOX_COLOR     = (0, 255, 0)
BBOX_THICKNESS = 2
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.6
FONT_THICKNESS = 2
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

        # Latest raw frame bytes — stored for the initial client sync frame
        self._latest_raw_frame: bytes | None = None
        self._raw_frame_lock = threading.Lock()

        # Last known detections from the laptop.
        # Applied to every camera frame. Replaced on each laptop message.
        # Cleared to [] only when the laptop sends an empty list, or disconnects.
        self._latest_detections: list = []
        self._detections_lock = threading.Lock()

        # Connected /video clients
        self._video_clients: set = set()
        self._video_lock = None   # asyncio.Lock — created inside event loop

        # Capture the caller's event loop (uvicorn/main — loop A).
        self._main_loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()

        # Start Vision's private async event loop on its own thread (loop B)
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="VisionLoop"
        )
        self._loop_thread.start()

    # ------------------------------------------------------------------
    # Cross-loop publish helper
    # ------------------------------------------------------------------

    def _publish_to_main(self, coro, label: str = "publish") -> None:
        """
        Schedule a coroutine onto loop A (uvicorn/main) from loop B (VisionLoop).
        Blocks until the coroutine completes or times out, then returns.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._main_loop)
        try:
            future.result(timeout=1.0)
        except TimeoutError:
            print(f"[Vision] {label} timed out — data dropped")
        except Exception as e:
            print(f"[Vision] {label} error: {e}")

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
    # Pi-side annotation
    # ------------------------------------------------------------------

    def _annotate(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw bounding boxes onto a decoded frame in-place and return it.
        Mirrors the annotation style of yolo_detector.annotate() on the laptop.
        """
        for det in detections:
            try:
                x1, y1, x2, y2 = map(int, det["bbox"])
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)
                cv2.putText(
                    frame, label,
                    (x1, max(20, y1 - 8)),
                    FONT, FONT_SCALE, BBOX_COLOR, FONT_THICKNESS,
                )
            except (KeyError, TypeError, ValueError) as e:
                print(f"[Vision] Annotation error for det={det}: {e}")
        return frame

    # ------------------------------------------------------------------
    # Event loop (loop B)
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

                # Snapshot detections under lock — don't hold lock during encode
                with self._detections_lock:
                    detections = list(self._latest_detections)

                # Annotate with last known detections (empty list = clean frame)
                if detections:
                    frame = self._annotate(frame, detections)

                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if not ok:
                    continue

                jpeg = buf.tobytes()

                with self._raw_frame_lock:
                    self._latest_raw_frame = jpeg

                await self._broadcast_video(jpeg)

                if self._video_pub is not None:
                    await loop.run_in_executor(
                        None,
                        self._publish_to_main,
                        self._video_pub.publish(jpeg),
                        "VideoFeed publish",
                    )

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
            await asyncio.Future()

    async def _video_handler(self, ws: WebSocketServerProtocol, path: str = ""):
        async with self._video_lock:
            self._video_clients.add(ws)

        # Sync new client with the latest frame immediately
        with self._raw_frame_lock:
            latest = self._latest_raw_frame
        if latest:
            try:
                await ws.send(latest)
            except Exception:
                pass

        print(f"[Vision] Video client connected: {ws.remote_address}")
        try:
            async for message in ws:
                # Laptop sends BBoxes JSON on this socket.
                # Just update _latest_detections — the camera loop does the rest.
                raw = message if isinstance(message, str) else message.decode("utf-8")
                try:
                    payload = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"[Vision] BBoxes parse error: {e}")
                    continue

                if payload.get("Id") != "BBoxes":
                    print(f"[Vision] Unexpected payload Id on port {VIDEO_PORT}: {payload.get('Id')}")
                    continue

                with self._detections_lock:
                    self._latest_detections = payload.get("detections", [])

        except websockets.ConnectionClosed:
            pass
        finally:
            async with self._video_lock:
                self._video_clients.discard(ws)
            # Laptop disconnected — clear overlays immediately
            with self._detections_lock:
                self._latest_detections = []
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
    # Detection WebSocket server  (port 8766) — unchanged
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
        Receives camera-local DynamicObjects payload from laptop on port 8766.
        Applies world-frame transform, publishes to VisionObject_Vision_pub.
        Unchanged from original.
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
            loop_b = asyncio.get_event_loop()
            await loop_b.run_in_executor(
                None,
                self._publish_to_main,
                self._detect_pub.publish(world_payload),
                "Detection publish",
            )