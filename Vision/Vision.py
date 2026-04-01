import asyncio
import threading
import queue
import time
import cv2
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(__file__))

from camera import USBCamera
from model import YOLODetector
from processing import TemporalManager, PositionEstimator
from output import CoordinateOutput


class VisionObject:
    """
    Vision pipeline. Define it, start it, read from it.

    Usage:
        vision = VisionObject()
        vision.start()

        outputs = vision.outputs      # latest detections
        frame   = vision.frame        # latest camera frame

        vision.stop()
    """

    # CAMERA_FPS_CAP    = 3
    # VIDEO_PUB_FPS_CAP = 2
    # INFERENCE_FPS_CAP = 0.5

    CAMERA_FPS_CAP    = 30
    VIDEO_PUB_FPS_CAP = 10
    INFERENCE_FPS_CAP = 10

    def __init__(self, VideoFeed_Vision_pub, VisionObject_Vision_pub, Verbose=False, debug_draw=False, debug_print=False):
        self.VideoFeed_Vision_pub    = VideoFeed_Vision_pub
        self.VisionObject_Vision_pub = VisionObject_Vision_pub

        self.Verbose     = Verbose
        self.debug_draw  = debug_draw
        self.debug_print = debug_print

        self.outputs = []
        self.frame   = None

        self._cam       = USBCamera()
        self._detector  = YOLODetector()
        self._manager   = TemporalManager(
            init_window=10,
            init_min_hits=7,
            delete_missing_frames=30,   
        )
        self._estimator = PositionEstimator()
        self._outputter = CoordinateOutput()

        self._outputs_lock  = threading.Lock()
        self._frame_lock    = threading.Lock()

        # Snapshot of outputs for video publisher — always in sync with inference
        self._display_outputs      = []
        self._display_outputs_lock = threading.Lock()

        # Latest robot pose [X, Y, theta] in world frame
        self._robot_pose      = np.array([0.0, 0.0, 0.0])
        self._robot_pose_lock = threading.Lock()

        self._infer_queue  = queue.Queue(maxsize=1)
        self._vidpub_queue = queue.Queue(maxsize=1)

        self._running          = False
        self._inference_thread = None
        self._event_loop       = None
        self._loop_thread      = None

    # ------------------------------------------------------------------
    # RobotPose Receiver
    # Protocol 4: 1x3 numpy array [X (m), Y (m), theta (rad)]
    # ------------------------------------------------------------------

    async def RobotPose_Receiver(self, pose_array):
        try:
            pose = np.asarray(pose_array, dtype=float).flatten()
            if pose.shape[0] != 3:
                print(f"[Vision] RobotPose_Receiver: unexpected shape {pose.shape}")
                return
            with self._robot_pose_lock:
                self._robot_pose = pose
        except Exception as e:
            print(f"[Vision] RobotPose_Receiver error: {e}")

    # ------------------------------------------------------------------
    # Pose transform helper
    # ------------------------------------------------------------------

    def _to_world_frame(self, x_cam, y_cam):
        """
        Transform camera-relative position into world frame.

        Map convention:
          - Origin at top-left
          - Positive X = right
          - Positive Y = down
          - Positive theta = CCW
          - theta = 0 points RIGHT (East)

        Camera mounted on front of robot:
          - x_cam = lateral (positive = right)
          - y_cam = forward (positive = away from camera)
        """
        with self._robot_pose_lock:
            rx, ry, rtheta = self._robot_pose

        t = rtheta + (np.pi / 2.0)

        cos_t = np.cos(t)
        sin_t = np.sin(t)

        world_x = rx + cos_t * x_cam + sin_t * y_cam
        world_y = ry - sin_t * x_cam + cos_t * y_cam

        return float(world_x), float(world_y)

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    async def VisionObject_sender(self, coord_outputs, frame_w):
        """
        Publishes detections in world frame coordinates.
        Protocol 1A: { Id, Turkeys, Humans, Ghosts }
        """
        turkeys = []
        humans  = []

        for out in coord_outputs:
            x1, y1, x2, y2 = out.bbox
            if x2 <= 0 or x1 >= frame_w:
                if self.Verbose:
                    print(f"[Vision] Dropping off-screen track: {out.class_name} id={out.target_id}")
                continue

            world_x, world_y = self._to_world_frame(out.x_m, out.y_m)
            angle_rad = out.angle_deg * (np.pi / 180.0)
            entry = [world_x, world_y, angle_rad]

            if out.class_name == "turkey":
                turkeys.append(entry)
            elif out.class_name == "person":
                humans.append(entry)

        payload = {
            "Id":      "DynamicObjects",
            "Turkeys": np.array(turkeys, dtype=float).tolist() if turkeys else [],
            "Humans":  np.array(humans,  dtype=float).tolist() if humans  else [],
            "Ghosts":  [],
        }

        if self.Verbose:
            print(f"[VisionObject] Turkeys={len(turkeys)} Humans={len(humans)}")
            for out in coord_outputs:
                print(
                    f"  [{out.class_name}] "
                    f"x={out.x_m:.2f}m  y={out.y_m:.2f}m  "
                    f"angle={out.angle_deg:.1f}deg  "
                    f"conf={out.confidence:.2f}"
                )

        await self.VisionObject_Vision_pub.publish(payload)

    async def VideoFeed_sender(self, frame, outputs):
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for out in outputs:
            x1, y1, x2, y2 = out.bbox

            # Skip off-screen tracks
            if x2 <= 0 or x1 >= w or y2 <= 0 or y1 >= h:
                continue

            label = f"{out.class_name} | x={out.x_m:.2f}, y={out.y_m:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

        if self.Verbose:
            if outputs:
                for out in outputs:
                    print(f"[VideoFeed] bbox={out.bbox}")
            else:
                print("[VideoFeed] No targets")

        success, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 40])
        if not success:
            print("[VideoFeed] Failed to encode frame")
            return

        encoded_bytes = encoded.tobytes()
        if self.Verbose:
            print(f"[VideoFeed] Encoded frame: {len(encoded_bytes)} bytes")

        await self.VideoFeed_Vision_pub.publish(encoded_bytes)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        self._cam.open()
        self._cam.print_actual_settings()
        self._running = True

        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="InferenceLoop"
        )
        self._inference_thread.start()

        self._loop_thread = threading.Thread(
            target=self._run_event_loop, daemon=True, name="EventLoop"
        )
        self._loop_thread.start()

        print("[Vision] Started.")

    def stop(self):
        self._running = False

        if self._event_loop:
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)

        if self._loop_thread:
            self._loop_thread.join(timeout=2.0)

        if self._inference_thread:
            self._inference_thread.join(timeout=2.0)

        self._cam.release()
        cv2.destroyAllWindows()
        print("[Vision] Stopped.")

    # ------------------------------------------------------------------
    # Output accessors
    # ------------------------------------------------------------------

    @property
    def outputs(self):
        with self._outputs_lock:
            return list(self._outputs)

    @outputs.setter
    def outputs(self, value):
        if not hasattr(self, '_outputs_lock'):
            self._outputs = value
        else:
            with self._outputs_lock:
                self._outputs = value

    @property
    def frame(self):
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    @frame.setter
    def frame(self, value):
        if not hasattr(self, '_frame_lock'):
            self._frame = value
        else:
            with self._frame_lock:
                self._frame = value

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def _run_event_loop(self):
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)
        self._event_loop.run_until_complete(self._run_async_tasks())

    async def _run_async_tasks(self):
        await asyncio.gather(
            self._camera_loop(),
            self._vidpub_loop(),
        )

    # ------------------------------------------------------------------
    # Async camera loop
    # ------------------------------------------------------------------

    async def _camera_loop(self):
        frame_interval = 1.0 / self.CAMERA_FPS_CAP
        prev_time = time.time()

        print(f"[Vision] Camera loop running ({self.CAMERA_FPS_CAP} FPS cap)")

        while self._running:
            loop_start = time.time()

            loop = asyncio.get_event_loop()
            ret, frame = await loop.run_in_executor(None, self._cam.read)

            if not ret:
                await asyncio.sleep(0.1)
                continue

            h, w = frame.shape[:2]

            with self._frame_lock:
                self._frame = frame

            for q in (self._infer_queue, self._vidpub_queue):
                try:
                    q.put_nowait((frame, w, h))
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                    q.put_nowait((frame, w, h))

            if self.debug_draw:
                vis = frame.copy()
                with self._display_outputs_lock:
                    draw_outputs = list(self._display_outputs)
                vis = self._draw(vis, draw_outputs)

                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-6)
                prev_time = curr_time

                cv2.putText(vis, f"FPS: {fps:.2f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow("CV Pipeline", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                self._running = False
                break

            elapsed = time.time() - loop_start
            await asyncio.sleep(max(0.0, frame_interval - elapsed))

    # ------------------------------------------------------------------
    # Async video publisher loop
    # ------------------------------------------------------------------

    async def _vidpub_loop(self):
        pub_interval = 1.0 / self.VIDEO_PUB_FPS_CAP

        print(f"[Vision] Video publisher running ({self.VIDEO_PUB_FPS_CAP} FPS cap)")

        while self._running:
            loop_start = time.time()

            try:
                frame, w, h = self._vidpub_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            # Use snapshot set by inference — never reads stale self.outputs
            with self._display_outputs_lock:
                current_outputs = list(self._display_outputs)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: None)
            await self.VideoFeed_sender(frame, current_outputs)

            elapsed = time.time() - loop_start
            await asyncio.sleep(max(0.0, pub_interval - elapsed))

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------

    def _inference_loop(self):
        inference_interval = 1.0 / self.INFERENCE_FPS_CAP
        print(f"[Vision] Inference loop running ({self.INFERENCE_FPS_CAP} inference/sec cap)")

        while self._running:
            loop_start = time.time()

            try:
                frame, w, h = self._infer_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            detections    = self._detector.detect(frame)
            confirmed     = self._manager.update(detections)
            positions     = self._estimator.estimate_all(confirmed, w, h)
            coord_outputs = self._outputter.convert_all(positions)

            # Filter off-screen tracks
            visible_outputs = [
                out for out in coord_outputs
                if not (out.bbox[2] <= 0 or out.bbox[0] >= w or
                        out.bbox[3] <= 0 or out.bbox[1] >= h)
            ]

            # Update both outputs and display snapshot atomically
            with self._outputs_lock:
                self._outputs = visible_outputs

            with self._display_outputs_lock:
                self._display_outputs = visible_outputs   # vidpub reads this

            if visible_outputs:
                for out in visible_outputs:
                    print(f"[Inference] {out.class_name} bbox={out.bbox}")
            else:
                print("[Inference] No targets")

            if self.debug_print:
                self._outputter.print_all(visible_outputs)

            # Always publish — downstream gets empty arrays when nothing visible
            if self._event_loop and self._event_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.VisionObject_sender(visible_outputs, w),
                    self._event_loop,
                )

            elapsed = time.time() - loop_start
            sleep_time = inference_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Draw helper
    # ------------------------------------------------------------------

    @staticmethod
    def _draw(frame, outputs):
        h, w = frame.shape[:2]
        for out in outputs:
            x1, y1, x2, y2 = out.bbox
            if x2 <= 0 or x1 >= w or y2 <= 0 or y1 >= h:
                continue
            label = f"{out.class_name} | x={out.x_m:.2f}, y={out.y_m:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    vision = VisionObject(
        VideoFeed_Vision_pub=None,
        VisionObject_Vision_pub=None,
        Verbose=True,
        debug_draw=True,
        debug_print=True,
    )
    vision.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        vision.stop()