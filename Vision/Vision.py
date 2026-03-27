import asyncio
import threading
import queue
import time
import cv2

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

    CAMERA_FPS_CAP    = 3    # how fast we read from camera
    VIDEO_PUB_FPS_CAP = 2    # how fast we encode + publish frames (slightly under camera)
    INFERENCE_FPS_CAP = 0.5  # classification twice per second — less priority

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
        self._manager   = TemporalManager(init_window=10, init_min_hits=7, delete_missing_frames=30)
        self._estimator = PositionEstimator()
        self._outputter = CoordinateOutput()

        self._outputs_lock  = threading.Lock()
        self._frame_lock    = threading.Lock()

        # Separate queues — both maxsize=1 so stale data is always dropped
        self._infer_queue   = queue.Queue(maxsize=1)
        self._vidpub_queue  = queue.Queue(maxsize=1)  # camera -> video publisher

        self._running          = False
        self._inference_thread = None
        self._event_loop       = None
        self._loop_thread      = None

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    async def VisionObject_sender(self, coord_outputs):
        if self.Verbose:
            print(f"[VisionObject] {len(coord_outputs)} target(s) detected:")
            for out in coord_outputs:
                print(
                    f"  [{out.class_name}] "
                    f"x={out.x_m:.2f}m  y={out.y_m:.2f}m  "
                    f"bbox={out.bbox}"
                )

        # TODO FIX THIS WITH CORRECT INTERFACE!!
        

        # print({"Id": "DynamicObjects", "Turkeys"})
        # await self.VisionObject_Vision_pub.publish(coord_outputs)

    async def VideoFeed_sender(self, frame, outputs):
        annotated = frame.copy()

        for out in outputs:
            x1, y1, x2, y2 = out.bbox
            label = f"{out.class_name} | x={out.x_m:.2f}, y={out.y_m:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

        # success, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])

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
    # Event loop — runs camera loop + video publisher loop as concurrent tasks
    # ------------------------------------------------------------------

    def _run_event_loop(self):
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)
        self._event_loop.run_until_complete(self._run_async_tasks())

    async def _run_async_tasks(self):
        await asyncio.gather(
            self._camera_loop(),
            self._vidpub_loop(),   # publisher runs independently, never blocks camera
        )

    # ------------------------------------------------------------------
    # Async camera loop — reads frames only, no encoding here
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

            # Drop stale frames — always keep newest only
            for q in (self._infer_queue, self._vidpub_queue):
                try:
                    q.put_nowait((frame, w, h))
                except queue.Full:
                    try:
                        q.get_nowait()       # evict stale frame
                    except queue.Empty:
                        pass
                    q.put_nowait((frame, w, h))  # put fresh frame

            if self.debug_draw:
                vis = frame.copy()
                vis = self._draw(vis, self.outputs)

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
    # Async video publisher loop — encodes + publishes independently
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

            # Encode in executor so it doesn't block the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: None)  # yield before heavy work
            await self.VideoFeed_sender(frame, self.outputs)

            elapsed = time.time() - loop_start
            await asyncio.sleep(max(0.0, pub_interval - elapsed))

    # ------------------------------------------------------------------
    # Inference loop (thread) — lowest priority, fire and forget publish
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

            with self._outputs_lock:
                self._outputs = coord_outputs

            if self.debug_print:
                self._outputter.print_all(coord_outputs)

            # Fire and forget — don't block inference waiting for publish
            if self._event_loop and self._event_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.VisionObject_sender(coord_outputs),
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
        for out in outputs:
            x1, y1, x2, y2 = out.bbox
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