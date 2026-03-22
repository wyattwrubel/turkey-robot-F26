"""
Vision.py — Static image carousel + simulated VisionObject (development fallback)

Publishes two streams:
  1. VideoFeed  — cycles turkey1.jpg / turkey2.jpg / turkey3.jpg as JPEG bytes
  2. VisionObject — fake turkey and human detections moving on sine wave paths

Place the three images in the same directory as this file.

Usage in main.py:
    vision = Vision(video_publisher, vision_object_publisher)
    await asyncio.gather(..., vision.run())
"""

import asyncio
import os
import math
import cv2


class Vision:

    # VideoFeed rate
    CAPTURE_RATE_HZ = 0.02
    CAPTURE_PERIOD  = 1.0 / CAPTURE_RATE_HZ
    JPEG_QUALITY    = 80
    IMAGE_DIR       = os.path.dirname(os.path.abspath(__file__))
    IMAGE_FILES     = ["turkey1.jpg", "turkey2.jpg", "turkey3.jpg"]

    # VisionObject publish rate
    VISION_RATE_HZ  = 2.0
    VISION_PERIOD   = 1.0 / VISION_RATE_HZ

    def __init__(self, video_publisher, vision_object_publisher=None):
        self._video_pub  = video_publisher
        self._vision_pub = vision_object_publisher
        self._frames     = []
        self._index      = 0

    # ------------------------------------------------------------------
    # VideoFeed
    # ------------------------------------------------------------------

    def _load_images(self) -> bool:
        for filename in self.IMAGE_FILES:
            path = os.path.join(self.IMAGE_DIR, filename)
            img  = cv2.imread(path)
            if img is None:
                print(f"[VISION] Could not load {path}")
                return False
            ok, buf = cv2.imencode(
                '.jpg', img,
                [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY]
            )
            if not ok:
                print(f"[VISION] Could not encode {filename}")
                return False
            self._frames.append(buf.tobytes())
            print(f"[VISION] Loaded {filename}")
        return True

    def _next_frame(self) -> bytes:
        frame       = self._frames[self._index]
        self._index = (self._index + 1) % len(self._frames)
        return frame

    async def _run_video(self):
        if not self._load_images():
            print("[VISION] Failed to load images — video feed stopped")
            return
        print(f"[VISION] Carousel running at {self.CAPTURE_RATE_HZ} Hz")
        loop = asyncio.get_event_loop()
        while True:
            start = loop.time()
            await self._video_pub.publish(self._next_frame())
            elapsed = loop.time() - start
            await asyncio.sleep(max(0, self.CAPTURE_PERIOD - elapsed))

    # ------------------------------------------------------------------
    # VisionObject — fake moving detections
    # ------------------------------------------------------------------
    #
    # Zone boundary (from Boundary.csv): X in [0,41], Y in [0,77]
    # Zone centre:  ~(20, 38)
    #
    # Turkey A: orbits inside the zone on a slow ellipse
    # Turkey B: crosses in and out of the zone on a horizontal sine path
    # Human:    moves slowly along the left edge (outside zone at x<0)
    #
    # All positions are world coordinates [X, Y] in metres.
    # Relative angle is the bearing from zone centre to the object (radians).
    # ------------------------------------------------------------------

    def _compute_detections(self, t: float) -> dict:
        """
        Compute fake world-space detections at time t (seconds).

        Returns a VisionObject dict:
            {
                "Turkeys": [[X, Y, angle_rad], ...],
                "Humans":  [[X, Y, angle_rad], ...]
            }
        """
        zone_cx, zone_cy = 20.0, 38.0

        def bearing(x, y):
            """Relative angle from zone centre to object (radians)."""
            return math.atan2(y - zone_cy, x - zone_cx)

        turkeys = []
        humans  = []

        # -- Turkey A: slow ellipse inside zone, always visible --
        # Period 30s, stays well inside [2-38] x [5-71]
        ax = zone_cx + 16.0 * math.cos(2 * math.pi * t / 30.0)
        ay = zone_cy + 28.0 * math.sin(2 * math.pi * t / 30.0)
        turkeys.append([int(round(ax)), int(round(ay)), 0])

        # -- Turkey B: crosses zone boundary on X axis --
        # Period 20s, oscillates X between -5 and 45 (zone boundary at x=0 and x=41)
        bx = 20.0 + 25.0 * math.sin(2 * math.pi * t / 20.0)
        by = 20.0
        turkeys.append([int(round(bx)), int(round(by)), 0])

        # -- Human: slow drift outside zone on negative X side --
        # Always outside zone (x stays negative)
        hx = -3.0 + 1.5 * math.sin(2 * math.pi * t / 45.0)
        hy = 10.0 + 30.0 * math.sin(2 * math.pi * t / 60.0)
        humans.append([int(round(hx)), int(round(hy)), 0])

        return {"Id": "DynamicObjects", "Turkeys": turkeys, "Humans": humans, "Ghosts": []}

    async def _run_vision_object(self):
        if self._vision_pub is None:
            return
        print(f"[VISION] VisionObject simulation running at {self.VISION_RATE_HZ} Hz")
        loop = asyncio.get_event_loop()
        t    = 0.0
        while True:
            start      = loop.time()
            detections = self._compute_detections(t)
            # print(
            #     f"[VISION] t={t:6.1f}s | "
            #     f"Turkeys={[(round(tk[0],1), round(tk[1],1)) for tk in detections['Turkeys']]} | "
            #     f"Humans={[(round(h[0],1), round(h[1],1)) for h in detections['Humans']]}"
            # )
            await self._vision_pub.publish(detections)
            elapsed = loop.time() - start
            await asyncio.sleep(max(0, self.VISION_PERIOD - elapsed))
            t += self.VISION_PERIOD

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self):
        await asyncio.gather(
            self._run_video(),
            self._run_vision_object(),
        )