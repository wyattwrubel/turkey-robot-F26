"""
Vision.py — Static image carousel (development fallback)

Cycles through turkey1.jpg, turkey2.jpg, turkey3.jpg at CAPTURE_RATE_HZ
and publishes each frame as JPEG bytes — identical interface to the real
camera version so main.py needs no changes when you swap in the real one.

Place the three images in the same directory as this file.

Usage in main.py:
    vision = Vision(video_publisher)
    await asyncio.gather(..., vision.run())
"""

import asyncio
import os
import cv2


class Vision:

    CAPTURE_RATE_HZ = 0.02 #0.01                       # slow enough to see each image
    CAPTURE_PERIOD  = 1.0 / CAPTURE_RATE_HZ

    JPEG_QUALITY    = 80
    IMAGE_DIR       = os.path.dirname(os.path.abspath(__file__))
    IMAGE_FILES     = ["turkey1.jpg", "turkey2.jpg", "turkey3.jpg"]

    def __init__(self, video_publisher, visionobject_publisher):
        self._publisher = video_publisher
        self._visionobject_pub = visionobject_publisher
        self._frames    = []
        self._index     = 0

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
        frame        = self._frames[self._index]
        self._index  = (self._index + 1) % len(self._frames)
        return frame

    async def VisionRunner(self):
        await asyncio.gather(self.run(), self.faketurkeys())


    async def run(self):
        if not self._load_images():
            print("[VISION] Failed to load images — stopping")
            return

        print(f"[VISION] Carousel running at {self.CAPTURE_RATE_HZ} Hz")

        loop = asyncio.get_event_loop()
        while True:
            start = loop.time()
            await self._publisher.publish(self._next_frame())
            elapsed = loop.time() - start
            await asyncio.sleep(max(0, self.CAPTURE_PERIOD - elapsed))

    async def faketurkeys(self):
        await asyncio.sleep(4)
        while True: 
            # await self._visionobject_pub.publish({"Id": "DynamicObjects", "Turkeys": [], "Humans": []})
            # await asyncio.sleep(5)
            turkeys = [[30, 10 , 0.0]]
            # await self._visionobject_pub.publish({"Id": "DynamicObjects", "Turkeys": turkeys, "Humans": [[70.0, 30.0, 0.0]]})
            await self._visionobject_pub.publish({"Id": "DynamicObjects", "Turkeys": turkeys, "Humans": []})
            await asyncio.sleep(20)     
            # turkeys = [
            #     [60 , 10, 0.0],
            #     # [30, 50 , 0.0],
            # ]
            # await self._visionobject_pub.publish({"Id": "DynamicObjects", "Turkeys": turkeys, "Humans": []})
            # await asyncio.sleep(60)
