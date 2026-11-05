"""
Vision.py — Publishes a fake turkey at a fixed position.

Set TURKEY_X and TURKEY_Y to whatever coordinates you want (0-100 range).
Set CLOAKED = True to publish an empty turkey list instead.
"""

import asyncio

TURKEY_X = 36.0   # <-- change me
TURKEY_Y = 22.0   # <-- change me
CLOAKED  = False  # True = publish Turkeys: []

PUBLISH_HZ =1


class FakeVision:

    def __init__(self, video_publisher, visionobject_publisher):
        self._visionobject_pub = visionobject_publisher

    async def run(self):
        return

    async def faketurkeys(self):
        period = 1.0 / PUBLISH_HZ
        print(f"[VISION] Publishing fake turkey at ({TURKEY_X}, {TURKEY_Y})  cloaked=False")

        while True:
            start = asyncio.get_event_loop().time()
            payload = {
                "Id": "DynamicObjects",
                "Turkeys": [[round(TURKEY_X, 1), round(TURKEY_Y, 1), 0.0]],
                "Humans": [],
            }
            try:
                await self._visionobject_pub.publish(payload)
                print(f"[VISION] Published: {payload}")
            except Exception as e:
                print(f"[VISION] Publish error: {e}")

            elapsed = asyncio.get_event_loop().time() - start
            await asyncio.sleep(max(0, period - elapsed))

    async def VisionRunner(self):
        await asyncio.gather(self.run(), self.faketurkeys())