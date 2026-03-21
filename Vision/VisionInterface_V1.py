"""
Vision.py — Frame transmission interface with JPEG conversion

The caller is responsible for providing image data.
This class handles publishing and optional JPEG conversion.

Usage:
    vision = Vision(video_publisher)

    # If you already have JPEG bytes:
    await vision.send(jpeg_bytes)

    # If you have a file path (PNG, BMP, etc.):
    await vision.send_file("test.png")

    # If you have a numpy array (e.g. from cv2 processing):
    await vision.send_array(numpy_array)
"""

import cv2
import numpy as np


class Vision:

    JPEG_QUALITY = 80

    def __init__(self, video_publisher):
        self._publisher = video_publisher

    async def send(self, frame: bytes) -> None:
        """Publish a single frame. Expects JPEG-encoded bytes."""
        await self._publisher.publish(frame)

    async def send_file(self, path: str) -> None:
        """Load an image file, convert to JPEG bytes, and publish."""
        await self.send(self._encode(cv2.imread(path)))

    async def send_array(self, frame: np.ndarray) -> None:
        """Convert a numpy array (e.g. from cv2) to JPEG bytes and publish."""
        await self.send(self._encode(frame))

    # ------------------------------------------------------------------
    def _encode(self, frame: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(
            '.jpg', frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY]
        )
        if not ok:
            raise ValueError("[VISION] JPEG encoding failed")
        return buf.tobytes()