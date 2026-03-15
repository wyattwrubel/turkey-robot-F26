"""
PanelCamera.py — Live Camera Feed

Owns everything related to the camera panel:
  - Receiving pre-rendered OpenCV frames from the video pipeline
  - Storing the latest frame so late-joining clients get it on connect
  - Broadcasting new frames to all connected browser clients

Inbound receiver (subscribe in main.py):
    video_pub.subscribe(ui.camera.VideoFeed_Receiver)

No outbound publishing — camera is display only.
"""

import base64
import json

from fastapi import WebSocket


class PanelCamera:

    def __init__(self):
        self._current_frame: str | None = None   # latest base64 JPEG

    # ── Inbound receiver ──────────────────────────────────────────────────────

    async def VideoFeed_Receiver(self, data, broadcast_fn) -> None:
        """
        Accepts either raw JPEG bytes or an already-base64-encoded string.
        Stores the frame and broadcasts it to all connected clients immediately.

        broadcast_fn — pass CDSControlServer._broadcast
        """
        if isinstance(data, (bytes, bytearray)):
            b64 = base64.b64encode(data).decode("ascii")
        else:
            b64 = str(data)

        self._current_frame = b64
        await broadcast_fn({"type": "frame", "image": b64})

    # ── On-connect sync ───────────────────────────────────────────────────────

    async def sync_client(self, ws: WebSocket) -> None:
        """Send the latest frame to a newly connected client, if one exists."""
        if self._current_frame:
            await ws.send_text(json.dumps({
                "type":  "frame",
                "image": self._current_frame,
            }))