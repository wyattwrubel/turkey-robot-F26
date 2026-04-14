"""
PanelCamera.py — Live Camera Feed (MJPEG over HTTP)

Camera frames are now served over a dedicated GET /camera HTTP endpoint
as a multipart/x-mixed-replace MJPEG stream, completely separate from
the WebSocket used for state/map/controls.

Benefits vs the old base64-over-WebSocket approach:
  • Large JPEG frames can no longer stall state messages on the shared WS
  • The browser's native MJPEG decoder handles partial frames and reconnects
  • A keepalive boundary is sent every 5 s to prevent the ESP32 AP from
    killing the idle TCP connection between frames
"""

import asyncio
import base64

from fastapi import WebSocket
from fastapi.responses import StreamingResponse


class PanelCamera:

    def __init__(self):
        self._current_frame: bytes | None = None   # raw JPEG bytes
        self._new_frame = asyncio.Event()

    # ── Inbound receiver ──────────────────────────────────────────────────────

    async def VideoFeed_Receiver(self, data, broadcast_fn) -> None:
        """
        Accepts raw JPEG bytes or an already-base64-encoded string.
        Stores the latest frame and wakes any waiting MJPEG streamers.
        broadcast_fn is intentionally ignored — camera no longer uses WebSocket.
        """
        if isinstance(data, (bytes, bytearray)):
            self._current_frame = bytes(data)
        else:
            # decode if it arrived as a base64 string
            self._current_frame = base64.b64decode(data)

        # Wake all active MJPEG generator coroutines
        self._new_frame.set()
        self._new_frame.clear()

    # ── MJPEG stream endpoint ─────────────────────────────────────────────────

    async def mjpeg_endpoint(self) -> StreamingResponse:
        """
        Mount on GET /camera in app.py:
            @app.get("/camera")
            async def camera_feed():
                return await self.camera.mjpeg_endpoint()

        Each browser <img> tag opens one long-lived HTTP response.
        The browser reconnects automatically if the connection drops.
        """
        return StreamingResponse(
            self._mjpeg_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma":        "no-cache",
            },
        )

    async def _mjpeg_generator(self):
        BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        TAIL     = b"\r\n"
        TIMEOUT  = 5.0   # seconds before sending keepalive

        # Send the latest frame immediately so the <img> isn't blank on load
        if self._current_frame:
            yield BOUNDARY + self._current_frame + TAIL

        while True:
            try:
                await asyncio.wait_for(
                    self._new_frame.wait(),
                    timeout=TIMEOUT,
                )
            except asyncio.TimeoutError:
                # No new frame — send an empty boundary so the TCP connection
                # doesn't idle-timeout on the ESP32 AP
                yield b"--frame\r\n\r\n"
                continue

            frame = self._current_frame
            if frame is None:
                continue

            try:
                yield BOUNDARY + frame + TAIL
            except Exception:
                break   # client disconnected — exit generator cleanly

    # ── On-connect sync (no-op — camera is now HTTP, not WebSocket) ───────────

    async def sync_client(self, ws: WebSocket) -> None:
        """
        Called by app.py on every new WebSocket connection.
        Camera no longer uses the WebSocket, so this is intentionally empty.
        app.py does not need to be changed.
        """
        pass