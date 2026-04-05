"""
PanelCamera.py — Live Camera Feed
"""

import asyncio
import base64
import json

from fastapi import WebSocket


class PanelCamera:

    def __init__(self):
        self._current_frame: str | None = None
        self._broadcast_fn = None

    # ── Inbound receiver ──────────────────────────────────────────────────────

    async def VideoFeed_Receiver(self, data, broadcast_fn) -> None:
        """
        Accepts either raw JPEG bytes or an already-base64-encoded string.
        Stores frame immediately, then broadcasts to clients in the background
        so a slow client never stalls the next incoming frame.
        """
        if isinstance(data, (bytes, bytearray)):
            b64 = base64.b64encode(data).decode("ascii")
        else:
            b64 = str(data)

        self._current_frame = b64

        # Fire and forget — don't await broadcast, so slow clients
        # can't back up the receiver and stall the video pipeline
        asyncio.create_task(
            self._broadcast_latest(broadcast_fn, b64)
        )

    async def _broadcast_latest(self, broadcast_fn, b64: str) -> None:
        """
        Broadcasts to all clients. Each client gets a timeout so one
        frozen browser can't hang everyone else.
        """
        try:
            await asyncio.wait_for(
                broadcast_fn({"type": "frame", "image": b64}),
                timeout=0.3,   # drop frame to this client if it can't receive in 300ms
            )
        except asyncio.TimeoutError:
            pass  # slow client — frame dropped, next one will come
        except Exception as e:
            print(f"[PanelCamera] Broadcast error: {e}")

    # ── On-connect sync ───────────────────────────────────────────────────────

    async def sync_client(self, ws: WebSocket) -> None:
        """Send the latest frame to a newly connected client, if one exists."""
        if self._current_frame:
            try:
                await asyncio.wait_for(
                    ws.send_text(json.dumps({
                        "type":  "frame",
                        "image": self._current_frame,
                    })),
                    timeout=1.0,
                )
            except (asyncio.TimeoutError, Exception) as e:
                print(f"[PanelCamera] sync_client error: {e}")