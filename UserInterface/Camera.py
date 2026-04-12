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
        self._frame_event = asyncio.Event()
        self._broadcast_fn = None
        self._loop_task: asyncio.Task | None = None

    # ── Start broadcast loop (call once after the event loop is running) ──────

    def start(self, broadcast_fn) -> None:
        """Call this from the FastAPI lifespan, after the event loop exists."""
        self._broadcast_fn = broadcast_fn
        self._loop_task = asyncio.create_task(self._broadcast_loop())

    # ── Inbound receiver ──────────────────────────────────────────────────────

    async def VideoFeed_Receiver(self, data, broadcast_fn) -> None:
        """
        Stores the latest frame and signals the broadcast loop.
        Never spawns tasks — just updates the slot and sets an event.
        If frames arrive faster than the loop can send, intermediate
        frames are naturally dropped (only latest is kept).
        """
        if self._loop_task is None:
            self.start(broadcast_fn)

        if isinstance(data, (bytes, bytearray)):
            b64 = base64.b64encode(data).decode("ascii")
        else:
            b64 = str(data)

        self._current_frame = b64      # overwrite — only latest matters
        self._frame_event.set()        # wake the broadcast loop

    # ── Single long-running broadcast loop ───────────────────────────────────

    async def _broadcast_loop(self) -> None:
        """
        Wakes whenever a new frame arrives, broadcasts it once, then sleeps.
        This serializes all video sends through a single coroutine, so
        there's no task pile-up and backpressure is implicit.
        """
        while True:
            await self._frame_event.wait()
            self._frame_event.clear()

            frame = self._current_frame
            if frame is None or self._broadcast_fn is None:
                continue

            try:
                await asyncio.wait_for(
                    self._broadcast_fn({"type": "frame", "image": frame}),
                    timeout=0.15,   # whole-broadcast budget; tune to your fps target
                )
            except asyncio.TimeoutError:
                pass   # fell behind — next frame will catch up
            except Exception as e:
                print(f"[PanelCamera] Broadcast error: {e}")

    # ── On-connect sync ───────────────────────────────────────────────────────

    async def sync_client(self, ws: WebSocket) -> None:
        """Send the latest frame to a newly connected client, if one exists."""
        if self._current_frame is None:
            return
        try:
            await asyncio.wait_for(
                ws.send_text(json.dumps({
                    "type":  "frame",
                    "image": self._current_frame,
                })),
                timeout=1.0,
            )
        except Exception as e:
            print(f"[PanelCamera] sync_client error: {e}")