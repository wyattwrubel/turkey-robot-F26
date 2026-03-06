"""
broadcast.py — Background broadcast loops

Two async tasks run continuously:
    broadcast_state()   pushes robot state  at 10 Hz
    broadcast_frames()  pushes camera frame at  1 Hz

Both are started in app.py on server startup.
"""

import asyncio
import json

from fastapi import WebSocket

from robot import RobotInterface


# Minimal 1x1 grey JPEG in base64.
# Activates the camera panel in the UI without a real camera.
# Replace with real OpenCV capture in broadcast_frames().
_PLACEHOLDER_FRAME = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
    "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
    "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAAAAAABgUE/8QAIhAA"
    "AgIBBAMAAAAAAAAAAAAAAQIDBAUREiExQf/EABQBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEQEA"
    "AAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCv1+xt622jDb2KLKJ1aMsU+nPf9E7IAP/Z"
)


async def _send_all(connections: list[WebSocket], msg: str) -> None:
    """Send a message to all connections, pruning dead ones."""
    dead = []
    for ws in connections:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connections.remove(ws)


async def broadcast_state(connections: list[WebSocket], robot: RobotInterface) -> None:
    """
    Push robot state to all connected clients at 10 Hz.
    robot.tick() advances the simulation (or reads live hardware).
    """
    while True:
        await asyncio.sleep(0.1)
        robot.tick(0.1)

        if connections:
            msg = json.dumps(robot.to_state_msg())
            await _send_all(connections, msg)


async def broadcast_frames(connections: list[WebSocket], robot: RobotInterface) -> None:
    """
    Push camera frames to all connected clients at 1 Hz.

    To use a real camera, replace the block below:

        import cv2
        camera = cv2.VideoCapture(0)

        success, frame = camera.read()
        if not success:
            continue
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buf).decode()
    """
    while True:
        await asyncio.sleep(1.0)

        if not connections:
            continue

        # ── TODO: replace with real camera capture ────────────
        b64 = _PLACEHOLDER_FRAME
        # ─────────────────────────────────────────────────────

        msg = json.dumps({"type": "frame", "image": b64})
        await _send_all(connections, msg)
