"""
EVENTUALLY DELETE AND PORT OVER ELSEWHERE

broadcast.py — Background broadcast loops

broadcast_state()   pushes robot state at 10 Hz
broadcast_frames()  pushes vision detections at 2 Hz

Single source of truth for vision data: _current_vision_msg.
  - On startup, pre-loaded with demo bboxes for turkey.jpg
  - VisionObject_Receiver in app.py calls set_vision() to update it
  - broadcast_frames just sends whatever _current_vision_msg holds
  - No competing message sources — one writer, one broadcaster

Set DEMO_BBOXES = False once real vision module is publishing.
"""

import asyncio
import base64
import io
import json
from pathlib import Path

from fastapi import WebSocket
from robot import RobotInterface

# ── Demo mode ──────────────────────────────────────────────────────────────────
# Set False once the real vision module is publishing via VisionObject_Receiver.
DEMO_BBOXES = True

# ── Shared state ───────────────────────────────────────────────────────────────
_current_frame_msg:  str | None = None   # base64 JPEG frame
_current_vision_msg: str | None = None   # latest vision/bbox JSON


def set_frame(b64: str, w: int, h: int) -> None:
    """Store a new base64-encoded frame. Sent once to each client on connect."""
    global _current_frame_msg
    _current_frame_msg = json.dumps({"type": "frame", "image": b64, "w": w, "h": h})


def set_vision(turkeys: list, humans: list, bboxes: list) -> None:
    """
    Update the vision state broadcast to all clients.
    Called by VisionObject_Receiver in app.py whenever the vision module publishes.
    Also called at startup to set demo bboxes.
    """
    global _current_vision_msg
    _current_vision_msg = json.dumps({
        "type":    "vision",
        "turkeys": turkeys,
        "humans":  humans,
        "bboxes":  bboxes,
    })


def clear_frame() -> None:
    global _current_frame_msg
    _current_frame_msg = None


def _load_static_image(filename: str) -> bool:
    """Load Website/<filename> at startup as the initial camera frame."""
    path = Path(__file__).parent / "Website" / filename
    if not path.exists():
        print(f"[CAM] {path} not found — camera panel will show placeholder")
        return False
    try:
        from PIL import Image
        img  = Image.open(path).convert("RGB")
        w, h = img.size
        buf  = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64  = base64.b64encode(buf.getvalue()).decode("ascii")
        set_frame(b64, w, h)
        print(f"[CAM] Loaded {path.name}  ({w}x{h})")
        return True
    except ImportError:
        pass
    try:
        raw = path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        set_frame(b64, 640, 480)
        print(f"[CAM] Loaded {path.name} (raw — install Pillow for exact dims)")
        return True
    except Exception as e:
        print(f"[CAM] Failed to load {path.name}: {e}")
        return False


# ── Startup initialisation ────────────────────────────────────────────────────
_load_static_image("turkey.jpg")

if DEMO_BBOXES:
    # Pre-load demo bboxes for turkey.jpg (748x499).
    # VisionObject_Receiver will overwrite this via set_vision() when real
    # detections arrive — at that point set DEMO_BBOXES = False.
    set_vision(
        turkeys=[],
        humans=[],
        bboxes=[
            {"label": "Turkey", "conf": 0.96, "x1": 200, "y1":  120, "x2": 400, "y2": 400}, 
            {"label": "Turkey", "conf": 0.91, "x1": 400, "y1":  120, "x2": 650, "y2": 400},
        ],
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
async def _send_all(connections: list[WebSocket], msg: str) -> None:
    dead = []
    for ws in connections:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connections.remove(ws)


# ── Broadcast loops ───────────────────────────────────────────────────────────
async def broadcast_state(connections: list[WebSocket], robot: RobotInterface) -> None:
    """Push robot state to all connected clients at 10 Hz."""
    while True:
        await asyncio.sleep(0.1)
        robot.tick(0.1)
        if connections:
            await _send_all(connections, json.dumps(robot.to_state_msg()))


async def broadcast_frames(connections: list[WebSocket], robot: RobotInterface) -> None:
    """
    Push vision detections at 2 Hz.
    Sends _current_vision_msg — the single source of truth.
    The camera frame itself is only sent once per client on connect (see app.py).
    """
    while True:
        await asyncio.sleep(0.2)
        if connections and _current_vision_msg is not None:
            await _send_all(connections, _current_vision_msg)


async def broadcast_new_frame(connections: list[WebSocket]) -> None:
    """Push a newly captured live frame to all connected clients immediately."""
    if _current_frame_msg is not None:
        await _send_all(connections, _current_frame_msg)