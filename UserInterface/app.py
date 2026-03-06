"""
app.py — FastAPI application, routes, and WebSocket endpoint

Exposes:
    GET  /           serves index.html
    GET  /styles.css serves styles.css
    GET  /script.js  serves script.js
    WS   /ws         bidirectional robot control channel
"""

import asyncio
import json
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from broadcast import broadcast_frames, broadcast_state
from handlers  import handle_message
from logger    import log, log_warn, log_err, log_info, C
from robot     import MockRobotState, RobotInterface


# ── Shared state ──────────────────────────────────────────────
# Swap MockRobotState() for your real implementation here.
robot: RobotInterface = MockRobotState()
connections: list[WebSocket] = []

STATIC_DIR = Path(__file__).parent / "Website"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    log_info("SERVER", "CDS Control Server starting")
    log_info("SERVER", "UI  ->  http://localhost:8000")
    log_info("SERVER", "WS  ->  ws://localhost:8000/ws")
    asyncio.create_task(broadcast_state(connections, robot))
    asyncio.create_task(broadcast_frames(connections, robot))
    yield
    # shutdown (add cleanup here if needed)


app = FastAPI(title="CDS Control Server", lifespan=lifespan)


# ── Static file routes ────────────────────────────────────────
# Explicit routes instead of app.mount("/", StaticFiles(...))
# because StaticFiles at "/" intercepts WebSocket upgrades and crashes.

@app.get("/")
async def root():
    f = STATIC_DIR / "index.html"
    if f.exists():
        return FileResponse(str(f))
    return HTMLResponse("<h2>index.html not found</h2>", status_code=404)


@app.get("/styles.css")
async def styles():
    f = STATIC_DIR / "styles.css"
    if f.exists():
        return FileResponse(str(f), media_type="text/css")
    return HTMLResponse("/* styles.css not found */", status_code=404, media_type="text/css")


@app.get("/script.js")
async def script():
    f = STATIC_DIR / "script.js"
    if f.exists():
        return FileResponse(str(f), media_type="application/javascript")
    return HTMLResponse("// script.js not found", status_code=404, media_type="application/javascript")


# Add additional asset routes here if needed:
# @app.get("/icon.png")
# async def icon():
#     return FileResponse(str(STATIC_DIR / "icon.png"), media_type="image/png")


# ── WebSocket endpoint ────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connections.append(ws)
    client = ws.client
    log("WS", f"Client connected  {client.host}:{client.port}", C.GREEN)

    # sync servo positions to UI immediately on connect
    await ws.send_text(json.dumps(robot.to_servo_state_msg()))

    try:
        while True:
            raw  = await ws.receive_text()
            data = json.loads(raw)
            await handle_message(ws, data, robot)

    except WebSocketDisconnect:
        log_warn("WS", f"Client disconnected  {client.host}:{client.port}")
    except Exception as e:
        log_err("WS", f"Unhandled error: {e}")
    finally:
        if ws in connections:
            connections.remove(ws)

# ── Test broadcast endpoint ───────────────────────────────────
# Used by test_ui_feedback.py to push state to all browser clients.
# POST /test/broadcast with a JSON body — server fans it out to all connections.


@app.post("/test/broadcast")
async def test_broadcast(request: Request):
    data = await request.json()
    msg  = json.dumps(data)
    dead = []
    for ws in connections:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connections.remove(ws)
    return {"sent_to": len(connections), "msg_type": data.get("type")}

# ── Test routes ───────────────────────────────────────────────
# Hit these in a browser tab while the UI is open.
# The broadcast loop picks up the changed robot state at 10Hz.
# All routes return JSON so you can see what happened.
#
#   http://localhost:8000/test/wasd
#   http://localhost:8000/test/servos
#   http://localhost:8000/test/speed
#   http://localhost:8000/test/estop
#   http://localhost:8000/test/full


@app.get("/test/wasd")
async def test_wasd():
    """Cycle W → S → A → D → stop, 1s each. Watch WASD keys light up."""
    log_info("TEST", "WASD sequence started")
    moves = [
        ("Forward  W",  0.8,  0.8),
        ("Reverse  S", -0.8, -0.8),
        ("Left     A", -0.8,  0.8),
        ("FrontRight    D",  0.8, 0.6),
        ("Right    D",  3.0, -2.1),
        ("Right    D",  3.0, 2.1),
        ("Right    D",  -3.0, 2.1),
        ("Stop      ",  -3.0, -2.1),
    ]
    for label, left, right in moves:
        robot.apply_motor_command(left, right, 1.5)
        log_info("TEST", f"WASD  {label}  L={left:+.1f}  R={right:+.1f}")
        await asyncio.sleep(1.0)
    return {"done": True, "sequence": "wasd"}


@app.get("/test/servos")
async def test_servos():
    """Sweep all servo sliders from min to max and back. 20 steps, 0.3s each = 12s total."""
    log_info("TEST", "Servo sweep started")
    ranges = {
        "SERVO_1": (0,   180),
        "SERVO_2": (0,   180),
        "SERVO_3": (0,   270),
        "SERVO_4": (-90,  90),
        "SERVO_5": (0,   180),
    }
    for direction in [range(0, 21), range(20, -1, -1)]:
        for step in direction:
            t = step / 20.0
            for sid, (lo, hi) in ranges.items():
                robot.apply_servo_angle(sid, round(lo + (hi - lo) * t, 1))
            log_info("TEST", f"Servo sweep {int(t*100):3d}%")
            await asyncio.sleep(0.3)   # slow enough to clearly see in UI

    # return to defaults
    robot.apply_servo_angle("SERVO_1", 90)
    robot.apply_servo_angle("SERVO_2", 90)
    robot.apply_servo_angle("SERVO_3", 135)
    robot.apply_servo_angle("SERVO_4", 0)
    robot.apply_servo_angle("SERVO_5", 90)
    log_info("TEST", "Servo sweep complete — centred")
    return {"done": True, "sequence": "servos"}


@app.get("/test/poses")
async def test_poses():
    """Cycle through all 5 poses, 2s each. Watch pose buttons highlight."""
    log_info("TEST", "Pose sequence started")
    for pose in ["POSE_1", "POSE_2", "POSE_3", "POSE_4", "POSE_5"]:
        robot.apply_pose(pose)
        log_info("TEST", f"Pose -> {pose}")
        await asyncio.sleep(2.0)
    robot.apply_pose(None)
    return {"done": True, "sequence": "poses"}


@app.get("/test/audio")
async def test_audio():
    """Fire each audio clip, 2s apart. Watch audio buttons highlight."""
    log_info("TEST", "Audio sequence started")
    for sound in ["BARK", "ALARM", "WHISTLE", "HOWL"]:
        robot.apply_audio(sound)
        log_info("TEST", f"Audio -> {sound}")
        await asyncio.sleep(2.0)
    return {"done": True, "sequence": "audio"}


@app.get("/test/speed")
async def test_speed():
    """Ramp speed display 0.5 -> 3.0 -> 0.5 m/s."""
    log_info("TEST", "Speed ramp started")
    for speed in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]:
        robot.apply_speed_limit(speed)
        log_info("TEST", f"Speed  {speed:.1f} m/s")
        await asyncio.sleep(0.3)
    return {"done": True, "sequence": "speed"}


@app.get("/test/estop")
async def test_estop():
    """Fire motors then trigger estop — watch motors zero out."""
    log_info("TEST", "E-stop test started")
    robot.apply_motor_command(0.8, 0.8, 1.5)
    log_info("TEST", "Motors running at 0.8")
    await asyncio.sleep(1.5)
    robot.apply_estop()
    log_info("TEST", "ESTOP fired — motors zeroed")
    await asyncio.sleep(1.0)
    robot.apply_motor_command(0.0, 0.0, 0.0)
    return {"done": True, "sequence": "estop"}


@app.get("/test/full")
async def test_full():
    """
    Run all sequences back to back.
    Open the UI then hit this once — takes ~60s to complete.
    """
    log_info("TEST", "Full sequence started")
    await test_wasd()
    await asyncio.sleep(0.5)
    await test_speed()
    await asyncio.sleep(0.5)
    await test_servos()
    await asyncio.sleep(0.5)
    await test_poses()
    await asyncio.sleep(0.5)
    await test_audio()
    await asyncio.sleep(0.5)
    await test_estop()
    log_info("TEST", "Full sequence complete")
    return {"done": True, "sequence": "full"}