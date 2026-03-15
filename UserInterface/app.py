"""
app.py — CDS Control Server

Single entry point for the UI backend. Handles:
  - WebSocket connection to browser
  - Inbound receivers (robot → UI state)
  - Outbound publishing (UI → robot hardware)
  - 10 Hz state broadcast to all connected clients

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE (in your main.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    from app import CDSControlServer

    ui = CDSControlServer(
        drivetrain_publisher,   # outbound: motor + speed commands
        canine_publisher,       # outbound: pose, servo, audio commands
        macrostate_publisher,   # outbound: Chase / Scan / Alert commands
    )

    # Wire up whichever inbound channels your system publishes:
    robot_pose_pub.subscribe(ui.RobotPose_Receiver)
    uwb_pub.subscribe(ui.UWB_Sensors_Receiver)
    imu_pub.subscribe(ui.Accelerometer_Receiver)
    vision_pub.subscribe(ui.VisionObject_Receiver)
    macrostate_pub.subscribe(ui.MacroState_Receiver)
    navigation_pub.subscribe(ui.Navigation_Receiver)
    canine_pub.subscribe(ui.Canine_Receiver)
    video_pub.subscribe(ui.VideoFeed_Receiver)

    await asyncio.gather(..., ui.serve())

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HTTP routes served
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    GET /           → Website/index.html
    GET /styles.css → Website/styles.css
    GET /script.js  → Website/script.js
    WS  /ws         → bidirectional control channel
"""

import asyncio
import base64
import json
import sys
import os
sys.path.append(os.path.dirname(__file__))

from contextlib import asynccontextmanager
from pathlib    import Path

import numpy as np
import uvicorn
from fastapi              import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses    import FileResponse, HTMLResponse

from robot import RobotState

STATIC_DIR = Path(__file__).parent / "Website"
MAP_CSV    = Path("/home/turkeysquad/ME450/MainDesign/Navigation/Zone.csv")

# Direction vectors for each key: [left_motor, right_motor] normalised to [-1, 1]
# Backend applies speed_limit scaling before publishing to hardware
KEY_DIRECTIONS = {
    "w": ( 1.0,  1.0),   # forward
    "s": (-1.0, -1.0),   # reverse
    "a": (-1.0,  1.0),   # turn left
    "d": ( 1.0, -1.0),   # turn right
}


class CDSControlServer:

    def __init__(self, drivetrain_publisher, canine_publisher, macrostate_publisher):
        # ── Outbound publishers (UI → robot) ───────────────────
        self._drivetrain_pub  = drivetrain_publisher
        self._canine_pub      = canine_publisher
        self._macrostate_pub  = macrostate_publisher

        # ── State + connections ─────────────────────────────────
        self.robot       = RobotState()
        self.connections: list[WebSocket] = []

        # ── Camera: last frame held for late-joining clients ────
        self._current_frame: str | None = None

        # ── Map obstacles: loaded once from CSV at startup ──────
        self._map_obstacles: list[dict] = []
        self._map_w: int = 100
        self._map_h: int = 100

        self.app = self._build_app()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # INBOUND RECEIVERS
    # Subscribe these to your system publishers in main.py.
    # Each receiver unpacks its protocol array and updates robot state.
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def RobotPose_Receiver(self, data) -> None:
        """1×3 array [X (m), Y (m), theta (rad)]"""
        arr = np.asarray(data, dtype=float).flatten()
        if arr.size >= 3:
            self.robot.apply_robot_pose(arr[0], arr[1], arr[2])

    async def UWB_Sensors_Receiver(self, data) -> None:
        """1×2 array [UWB_X (m), UWB_Y (m)]"""
        arr = np.asarray(data, dtype=float).flatten()
        if arr.size >= 2:
            self.robot.apply_uwb(arr[0], arr[1])

    async def Accelerometer_Receiver(self, data) -> None:
        """1×1 array [theta (rad)]"""
        arr = np.asarray(data, dtype=float).flatten()
        if arr.size >= 1:
            self.robot.apply_imu(arr[0])

    async def VisionObject_Receiver(self, data: dict) -> None:
        """
        Dict with keys:
          Turkeys: T×3 array [x, y, angle_rad]
          Humans:  P×3 array [x, y, angle_rad]
        Broadcasts a 'vision' message to all connected clients.
        """
        turkeys = np.asarray(data.get("Turkeys", []), dtype=float)
        humans  = np.asarray(data.get("Humans",  []), dtype=float)
        await self._broadcast({
            "type":    "vision",
            "turkeys": turkeys.tolist() if turkeys.size > 0 else [],
            "humans":  humans.tolist()  if humans.size  > 0 else [],
        })

    async def MacroState_Receiver(self, data: dict) -> None:
        """
        Dict with key Id:
          "Scan"     → Scanning
          "Alert"    → Alert
          "Chase"    → Chase
          "UnlockUI" → Scanning (auto mode restored)
        """
        mapping = {"Scan": "Scanning", "Alert": "Alert",
                   "Chase": "Chase",   "UnlockUI": "Scanning"}
        state = mapping.get(data.get("Id", ""))
        if state:
            self.robot.apply_sys_state(state)

    async def Navigation_Receiver(self, data: dict) -> None:
        """
        Dict with keys Id and Cmd:
          Id="Move"     Cmd=[X, Y, _]      → update target x, y (theta=0)
          Id="Orient"   Cmd=[X, Y, theta]  → update target x, y, theta
          Id="SpeedCap" Cmd=[speed_m/s]    → update speed limit
        """
        cmd    = np.asarray(data.get("Cmd", [0, 0, 0]), dtype=float).flatten()
        msg_id = data.get("Id", "")

        if msg_id == "Move" and cmd.size >= 2:
            self.robot.apply_target(cmd[0], cmd[1], theta=0.0)
        elif msg_id == "Orient" and cmd.size >= 3:
            self.robot.apply_target(cmd[0], cmd[1], cmd[2])
        elif msg_id == "SpeedCap" and cmd.size >= 1:
            self.robot.apply_speed_limit(float(cmd[0]))

    async def Canine_Receiver(self, data: dict) -> None:
        """
        Dict with keys Id, Type, Name, Data:
          Type="Pose"   Data="Pose1".."Pose5"
          Type="Servo"  Name=servo_id  Data=angle (float)
          Type="Audio"  Data=audio_id
          Id="UnlockUI" → ignored
        """
        if data.get("Id") == "UnlockUI":
            return
        msg_type = data.get("Type", "")
        if msg_type == "Pose":
            self.robot.apply_pose(data.get("Data", ""))
        elif msg_type == "Servo":
            self.robot.apply_servo_angle(data.get("Name", ""), float(data.get("Data", 0.0)))
        elif msg_type == "Audio":
            self.robot.apply_audio(data.get("Data", ""))

    async def VideoFeed_Receiver(self, data) -> None:
        """
        Raw JPEG bytes or a base64-encoded string.
        Stores the latest frame and pushes it to all connected clients.
        """
        if isinstance(data, (bytes, bytearray)):
            b64 = base64.b64encode(data).decode("ascii")
        else:
            b64 = str(data)
        self._current_frame = b64
        await self._broadcast({"type": "frame", "image": b64})

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # INBOUND UI HANDLERS  (browser → server, routed from /ws)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _handle_ui_message(self, data: dict) -> None:
        """Route a WebSocket message from the browser to state + publisher."""
        msg_type = data.get("type", "")

        if msg_type == "key_command":
            key   = data.get("key",   "")
            state = data.get("state", "")

            if state == "press" and key in KEY_DIRECTIONS:
                left, right = KEY_DIRECTIONS[key]
                # Scale by current speed limit before sending to hardware
                left  *= self.robot.speed_limit
                right *= self.robot.speed_limit
                self.robot.apply_motor_command(left, right)
                await self._pub_drivetrain("UI", left, right)

            elif state == "release":
                # Handles single key release and 'all' bulk release (mode switch)
                self.robot.apply_motor_command(0.0, 0.0)
                await self._pub_drivetrain("UI", 0.0, 0.0)

        elif msg_type == "speed_limit":
            speed = float(data.get("speed", 1.5))
            self.robot.apply_speed_limit(speed)
            await self._pub_speed(speed)

        elif msg_type == "servo_angle":
            servo_id = data.get("servo", "")
            angle    = float(data.get("angle", 0.0))
            self.robot.apply_servo_angle(servo_id, angle)
            await self._pub_servo(servo_id, angle)

        elif msg_type == "pose_command":
            pose = data.get("pose", "")
            self.robot.apply_pose(pose)
            await self._pub_pose(pose)

        elif msg_type == "audio_play":
            sound = data.get("sound", "")
            self.robot.apply_audio(sound)
            await self._pub_audio(sound)

        elif msg_type == "macrostate_command":
            state = data.get("state", "")
            self.robot.apply_sys_state(state)
            await self._macrostate_pub.publish({"Id": state})

        elif msg_type == "estop":
            self.robot.apply_estop_engage()
            await self._pub_drivetrain("UI", 0.0, 0.0)

        elif msg_type == "estop_reset":
            self.robot.apply_estop_release()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # OUTBOUND PUBLISHER STUBS
    # These format and dispatch protocol messages to your hardware.
    # Fill in any additional logic (e.g. scaling, limits) here.
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _pub_drivetrain(self, source: str, left: float, right: float) -> None:
        """Drivetrain Channel — motor command."""
        await self._drivetrain_pub.publish({
            "Id":  source,
            "Cmd": [left, right],
        })

    async def _pub_speed(self, speed: float) -> None:
        """Drivetrain Channel — speed cap."""
        await self._drivetrain_pub.publish({
            "Id":  "SpeedCap",
            "Cmd": [speed],
        })

    async def _pub_pose(self, pose_id: str) -> None:
        """Canine Channel — pose command."""
        await self._canine_pub.publish({
            "Id":   "UI",
            "Type": "Pose",
            "Data": pose_id,
        })

    async def _pub_servo(self, servo_id: str, angle: float) -> None:
        """Canine Channel — raw servo command."""
        await self._canine_pub.publish({
            "Id":   "UI",
            "Type": "Servo",
            "Name": servo_id,
            "Data": angle,
        })

    async def _pub_audio(self, audio_id: str) -> None:
        """Canine Channel — audio command."""
        await self._canine_pub.publish({
            "Id":   "UI",
            "Type": "Audio",
            "Data": audio_id,
        })

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # INTERNAL — broadcast, map loader, server
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _broadcast(self, msg: dict) -> None:
        """Serialize and send a dict to every connected browser client."""
        text = json.dumps(msg)
        dead = []
        for ws in self.connections:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connections.remove(ws)

    async def _broadcast_state(self) -> None:
        """Push robot state to all clients at 10 Hz."""
        while True:
            await asyncio.sleep(0.1)
            self.robot.tick(0.1)
            if self.connections:
                await self._broadcast(self.robot.to_state_msg())

    def _load_map_csv(self) -> None:
        """
        Load Website/map.csv at startup.

        Expected format — occupancy grid:
          Row 0 (header): blank, 0, 1, 2, ... N   ← column X indices
          Rows 1+:        Y, cell, cell, ...       ← row Y index, then 1=wall blank=free

        Example (5×3 grid):
          ,0,1,2,3,4
          0,1,1,1,1,1
          1,1,,,1,
          2,1,1,1,1,1

        Grid dimensions are inferred from the data and sent to the browser
        alongside the obstacle list so the map canvas resizes automatically.
        """
        if not MAP_CSV.exists():
            print(f"[MAP] {MAP_CSV} not found — map will be empty")
            return

        import csv as _csv
        walls   = []
        max_x   = 0
        max_y   = 0

        with open(MAP_CSV, newline="") as f:
            reader = _csv.reader(f)
            rows   = list(reader)

        if len(rows) < 2:
            print("[MAP] map.csv is empty or has no data rows")
            return

        # Row 0: header — first cell blank, rest are column X indices
        col_headers = [c.strip() for c in rows[0][1:]]

        for row in rows[1:]:
            if not row or not row[0].strip():
                continue
            try:
                y = int(row[0].strip())
            except ValueError:
                continue
            for c_idx, val in enumerate(row[1:]):
                if val.strip() == "1" and c_idx < len(col_headers):
                    try:
                        x     = int(col_headers[c_idx])
                        max_x = max(max_x, x)
                        max_y = max(max_y, y)
                        walls.append({"type": "wall", "x": x, "y": y})
                    except (ValueError, IndexError):
                        continue

        self._map_obstacles = walls
        # +1 so the grid edge aligns with the last cell boundary
        self._map_w = max_x + 1
        self._map_h = max_y + 1
        print(f"[MAP] Loaded {len(walls)} wall cells  "
              f"(grid {self._map_w} × {self._map_h})")

    async def serve(self) -> None:
        """Drop into asyncio.gather() to start the server."""
        config = uvicorn.Config(self.app, host="0.0.0.0", port=8000,
                                log_level="warning", reload=False)
        await uvicorn.Server(config).serve()

    def _build_app(self) -> FastAPI:

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            self._load_map_csv()
            print("[SERVER] Starting — http://localhost:8000")
            asyncio.create_task(self._broadcast_state())
            yield

        app = FastAPI(lifespan=lifespan)

        @app.get("/")
        async def root():
            f = STATIC_DIR / "index.html"
            return FileResponse(str(f)) if f.exists() else HTMLResponse("index.html not found", 404)

        @app.get("/styles.css")
        async def styles():
            f = STATIC_DIR / "styles.css"
            return FileResponse(str(f), media_type="text/css") if f.exists() else HTMLResponse("", 404)

        @app.get("/script.js")
        async def script():
            f = STATIC_DIR / "script.js"
            return FileResponse(str(f), media_type="application/javascript") if f.exists() else HTMLResponse("", 404)

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            self.connections.append(ws)
            print(f"[WS] Connected    {ws.client.host}:{ws.client.port}")

            # Sync client on connect
            await ws.send_text(json.dumps(self.robot.to_servo_state_msg()))
            await ws.send_text(json.dumps({"type": "map_obstacles", "obstacles": self._map_obstacles, "map_w": self._map_w, "map_h": self._map_h}))
            if self._current_frame:
                await ws.send_text(json.dumps({"type": "frame", "image": self._current_frame}))

            try:
                while True:
                    data = json.loads(await ws.receive_text())
                    await self._handle_ui_message(data)
            except WebSocketDisconnect:
                print(f"[WS] Disconnected {ws.client.host}:{ws.client.port}")
            except Exception as e:
                print(f"[WS] Error: {e}")
            finally:
                if ws in self.connections:
                    self.connections.remove(ws)

        return app