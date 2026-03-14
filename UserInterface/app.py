"""
app.py — FastAPI application, routes, and WebSocket endpoint

Exposes:
    GET  /           serves index.html
    GET  /styles.css serves styles.css
    GET  /script.js  serves script.js
    WS   /ws         bidirectional robot control channel

Usage in main.py
----------------
    publisher1 = AsyncPublisher()   # e.g. VisionObject publisher
    publisher2 = AsyncPublisher()   # e.g. RobotPose publisher
    publisher3 = AsyncPublisher()   # e.g. UWB_Sensors publisher
    publisher4 = AsyncPublisher()   # e.g. StatePlanner publisher

    ui = CDSControlServer(publisher1, publisher2, publisher3, publisher4)

    # Subscribe the UI's inbound receivers to whichever publishers feed them:
    publisher1.subscribe(ui.VisionObject_Receiver)
    publisher2.subscribe(ui.RobotPose_Receiver)
    publisher3.subscribe(ui.UWB_Sensors_Receiver)
    publisher3.subscribe(ui.Accelerometer_Receiver)
    publisher4.subscribe(ui.StatePlanner_Receiver)
    publisher4.subscribe(ui.state_receiver)         # raw state → all WS clients

    await asyncio.gather(
        ...,
        ui.serve(),
    )
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from broadcast import broadcast_frames, broadcast_state
from handlers  import handle_message
from robot     import RobotState, RobotInterface

STATIC_DIR = Path(__file__).parent / "Website"

class CDSControlServer:
    """
    FastAPI server wrapped as a pub/sub participant.

    Parameters
    ----------
    publisher1 : AsyncPublisher
        First inbound publisher (e.g. VisionObject).
    publisher2 : AsyncPublisher
        Second inbound publisher (e.g. RobotPose).
    publisher3 : AsyncPublisher
        Third inbound publisher (e.g. UWB / Accelerometer).
    publisher4 : AsyncPublisher
        Fourth inbound publisher (e.g. StatePlanner).

    The server also exposes an internal command_publisher that fires
    for every WebSocket message received from the browser — subscribe
    your downstream handlers to it via ui.command_publisher.subscribe(...).

    Subscriber methods (wire these to publishers in main.py)
    --------------------------------------------------------
        ui.state_receiver          — fans any dict out to all WS clients as JSON
        ui.VisionObject_Receiver   — handles VisionObject messages
        ui.RobotPose_Receiver      — handles RobotPose messages
        ui.UWB_Sensors_Receiver    — handles UWB sensor messages
        ui.Accelerometer_Receiver  — handles Accelerometer messages
        ui.StatePlanner_Receiver   — handles StatePlanner messages
    """

    def __init__(
        self,
        publisher1,
        publisher2,
        publisher3,
        publisher4,
    ):
        # ── Inbound publishers (data arriving from the rest of the system) ──
        self.publisher1 = publisher1
        self.publisher2 = publisher2
        self.publisher3 = publisher3
        self.publisher4 = publisher4

        # ── Outbound publisher (UI → system) ───────────────────────────────
        # Subscribe your downstream handlers here, e.g.:
        #   ui.command_publisher.subscribe(motor_driver.command_receiver)
        from broadcast import broadcast_frames, broadcast_state  # local import avoids circular
        # (command_publisher imported dynamically to keep this module self-contained)
        self._command_subscribers = []

        # ── WebSocket connection list ───────────────────────────────────────
        self.connections: list[WebSocket] = []
        self._last_vision: dict = {"type": "vision", "turkeys": [], "humans": []}  # most recent detection frame

        # ── Robot state (mock by default — swap in your real class) ─────────
        self.robot: RobotInterface = RobotState()

        self.app = self._build_app()

    # ── Outbound command publisher shim ──────────────────────────────────────
    # Allows callers to do:  ui.command_publisher.subscribe(my_fn)
    # without needing a real AsyncPublisher injected for outbound traffic.

    class _CommandPublisher:
        def __init__(self, server):
            self._server = server

        def subscribe(self, cb):
            self._server._command_subscribers.append(cb)

        async def publish(self, data: dict):
            for cb in self._server._command_subscribers:
                try:
                    await cb(data)
                except Exception as e:
                    log_err("CMD_PUB", f"Subscriber error: {e}")

    @property
    def command_publisher(self):
        return self._CommandPublisher(self)

    # ── Subscriber callbacks ──────────────────────────────────────────────────

    async def state_receiver(self, data: dict) -> None:
        """
        Generic state fan-out — subscribe this to any publisher whose output
        should be forwarded verbatim to every connected browser client.

        Example:
            pose_publisher.subscribe(ui.state_receiver)
        """
        msg  = json.dumps(data)
        dead = []
        for ws in self.connections:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connections.remove(ws)

    async def VisionObject_Receiver(self, data: dict) -> None:
        """
        VisionObject Channel (Ch. 1)

        Expected keys:
          Id       = "DynamicObjects"
          Turkeys  = [[world_x, world_y], ...]   — world coords for map overlay
          Humans   = [[world_x, world_y], ...]
          Bboxes   = [                            — pixel coords for camera overlay
            { "label": str, "conf": float,
              "x1": int, "y1": int, "x2": int, "y2": int },
            ...
          ]

        Writes to broadcast.set_vision() — the single source of truth.
        broadcast_frames() picks it up and delivers to all clients at 2 Hz.
        No direct WebSocket pushes here — that was causing competing messages
        and flickering when test sequences also sent vision data.
        """
        from broadcast import set_vision
        turkeys = data.get("Turkeys", [])
        humans  = data.get("Humans",  [])
        bboxes  = data.get("Bboxes",  [])
        # Only overwrite bboxes if the message actually contains them.
        # test_vision() sends world positions only (no Bboxes key) — we keep
        # the existing demo/real bboxes so they don't disappear during chase.
        from broadcast import _current_vision_msg
        import json as _json
        if "Bboxes" not in data and _current_vision_msg is not None:
            existing = _json.loads(_current_vision_msg)
            bboxes = existing.get("bboxes", [])

        set_vision(turkeys, humans, bboxes)

    async def RobotPose_Receiver(self, data: dict) -> None:
        """
        RobotPose Channel (Ch. 4)
        Expected format: numpy array [X, Y, theta] serialised as a list.
        Also handles UncertaintyPose if sigma fields are present.
        """
        import numpy as np
        arr = np.array(data, dtype=np.float64).flatten()

        if arr.size >= 3:
            self.robot.apply_robot_pose(
                x     = float(arr[0]),
                y     = float(arr[1]),
                theta = float(arr[2]),
            )

        # UncertaintyPose piggybacks here if published as [x,y,theta,sx,sy,stheta]
        if arr.size >= 6:
            self.robot.apply_uncertainty(
                sx     = float(arr[3]),
                sy     = float(arr[4]),
                stheta = float(arr[5]),
            )

    async def UWB_Sensors_Receiver(self, data: dict) -> None:
        """
        UWB_Values Channel (Ch. 8.1)
        Expected format: array [UWB_X, UWB_Y]
        uwb_age is derived automatically from time-since-last-message.
        """
        import numpy as np
        arr = np.array(data, dtype=np.float64).flatten()
        if arr.size >= 2:
            self.robot.apply_uwb(
                uwb_x = float(arr[0]),
                uwb_y = float(arr[1]),
            )

    async def Accelerometer_Receiver(self, data: dict) -> None:
        """
        Acceler_Values Channel (Ch. 8.2)
        Expected format: array [theta (rad)]
        """
        import numpy as np
        arr = np.array(data, dtype=np.float64).flatten()
        if arr.size >= 1:
            self.robot.apply_imu(imu_theta=float(arr[0]))

    async def StatePlanner_Receiver(self, data: dict) -> None:
        """
        Navigation Channel — extended mission fields (Ch. 6 + extensions)
        Handles all Navigation message Id types:

          Id="Move"     → nav_cmd={"type":"Move",   "x":..., "y":...}
          Id="Orient"   → nav_cmd={"type":"Orient", "x":..., "y":..., "theta":...}
          Id="SpeedCap" → speed limit             (routes to apply_speed_limit)
          Id="Chase"    → sys_state=Chase,    destination=hypot(Cmd),  nav_cmd cleared
          Id="Scan"     → sys_state=Scanning, destination cleared,       nav_cmd cleared
          Id="Alert"    → sys_state=Alert,    destination cleared,       nav_cmd cleared
          Id="UnlockUI" → relinquish control      (no robot state change needed)

        wp_id and wp_dist come from the separate WaypointInfo protocol
        via WaypointInfo_Receiver — not from Move/Orient/Chase.
        """
        import numpy as np
        msg_id = data.get("Id", "")

        if msg_id == "SpeedCap":
            cmd = np.array(data.get("Cmd", [1.5]), dtype=np.float64).flatten()
            self.robot.apply_speed_limit(float(cmd[0]))

        elif msg_id == "Move":
            cmd = np.array(data.get("Cmd", [0, 0, 0]), dtype=np.float64).flatten()
            x = round(float(cmd[0]), 3) if cmd.size > 0 else 0.0
            y = round(float(cmd[1]), 3) if cmd.size > 1 else 0.0
            self.robot.apply_mission_state(
                nav_cmd = {"type": "Move", "x": x, "y": y},
            )

        elif msg_id == "Orient":
            cmd = np.array(data.get("Cmd", [0, 0, 0]), dtype=np.float64).flatten()
            x     = round(float(cmd[0]), 3) if cmd.size > 0 else 0.0
            y     = round(float(cmd[1]), 3) if cmd.size > 1 else 0.0
            theta = round(float(cmd[2]), 4) if cmd.size > 2 else 0.0
            self.robot.apply_mission_state(
                nav_cmd = {"type": "Orient", "x": x, "y": y, "theta": theta},
            )

        elif msg_id == "Chase":
            import math
            cmd = np.array(data.get("Cmd", [0, 0, 0]), dtype=np.float64).flatten()
            destination = round(math.hypot(float(cmd[0]), float(cmd[1])), 2) if cmd.size >= 2 else None
            self.robot.apply_mission_state(
                sys_state   = "Chase",
                destination = destination,
                nav_cmd     = None,
            )

        elif msg_id == "Scan":
            self.robot.apply_mission_state(
                sys_state   = "Scanning",
                destination = None,
                nav_cmd     = None,
            )

        elif msg_id == "Alert":
            self.robot.apply_mission_state(
                sys_state   = "Alert",
                destination = None,
                nav_cmd     = None,
            )

        elif msg_id == "UnlockUI":

        else:


    async def WaypointInfo_Receiver(self, data: dict) -> None:
        """
        WaypointInfo Protocol — separate channel for waypoint progress display.
        Wire up: waypoint_publisher.subscribe(ui.WaypointInfo_Receiver)

        Expected keys:
          wp_id   — string label e.g. 'WP 3 / 7'
          wp_dist — float, metres remaining to the waypoint
        """
        wp_id   = data.get("wp_id",   "—")
        wp_dist = float(data.get("wp_dist", 0.0))
        self.robot.apply_waypoint_info(wp_id=wp_id, wp_dist=wp_dist)

    async def DriveTrain_Receiver(self, data: dict) -> None:
        """
        DriveTrain Channel (Ch. 3) — inbound motor command echo.
        Wire up: drivetrain_publisher.subscribe(ui.DriveTrain_Receiver)

        Reads the Id field to set drive_source:
          Id="UI"      → manual override from browser
          Id="System"  → autonomous command from navigation
        ESTOP is internal to the UI — handled by handlers.py, not this channel.
        """
        import numpy as np
        msg_id = data.get("Id", "System")
        cmd    = np.array(data.get("Cmd", [0.0, 0.0]), dtype=np.float64).flatten()
        left   = float(cmd[0]) if cmd.size > 0 else 0.0
        right  = float(cmd[1]) if cmd.size > 1 else 0.0
        self.robot.apply_motor_command(left, right, source=msg_id)


    async def Canine_Receiver(self, data: dict) -> None:
        """
        Canine Channel (Ch. 2) — Pose, Servo, Audio commands from system.
        Wire up: canine_publisher.subscribe(ui.Canine_Receiver)

          Type="Pose"   → Id, Data in ["Pose1".."Pose5"]
          Type="Servo"  → Id, Name in servo_angles keys, Data = angle (float)
          Type="Audio"  → Id, Data in ["Audio1".."Audio5"]
          Id="UnlockUI" → relinquish canine control back to software
        """
        msg_id   = data.get("Id", "")
        msg_type = data.get("Type", "")

        if msg_id == "UnlockUI":
            return

        if msg_type == "Pose":
            pose = data.get("Data", "")
            self.robot.apply_pose(pose)

        elif msg_type == "Servo":
            name  = data.get("Name", "")
            angle = float(data.get("Data", 0.0))
            self.robot.apply_servo_angle(name, angle)

        elif msg_type == "Audio":
            sound = data.get("Data", "")
            self.robot.apply_audio(sound)

        else:

    # ── Serve coroutine (drop into asyncio.gather) ────────────────────────────
    async def serve(self):
        config = uvicorn.Config(
            self.app,
            host      = "0.0.0.0",
            port      = 8000,
            log_level = "warning",
            reload    = False,
        )
        server = uvicorn.Server(config)
        await server.serve()

    # ── Internal app builder ──────────────────────────────────────────────────
    def _build_app(self) -> FastAPI:

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            print("SERVER", "CDS Control Server starting")
            print("SERVER", "UI  ->  http://localhost:8000")
            print("SERVER", "WS  ->  ws://localhost:8000/ws")
            asyncio.create_task(broadcast_state(self.connections, self.robot))
            asyncio.create_task(broadcast_frames(self.connections, self.robot))
            yield
            # shutdown cleanup here if needed

        app = FastAPI(title="CDS Control Server", lifespan=lifespan)

        # ── Static file routes ────────────────────────────────────────────────
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

        @app.get("/map.csv")
        async def map_csv():
            """
            Serves the static occupancy-grid CSV for the Zone Map panel.
            Drop your CSV file as Website/map.csv and the map will auto-load
            on page open (controlled by MAP_CSV constant in script.js).
            Returns 404 silently if no file is present — map just stays empty.
            """
            f = STATIC_DIR / "map.csv"
            if f.exists():
                return FileResponse(str(f), media_type="text/csv")
            return HTMLResponse("", status_code=404)

        # ── WebSocket endpoint ────────────────────────────────────────────────────
        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            self.connections.append(ws)
            client = ws.client
            log("WS", f"Client connected  {client.host}:{client.port}", C.GREEN)

            # Sync servo positions on connect
            await ws.send_text(json.dumps(self.robot.to_servo_state_msg()))
            # Send current frame + vision state once on connect.
            # Both are maintained as single sources of truth in broadcast.py.
            # The frame is never re-sent in the loop (static image = send once).
            # Vision is re-sent at 2 Hz by broadcast_frames.
            from broadcast import _current_frame_msg, _current_vision_msg
            if _current_frame_msg  is not None:
                await ws.send_text(_current_frame_msg)
            if _current_vision_msg is not None:
                await ws.send_text(_current_vision_msg)

            try:
                while True:
                    raw  = await ws.receive_text()
                    data = json.loads(raw)

                    # Always route UI messages through handle_message so that
                    # buttons, sliders, and controls update robot state.
                    await handle_message(ws, data, self.robot)

                    # Also fan out to any external subscribers (e.g. drivetrain).
                    for cb in self._command_subscribers:
                        try:
                            await cb(data)
                        except Exception as e:
                            log_err("WS", f"Command subscriber error: {e}")

            except WebSocketDisconnect:
                log_warn("WS", f"Client disconnected  {client.host}:{client.port}")
            except Exception as e:
                log_err("WS", f"Unhandled error: {e}")
            finally:
                if ws in self.connections:
                    self.connections.remove(ws)