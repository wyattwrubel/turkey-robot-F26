"""
app.py — CDS Control Server

Wires together the four UI panels and owns the WebSocket endpoint,
the 10 Hz state broadcast loop, and the HTTP static file routes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE (in your main.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    from app import CDSControlServer

    ui = CDSControlServer(
        drivetrain_publisher,
        canine_publisher,
        macrostate_publisher,
    )

    # Wire inbound receivers to your system publishers:
    pose_pub.subscribe(ui.diagnostics.RobotPose_Receiver)
    uwb_pub.subscribe(ui.diagnostics.UWB_Sensors_Receiver)
    imu_pub.subscribe(ui.diagnostics.Accelerometer_Receiver)
    vision_pub.subscribe(ui.diagnostics.VisionObject_Receiver)
    macrostate_pub.subscribe(ui.diagnostics.MacroState_Receiver)
    navigation_pub.subscribe(ui.map.Navigation_Receiver)
    canine_pub.subscribe(ui.controls.Canine_Receiver)
    video_pub.subscribe(ui.camera.VideoFeed_Receiver)

    await asyncio.gather(..., ui.serve())

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Panel modules
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ui.camera       → PanelCamera.py      (VideoFeed)
    ui.map          → PanelMap.py         (Navigation, CSV obstacles)
    ui.diagnostics  → PanelDiagnostics.py (Pose, UWB, IMU, MacroState, Vision)
    ui.controls     → PanelControls.py    (Canine, UI handlers, publishers)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HTTP routes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    GET /           → Website/index.html
    GET /styles.css → Website/styles.css
    GET /script.js  → Website/script.js
    WS  /ws         → bidirectional control channel
"""

import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(__file__))

from contextlib       import asynccontextmanager
from pathlib          import Path

import uvicorn
from fastapi          import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from Robot             import Robot
from Camera      import PanelCamera
from Zonemap     import PanelMap
from Diagnostics import PanelDiagnostics
from Controls    import PanelControls

STATIC_DIR = Path(__file__).parent / "Website"
MAP_CSV       = Path("/home/turkeysquad/ME450/MainDesign/Navigation/Zone.csv")
WAYPOINTS_CSV = Path("/home/turkeysquad/ME450/MainDesign/Navigation/Waypoints.csv")


class CDSControlServer:

    def __init__(self, drivetrain_publisher, canine_publisher, macrostate_publisher, navigation_publisher):
        # ── Shared state ───────────────────────────────────────────────────────
        self.robot = Robot()

        # ── Panel modules ──────────────────────────────────────────────────────
        self.camera      = PanelCamera()
        self.map         = PanelMap(MAP_CSV, WAYPOINTS_CSV, self.robot)
        self.diagnostics = PanelDiagnostics(self.robot)
        self.controls    = PanelControls(
            self.robot,
            drivetrain_publisher,
            canine_publisher,
            macrostate_publisher,
            navigation_publisher,
        )

        # ── WebSocket connections ──────────────────────────────────────────────
        self.connections: list[WebSocket] = []

        self.app = self._build_app()

    # ── Broadcast helper (passed into panels that need to push directly) ──────

    async def _broadcast(self, msg: dict) -> None:
        text = json.dumps(msg)
        dead = []
        for ws in list(self.connections):   # ← iterate a snapshot
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.connections:      # ← guard against double-remove
                self.connections.remove(ws)

    # ── Receiver shims ────────────────────────────────────────────────────────
    # Panels that need to broadcast require _broadcast passed in.
    # These thin wrappers keep the subscribe() calls in main.py clean.

    async def VisionObject_Receiver(self, data: dict) -> None:
        await self.diagnostics.VisionObject_Receiver(data, self._broadcast)

    async def VideoFeed_Receiver(self, data) -> None:
        await self.camera.VideoFeed_Receiver(data, self._broadcast)

    # ── 10 Hz state broadcast ─────────────────────────────────────────────────

    async def _broadcast_state(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            # self.robot.tick(0.1)
            if self.connections:
                await self._broadcast(self.robot.to_state_msg())

    # ── Serve ─────────────────────────────────────────────────────────────────

    async def serve(self) -> None:
        """Drop into asyncio.gather() to start the server."""
        config = uvicorn.Config(self.app, host="0.0.0.0", port=8000,
                                log_level="warning", reload=False)
        await uvicorn.Server(config).serve()

    # ── App builder ───────────────────────────────────────────────────────────

    def _build_app(self) -> FastAPI:

        @asynccontextmanager
        async def lifespan(app: FastAPI):
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

            # Sync all panels on connect
            await self.controls.sync_client(ws)
            await self.map.sync_client(ws)
            await self.camera.sync_client(ws)
            # Send manual mode state so browser restores correctly after refresh
            await ws.send_text(json.dumps({
                "type":      "mode_state",
                "is_manual": self.robot.ui_locked,
            }))

            try:
                while True:
                    data = json.loads(await ws.receive_text())
                    await self.controls.handle_ui_message(data)
            except WebSocketDisconnect:
                print(f"[WS] Disconnected {ws.client.host}:{ws.client.port}")
            except Exception as e:
                print(f"[WS] Error: {e}")
            finally:
                if ws in self.connections:
                    self.connections.remove(ws)

        return app