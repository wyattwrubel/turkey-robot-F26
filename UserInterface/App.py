"""
app.py — CDS Control Server
"""

import asyncio
import json
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(__file__))

from contextlib       import asynccontextmanager
from pathlib          import Path

import uvicorn
from fastapi          import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from starlette.websockets import WebSocketState

from Robot       import Robot
from Camera      import PanelCamera
from Zonemap     import PanelMap
from Diagnostics import PanelDiagnostics
from Controls    import PanelControls

from config.paths import ZONE_CSV, WAYPOINTS_CSV

STATIC_DIR    = Path(__file__).parent / "Website"
MAP_CSV       = ZONE_CSV


# ------------------------------------------------------------------
# JSON serializer — handles numpy types that slip through from CV
# ------------------------------------------------------------------

def _to_json(msg: dict) -> str:
    def _default(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.bool_):    return bool(obj)
        raise TypeError(f"[JSON] Not serializable: {type(obj)} = {obj}")
    return json.dumps(msg, default=_default)


class CDSControlServer:

    def __init__(self, drivetrain_publisher, canine_publisher, macrostate_publisher, navigation_publisher):
        self.robot = Robot()
        self._last_frame_time: float = 0.0
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

        self.connections: set[WebSocket] = set()
        self.app = self._build_app()

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    async def _broadcast(self, msg: dict) -> None:
        if not self.connections:
            return
        try:
            text = _to_json(msg)
        except TypeError as e:
            print(f"[Broadcast] Serialization error: {e}")
            return

        await asyncio.gather(
            *[self._safe_send(ws, text) for ws in set(self.connections)],
            return_exceptions=True,
        )

    async def _safe_send(self, ws: WebSocket, text: str) -> None:
        if ws.client_state != WebSocketState.CONNECTED:
            self.connections.discard(ws)
            return
        try:
            await asyncio.wait_for(ws.send_text(text), timeout=2)
        except (asyncio.TimeoutError, WebSocketDisconnect, AssertionError):
            self.connections.discard(ws)
            print(f"[WS] Removed dead client {ws.client.host}:{ws.client.port}")
        except Exception as e:
            self.connections.discard(ws)
            print(f"[WS] Client error, removed: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Receiver shims
    # ------------------------------------------------------------------

    async def VisionObject_Receiver(self, data: dict) -> None:
        await self.diagnostics.VisionObject_Receiver(data, self._broadcast)

    async def Geofence_Receiver(self, data: dict) -> None:
        await self.diagnostics.Geofence_Receiver(data, self._broadcast)

    async def VideoFeed_Receiver(self, data) -> None:
        await self.camera.VideoFeed_Receiver(data, self._broadcast)

    # ------------------------------------------------------------------
    # 10 Hz state broadcast
    # ------------------------------------------------------------------

    async def _broadcast_state(self) -> None:
        while True:
            await asyncio.sleep(0.1)

            if not self.robot.check_heartbeat() and not self.robot._estop:
                print("[UI WATCHDOG] Heartbeat timeout — engaging estop")
                self.robot.estop_engage()
                await self.controls._pub_estop()

            if not self.connections:
                continue
            try:
                msg  = self.robot.to_state_msg()
                text = _to_json(msg)
            except TypeError as e:
                print(f"[State] Serialization error: {e}")
                continue
            except Exception as e:
                print(f"[State] Unexpected error: {type(e).__name__}: {e}")
                continue

            await asyncio.gather(
                *[self._safe_send(ws, text) for ws in set(self.connections)],
                return_exceptions=True,
            )

    # ------------------------------------------------------------------
    # Serve
    # ------------------------------------------------------------------

    async def serve(self) -> None:
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=8000,
            log_level="warning",
            reload=False,
            timeout_graceful_shutdown=0,
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None
        await server.serve()

    # ------------------------------------------------------------------
    # App builder
    # ------------------------------------------------------------------

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

        @app.get("/camera")
        async def camera_feed():
            return await self.camera.mjpeg_endpoint()

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):

            # Drop plain HTTP probes (e.g. ESP32 router) that aren't
            # actual WebSocket upgrade requests
            if ws.headers.get("upgrade", "").lower() != "websocket":
                await ws.close()
                return

            await ws.accept()
            print(f"[WS] Connected    {ws.client.host}:{ws.client.port}")

            try:
                for coro in [
                    self.controls.sync_client(ws),
                    self.map.sync_client(ws),
                    self.camera.sync_client(ws),
                ]:
                    await coro

                await ws.send_text(_to_json({
                    "type":      "mode_state",
                    "is_manual": self.robot.ui_locked,
                }))

            except Exception as e:
                print(f"[WS] Sync failed, dropping client: {type(e).__name__}: {e}")
                return

            self.connections.add(ws)

            try:
                while True:
                    data = json.loads(await ws.receive_text())
                    await self.controls.handle_ui_message(data)
            except WebSocketDisconnect:
                print(f"[WS] Disconnected {ws.client.host}:{ws.client.port}")
            except Exception as e:
                print(f"[WS] Error: {type(e).__name__}: {e}")
            finally:
                self.connections.discard(ws)

        return app