"""
zonemap.py — Zone Map

Owns everything related to the zone map panel:
  - Loading the occupancy grid from Zone.csv at startup
  - Loading waypoints from Waypoints.csv at startup
  - Receiving Navigation channel messages (Move/Orient/SpeedCap)
  - Sending obstacles, waypoints, and grid dimensions to new clients

Inbound receiver (subscribe in main.py):
    navigation_pub.subscribe(ui.map.Navigation_Receiver)

Zone.csv format — occupancy grid:
    Row 0 (header):  blank, 0, 1, 2, ... N
    Rows 1+:         Y, cell, ...   (1=wall, blank=free)

Waypoints.csv format — one waypoint per row:
    label, x, y
    WP1, 20.0, 15.0
    WP2, 45.0, 30.0
"""

import csv
import json
from pathlib import Path

import numpy as np
from fastapi import WebSocket

from Robot import Robot


class PanelMap:

    def __init__(self, csv_path: Path, waypoints_path: Path, robot: Robot):
        self._robot             = robot
        self._csv_path          = csv_path
        self._waypoints_path    = waypoints_path
        self._obstacles:  list[dict] = []
        self._waypoints:  list[dict] = []
        self._map_w: int  = 100
        self._map_h: int  = 100
        self._load_csv()
        self._load_waypoints()

    # ── CSV loader ────────────────────────────────────────────────────────────

    def _load_csv(self) -> None:
        if not self._csv_path.exists():
            print(f"[MAP] {self._csv_path} not found — map will be empty")
            return

        walls = []
        max_x = max_y = 0

        with open(self._csv_path, newline="") as f:
            rows = list(csv.reader(f))

        if len(rows) < 2:
            print("[MAP] CSV is empty or has no data rows")
            return

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

        self._obstacles = walls
        self._map_w     = max_x + 1   # +1 aligns grid edge with last cell boundary
        self._map_h     = max_y + 1
        print(f"[MAP] Loaded {len(walls)} wall cells  "
              f"(grid {self._map_w} × {self._map_h})")

    def _load_waypoints(self) -> None:
        if not self._waypoints_path.exists():
            print(f"[MAP] {self._waypoints_path} not found — no waypoints")
            return

        waypoints = []
        with open(self._waypoints_path, newline="") as f:
            for row in csv.reader(f):
                row = [c.strip() for c in row]
                if not row or row[0].startswith("#"):
                    continue
                try:
                    label = row[0]
                    x     = float(row[1])
                    y     = float(row[2])
                    waypoints.append({"label": label, "x": x, "y": y})
                except (ValueError, IndexError):
                    continue

        self._waypoints = waypoints
        print(f"[MAP] Loaded {len(waypoints)} waypoints")

    # ── Inbound receiver ──────────────────────────────────────────────────────

    async def Navigation_Receiver(self, data: dict) -> None:
        """
        Navigation channel — Move, Orient, SpeedCap.

          Id="Move"     Cmd=[X, Y, _]       → target x, y  (theta defaults to 0)
          Id="Orient"   Cmd=[X, Y, theta]   → target x, y, theta
          Id="SpeedCap" Cmd=[speed_m/s]     → speed limit update
        """
        cmd    = np.asarray(data.get("Cmd", [0, 0, 0]), dtype=float).flatten()
        msg_id = data.get("Id", "")

        # In manual mode the UI owns the target — ignore Move/Orient from system
        if msg_id == "Move" and cmd.size >= 2:
            if not self._robot.ui_locked:
                self._robot.set_target(cmd[0], cmd[1], theta=0.0, nav_type="Move")

        elif msg_id == "Orient" and cmd.size >= 3:
            if not self._robot.ui_locked:
                self._robot.set_target(cmd[0], cmd[1], cmd[2], nav_type="Orient")

        elif msg_id == "SpeedCap" and cmd.size >= 1:
            self._robot.set_speed(float(cmd[0]))

    # ── On-connect sync ───────────────────────────────────────────────────────

    async def sync_client(self, ws: WebSocket) -> None:
        """Send obstacle list and grid dimensions to a newly connected client."""
        await ws.send_text(json.dumps({
            "type":      "map_obstacles",
            "obstacles": self._obstacles,
            "waypoints": self._waypoints,
            "map_w":     self._map_w,
            "map_h":     self._map_h,
        }))