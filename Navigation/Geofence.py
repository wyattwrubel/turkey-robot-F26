"""
Geofence.py: the robot's own geofence (DR2 plan W2, contract C4).

The fence is a convex polygon in zone metres (x right, y down the grid, the same frame as
RobotPose and Zone.csv), read from Navigation/geofence.json. Every 0.1 s it compares the
robot's latest pose with the polygon and publishes

    {"Id": "Geofence", "State": "inside" | "near_edge" | "outside" | "unknown",
     "DistToEdge_m": float,     # + inside, - outside
     "Buffer_m": 1.4, "Fix": "sim" | "none" | "gps" | "rtk_float" | "rtk_fixed", "t": float}

    inside     more than Buffer_m from every edge
    near_edge  inside, but within Buffer_m of an edge (the planner slows to 0.5 m/s)
    outside    past an edge
    unknown    no pose yet, no pose for STALE_S, or Fix "none"

The planner acts on it (StatePlanner/GeofenceGate.py: near_edge caps speed, outside or
unknown latch STOP) and keeps Move targets inside with clip_target(x, y).

Why 1.4 m: at the 2 m/s chase cap the robot needs about 0.02 m (drivetrain braking) + 0.4 m
(E-stop latency) + 1.0 m (GPS error) to stop, so it slows down 1.4 m before the edge.

Why convex: the stop-distance and clip math below are only guaranteed for convex polygons,
and a parking-lot test area is a rectangle. A non-convex file is refused at load.

The polygon in geofence.json is a placeholder in W26's North Campus zone, for the sim. The
real one gets surveyed with RTK in the test lot (lat/lon -> zone metres comes with the F9P).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Optional

import numpy as np

from config.paths import GEOFENCE_JSON

BUFFER_M      = 1.4    # near_edge band (see "Why 1.4 m")
CLIP_MARGIN_M = 0.3    # clip_target puts targets at least this far inside the edge
STALE_S       = 1.0    # no pose for this long -> "unknown"
PERIOD_S      = 0.1    # 10 Hz
FIXES         = ("sim", "none", "gps", "rtk_float", "rtk_fixed")


class Geofence:

    def __init__(self, publisher=None, polygon=None, buffer_m: Optional[float] = None,
                 path=GEOFENCE_JSON, fix: str = "sim"):
        if polygon is None:
            with open(path) as f:
                cfg = json.load(f)
            polygon = cfg["polygon"]
            buffer_m = cfg.get("buffer_m", BUFFER_M) if buffer_m is None else buffer_m
        self.poly     = _checked_convex(np.asarray(polygon, dtype=float))
        self.centroid = self.poly.mean(axis=0)
        self.buffer_m = BUFFER_M if buffer_m is None else float(buffer_m)
        self._pub     = publisher
        self._pose: Optional[tuple] = None      # (x, y, monotonic time)
        self.set_fix(fix)

    # ── Geometry ─────────────────────────────────────────────────────────────

    def _edges(self):
        return zip(self.poly, np.roll(self.poly, -1, axis=0))

    def contains(self, x, y) -> bool:
        return self.signed_dist(x, y) >= 0.0

    def signed_dist(self, x, y) -> float:
        """Distance to the nearest edge: + inside, - outside."""
        p = np.array([x, y], dtype=float)
        d = min(np.linalg.norm(p - (a + np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1) * (b - a)))
                for a, b in self._edges())
        return float(d) if self._inside(p) else -float(d)

    def _inside(self, p) -> bool:
        """Point in polygon (ray casting; edges count as inside)."""
        x, y = p
        inside = False
        for (x1, y1), (x2, y2) in self._edges():
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
        return inside or any(_on_segment(p, a, b) for a, b in self._edges())

    def _along_ray(self, angle, level):
        """Point on the ray from the centroid at zone heading `angle` where signed_dist = level."""
        u = np.array([math.cos(angle), -math.sin(angle)])
        lo, hi = 0.0, float(np.max(np.linalg.norm(self.poly - self.centroid, axis=1))) + abs(level) + 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if self.signed_dist(*(self.centroid + mid * u)) >= level else (lo, mid)
        return self.centroid + lo * u

    def clip_target(self, x, y):
        """A Move target, pulled in along the line from the fence's centre until it is at least
        CLIP_MARGIN_M inside the edge. Targets already that far inside come back unchanged."""
        if self.signed_dist(x, y) >= CLIP_MARGIN_M:
            return float(x), float(y)
        v = np.array([x, y], dtype=float) - self.centroid
        if not v.any():
            return float(x), float(y)
        p = self._along_ray(math.atan2(-v[1], v[0]), CLIP_MARGIN_M)
        return float(p[0]), float(p[1])

    # ── State and message (C4) ───────────────────────────────────────────────

    def state_for(self, d: float) -> str:
        return "inside" if d > self.buffer_m else "near_edge" if d >= 0.0 else "outside"

    def message(self, x=None, y=None, fix=None) -> dict:
        """The C4 message for a position (default: the latest pose). "unknown" when there is
        no fresh position or the fix is "none"."""
        fix = self.fix if fix is None else fix
        if x is None:
            fresh = self._pose is not None and time.monotonic() - self._pose[2] <= STALE_S
            x, y = (self._pose[0], self._pose[1]) if fresh else (None, None)
        if x is None or fix == "none":
            return {"Id": "Geofence", "State": "unknown", "DistToEdge_m": float("nan"),
                    "Buffer_m": self.buffer_m, "Fix": fix, "t": time.monotonic()}
        d = self.signed_dist(x, y)
        return {"Id": "Geofence", "State": self.state_for(d), "DistToEdge_m": round(d, 3),
                "Buffer_m": self.buffer_m, "Fix": fix, "t": time.monotonic()}

    def set_fix(self, fix: str) -> None:
        if fix not in FIXES:
            raise ValueError(f"fix must be one of {FIXES}, got {fix!r}")
        self.fix = fix

    # ── Wiring (Main.py) ─────────────────────────────────────────────────────

    async def RobotPose_Receiver(self, pose) -> None:
        p = np.asarray(pose, dtype=float)
        self._pose = (float(p[0]), float(p[1]), time.monotonic())

    async def Runner(self) -> None:
        while True:
            t0 = time.monotonic()
            if self._pub is not None:
                await self._pub.publish(self.message())
            await asyncio.sleep(max(0.0, PERIOD_S - (time.monotonic() - t0)))


def _on_segment(p, a, b, tol=1e-9) -> bool:
    ab, ap = b - a, p - a
    cross = ab[0] * ap[1] - ab[1] * ap[0]
    return abs(cross) <= tol * max(1.0, float(np.dot(ab, ab))) and -tol <= np.dot(ap, ab) <= np.dot(ab, ab) + tol


def _checked_convex(poly):
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError("geofence polygon needs at least 3 [x, y] points")
    crosses = []
    for i in range(len(poly)):
        a, b, c = poly[i], poly[(i + 1) % len(poly)], poly[(i + 2) % len(poly)]
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    crosses = np.array(crosses)
    if not (np.all(crosses > 0) or np.all(crosses < 0)):
        raise ValueError("geofence polygon must be convex, with no repeated or collinear points")
    return poly


# ── Module-level helpers (C4: Navigation/Geofence.py exposes contains and clip_target) ──
# The planner falls back to these when Main.py doesn't hand it a Geofence object. They use
# the polygon in geofence.json, loaded on first use.

_default: Optional[Geofence] = None


def _fence() -> Geofence:
    global _default
    if _default is None:
        _default = Geofence()
    return _default


def contains(x, y) -> bool:
    return _fence().contains(x, y)


def clip_target(x, y):
    return _fence().clip_target(x, y)
