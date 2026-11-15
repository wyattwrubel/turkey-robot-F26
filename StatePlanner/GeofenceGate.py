"""
GeofenceGate.py

The planner's side of the robot geofence (DR2 plan X6, contract C4).

Navigation/Geofence.py (Wyatt, W2) publishes at 10 Hz
    {"Id": "Geofence", "State": "inside" | "near_edge" | "outside" | "unknown",
     "DistToEdge_m": float, "Buffer_m": 1.4,
     "Fix": "sim" | "none" | "gps" | "rtk_float" | "rtk_fixed", "t": float}
and exposes clip_target(x, y) -> (x, y).

Planner rules (C4):
    near_edge                                -> SpeedCap 0.5, no Move target outside
    outside / unknown                        -> STOP
    --gps real and Fix worse than rtk_float  -> STOP

GeofenceGate wraps the real MotionCommand publisher, so every Move, Orient and
SpeedCap the planner sends (scan, chase, zig-zag, alert) passes one checkpoint:
    latched (STOP)  Move and Orient are dropped; SpeedCap is forced to 0
    near_edge       SpeedCap is capped at NEAR_EDGE_SPEED
    active          Move targets go through clip_target
It remembers the speed the planner last asked for, so leaving near_edge
restores it. A STOP stays latched until the operator presses AUTO again
(StatePlannerReceiver releases it only if the geofence is OK by then).

Nothing is enforced until the first Geofence message arrives, so profiles
without a geofence (deter, sim before W2) run unchanged (C1). After that,
silence for more than STALE_S counts as "unknown".
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

STATES      = ("inside", "near_edge", "outside", "unknown")
STOP_STATES = ("outside", "unknown")
GOOD_FIXES  = ("rtk_float", "rtk_fixed")


def evaluate(msg: dict, gps_real: bool) -> Tuple[str, str]:
    """
    Planner decision for one Geofence message.
    Returns (decision, reason), decision in "inside" | "near_edge" | "stop".
    """
    state = msg.get("State")
    if state not in STATES:
        return "stop", f"bad State {state!r}"
    if state in STOP_STATES:
        return "stop", state
    if gps_real and msg.get("Fix") not in GOOD_FIXES:
        return "stop", f"fix {msg.get('Fix')!r} is worse than rtk_float"
    return state, state


class GeofenceGate:

    NEAR_EDGE_SPEED = 0.5    # m/s (C4)
    STALE_S         = 1.0    # s without a Geofence message -> "unknown" (publisher runs at 10 Hz)

    def __init__(self, publisher, clip_target: Optional[Callable] = None, log=print):
        self._pub  = publisher
        self._clip = clip_target
        self._log  = log
        self.active    = False   # set when the first Geofence message arrives
        self.latched   = False   # STOP: wheels held at 0 until release()
        self.near_edge = False
        self.requested_speed: Optional[float] = None   # last SpeedCap the planner asked for
        self._warned_no_clip = False

    # Same interface as Main.py's AsyncPublisher.
    def subscribe(self, callback) -> None:
        self._pub.subscribe(callback)

    def _limit(self) -> Optional[float]:
        if self.latched:
            return 0.0
        if self.near_edge:
            return self.NEAR_EDGE_SPEED
        return None

    async def publish(self, data) -> None:
        kind = data.get("Id") if isinstance(data, dict) else None

        if kind == "SpeedCap":
            try:
                self.requested_speed = float(data["Cmd"])
            except (KeyError, TypeError, ValueError):
                pass                              # Navigation reports malformed commands
            limit = self._limit()
            if limit is not None and (self.requested_speed is None or self.requested_speed > limit):
                data = {**data, "Cmd": limit}

        elif kind in ("Move", "Orient") and self.latched:
            self._log(f"[Geofence] STOP latched — dropped {kind} {data.get('Cmd')}")
            return

        elif kind == "Move" and self.active:
            data = self._clipped(data)

        await self._pub.publish(data)

    def _clipped(self, data: dict) -> dict:
        if self._clip is None:
            if not self._warned_no_clip:
                self._log("[Geofence] no clip_target available — Move targets are not clipped")
                self._warned_no_clip = True
            return data
        try:
            x, y = float(data["Cmd"][0]), float(data["Cmd"][1])
        except (KeyError, IndexError, TypeError, ValueError):
            return data
        cx, cy = self._clip(x, y)
        cx, cy = float(cx), float(cy)
        if abs(cx - x) > 0.05 or abs(cy - y) > 0.05:
            self._log(f"[Geofence] Move target ({x:.1f}, {y:.1f}) clipped to ({cx:.1f}, {cy:.1f})")
        return {**data, "Cmd": [cx, cy]}

    async def latch(self, reason: str) -> None:
        """STOP: hold the wheels now and drop motion until release()."""
        if not self.latched:
            self._log(f"[Geofence] STOP ({reason}) — wheels held until AUTO")
        self.latched = True
        await self._pub.publish({"Id": "SpeedCap", "Cmd": 0.0})

    def release(self) -> None:
        """Called on AUTO. The routine that starts next publishes its own SpeedCap."""
        self.latched = False

    async def set_near_edge(self, on: bool) -> None:
        if on == self.near_edge:
            return
        self.near_edge = on
        if self.latched:
            return
        if on:
            speed = self.NEAR_EDGE_SPEED
            if self.requested_speed is not None:
                speed = min(self.requested_speed, speed)
        elif self.requested_speed is not None:
            speed = self.requested_speed
        else:
            return
        await self._pub.publish({"Id": "SpeedCap", "Cmd": speed})
