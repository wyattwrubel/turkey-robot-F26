"""
ZigZag.py

Randomized zig-zag approach toward a turkey (DR2 plan X2, contracts C5, C9).

Instead of driving straight at the bird, the robot drives a chain of legs that
alternate left and right of the line of sight. Each leg is one ordinary Move
(C5) that Navigation (A* + MPC) drives. Offset angle and leg length are random
within limits, so no two approaches look the same.

This module is geometry only: it returns legs and never publishes.
StatePlannerReceiver owns all movement commands and calls next_leg().

Frames (same as the rest of the code):
    heading of a vector (dx, dy) = atan2(-dy, dx)      (+Y is down the grid)
    point at distance L along heading h from (x, y) = (x + L cos h, y - L sin h)
    bearing = target heading - robot heading, + = left (C2)

Camera limit. Driving a leg at offset a from the line of sight, the bird's
bearing grows as the robot moves sideways and closer. After a leg of length L
that starts at distance D, the bearing is a + atan2(L sin a, D - L cos a).
Keeping that <= B gives, by the law of sines,
        L <= D * sin(B - a) / sin(B)
so legs get shorter or flatter as the robot closes in. The offset itself is
capped at HFOV_DEG/2 - 10 deg (plan X2), and B = HFOV_DEG/2 - FOV_MARGIN_DEG
leaves room for the robot not tracking the leg perfectly.

Navigation limits (Navigation/ is not changed for DR2):
  - The MPC aims at a point LOOKAHEAD_M ahead on the path and swings toward
    the goal once it is closer than that, which would swing the camera off
    the bird at the end of every leg. So each leg's Move target is the vertex
    extended EXTEND_M further along the leg; the planner starts the next leg
    when the robot passes the real vertex, before it ever gets there.
  - TrajectoryTracker drops a Move whose target is within 3 m of the previous
    one while the MPC is driving (REPLAN_THRESHOLD), so consecutive targets
    are at least MIN_TARGET_SEP_M apart.
  - A* stops 4 m from the bird and the MPC stops 1 m short of a plain Move
    target, so a plain Move can't end exactly at the 4 m hold. When no further
    leg fits, next_leg() returns None and the planner finishes with W26's
    straight chase (Move Mode "Chase"), which holds at CHASE_STOP_DIST_M.
"""

from __future__ import annotations

import math
import os
import random
from typing import List, Optional

import numpy as np

try:
    from Vision.config import HFOV_DEG   # contract C9 (arrives with cv/onboard-detector)
except ImportError:
    # Same override as Vision/config.py; 60 is the plan's placeholder until it's measured.
    HFOV_DEG = float(os.environ.get("VISION_HFOV_DEG", 60.0))


def heading_to(from_xy, to_xy) -> float:
    """Heading (rad, zone frame) of the vector from from_xy to to_xy."""
    return math.atan2(-(to_xy[1] - from_xy[1]), to_xy[0] - from_xy[0])


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def bearing_of(pose, target_xy) -> float:
    """Bearing (rad, + = left) of target_xy seen from pose [x, y, theta]."""
    return wrap(heading_to(pose[:2], target_xy) - pose[2])


class ZigZag:

    MIN_OFFSET_DEG    = 8.0    # smallest zig worth doing
    OFFSET_MARGIN_DEG = 10.0   # offset cap = HFOV/2 - this (plan X2)
    FOV_MARGIN_DEG    = 7.0    # bearing limit B = HFOV/2 - this, at the end of a leg
    MIN_LEG_M         = 3.5
    MAX_LEG_M         = 8.0
    EXTEND_M          = 2.5    # = MovementExecutor.LOOKAHEAD_DIST
    MIN_TARGET_SEP_M  = 3.2    # > Navigation REPLAN_THRESHOLD (3.0 m)
    END_MARGIN_M      = 0.5    # a leg ends at least hold_m + this from the bird
    TRIES             = 12     # random draws before giving up on a leg

    def __init__(self, hold_m: float = 4.0, hfov_deg: float = HFOV_DEG,
                 rng: Optional[random.Random] = None):
        self.hold_m   = hold_m
        self.hfov_deg = hfov_deg
        self.rng      = rng or random.Random()
        half = hfov_deg / 2.0
        self.offset_max    = math.radians(half - self.OFFSET_MARGIN_DEG)
        self.offset_min    = min(math.radians(self.MIN_OFFSET_DEG), self.offset_max / 2.0)
        self.bearing_limit = math.radians(half - self.FOV_MARGIN_DEG)
        self.reset()

    def reset(self) -> None:
        """Start a new approach: the next leg picks a random side."""
        self._side: Optional[int] = None
        self.legs: List[dict] = []

    # ------------------------------------------------------------------
    # Limits for one leg
    # ------------------------------------------------------------------

    def _offset_max_at(self, dist: float) -> float:
        """Largest offset that still allows a MIN_LEG_M leg inside the camera limit."""
        s = self.MIN_LEG_M * math.sin(self.bearing_limit) / dist
        if s >= 1.0:
            return -1.0
        return min(self.offset_max, self.bearing_limit - math.asin(s))

    def _leg_max(self, dist: float, a: float) -> float:
        """Longest leg at offset a: camera limit, MAX_LEG_M, and end >= hold + margin."""
        l_fov = dist * math.sin(self.bearing_limit - a) / math.sin(self.bearing_limit)
        end_min = self.hold_m + self.END_MARGIN_M
        lateral = dist * math.sin(a)
        if lateral >= end_min:
            l_room = math.inf
        else:
            l_room = dist * math.cos(a) - math.sqrt(end_min ** 2 - lateral ** 2)
        return min(self.MAX_LEG_M, l_fov, l_room)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def next_leg(self, start_xy, turkey_xy, last_target=None) -> Optional[dict]:
        """
        Next zig-zag leg from start_xy (the robot, or the vertex it just passed).

        Returns a dict with
            start, end   leg start and vertex (np arrays)
            dir          unit vector along the leg
            length       metres from start to end
            target       end + EXTEND_M * dir: what to send as the Move
            side         +1 = left of the line of sight, -1 = right
        or None when no leg fits any more: time for the final straight approach.
        """
        start  = np.asarray(start_xy, dtype=float)[:2]
        turkey = np.asarray(turkey_xy, dtype=float)[:2]
        dist   = float(np.linalg.norm(turkey - start))
        a_hi   = self._offset_max_at(dist) if dist > 0 else -1.0
        if a_hi < self.offset_min:
            return None

        side = self.rng.choice((-1, 1)) if self._side is None else -self._side
        los  = heading_to(start, turkey)

        for _ in range(self.TRIES):
            a = self.rng.uniform(self.offset_min, a_hi)
            l_hi = self._leg_max(dist, a)
            if l_hi < self.MIN_LEG_M:
                continue
            length = self.rng.uniform(self.MIN_LEG_M, l_hi)
            h      = los + side * a
            u      = np.array([math.cos(h), -math.sin(h)])
            end    = start + length * u
            target = end + self.EXTEND_M * u
            if (last_target is not None
                    and np.linalg.norm(target - np.asarray(last_target, dtype=float)[:2])
                    < self.MIN_TARGET_SEP_M):
                continue
            end_bearing = a + math.atan2(length * math.sin(a), dist - length * math.cos(a))
            self._side = side
            self.legs.append({
                "start": start.round(2).tolist(), "end": end.round(2).tolist(),
                "side": "left" if side > 0 else "right",
                "offset_deg": round(math.degrees(a), 1), "length_m": round(length, 2),
                "dist_m": round(dist, 2), "end_bearing_deg": round(math.degrees(end_bearing), 1),
            })
            return {"start": start, "end": end, "dir": u, "length": length,
                    "target": target, "side": side}
        return None
