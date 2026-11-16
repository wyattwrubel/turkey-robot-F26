"""
ModuleDeter.py

Stationary detect-and-deter mode (DR2 plan X5, contracts C6 and C8).

Behaviour, with the wheels off:
  - A confirmed turkey appears   -> AlertPose, head toward its bearing, bark.
  - While it stays in view       -> the head follows the bearing; bark again
                                    every BARK_COOLDOWN_S seconds.
  - No turkey for LOST_TIMEOUT_S -> IdlePose (head back to centre).
  A new sighting after going idle barks right away; the cooldown only spaces
  barks within one sighting.

Arming: StatePlannerReceiver starts run() on UnlockUI (the UI's AUTO button)
and cancels it on any other MacroState (MANUAL, Stop, a routine button). AUTO
also clears CanineDriver's UI lock, so the dog listens to "System" commands.

Ownership: like the other modules, this one only drives poses, audio and the
head servo. It never publishes MotionCommand (C8). StatePlannerReceiver also
swaps its MotionCommand publisher for a no-op in deter mode, so nothing can
move the wheels by accident.

Input: the planner forwards every DynamicObjects message to on_vision(). Only
the bearing is used (column 2, radians, + = left of the optical axis, C2).
World x, y are ignored: the drivetrain is off, so the pose is not real.
on_vision() only stores data and returns. All awaiting happens in run(),
because Vision waits on its publish (up to 1 s) before the next frame.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Optional


class ModuleDeter:

    # ── Head aiming ──────────────────────────────────────────────────────────
    # Head servo command (deg) for a turkey at bearing b (rad):
    #     head_deg = HEAD_SIGN * degrees(b), clipped to +-HEAD_LIMIT_DEG
    #
    # HEAD_SIGN IS NOT MEASURED YET (C6). Bench check, Mon 11-16:
    #   from the UI, set the Head servo to +30.
    #   Head turns to the dog's LEFT  -> HEAD_SIGN = +1.0
    #   Head turns to the dog's RIGHT -> HEAD_SIGN = -1.0
    # Measured: not yet. Write the date and result here when it is.
    HEAD_SIGN         = 1.0
    HEAD_LIMIT_DEG    = 60.0   # = ServoDriver.Max_Head_Command
    HEAD_DEADBAND_DEG = 3.0    # skip head updates smaller than this (detector jitter)

    # ── Timing ───────────────────────────────────────────────────────────────
    BARK_SOUND        = "BarkShort_3x"   # 1.2 s clip; tune by ear on the robot
    BARK_COOLDOWN_S   = 5.0
    LOST_TIMEOUT_S    = 2.0
    TICK_S            = 0.05

    def __init__(self, parent):
        self.parent = parent
        self._bearing: float = 0.0
        self._last_seen: Optional[float] = None   # time.monotonic() of the last turkey

    # ------------------------------------------------------------------
    # Input (called by StatePlannerReceiver.VisionObject_Receiver)
    # ------------------------------------------------------------------

    def on_vision(self, data: dict) -> None:
        """Remember the latest bearing. Must not await: it runs inside Vision's publish."""
        bearing = self.pick_bearing(data.get("Turkeys", []))
        if bearing is not None:
            self._bearing   = bearing
            self._last_seen = time.monotonic()

    @staticmethod
    def pick_bearing(turkeys) -> Optional[float]:
        """Bearing of the most centred turkey, or None if there is none.

        Most centred = smallest |bearing|. It needs no pose, and with several
        birds the head stays on the one the camera sees best.
        """
        best = None
        for t in turkeys or []:
            try:
                b = float(t[2])
            except (TypeError, ValueError, IndexError):
                continue
            if not math.isfinite(b):
                continue
            if best is None or abs(b) < abs(best):
                best = b
        return best

    def head_deg(self, bearing_rad: float) -> float:
        deg = self.HEAD_SIGN * math.degrees(bearing_rad)
        return max(-self.HEAD_LIMIT_DEG, min(self.HEAD_LIMIT_DEG, deg))

    def _turkey_in_view(self) -> bool:
        return (self._last_seen is not None
                and time.monotonic() - self._last_seen < self.LOST_TIMEOUT_S)

    # ------------------------------------------------------------------
    # Main loop (started on UnlockUI, cancelled on any other MacroState)
    # ------------------------------------------------------------------

    async def run(self) -> None:
        p = self.parent
        engaged   = False
        last_bark: Optional[float] = None
        last_head: Optional[float] = None

        await p.MacroState_Publisher.publish({"Id": "DeterIdle"})
        await p._Pose_Publish("IdlePose")

        while True:
            seen = self._turkey_in_view()

            if seen and not engaged:
                engaged, last_bark = True, None
                p._dbg(f"[Deter] turkey at {math.degrees(self._bearing):+.1f} deg — engage")
                await p.MacroState_Publisher.publish({"Id": "Deter"})
                await p._Pose_Publish("AlertPose")   # takes ~0.5 s and centres the head
                last_head = 0.0
            elif not seen and engaged:
                engaged = False
                p._dbg(f"[Deter] no turkey for {self.LOST_TIMEOUT_S:.0f} s — idle")
                await p._Pose_Publish("IdlePose")
                await p.MacroState_Publisher.publish({"Id": "DeterIdle"})
                last_head = 0.0

            if engaged:
                head = self.head_deg(self._bearing)
                if last_head is None or abs(head - last_head) >= self.HEAD_DEADBAND_DEG:
                    await p._ServoCommand_Publish("Head", head)
                    last_head = head

                now = time.monotonic()
                if last_bark is None or now - last_bark >= self.BARK_COOLDOWN_S:
                    await p._Audio_Publish(self.BARK_SOUND)
                    last_bark = now

            await asyncio.sleep(self.TICK_S)
