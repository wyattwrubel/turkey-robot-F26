"""
sim_resume.py — check what AUTO does when the bird is already in the zone (~5 s).

    python3 -m StatePlanner.sim_resume

The UI's MANUAL button and the E-stop (incl. the heartbeat watchdog) both send
{"Id": "Stop"}; AUTO sends {"Id": "UnlockUI"} (UserInterface/Controls.py).
UnlockUI puts the planner in IDLE and arbitrates.
  1. First AUTO with the bird already in view          -> CHASE
  2. AUTO, chase, MANUAL or E-stop + RESET, AUTO, bird still there
                                                       -> CHASE (W26 stayed IDLE)
  3. MANUAL, AUTO with no bird                         -> SCAN
  4. Geofence STOP with the bird in view               -> stays STOP (no chase)
"""

import asyncio
import contextlib
import io
import sys
import time

from StatePlanner.StatePlannerReceiver import Mode
from StatePlanner.sim_combos import FAR, feed, make_planner

AUTO   = {"Id": "UnlockUI"}
MANUAL = {"Id": "Stop"}


async def no_bird(p, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        await p.VisionObject_Receiver({"Id": "DynamicObjects", "Turkeys": [], "Humans": [], "Ghosts": []})
        await asyncio.sleep(0.1)


def outside():
    return {"Id": "Geofence", "State": "outside", "DistToEdge_m": -0.5,
            "Buffer_m": 1.4, "Fix": "sim", "t": time.time()}


async def scenarios(check):
    async def fresh():
        p, _, _ = make_planner()
        await p.RobotPose_Receiver(FAR)
        return p

    def done(p):
        p._cancel_current_task()
        if getattr(p, "_geofence_watchdog", None):
            p._geofence_watchdog.cancel()

    # 1. First AUTO, bird already in view.
    p = await fresh()
    await feed(p, 0.3)                      # manual: frames are ignored
    await p.MacroState_Receiver(AUTO)
    await feed(p, 0.5)
    check(p._mode == Mode.CHASE, f"first AUTO, bird in view -> {p._mode.name} (want CHASE)")
    done(p)

    # 2. AUTO, chase, MANUAL, AUTO again while the bird is still in the zone.
    p = await fresh()
    await p.MacroState_Receiver(AUTO)
    await feed(p, 0.5)
    chasing = p._mode == Mode.CHASE
    await p.MacroState_Receiver(MANUAL)
    await feed(p, 0.5)
    manual_idle = p.IsManualMode and p._mode == Mode.IDLE
    await p.MacroState_Receiver(AUTO)
    await feed(p, 1.0)
    check(chasing and manual_idle and p._mode == Mode.CHASE,
          f"AUTO -> MANUAL/E-stop -> AUTO, bird still in view -> {p._mode.name} (want CHASE)")
    done(p)

    # 3. MANUAL, then AUTO with no bird.
    p = await fresh()
    await p.MacroState_Receiver(AUTO)
    await feed(p, 0.3)
    await p.MacroState_Receiver(MANUAL)
    await no_bird(p, 0.3)
    await p.MacroState_Receiver(AUTO)
    await no_bird(p, 0.5)
    check(p._mode == Mode.SCAN, f"MANUAL -> AUTO, no bird -> {p._mode.name} (want SCAN)")
    done(p)

    # 4. Geofence STOP must win over the bird.
    p = await fresh()
    await p.MacroState_Receiver(AUTO)
    await feed(p, 0.3)
    await p.Geofence_Receiver(outside())
    await feed(p, 0.5)
    stopped = p._mode == Mode.STOP
    await p.MacroState_Receiver(AUTO)     # refused: still outside
    await p.Geofence_Receiver(outside())
    await feed(p, 0.5)
    check(stopped and p._mode == Mode.STOP,
          f"geofence STOP, bird in view, AUTO pressed -> {p._mode.name} (want STOP)")
    done(p)


def main() -> int:
    results = []

    def check(ok, what):
        results.append((bool(ok), what))

    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(scenarios(check))

    for ok, what in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    failed = sum(1 for ok, _ in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
