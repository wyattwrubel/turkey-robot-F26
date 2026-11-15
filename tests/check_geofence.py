"""
check_geofence.py: checks for the robot geofence (DR2 plan W2, contract C4). No robot needed.
Run from the repo root with the laptop venv:

    python tests/check_geofence.py                       # rules and geometry, a few seconds
    python tests/check_geofence.py --drive --plot g.png  # also the closed loop (F9 in sim)

Part 1, rules and geometry:
  inside / near_edge / outside / unknown and the signed distance; clip_target keeps targets
  CLIP_MARGIN_M inside; the C4 message has exactly its keys; "unknown" with no pose, a stale
  pose, or fix "none"; non-convex polygons refused; Runner publishes at 10 Hz; geofence.json
  loads and holds the sim start pose; and, when Xinyuan's GeofenceGate is present, her
  evaluate() reads these messages the way C4 says.
Part 2, --drive (needs behavior/geofence-receiver merged, plus scipy and casadi):
  Xinyuan's closed-loop drive from StatePlanner/sim_geofence.py, with her stand-in fence
  swapped for this module: the planner, A* + MPC and the simulated drivetrain drive at the
  fence from 5 angles and must stop inside; one run without clip_target checks the STOP
  backstop. --plot saves the trajectories for the DR2 slides.
Exit code 0 means every check passed.
"""

import argparse
import asyncio
import importlib.util
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import Navigation.Geofence as gf   # noqa: E402
from Navigation.Geofence import Geofence, BUFFER_M, CLIP_MARGIN_M   # noqa: E402

FAILS = 0
C4_KEYS = {"Id", "State", "DistToEdge_m", "Buffer_m", "Fix", "t"}


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    FAILS += 0 if ok else 1


def raises(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


class Recorder:
    def __init__(self):
        self.msgs = []

    async def publish(self, data):
        self.msgs.append(data)


def part1():
    sq = Geofence(polygon=[(0, 0), (50, 0), (50, 50), (0, 50)])

    # Geometry and states
    check("centre is 25 m inside", abs(sq.signed_dist(25, 25) - 25) < 1e-9)
    check("1 m from an edge is near_edge", sq.message(1, 25)["State"] == "near_edge")
    check("exactly on the edge is near_edge", sq.message(0, 25)["State"] == "near_edge")
    check(f"{BUFFER_M} m in is still near_edge, a bit more is inside",
          sq.message(BUFFER_M, 25)["State"] == "near_edge" and sq.message(BUFFER_M + 0.01, 25)["State"] == "inside")
    check("2 m past an edge is outside, distance -2",
          sq.message(-2, 25)["State"] == "outside" and abs(sq.message(-2, 25)["DistToEdge_m"] + 2) < 1e-9)
    check("contains: edge yes, outside no", sq.contains(0, 25) and not sq.contains(-0.01, 25))

    # clip_target
    check("a deep target is unchanged", sq.clip_target(25, 25) == (25.0, 25.0))
    for tx, ty in [(80, 25), (60, 60), (49.9, 25), (-30, -5), (25, 0.1)]:
        cx, cy = sq.clip_target(tx, ty)
        d = sq.signed_dist(cx, cy)
        check(f"target ({tx}, {ty}) clipped to >= {CLIP_MARGIN_M} m inside",
              d >= CLIP_MARGIN_M - 1e-3 and d <= CLIP_MARGIN_M + 0.05, f"-> ({cx:.2f}, {cy:.2f}), {d:.3f} m")

    # C4 message
    m = sq.message(10, 10)
    check("C4 message has exactly its keys", set(m) == C4_KEYS, str(sorted(m)))
    check("C4 values", m["Id"] == "Geofence" and m["Buffer_m"] == BUFFER_M and m["Fix"] == "sim")

    # unknown
    fresh = Geofence(polygon=sq.poly)
    check("no pose yet -> unknown", fresh.message()["State"] == "unknown")
    asyncio.run(fresh.RobotPose_Receiver([10.0, 10.0, 0.0]))
    check("fresh pose -> inside", fresh.message()["State"] == "inside")
    fresh._pose = (10.0, 10.0, time.monotonic() - 5.0)
    check("pose 5 s old -> unknown", fresh.message()["State"] == "unknown")
    asyncio.run(fresh.RobotPose_Receiver([10.0, 10.0, 0.0]))
    fresh.set_fix("none")
    check('fix "none" -> unknown', fresh.message()["State"] == "unknown")
    check("bad fix name refused", raises(lambda: fresh.set_fix("great")))

    # polygon validation
    check("non-convex polygon refused", raises(lambda: Geofence(polygon=[(0, 0), (10, 0), (5, 2), (10, 10), (0, 10)])))
    check("two points refused", raises(lambda: Geofence(polygon=[(0, 0), (10, 0)])))

    # 10 Hz
    rec = Recorder()
    live = Geofence(rec, polygon=sq.poly)

    async def run_1s():
        await live.RobotPose_Receiver([25.0, 25.0, 0.0])
        task = asyncio.create_task(live.Runner())
        await asyncio.sleep(1.0)
        task.cancel()
    asyncio.run(run_1s())
    check("Runner publishes at 10 Hz", 9 <= len(rec.msgs) <= 11, f"{len(rec.msgs)} in 1 s")

    # geofence.json and the module-level helpers
    dflt = Geofence()
    check("geofence.json loads (convex)", len(dflt.poly) >= 3)
    check("it holds the sim start pose (15, 75)", dflt.message(15, 75)["State"] == "inside")
    check("module-level contains / clip_target", gf.contains(15, 75) and gf.contains(*gf.clip_target(30, 27)))

    # Xinyuan's side, when present
    if importlib.util.find_spec("StatePlanner.GeofenceGate") is None:
        print("[SKIP] GeofenceGate.evaluate: behavior/geofence-receiver not merged here")
        return
    from StatePlanner.GeofenceGate import evaluate
    cases = [(sq.message(25, 25), False, "inside"), (sq.message(1, 25), False, "near_edge"),
             (sq.message(-1, 25), False, "stop"), (fresh.message(), False, "stop"),
             (sq.message(25, 25), True, "stop"), (sq.message(25, 25, fix="rtk_fixed"), True, "inside")]
    got = [evaluate(msg, gps_real)[0] for msg, gps_real, _ in cases]
    check("Xinyuan's evaluate() reads these messages per C4", got == [c[2] for c in cases], str(got))


def part2(plot):
    if importlib.util.find_spec("StatePlanner.sim_geofence") is None:
        print("[SKIP] --drive: behavior/geofence-receiver not merged here")
        return
    from StatePlanner import sim_geofence as sg
    real = Geofence(polygon=sg.PENTAGON.poly)
    sg.PENTAGON = real          # her closed loop, with this module in place of FakeFence
    print("\n── Xinyuan's closed-loop drive with Navigation/Geofence.py ──")
    failures = sg.run_drive(plot)
    check("closed loop: stops inside from 5 angles, STOP backstop works", failures == 0,
          f"{failures} of her checks failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", action="store_true")
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    part1()
    if a.drive:
        part2(a.plot)
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)
