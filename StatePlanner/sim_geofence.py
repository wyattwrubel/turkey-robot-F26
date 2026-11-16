"""
sim_geofence.py — check the geofence receiver (X6) on a laptop.

  python3 -m StatePlanner.sim_geofence
      Rule checks with stub publishers (~10 s, no extra packages): clipping,
      near_edge speed cap and restore, STOP latch, AUTO refused / accepted,
      stale feed, bad fix with gps_real, unknown at start, Stop holds the wheels.

  .venv/bin/python -m StatePlanner.sim_geofence --drive [--plot fence.png]
      Closed loop (planner + A* + MPC + simulated drivetrain, ~2 min): the robot
      scans toward a waypoint beyond the fence from 5 angles and must stop
      inside, slowing to 0.5 m/s in the buffer. One extra run without
      clip_target checks the STOP backstop.

FakeFence stands in for Wyatt's Navigation/Geofence.py (W2): a convex polygon in
zone metres, 1.4 m buffer, the C4 message, and clip_target(). Replace it with the
real module when W2 lands.
"""

import argparse
import asyncio
import contextlib
import io
import math
import sys
import time

import numpy as np

import StatePlanner.StatePlannerReceiver as spr
from StatePlanner.GeofenceGate import GeofenceGate


# ─────────────────────────────────────────────────────────────────────────────
# Stand-in for Navigation/Geofence.py (W2)
# ─────────────────────────────────────────────────────────────────────────────

class FakeFence:
    BUFFER_M      = 1.4
    CLIP_MARGIN_M = 0.3      # clip_target puts targets this far inside the edge

    def __init__(self, polygon):
        self.poly = np.asarray(polygon, dtype=float)
        self.centroid = self.poly.mean(axis=0)

    def _edges(self):
        return zip(self.poly, np.roll(self.poly, -1, axis=0))

    def contains(self, x, y) -> bool:
        inside = False
        for (x1, y1), (x2, y2) in self._edges():
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
        return inside

    def signed_dist(self, x, y) -> float:
        p = np.array([x, y], dtype=float)
        d = min(np.linalg.norm(p - (a + np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1) * (b - a)))
                for a, b in self._edges())
        return d if self.contains(x, y) else -d

    def _along_ray(self, angle, level):
        """Point on the ray from the centroid at `angle` (zone heading) where signed_dist = level."""
        u = np.array([math.cos(angle), -math.sin(angle)])
        lo, hi = 0.0, 500.0
        for _ in range(60):
            mid = (lo + hi) / 2
            p = self.centroid + mid * u
            lo, hi = (mid, hi) if self.signed_dist(*p) >= level else (lo, mid)
        return self.centroid + lo * u

    def clip_target(self, x, y):
        if self.signed_dist(x, y) >= self.CLIP_MARGIN_M:
            return float(x), float(y)
        v = np.array([x, y]) - self.centroid
        p = self._along_ray(math.atan2(-v[1], v[0]), self.CLIP_MARGIN_M)
        return float(p[0]), float(p[1])

    def message(self, x, y, fix="sim"):
        d = self.signed_dist(x, y)
        state = "inside" if d > self.BUFFER_M else "near_edge" if d >= 0 else "outside"
        return {"Id": "Geofence", "State": state, "DistToEdge_m": round(d, 3),
                "Buffer_m": self.BUFFER_M, "Fix": fix, "t": time.monotonic()}


SQUARE = FakeFence([(0, 0), (50, 0), (50, 50), (0, 50)])


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: rule checks with stub publishers
# ─────────────────────────────────────────────────────────────────────────────

class Recorder:
    def __init__(self):
        self.events = []

    def subscribe(self, callback):
        pass

    async def publish(self, data):
        self.events.append(data)

    def since(self, n, kind=None):
        return [e for e in self.events[n:] if kind is None or e.get("Id") == kind]


class _NoZone:
    def is_inside(self, x, y):
        return True

    def is_outside(self, x, y):
        return False


def make_planner(waypoint=(80.0, 25.0), **kw):
    spr.ZoneChecker = _NoZone
    spr.StatePlannerReceiver._load_waypoints = lambda self, path: [np.array(waypoint, dtype=float)]
    macro, canine, motion = Recorder(), Recorder(), Recorder()
    p = spr.StatePlannerReceiver(macro, canine, motion, **kw)
    return p, macro, motion


async def rules(check):
    fence = SQUARE
    p, macro, motion = make_planner(geofence=fence)
    mc = p.MotionCommand_Publisher

    async def at(x, y, fix="sim"):
        await p.Geofence_Receiver(fence.message(x, y, fix))

    # 1. No Geofence messages yet: nothing enforced.
    n = len(motion.events)
    await mc.publish({"Id": "Move", "Cmd": [80.0, 25.0]})
    await mc.publish({"Id": "SpeedCap", "Cmd": 2.0})
    ev = motion.since(n)
    check(ev[0]["Cmd"] == [80.0, 25.0] and ev[1]["Cmd"] == 2.0,
          "no Geofence messages yet: Move and SpeedCap pass untouched")

    # 2. Inside: Move targets are clipped into the polygon.
    await at(25, 25)
    n = len(motion.events)
    await mc.publish({"Id": "Move", "Cmd": [80.0, 25.0]})
    x = motion.since(n, "Move")[0]["Cmd"][0]
    check(49.0 < x < 50.0, f"inside: Move (80, 25) clipped to x = {x:.2f}")

    # 3. near_edge: speed capped at 0.5 at once, requests capped, restored on leaving.
    n = len(motion.events)
    await at(49, 25)
    capped_now = motion.since(n, "SpeedCap")
    await mc.publish({"Id": "SpeedCap", "Cmd": 2.0})
    capped_req = motion.since(n, "SpeedCap")[-1]["Cmd"]
    await at(25, 25)
    restored = motion.since(n, "SpeedCap")[-1]["Cmd"]
    check(capped_now and capped_now[0]["Cmd"] == GeofenceGate.NEAR_EDGE_SPEED,
          "near_edge: SpeedCap 0.5 sent immediately")
    check(capped_req == GeofenceGate.NEAR_EDGE_SPEED, "near_edge: SpeedCap 2.0 request forwarded as 0.5")
    check(restored == 2.0, "back inside: SpeedCap 2.0 restored")

    # 4. Outside: STOP latched, wheels held, motion dropped.
    n, m = len(motion.events), len(macro.events)
    await at(52, 25)
    await asyncio.sleep(0.05)
    check(motion.since(n, "SpeedCap") and motion.since(n, "SpeedCap")[0]["Cmd"] == 0.0,
          "outside: SpeedCap 0 sent at once")
    check(p._mode == spr.Mode.STOP and {"Id": "Stop"} in macro.events[m:],
          "outside: planner in STOP, MacroState Stop published")
    n = len(motion.events)
    await mc.publish({"Id": "Move", "Cmd": [10.0, 10.0]})
    await mc.publish({"Id": "Orient", "Cmd": [0.0, 0.0, 1.0]})
    await mc.publish({"Id": "SpeedCap", "Cmd": 1.0})
    ev = motion.since(n)
    check(len(ev) == 1 and ev[0] == {"Id": "SpeedCap", "Cmd": 0.0},
          "latched: Move and Orient dropped, SpeedCap forced to 0")

    # 5. Back inside: still latched until AUTO.
    await at(25, 25)
    n = len(motion.events)
    await mc.publish({"Id": "Move", "Cmd": [10.0, 10.0]})
    check(not motion.since(n, "Move"), "back inside: still latched (no Move) until AUTO")

    # 6. AUTO while outside is refused; AUTO while inside resumes with a clipped scan Move.
    await at(52, 25)
    await p.MacroState_Receiver({"Id": "UnlockUI"})
    check(p._geofence_gate.latched and p._mode == spr.Mode.STOP, "AUTO while outside: refused")
    await at(25, 25)
    n = len(motion.events)
    await p.MacroState_Receiver({"Id": "UnlockUI"})
    await asyncio.sleep(0.1)
    moves = motion.since(n, "Move")
    check(not p._geofence_gate.latched and p._mode == spr.Mode.SCAN, "AUTO while inside: released, SCAN")
    check(moves and moves[0]["Cmd"][0] < 50.0, "after AUTO: the scan Move to (80, 25) is clipped")

    # 7. Feed goes silent: STOP after STALE_S.
    n = len(motion.events)
    await asyncio.sleep(GeofenceGate.STALE_S + 0.5)
    check(p._geofence_gate.latched and p._mode == spr.Mode.STOP
          and {"Id": "SpeedCap", "Cmd": 0.0} in motion.since(n),
          f"no Geofence message for {GeofenceGate.STALE_S:.0f} s: STOP")
    p._cancel_current_task()


async def rules_fix(check):
    fence = SQUARE
    p, _, _ = make_planner(geofence=fence, gps_real=True)
    await p.Geofence_Receiver(fence.message(25, 25, fix="gps"))
    check(p._geofence_gate.latched, "gps_real: Fix 'gps' (worse than rtk_float) means STOP")
    await p.Geofence_Receiver(fence.message(25, 25, fix="rtk_float"))
    await p.MacroState_Receiver({"Id": "UnlockUI"})
    check(not p._geofence_gate.latched, "gps_real: Fix 'rtk_float' and AUTO: released")
    p._cancel_current_task()

    p2, _, _ = make_planner(geofence=fence)
    await p2.Geofence_Receiver({"Id": "Geofence", "State": "unknown", "DistToEdge_m": 0.0,
                                "Buffer_m": 1.4, "Fix": "none", "t": time.monotonic()})
    check(p2._geofence_gate.latched, "State 'unknown' at start: STOP")
    p2._cancel_current_task()

    p3, _, motion3 = make_planner()
    await p3.MacroState_Receiver({"Id": "Stop"})
    await asyncio.sleep(0.05)
    check({"Id": "SpeedCap", "Cmd": 0.0} in motion3.events,
          "UI Stop / MANUAL (no geofence at all): wheels held with SpeedCap 0")


def run_rules() -> int:
    results = []

    def check(ok, what):
        results.append((bool(ok), what))

    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        asyncio.run(rules(check))
        asyncio.run(rules_fix(check))
    lines = [l for l in log.getvalue().splitlines() if l.startswith("[Geofence]")]
    transitions = [l for l in lines if " -> " in l]
    check(any("-> near_edge" in l for l in transitions)
          and any("-> stop (outside" in l for l in transitions)
          and any("-> stop (no Geofence message" in l for l in transitions),
          f"state changes logged ({len(transitions)} transition lines)")

    print("Geofence log:")
    for l in lines:
        print("   ", l)
    print()
    for ok, what in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    failed = sum(1 for ok, _ in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return failed


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: closed loop (needs casadi + scipy, i.e. the .venv)
# ─────────────────────────────────────────────────────────────────────────────

PENTAGON = FakeFence([(30, 35), (65, 30), (75, 55), (55, 72), (28, 60)])
ANGLES   = [10.0, 80.0, 150.0, 215.0, 290.0]    # zone headings from the centroid
START_IN = 8.0                                   # start this far inside the edge


class _NoClip:
    """A geofence with no clip_target: only the STOP backstop protects the fence."""


async def drive_one(sz, angle_deg, clip=True):
    fence = PENTAGON
    a = math.radians(angle_deg)
    start = fence._along_ray(a, START_IN)
    waypoint = start + 30.0 * np.array([math.cos(a), -math.sin(a)])
    pose0 = np.array([*start, a])

    pubs = {n: sz.AsyncPublisher(n) for n in
            ("drive", "slip", "pose", "nav_motion", "mpc", "sp_motion", "macro", "canine", "vision", "fence")}
    spr.StatePlannerReceiver._load_waypoints = lambda self, path: [waypoint.copy()]
    drivetrain = sz.MotorDriver(pubs["pose"], pubs["slip"], Simulated=True)
    drivetrain._dr_state = pose0.astype(np.float64).copy()
    nav = sz.TrajectoryTracker(pubs["drive"], pubs["nav_motion"], pubs["mpc"], Verbose=False)
    planner = spr.StatePlannerReceiver(pubs["macro"], pubs["canine"], pubs["sp_motion"],
                                       geofence=fence if clip else _NoClip())
    cam = sz.FakeCamera(pubs["vision"], [-500.0, -500.0], sz.HFOV_DEG)   # no bird in this test
    pose = {"p": pose0.copy()}

    async def on_pose(d):
        pose["p"] = np.asarray(d, dtype=float)[:3].copy()

    async def fence_loop():
        while True:
            await pubs["fence"].publish(fence.message(*pose["p"][:2]))
            await asyncio.sleep(0.1)

    pubs["drive"].subscribe(drivetrain.MotorCommandReceiver)
    for cb in (nav.RobotPose_Receiver, planner.RobotPose_Receiver, cam.RobotPose_Receiver, on_pose):
        pubs["pose"].subscribe(cb)
    pubs["sp_motion"].subscribe(nav.TrajectoryCMD_Receiver)
    pubs["vision"].subscribe(nav.VisionObject_Receiver)
    pubs["vision"].subscribe(planner.VisionObject_Receiver)
    pubs["fence"].subscribe(planner.Geofence_Receiver)

    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(c) for c in (drivetrain.DriveTrainDriver_Runner(), cam.run(), fence_loop())]
    await asyncio.sleep(0.5)
    await planner.MacroState_Receiver({"Id": "UnlockUI"})

    t0, still_since = loop.time(), None
    while loop.time() - t0 < 40.0:
        await asyncio.sleep(0.2)
        s = cam.samples
        if len(s) > 10:
            speed = math.hypot(s[-1][1] - s[-6][1], s[-1][2] - s[-6][2]) / (s[-1][0] - s[-6][0])
            moved = math.hypot(s[-1][1] - start[0], s[-1][2] - start[1]) > 1.0
            if moved and speed < 0.05:
                still_since = still_since or loop.time()
                if loop.time() - still_since > 2.0:
                    break
            else:
                still_since = None
    for t in tasks:
        t.cancel()
    planner._cancel_current_task()
    return {"angle": angle_deg, "clip": clip, "samples": cam.samples, "start": start,
            "waypoint": waypoint, "stopped": planner._mode == spr.Mode.STOP,
            "latched": planner._geofence_gate.latched}


def run_drive(plot_path) -> int:
    from StatePlanner import sim_zigzag as sz      # patches map paths; needs casadi + scipy
    fence, buf = PENTAGON, FakeFence.BUFFER_M
    runs, failures = [], 0
    jobs = [(a, True) for a in ANGLES] + [(ANGLES[0], False)]
    for angle, clip in jobs:
        print(f"\n── toward the fence at {angle:.0f} deg{'' if clip else ', NO clip_target (backstop)'} ──")
        run = asyncio.run(drive_one(sz, angle, clip))
        runs.append(run)
        s = np.array([r[:3] for r in run["samples"]], dtype=float)
        d = np.array([fence.signed_dist(x, y) for _, x, y in s])
        v = np.r_[0.0, np.hypot(np.diff(s[:, 1]), np.diff(s[:, 2])) / np.maximum(np.diff(s[:, 0]), 1e-3)]
        # speed well inside the buffer (allow 0.4 m for the drivetrain to slow down)
        deep = d < buf - 0.4
        v_buf = float(v[deep].max()) if deep.any() else 0.0
        if clip:
            # Clipped targets stop the robot near the buffer, so the speed cap is
            # checked on the backstop run below, which drives through the buffer.
            checks = [
                (d.min() >= 0.0, f"never outside: closest {d.min():.2f} m inside the edge"),
                (not run["latched"], "no STOP needed (targets clipped)"),
            ]
        else:
            checks = [
                (deep.any() and v_buf <= GeofenceGate.NEAR_EDGE_SPEED + 0.1,
                 f"through the buffer: max speed {v_buf:.2f} m/s (cap {GeofenceGate.NEAR_EDGE_SPEED})"),
                (run["latched"] and run["stopped"], "crossed the edge: STOP latched"),
                (d.min() > -1.0, f"stopped {-d.min():.2f} m past the edge (< 1.0 m)"),
            ]
        for ok, what in checks:
            print(f"   {'PASS' if ok else 'FAIL'}  {what}")
            failures += not ok
    if plot_path:
        _plot(runs, plot_path)
    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} check(s) FAILED'}")
    return failures


def _plot(runs, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping the plot")
        return
    fence = PENTAGON
    fig, ax = plt.subplots(figsize=(7.5, 7))
    poly = np.vstack([fence.poly, fence.poly[:1]])
    ax.plot(poly[:, 0], poly[:, 1], color="#c0392b", lw=2, label="fence")
    inner = np.array([fence._along_ray(t, FakeFence.BUFFER_M) for t in np.linspace(0, 2 * math.pi, 200)])
    ax.plot(inner[:, 0], inner[:, 1], color="#c0392b", lw=1, ls=":", label="1.4 m buffer (near_edge)")
    for i, run in enumerate(runs):
        s = np.array([r[1:3] for r in run["samples"]], dtype=float)
        style = (dict(color="#2a78d6", lw=2, label="scan toward a waypoint outside" if i == 0 else None)
                 if run["clip"] else dict(color="#777777", lw=2, ls="--", label="no clip_target: STOP backstop"))
        ax.plot(s[:, 0], s[:, 1], **style)
        ax.plot(*run["start"], "o", color=style["color"], ms=5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m, down the grid)")
    ax.set_title("Robot geofence: stops inside from 5 angles")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"plot saved: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--drive", action="store_true", help="closed-loop runs (needs the .venv)")
    ap.add_argument("--plot", default=None, help="with --drive: save a plot here")
    args = ap.parse_args()
    sys.exit(1 if (run_drive(args.plot) if args.drive else run_rules()) else 0)
