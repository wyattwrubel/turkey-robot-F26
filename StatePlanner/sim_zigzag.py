"""
sim_zigzag.py — closed-loop check of the zig-zag approach (X2) on a laptop.

Runs the real StatePlannerReceiver, Navigation (A* + MPC) and the simulated
drivetrain (MotorDriver, Simulated=True) together, with a fake camera that
reports the decoy only while it is inside the horizontal field of view. For
each start the robot is armed (AUTO), sees the decoy and approaches it.

Checks the X2 "done when" rules:
  1. zig-zag paths are visibly not straight
  2. the bird's bearing never leaves the FOV (|bearing| <= HFOV_DEG / 2)
  3. the robot stops 4 m +- 0.5 m from the bird
  4. it never loses the bird (no Alert)
and saves a plot of W26's straight chase vs. the zig-zag for the slides.

Needs casadi and scipy (matplotlib for the plot). Run from the repo root:
    python3 -m StatePlanner.sim_zigzag                      # ~1.5 min
    python3 -m StatePlanner.sim_zigzag --plot zigzag.png --seed 7
It runs in real time on the open 100 x 100 m test field (Navigation/FakeZone.csv).
Until W0 lands, it points the map loaders at the repo's own files itself.
"""

import argparse
import asyncio
import math
import random
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

try:
    import serial  # noqa: F401
except ImportError:                     # DriveTrain imports pyserial even when simulated
    sys.modules["serial"] = types.ModuleType("serial")

MAP_CSV = REPO / "Navigation" / "FakeZone.csv"

from Navigation import MapUpdater                  # noqa: E402
from StatePlanner import ZoneChecker as zone_mod   # noqa: E402

MapUpdater.Map._Map_File = str(MAP_CSV)
zone_mod.ZoneChecker.CSV_PATH = str(MAP_CSV)

import StatePlanner.StatePlannerReceiver as spr        # noqa: E402
from DriveTrain.DriveTrainDriver import MotorDriver    # noqa: E402
from Navigation.NavigationDriver import TrajectoryTracker  # noqa: E402
from StatePlanner.ZigZag import HFOV_DEG, bearing_of, heading_to  # noqa: E402

TURKEY     = np.array([60.0, 50.0])
START_DIST = 20.0
HOLD_M     = spr.StatePlannerReceiver.CHASE_STOP_DIST_M
TIMEOUT_S  = 60.0


class AsyncPublisher:
    """Same as Main.py's."""

    def __init__(self, name=""):
        self._name, self._subscribers = name, []

    def subscribe(self, callback):
        self._subscribers.append(callback)

    async def publish(self, data):
        for cb in self._subscribers:
            try:
                await cb(data)
            except Exception as e:
                print(f"[{self._name}] Subscriber error: {e}")


class FakeCamera:
    """Publishes DynamicObjects at 10 Hz; the decoy is listed only while it's in view."""

    def __init__(self, pub, turkey_xy, hfov_deg, range_m=30.0, hz=10.0):
        self.pub, self.turkey = pub, np.asarray(turkey_xy, dtype=float)
        self.half = math.radians(hfov_deg / 2.0)
        self.range_m, self.period = range_m, 1.0 / hz
        self.pose = None
        self.samples = []          # (t, x, y, theta, bearing, visible)

    async def RobotPose_Receiver(self, pose):
        self.pose = np.asarray(pose, dtype=float)[:3].copy()

    async def run(self):
        loop = asyncio.get_event_loop()
        while True:
            if self.pose is not None:
                b = bearing_of(self.pose, self.turkey)
                visible = (abs(b) <= self.half
                           and np.linalg.norm(self.turkey - self.pose[:2]) <= self.range_m)
                self.samples.append((loop.time(), *self.pose, b, visible))
                await self.pub.publish({
                    "Id": "DynamicObjects",
                    "Turkeys": [[*self.turkey.tolist(), b]] if visible else [],
                    "Humans": [], "Ghosts": [],
                })
            await asyncio.sleep(self.period)


def start_pose(angle_deg, offset_deg=5.0):
    """Start START_DIST from the bird at angle_deg (zone heading), facing it +offset."""
    a = math.radians(angle_deg)
    xy = TURKEY + START_DIST * np.array([math.cos(a), -math.sin(a)])
    return np.array([*xy, heading_to(xy, TURKEY) + math.radians(offset_deg)])


async def run_one(pose0, approach, seed, verbose=False):
    pubs = {n: AsyncPublisher(n) for n in
            ("drive", "slip", "pose", "nav_motion", "mpc", "sp_motion", "macro", "canine", "vision")}
    spr.StatePlannerReceiver._load_waypoints = lambda self, path: [pose0[:2].copy()]

    drivetrain = MotorDriver(pubs["pose"], pubs["slip"], Simulated=True)
    drivetrain._dr_state = pose0.astype(np.float64).copy()
    nav = TrajectoryTracker(pubs["drive"], pubs["nav_motion"], pubs["mpc"], Verbose=False)
    planner = spr.StatePlannerReceiver(pubs["macro"], pubs["canine"], pubs["sp_motion"],
                                       Verbose=verbose)
    planner.CHASE_RANDOM_COMBOS = False      # this sim compares fixed approach shapes
    planner.CHASE_APPROACH = approach
    planner._zigzag.rng = random.Random(seed)
    cam = FakeCamera(pubs["vision"], TURKEY, HFOV_DEG)

    macro_ids, astar_paths = [], []

    async def on_macro(d):
        macro_ids.append(d.get("Id"))

    async def on_mpc(d):
        if d.get("Id") == "Reference":
            astar_paths.append(d["Cmd"])

    pubs["mpc"].subscribe(on_mpc)

    pubs["drive"].subscribe(drivetrain.MotorCommandReceiver)
    for cb in (nav.RobotPose_Receiver, planner.RobotPose_Receiver, cam.RobotPose_Receiver):
        pubs["pose"].subscribe(cb)
    pubs["sp_motion"].subscribe(nav.TrajectoryCMD_Receiver)
    pubs["vision"].subscribe(nav.VisionObject_Receiver)
    pubs["vision"].subscribe(planner.VisionObject_Receiver)
    pubs["macro"].subscribe(on_macro)

    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(drivetrain.DriveTrainDriver_Runner()), loop.create_task(cam.run())]
    await asyncio.sleep(0.5)
    await planner.MacroState_Receiver({"Id": "UnlockUI"})

    t0, held_since, outcome = loop.time(), None, "timeout"
    while loop.time() - t0 < TIMEOUT_S:
        await asyncio.sleep(0.2)
        if "Alert" in macro_ids:
            outcome = "lost the bird (Alert)"
            break
        s = cam.samples
        if len(s) >= 6:
            speed = math.hypot(s[-1][1] - s[-6][1], s[-1][2] - s[-6][2]) / (s[-1][0] - s[-6][0])
            dist = float(np.linalg.norm(TURKEY - np.array(s[-1][1:3])))
            if dist < HOLD_M + 1.0 and speed < 0.05:
                held_since = held_since or loop.time()
                if loop.time() - held_since > 2.0:
                    outcome = "held"
                    break
            else:
                held_since = None

    for t in tasks:
        t.cancel()
    planner._cancel_current_task()
    return {"approach": approach, "seed": seed, "pose0": pose0, "outcome": outcome,
            "samples": cam.samples, "legs": list(planner._zigzag.legs), "astar": astar_paths,
            "duration_s": loop.time() - t0}


def evaluate(run):
    s = np.array([r[1:5] for r in run["samples"]], dtype=float)      # x, y, theta, bearing
    xy = s[:, :2]
    p0 = run["pose0"][:2]
    line = (TURKEY - p0) / np.linalg.norm(TURKEY - p0)
    lateral = (xy[:, 0] - p0[0]) * line[1] - (xy[:, 1] - p0[1]) * line[0]
    final_dist = float(np.linalg.norm(TURKEY - xy[-1]))
    return {
        "max_lateral_m": float(np.max(np.abs(lateral))),
        "max_bearing_deg": float(np.degrees(np.max(np.abs(s[:, 3])))),
        "final_dist_m": final_dist,
    }


def plot(runs, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping the plot")
        return
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})
    colors = ["#2a78d6", "#e8743b", "#1a9e77", "#8f5bd6", "#d6455d"]
    zi = 0
    for run in runs:
        s = np.array([r[:6] for r in run["samples"]], dtype=float)
        t = s[:, 0] - s[0, 0]
        if run["approach"] == "straight":
            style = dict(color="#777777", ls="--", lw=2, label="W26 straight chase")
        else:
            style = dict(color=colors[zi % len(colors)], lw=2, label=f"zig-zag (seed {run['seed']})")
            zi += 1
        ax.plot(s[:, 1], s[:, 2], **style)
        ax.plot(s[0, 1], s[0, 2], "o", color=style["color"])
        axb.plot(t, np.degrees(s[:, 4]), **style)
    ax.plot(*TURKEY, marker="*", ms=16, color="#c0392b", ls="none", label="decoy")
    ax.add_patch(plt.Circle(TURKEY, HOLD_M, fill=False, ls=":", color="#c0392b"))
    ax.set_aspect("equal")
    ax.invert_yaxis()                      # +Y is down the zone grid
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m, down the grid)")
    ax.set_title(f"Approach paths (hold at {HOLD_M:.0f} m)")
    ax.legend(loc="best", fontsize=8)
    half = HFOV_DEG / 2.0
    axb.axhspan(-half, half, color="#2a78d6", alpha=0.07)
    for y in (-half, half):
        axb.axhline(y, color="#2a78d6", lw=1, ls=":")
    axb.set_xlabel("time (s)")
    axb.set_ylabel("bearing to decoy (deg, + = left)")
    axb.set_title(f"Decoy stays in view (HFOV {HFOV_DEG:.0f} deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"plot saved: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=1, help="first zig-zag seed")
    ap.add_argument("--angles", type=float, nargs="+", default=[180.0, 140.0, 220.0],
                    help="start angles around the decoy (deg, zone heading)")
    ap.add_argument("--no-straight", action="store_true", help="skip the W26 comparison run")
    ap.add_argument("--plot", default="zigzag_vs_straight.png", help="output image")
    ap.add_argument("--verbose", action="store_true", help="planner debug prints")
    args = ap.parse_args()

    print(f"HFOV_DEG = {HFOV_DEG}  (zig offsets <= {HFOV_DEG / 2 - 10:.0f} deg)")
    jobs = [("zigzag", a, args.seed + i) for i, a in enumerate(args.angles)]
    if not args.no_straight:
        jobs.append(("straight", args.angles[0], 0))

    runs, failures = [], 0
    for approach, angle, seed in jobs:
        print(f"\n── {approach}, start angle {angle:.0f} deg, seed {seed} ──")
        run = asyncio.run(run_one(start_pose(angle), approach, seed, args.verbose))
        ev = evaluate(run)
        runs.append(run)
        for leg in run["legs"]:
            print(f"   leg: {leg['side']:5s} {leg['offset_deg']:5.1f} deg  {leg['length_m']:4.1f} m"
                  f"  (planned end bearing {leg['end_bearing_deg']:.1f} deg)")
        checks = [
            (run["outcome"] == "held", f"outcome: {run['outcome']} after {run['duration_s']:.1f} s"),
            (abs(ev["final_dist_m"] - HOLD_M) <= 0.5, f"stopped {ev['final_dist_m']:.2f} m from the decoy"),
            (ev["max_bearing_deg"] <= HFOV_DEG / 2.0,
             f"max |bearing| {ev['max_bearing_deg']:.1f} deg (FOV limit {HFOV_DEG / 2:.0f})"),
        ]
        if approach == "zigzag":
            checks.append((ev["max_lateral_m"] >= 0.5 and len(run["legs"]) >= 2,
                           f"not straight: {len(run['legs'])} legs, max {ev['max_lateral_m']:.2f} m off the line"))
        else:
            print(f"   (straight run: max {ev['max_lateral_m']:.2f} m off the line)")
        for ok, what in checks:
            print(f"   {'PASS' if ok else 'FAIL'}  {what}")
            failures += not ok

    plot(runs, args.plot)
    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} check(s) FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
