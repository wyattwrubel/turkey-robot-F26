"""
sim_combos.py — check the F8 combos (X3) on a laptop; no robot or extra packages (~30 s).

    python3 -m StatePlanner.sim_combos

  1. Picker: 20 chases -> at least 10 distinct combos and no back-to-back repeat
     (also checked over 500 picks).
  2. 20 real chase starts in StatePlannerReceiver (stub publishers): each logs its
     combo, and the opening pose, opening sound, first SpeedCap and the kind of
     Move (zig-zag leg vs straight chase) match that combo.
  3. Speed profile over time: "pulse" goes slow / full / slow at 0 / 3 / 6 s; and
     while the robot holds at 4 m from the bird the profile never raises speed.
It ends with the 20-chase log, ready to paste into tests/log.md.
"""

import asyncio
import contextlib
import io
import sys
import time
import types

import numpy as np

import StatePlanner.StatePlannerReceiver as spr
from StatePlanner.Combos import COMBOS, SPEED_PROFILES, Combo, ComboPicker

TURKEY = [60.0, 50.0]
FAR    = [40.0, 50.0, 0.0]     # 20 m from the bird, facing it
NEAR   = [57.0, 50.0, 0.0]     # 3 m: inside the 4 m hold


class Recorder:
    def __init__(self):
        self.events = []

    def subscribe(self, callback):
        pass

    async def publish(self, data):
        self.events.append((time.monotonic(), data))

    def after(self, t0, kind=None, key="Id"):
        return [(t, d) for t, d in self.events
                if t >= t0 and (kind is None or d.get(key) == kind)]


class _NoZone:
    def is_inside(self, x, y):
        return True

    def is_outside(self, x, y):
        return False


def make_planner():
    spr.ZoneChecker = _NoZone
    spr.StatePlannerReceiver._load_waypoints = lambda self, path: [np.array(FAR[:2])]
    macro, canine, motion = Recorder(), Recorder(), Recorder()
    return spr.StatePlannerReceiver(macro, canine, motion), canine, motion


def vision():
    return {"Id": "DynamicObjects", "Turkeys": [[*TURKEY, 0.0]], "Humans": [], "Ghosts": []}


async def start_chase(p):
    """From SCAN, one vision frame with the bird in the zone starts a chase."""
    p._enter_scan()
    await asyncio.sleep(0.05)
    t0 = time.monotonic()
    await p.VisionObject_Receiver(vision())
    return t0


async def feed(p, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        await p.VisionObject_Receiver(vision())
        await asyncio.sleep(0.1)


async def chases(check, n=20):
    p, canine, motion = make_planner()
    await p.RobotPose_Receiver(FAR)
    await p.MacroState_Receiver({"Id": "UnlockUI"})
    rows = []
    for i in range(n):
        t0 = await start_chase(p)
        await feed(p, 0.4)
        c = p._combo
        pose  = canine.after(t0, "Pose", key="Type")
        sound = canine.after(t0, "Audio", key="Type")
        caps  = motion.after(t0, "SpeedCap")
        moves = motion.after(t0, "Move")
        ok_pose  = bool(pose)  and pose[0][1]["Data"] == c.pose
        ok_sound = bool(sound) and sound[0][1]["Data"] == c.sound
        want_cap = p._speed_level(SPEED_PROFILES[c.speed][0][1])
        ok_speed = bool(caps)  and caps[0][1]["Cmd"] == want_cap
        chase_mode = [d.get("Mode") == "Chase" for _, d in moves]
        ok_shape = bool(moves) and (all(chase_mode) if c.approach == "straight" else not chase_mode[0])
        rows.append((p._chase_count, c, ok_pose and ok_sound and ok_speed and ok_shape))
        check(ok_pose and ok_sound and ok_speed and ok_shape,
              f"chase {i + 1:2d} '{c.name}': pose {'ok' if ok_pose else 'WRONG'}, "
              f"sound {'ok' if ok_sound else 'WRONG'}, speed {'ok' if ok_speed else 'WRONG'}, "
              f"{c.approach} Move {'ok' if ok_shape else 'WRONG'}")
    p._cancel_current_task()
    names = [c.name for _, c, _ in rows]
    check(len(set(names)) >= 10, f"20 chases: {len(set(names))} distinct combos (>= 10)")
    check(all(a != b for a, b in zip(names, names[1:])), "20 chases: no back-to-back repeat")
    return rows


async def speed_profile(check):
    pulse = Combo("test pulse", "straight", "pulse", "AlertStep", "BarkBurst")

    # Robot far away: slow / full / slow at 0 / 3 / 6 s.
    p, _, motion = make_planner()
    p._combo_picker = types.SimpleNamespace(pick=lambda: pulse)
    await p.RobotPose_Receiver(FAR)
    await p.MacroState_Receiver({"Id": "UnlockUI"})
    t0 = await start_chase(p)
    await feed(p, 7.0)
    caps = [(round(t - t0, 1), d["Cmd"]) for t, d in motion.after(t0, "SpeedCap")]
    slow, full = p.SPEED_CHASE_GROWL, p.SPEED_CHASE
    want = [(0.0, slow), (3.0, full), (6.0, slow)]
    ok = len(caps) >= 3 and all(abs(ct - wt) <= 0.3 and cv == wv
                                for (ct, cv), (wt, wv) in zip(caps, want))
    check(ok, f"pulse profile: SpeedCaps {caps} (want slow/full/slow at 0/3/6 s)")
    p._cancel_current_task()

    # Robot inside the 4 m hold: never more than 0 until the bird is far again.
    p, _, motion = make_planner()
    p._combo_picker = types.SimpleNamespace(pick=lambda: pulse)
    await p.RobotPose_Receiver(NEAR)
    await p.MacroState_Receiver({"Id": "UnlockUI"})
    t0 = await start_chase(p)
    await feed(p, 7.0)
    caps = [(round(t - t0, 1), d["Cmd"]) for t, d in motion.after(t0, "SpeedCap")]
    check(caps and caps[0][1] == 0.0 and all(v == 0.0 for _, v in caps),
          f"holding at 3 m: SpeedCaps {caps} (all 0 while the profile steps at 3 and 6 s)")
    t1 = time.monotonic()
    await p.RobotPose_Receiver(FAR)
    await feed(p, 0.8)
    resumed = [d["Cmd"] for _, d in motion.after(t1, "SpeedCap")]
    check(resumed[:1] == [slow], f"bird far again: resumes at the profile's current speed {resumed[:1]}")
    p._cancel_current_task()


def main() -> int:
    results = []

    def check(ok, what):
        results.append((bool(ok), what))

    # 1. Picker alone.
    picker = ComboPicker()
    seq = [picker.pick().name for _ in range(500)]
    check(len(set(seq[:20])) >= 10 and all(a != b for a, b in zip(seq, seq[1:])),
          f"picker: {len(set(seq[:20]))} distinct in the first 20, no repeats in 500 picks")
    check(len(COMBOS) >= 10, f"table has {len(COMBOS)} combos (>= 10)")

    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        rows = asyncio.run(chases(check))
        asyncio.run(speed_profile(check))
    f8_lines = [l for l in log.getvalue().splitlines() if l.startswith("[F8]")]
    check(len(f8_lines) >= 20, f"every chase logged its combo ({len(f8_lines)} [F8] lines)")

    for ok, what in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    failed = sum(1 for ok, _ in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")

    print("\n20-chase log (for tests/log.md):")
    print("| chase | combo | approach | speed | pose | sound |")
    print("|---|---|---|---|---|---|")
    for n, c, _ in rows:
        print(f"| {n} | {c.name} | {c.approach} | {c.speed} | {c.pose} | {c.sound} |")
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
