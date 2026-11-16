"""
sim_deter.py — check deter mode (X5) on a laptop, no robot needed.

Drives StatePlannerReceiver(deter_mode=True) with a fake vision feed that
uncovers and covers a decoy several times at different bearings. A stub
canine driver holds each pose 0.5 s like the real PoseDriver. Checks the X5
"done when" rules plus C8:

  1. nothing happens before AUTO (UnlockUI)
  2. each time the decoy appears: AlertPose, head toward it, bark within 1 s
  3. head command = HEAD_SIGN * bearing in degrees, clipped to +-60
  4. while it stays in view, barks are BARK_COOLDOWN_S apart (no spam)
  5. after it disappears: IdlePose about LOST_TIMEOUT_S later
  6. no MotionCommand ever, and Scan / Alert requests are ignored
  7. VisionObject_Receiver returns at once (Vision waits on it)

Run from the repo root (about 35 s):
    python3 -m StatePlanner.sim_deter
Exit code 0 means every check passed.
"""

import asyncio
import math
import sys
import time

import numpy as np

import StatePlanner.StatePlannerReceiver as spr
from StatePlanner.ModuleDeter import ModuleDeter

VISION_HZ = 10.0

# (bearing_rad, seconds in view); each is followed by HIDE_S of empty frames.
SCENARIO = [
    (+0.35, 2.0),
    (-0.25, 2.0),
    (+0.10, 6.5),    # stays in view past the cooldown: expect a second bark
    (+1.20, 2.0),    # beyond the head limit: expect a clipped command
    (-0.60, 2.0),
]
HIDE_S = ModuleDeter.LOST_TIMEOUT_S + 1.0


class Recorder:
    """Stub publisher that records (time, message)."""

    def __init__(self, pose_delay_s: float = 0.0):
        self.events = []
        self.pose_delay_s = pose_delay_s

    def subscribe(self, callback):
        pass

    async def publish(self, data):
        self.events.append((time.monotonic(), data))
        if self.pose_delay_s and isinstance(data, dict) and data.get("Type") == "Pose":
            await asyncio.sleep(self.pose_delay_s)   # PoseDriver holds each pose ~0.5 s


class _NoZone:
    """Zone.csv is read from a Pi-only path until W0; deter mode doesn't use it."""
    def is_inside(self, x, y):
        return True

    def is_outside(self, x, y):
        return False


def make_planner(macro, canine, motion):
    spr.ZoneChecker = _NoZone
    spr.StatePlannerReceiver._load_waypoints = lambda self, path: [np.zeros(2)]
    return spr.StatePlannerReceiver(macro, canine, motion, Verbose=False, deter_mode=True)


async def feed(planner, bearing, seconds, recv_times):
    """Publish DynamicObjects at VISION_HZ for `seconds`; bearing None = empty frames."""
    turkeys = [] if bearing is None else [[3.0, 5.0, bearing]]
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        msg = {"Id": "DynamicObjects", "Turkeys": turkeys, "Humans": [], "Ghosts": []}
        t0 = time.monotonic()
        await planner.VisionObject_Receiver(msg)
        recv_times.append(time.monotonic() - t0)
        await asyncio.sleep(1.0 / VISION_HZ)


def between(rec, t0, t1, pred):
    return [(t, d) for t, d in rec.events if t0 <= t < t1 and pred(d)]


def is_canine(kind, value=None):
    def pred(d):
        if d.get("Type") != kind:
            return False
        if kind == "Servo":
            return d.get("Name") == "Head"
        return value is None or d.get("Data") == value
    return pred


async def main() -> int:
    macro  = Recorder()
    canine = Recorder(pose_delay_s=0.5)
    motion = Recorder()
    planner = make_planner(macro, canine, motion)
    recv_times, results = [], []

    def check(ok, what):
        results.append((bool(ok), what))

    # 1. Not armed: a turkey in view must do nothing.
    await feed(planner, 0.3, 1.0, recv_times)
    check(not canine.events, "before AUTO: no canine output")
    # Let that sighting age out, so sighting 1 starts from idle.
    await feed(planner, None, HIDE_S, recv_times)

    # Arm, like the UI's AUTO button.
    await planner.MacroState_Receiver({"Id": "UnlockUI"})
    await asyncio.sleep(0.7)

    windows = []
    for bearing, show_s in SCENARIO:
        t_show = time.monotonic()
        await feed(planner, bearing, show_s, recv_times)
        t_hide = time.monotonic()
        await feed(planner, None, HIDE_S, recv_times)
        windows.append((bearing, show_s, t_show, t_hide, time.monotonic()))

    # 6. Manual Scan / Alert requests must be ignored and disarm deter mode.
    await planner.MacroState_Receiver({"Id": "Scan"})
    await planner.MacroState_Receiver({"Id": "Alert"})
    t_disarmed = time.monotonic()
    await feed(planner, 0.3, 1.5, recv_times)
    check(not between(canine, t_disarmed, time.monotonic(), lambda d: True),
          "after Scan/Alert from the UI: disarmed, no canine output")
    check(not motion.events, f"no MotionCommand published (got {len(motion.events)})")

    # 2-5. Per sighting.
    print("\nSighting timeline (seconds after the decoy appears / disappears):")
    for i, (bearing, show_s, t_show, t_hide, t_end) in enumerate(windows, 1):
        tag = f"sighting {i} ({math.degrees(bearing):+.0f} deg)"
        alerts = between(canine, t_show, t_end, is_canine("Pose", "AlertPose"))
        heads  = between(canine, t_show, t_end, is_canine("Servo"))
        barks  = between(canine, t_show, t_end, is_canine("Audio", ModuleDeter.BARK_SOUND))
        idles  = between(canine, t_hide, t_end, is_canine("Pose", "IdlePose"))

        expected_head = max(-60.0, min(60.0, ModuleDeter.HEAD_SIGN * math.degrees(bearing)))
        bark_lat = barks[0][0] - t_show if barks else None
        idle_lat = idles[0][0] - t_hide if idles else None
        head_str = f"{heads[0][1]['Data']:+.1f} deg at +{heads[0][0] - t_show:.2f}" if heads else "none"
        print(f"  {tag}: bark +{bark_lat if bark_lat is None else round(bark_lat, 2)} s, "
              f"head {head_str}, {len(barks)} bark(s), idle +{idle_lat if idle_lat is None else round(idle_lat, 2)} s")

        check(len(alerts) == 1, f"{tag}: one AlertPose (got {len(alerts)})")
        check(bark_lat is not None and bark_lat <= 1.0, f"{tag}: bark within 1 s")
        if abs(expected_head) >= ModuleDeter.HEAD_DEADBAND_DEG:
            ok = heads and abs(heads[0][1]["Data"] - expected_head) < 0.5
            check(ok, f"{tag}: head command {expected_head:+.1f} deg")
            if heads and alerts and barks:
                check(alerts[0][0] <= heads[0][0] <= barks[0][0], f"{tag}: order AlertPose, head, bark")
        gaps = [b[0] - a[0] for a, b in zip(barks, barks[1:])]
        check(all(g >= ModuleDeter.BARK_COOLDOWN_S - 0.05 for g in gaps), f"{tag}: barks >= cooldown apart")
        if show_s > ModuleDeter.BARK_COOLDOWN_S + 1.0:
            check(len(barks) >= 2, f"{tag}: re-barks while it stays in view")
        check(idle_lat is not None
              and ModuleDeter.LOST_TIMEOUT_S - 0.2 <= idle_lat <= ModuleDeter.LOST_TIMEOUT_S + 0.5,
              f"{tag}: IdlePose about {ModuleDeter.LOST_TIMEOUT_S:.0f} s after it disappears")

    # 7. Receiver speed.
    check(max(recv_times) < 0.02, f"VisionObject_Receiver max {max(recv_times) * 1000:.1f} ms (< 20 ms)")

    print()
    for ok, what in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    failed = sum(1 for ok, _ in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
