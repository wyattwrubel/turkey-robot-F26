"""
check_profiles.py: checks for the run profiles and repo paths (DR2 plan W0, contracts C0 and C1).
No robot needed. Run from the repo root with the laptop venv:

    python tests/check_profiles.py              # all checks, sim run 60 s
    python tests/check_profiles.py --sim 10     # shorter sim run

  1. No absolute robot-home paths left in the repo's Python (C0).
  2. Each profile builds exactly its modules with the right flags (C1), using stand-ins
     for the hardware classes, so full / vision / deter are checked on a laptop too.
     deter must pass deter_mode=True (C8). The MPC loop is not started a second time.
  3. --profile sim runs for real for N seconds: exit 0 and no errors in the output.
Exit code 0 means every check passed.
"""

import argparse
import asyncio
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    FAILS += 0 if ok else 1


# ── 1. Paths ─────────────────────────────────────────────────────────────────
ROBOT_HOME = "/home/" + "turkeysquad"     # split so this file doesn't match itself


def check_paths():
    hits = [f"{p.relative_to(REPO)}:{i}" for p in REPO.rglob("*.py")
            if "casadi" not in p.relative_to(REPO).parts[:1] and ".venv" not in str(p)
            for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1)
            if ROBOT_HOME in line]
    check("no robot-home absolute paths in the repo's Python", not hits, ", ".join(hits[:5]))


# ── 2. Profiles with stand-in modules ────────────────────────────────────────
class Stub:
    """Stands in for any module: every attribute is another Stub, every call returns one,
    and awaiting a Stub sleeps, like a runner that never ends."""

    def __init__(self, name="stub"):
        self._name = name

    def __getattr__(self, attr):
        return Stub(f"{self._name}.{attr}")

    def __call__(self, *a, **k):
        return Stub(f"{self._name}()")

    def __await__(self):
        return asyncio.sleep(3600).__await__()


def recorder(label, built):
    def make(*args, **kwargs):
        built[label] = kwargs
        return Stub(label)
    return make


EXPECTED = {   # module -> flags it must be built with; absent = must not be built
    "full":   {"VisionObject": {"Simulated": False}, "CanineDriver": {"Simulated": False},
               "MotorDriver": {"Simulated": False}, "TrajectoryTracker": {},
               "StatePlannerReceiver": {}, "PositioningDriver": {"Simulation": True},
               "IMUDriver": {"Simulation": True}, "CDSControlServer": {}},
    "vision": {"VisionObject": {"Simulated": False}, "CDSControlServer": {}},
    "deter":  {"VisionObject": {"Simulated": False}, "CanineDriver": {"Simulated": False},
               "StatePlannerReceiver": {"deter_mode": True}, "CDSControlServer": {}},
    "sim":    {"FakeVision": {}, "CanineDriver": {"Simulated": True},
               "MotorDriver": {"Simulated": True}, "TrajectoryTracker": {},
               "StatePlannerReceiver": {}, "PositioningDriver": {"Simulation": True},
               "IMUDriver": {"Simulation": True}, "CDSControlServer": {}},
}
CLASSES = ["VisionObject", "FakeVision", "CanineDriver", "MotorDriver", "TrajectoryTracker",
           "StatePlannerReceiver", "PositioningDriver", "IMUDriver", "CDSControlServer"]


def check_profiles():
    import Main
    real = {c: getattr(Main, c) for c in CLASSES}
    try:
        for profile, want in EXPECTED.items():
            built = {}
            for c in CLASSES:
                setattr(Main, c, recorder(c, built))
            asyncio.run(Main.main(Main.parse_args(["--profile", profile, "--run-for", "0.05"])))
            check(f"{profile}: builds only {sorted(want)}", set(built) == set(want),
                  f"built {sorted(built)}")
            for c, flags in want.items():
                got = {k: built.get(c, {}).get(k) for k in flags}
                check(f"{profile}: {c} flags {flags}", got == flags, f"got {got}")
    finally:
        for c, v in real.items():
            setattr(Main, c, v)
    code = [l for l in (REPO / "Main.py").read_text().splitlines() if not l.strip().startswith("#")]
    check("MPC loop not started a second time", not any("DriveTrain_Sender(" in l for l in code))
    try:
        Main.parse_args(["--profile", "sim", "--gps", "real"])
        check("--gps real refused outside full", False)
    except SystemExit:
        check("--gps real refused outside full", True)


# ── 3. A real sim run ────────────────────────────────────────────────────────
ERROR_RE = re.compile(r"Traceback|Subscriber error|\bError\b|Exception")


def check_sim(seconds):
    out = subprocess.run([sys.executable, "Main.py", "--profile", "sim", "--run-for", str(seconds)],
                         cwd=REPO, capture_output=True, text=True, timeout=seconds + 120)
    text = out.stdout + out.stderr
    bad = [l for l in text.splitlines() if ERROR_RE.search(l)]
    check(f"sim runs {seconds:g} s: exit 0", out.returncode == 0, f"exit {out.returncode}")
    check(f"sim runs {seconds:g} s: no errors in output", not bad, " | ".join(bad[:3]))
    check("sim reached the timed stop", "--run-for" in text and "stopped" in text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=float, default=60.0, metavar="SECONDS")
    a = ap.parse_args()
    check_paths()
    check_profiles()
    check_sim(a.sim)
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)
