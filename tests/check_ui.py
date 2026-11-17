"""
check_ui.py: checks for the operator UI's state display and its heartbeat watchdog.
No robot needed. Run from the repo root with the laptop venv:

    python tests/check_ui.py            # all checks (starts --profile sim on port 8000, ~25 s)
    python tests/check_ui.py --no-sim   # static checks only

  1. Every MacroState Id the planner publishes is one the UI accepts (no "Invalid Data"),
     found by scanning StatePlanner/*.py, so a new planner state without a UI entry fails here.
  2. Deter and DeterIdle reach the page's state message as macrostate.
  4. Servo limits: the UI sliders' ranges equal the servo driver's command limits, the driver
     clamps an out-of-range leg command (Rear Right at -135 stalled the rail on 11-17), and every
     W26 pose stays inside the limits.
  3. The watchdog with a real server (--profile sim):
     a. heartbeats sent only in reply to the 10 Hz 'state' messages (what a hidden browser tab
        still does) keep AUTO alive for 8 s: no E-stop;
     b. a timer slowed to one beat per 1.2 s (what a hidden tab did before) trips the E-stop,
        so the watchdog still catches a page that stops answering.
Exit code 0 means every check passed.
"""

import argparse
import asyncio
import contextlib
import io
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "UserInterface"))   # Diagnostics.py imports "from Robot import Robot"

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    FAILS += 0 if ok else 1


# ── 1 + 2. Planner states the UI accepts ─────────────────────────────────────
PUBLISH = re.compile(r'MacroState_Publisher\.publish\(\{\s*"Id"\s*:\s*"([A-Za-z]+)"')


def planner_state_ids():
    ids = set()
    for p in (REPO / "StatePlanner").glob("*.py"):
        if p.name.startswith("sim_"):
            continue
        ids |= set(PUBLISH.findall(p.read_text()))
    return ids


def check_states():
    from Diagnostics import PanelDiagnostics
    from Robot import Robot

    ids = planner_state_ids()
    check("planner publishes Deter and DeterIdle (scan found them)", {"Deter", "DeterIdle"} <= ids,
          f"{len(ids)} ids")

    rejected = []
    for sid in sorted(ids):
        robot = Robot()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            asyncio.run(PanelDiagnostics(robot).MacroState_Receiver({"Id": sid}))
        if "Invalid Data" in out.getvalue():
            rejected.append(sid)
    check("UI accepts every planner MacroState", not rejected, f"rejected: {rejected}" if rejected else "")

    for sid in ("Deter", "DeterIdle"):
        robot = Robot()
        asyncio.run(PanelDiagnostics(robot).MacroState_Receiver({"Id": sid}))
        got = robot.to_state_msg()["macrostate"]
        check(f"{sid} reaches the page as macrostate", got == sid, f"got {got!r}")


# ── 4. Servo limits: UI sliders, driver clamp, poses ─────────────────────────
SLIDER = re.compile(r"\{\s*id:\s*'(\w+)'.*?min:\s*(-?\d+),\s*max:\s*(-?\d+)")
POSE_CMD = re.compile(r'self\.servos\["(\w+)"\]\.servo_control_receiver\(\s*"(\w+)"\s*,\s*'
                      r'(-?\d+(?:\.\d+)?)(?:\s*,\s*(-?\d+(?:\.\d+)?))?')


def check_servo_limits():
    from Canine.ServoDriver import Servo

    def limits(servo_id):
        if servo_id == "Head":
            return Servo.Min_Head_Command, Servo.Max_Head_Command
        return Servo.Min_Leg_Command, Servo.Max_Leg_Command

    js = (REPO / "UserInterface" / "Website" / "script.js").read_text()
    block = js[js.index("\nconst SERVO_DEFS"):]
    block = block[:block.index("]")]
    sliders = {sid: (int(lo), int(hi)) for sid, lo, hi in SLIDER.findall(block)}
    check("UI has a slider for all five servos", set(sliders) ==
          {"FrontLeft", "FrontRight", "RearLeft", "RearRight", "Head"}, str(sorted(sliders)))
    wrong = {s: r for s, r in sliders.items() if r != limits(s)}
    check("UI slider ranges equal the driver's command limits", not wrong,
          f"mismatch: {wrong}" if wrong else f"legs {limits('FrontLeft')}, head {limits('Head')}")

    leg = Servo("RearRight", "Leg", 0, -18, False, Simulated=True)
    leg.SetAngle(-135)
    head = Servo("Head", "Head", 1, -6, True, Simulated=True)
    head.SetAngle(90)
    check("driver clamps a -135 leg command to the leg limit", leg.Current_Angle == Servo.Min_Leg_Command,
          f"got {leg.Current_Angle}")
    check("driver clamps a 90 head command to the head limit", head.Current_Angle == Servo.Max_Head_Command,
          f"got {head.Current_Angle}")

    poses = (REPO / "Canine" / "PoseDriver.py").read_text()
    over = []
    n = 0
    for sid, _kind, a, b in POSE_CMD.findall(poses):
        lo, hi = limits(sid)
        for v in (a, b):
            if v:
                n += 1
                if not lo <= float(v) <= hi:
                    over.append(f"{sid} {v}")
    check("every W26 pose angle is inside the limits", n > 20 and not over,
          f"{n} angles checked" + (f"; over: {over}" if over else ""))


# ── 3. Watchdog against a real server ────────────────────────────────────────
PORT = 8000


def port_open():
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


async def watchdog_session():
    import websockets

    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws", open_timeout=10) as ws:
        state = {}

        async def drain(seconds, reply):
            """Read state for `seconds`. reply='state' answers each state message (rate-limited
            to 200 ms like script.js _beat); reply=<float> sends one beat every that many s."""
            end = time.monotonic() + seconds
            last_beat = 0.0
            estops = 0
            auto_seen = False
            while time.monotonic() < end:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.2)
                except asyncio.TimeoutError:
                    raw = None
                now = time.monotonic()
                if raw is not None:
                    m = json.loads(raw)
                    if m.get("type") == "state":
                        state.update(m)
                        auto_seen |= m.get("is_manual") is False
                        estops += bool(m.get("is_estopped"))
                        if reply == "state" and now - last_beat >= 0.2:
                            last_beat = now
                            await ws.send(json.dumps({"type": "heartbeat"}))
                if isinstance(reply, float) and now - last_beat >= reply:
                    last_beat = now
                    await ws.send(json.dumps({"type": "heartbeat"}))
            return auto_seen, estops

        await ws.send(json.dumps({"type": "heartbeat"}))
        await ws.send(json.dumps({"type": "toggle_mode"}))        # MANUAL -> AUTO arms the watchdog
        auto_seen, estops = await drain(8.0, reply="state")
        check("hidden-tab pattern (beat on each state message) keeps AUTO for 8 s, no E-stop",
              auto_seen and estops == 0, f"auto seen {auto_seen}, E-stop frames {estops}")

        auto_seen, estops = await drain(4.0, reply=1.2)
        check("old hidden-tab pattern (1 beat per 1.2 s) trips the E-stop",
              estops > 0, f"E-stop frames {estops}")


def check_watchdog():
    if port_open():
        check("port 8000 free for the sim server", False, "something is already listening")
        return
    log = REPO / "tests" / ".check_ui_sim.log"
    with open(log, "w") as f:
        proc = subprocess.Popen([sys.executable, "Main.py", "--profile", "sim", "--run-for", "40"],
                                cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    try:
        t0 = time.monotonic()
        while not port_open() and time.monotonic() - t0 < 30 and proc.poll() is None:
            time.sleep(0.3)
        if not port_open():
            check("sim server came up on port 8000", False, f"see {log.relative_to(REPO)}")
            return
        asyncio.run(watchdog_session())
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        if proc.poll() is None:
            proc.kill()
    log.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sim", action="store_true", help="skip the live watchdog checks")
    args = ap.parse_args()
    check_states()
    check_servo_limits()
    if not args.no_sim:
        check_watchdog()
    print("\nALL PASS" if FAILS == 0 else f"\n{FAILS} check(s) FAILED")
    sys.exit(0 if FAILS == 0 else 1)


if __name__ == "__main__":
    main()
