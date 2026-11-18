"""
check_ui.py: checks for the operator and presentation UIs and the heartbeat watchdog.
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
     c. --ui ops serves the operator UI and --ui present the presentation copy, on port 8000.
  5. Both page copies (UserInterface/Website, WebsitePresent) exist; only the presentation copy
     drops the zone map; both keep the servo limits (check 4).
  6. The GPS / IMU / drivetrain dots come from the run profile and data arrival, not hard-coded
     green (the drivetrain dot said 'on' with the drivetrain powered off, 11-17).
  7. --ui present hides the grey candidate boxes and serves the stripped page; both pages treat
     the detector's updates as proof the camera is live.
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
import urllib.request
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


# ── 5. The two page copies: operator UI and presentation UI ──────────────────
SITES = ("Website", "WebsitePresent")     # Main.py --ui ops / --ui present


def check_sites():
    for site in SITES:
        missing = [f for f in ("index.html", "script.js", "styles.css")
                   if not (REPO / "UserInterface" / site / f).exists()]
        check(f"{site} has index.html, script.js, styles.css", not missing, f"missing {missing}")
    ops = (REPO / "UserInterface" / "Website" / "index.html").read_text()
    pres = (REPO / "UserInterface" / "WebsitePresent" / "index.html").read_text()
    check("operator UI keeps the zone map", 'id="map-canvas"' in ops)
    check("presentation UI has no zone map", 'id="map-canvas"' not in pres and "Zone Map" not in pres)
    check("presentation UI keeps camera, Planner State and E-stop",
          all(k in pres for k in ('id="cam-ph"', 'id="planner-state"', 'id="estop-btn"')))


# ── 6. Header status dots reflect real sources ───────────────────────────────
def check_sources():
    from Robot import Robot
    r = Robot()
    msg = r.to_state_msg()["sources"]
    check("dots default to 'off' (nothing assumed running)",
          all(v == {"mode": "off", "age": None} for v in msg.values()), str(msg))
    r.set_sources({"gps": "off", "imu": "off", "drivetrain": "off"})      # what --profile deter passes
    check("deter profile: drivetrain reported off, not 'nominal'",
          r.to_state_msg()["sources"]["drivetrain"]["mode"] == "off" and r.drivetrain_status == 0)
    r.set_sources({"drivetrain": "real"})
    r.mark_seen("drivetrain")
    d = r.to_state_msg()["sources"]["drivetrain"]
    check("a source's age resets when its data arrives", d["mode"] == "real" and d["age"] is not None
          and d["age"] < 0.5, str(d))
    for site in SITES:
        page = (REPO / "UserInterface" / site / "index.html").read_text()
        js = (REPO / "UserInterface" / site / "script.js").read_text()
        check(f"{site}: GPS / IMU / drivetrain dots have ids the page updates",
              all(f'id="{i}"' in page for i in ("gps-dot", "imu-dot", "dt-dot"))
              and "applySources(m.sources)" in js and ">UWB<" not in page)


# ── 7. Presentation extras: no candidate boxes, stripped page, CAM light ─────
def check_presentation_extras():
    import Vision.config as vcfg
    check("candidate boxes on by default (operator UI)", vcfg.SHOW_CANDIDATES is True)
    vis = (REPO / "Vision" / "Vision.py").read_text()
    main_py = (REPO / "Main.py").read_text()
    check("Vision draws candidates only when config.SHOW_CANDIDATES",
          "if vcfg.SHOW_CANDIDATES:" in vis)
    check("--ui present turns candidate boxes off",
          'if args.ui == "present":' in main_py and "vcfg.SHOW_CANDIDATES = False" in main_py)
    pres = (REPO / "UserInterface" / "WebsitePresent" / "index.html").read_text()
    check("presentation page: big planner state and head angle, fluff hidden",
          'id="head-angle"' in pres and pres.count("present-hide") >= 7
          and 'id="servo-list"' in pres and 'id="audio-grid"' in pres and 'id="estop-btn"' in pres)
    for site in SITES:
        js = (REPO / "UserInterface" / site / ("script" + ".js")).read_text()
        check(f"{site}: CAM light counts detector updates as live (Chrome loads MJPEG once)",
              "_lastVisionMs = Date.now()" in js and "Math.max(lastLoad, _lastVisionMs)" in js)


def fetch(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=5) as r:
        return r.status, r.read().decode()


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

    for site in SITES:
        js = (REPO / "UserInterface" / site / "script.js").read_text()
        block = js[js.index("\nconst SERVO_DEFS"):]
        block = block[:block.index("]")]
        sliders = {sid: (int(lo), int(hi)) for sid, lo, hi in SLIDER.findall(block)}
        check(f"{site}: a slider for all five servos", set(sliders) ==
              {"FrontLeft", "FrontRight", "RearLeft", "RearRight", "Head"}, str(sorted(sliders)))
        wrong = {s: r for s, r in sliders.items() if r != limits(s)}
        check(f"{site}: slider ranges equal the driver's command limits", not wrong,
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
        src = state.get("sources", {})
        check("sim profile: dots get gps/imu/drivetrain as 'sim', drivetrain data fresh",
              {k: v.get("mode") for k, v in src.items()} == {"gps": "sim", "imu": "sim", "drivetrain": "sim"}
              and src.get("drivetrain", {}).get("age") is not None
              and src["drivetrain"]["age"] <= 2.0, str(src))
        check("hidden-tab pattern (beat on each state message) keeps AUTO for 8 s, no E-stop",
              auto_seen and estops == 0, f"auto seen {auto_seen}, E-stop frames {estops}")

        auto_seen, estops = await drain(4.0, reply=1.2)
        check("old hidden-tab pattern (1 beat per 1.2 s) trips the E-stop",
              estops > 0, f"E-stop frames {estops}")


@contextlib.contextmanager
def sim_server(ui):
    """Main.py --profile sim --ui <ui> on port 8000; yields once the port answers, or None."""
    log = REPO / "tests" / f".check_ui_sim_{ui}.log"
    with open(log, "w") as f:
        proc = subprocess.Popen([sys.executable, "Main.py", "--profile", "sim", "--ui", ui,
                                 "--run-for", "40"], cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    try:
        t0 = time.monotonic()
        while not port_open() and time.monotonic() - t0 < 30 and proc.poll() is None:
            time.sleep(0.3)
        ok = port_open()
        if not ok:
            check(f"sim server (--ui {ui}) came up on port 8000", False, f"see {log.relative_to(REPO)}")
        yield proc if ok else None
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        if proc.poll() is None:
            proc.kill()
        while port_open():          # let the port free up before the next server
            time.sleep(0.2)
    log.unlink(missing_ok=True)


def check_watchdog():
    if port_open():
        check("port 8000 free for the sim server", False, "something is already listening")
        return
    with sim_server("ops") as srv:
        if srv:
            status, page = fetch("/")
            check("--ui ops (default) serves the operator UI with the zone map",
                  status == 200 and 'id="map-canvas"' in page)
            asyncio.run(watchdog_session())
    with sim_server("present") as srv:
        if srv:
            status, page = fetch("/")
            js_status, _ = fetch("/script.js")
            check("--ui present serves the presentation UI on the same port, no zone map",
                  status == 200 and js_status == 200 and 'id="map-canvas"' not in page
                  and 'id="planner-state"' in page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sim", action="store_true", help="skip the live watchdog checks")
    args = ap.parse_args()
    check_states()
    check_servo_limits()
    check_sites()
    check_sources()
    check_presentation_extras()
    if not args.no_sim:
        check_watchdog()
    print("\nALL PASS" if FAILS == 0 else f"\n{FAILS} check(s) FAILED")
    sys.exit(0 if FAILS == 0 else 1)


if __name__ == "__main__":
    main()
