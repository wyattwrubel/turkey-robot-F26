"""
main.py — Test harness for CDS Control Server

Two parallel test tracks run continuously:

  TRACK A — Background fake data (inbound receivers)
  ───────────────────────────────────────────────────
  fake_sensors()     RobotPose / UWB / Accelerometer at 10 Hz
  fake_vision()      VisionObject at 5 Hz (orbiting turkeys + human)
  fake_navigation()  Navigation Move/Orient/SpeedCap every 8s
  fake_macrostate()  MacroState Scan/Chase/Alert every 12s
  fake_canine()      Canine Pose/Servo/Audio every 5s
  fake_video()       VideoFeed colour-shifting JPEG at 2 Hz

  TRACK B — Button sequence test (simulated UI presses)
  ───────────────────────────────────────────────────────
  button_sequence()  Loops through every UI message type in order,
                     pausing 1.5s between each step.
                     Prints SEND / RESULT for every action so you
                     can verify the full pipeline in the console.

Outbound publishers print:
  [DRIVETRAIN]   motor commands, speed cap, estop
  [CANINE]       pose, servo, audio
  [MACROSTATE]   Chase / Scan / Alert

Usage:
    python main.py
    Open http://localhost:8000
    Watch the console — you should see every feature fire in sequence.
"""

import asyncio
import io
import math
import random

from App import CDSControlServer


# ── AsyncPublisher ────────────────────────────────────────────────────────────

class AsyncPublisher:
    def __init__(self):
        self._subscribers = []

    def subscribe(self, callback):
        self._subscribers.append(callback)

    async def publish(self, data):
        for cb in self._subscribers:
            try:
                await cb(data)
            except Exception as e:
                print(f"[PUB ERROR] {e}")


# ── Debug helpers ─────────────────────────────────────────────────────────────

SEP  = "─" * 55
TICK = "  ✓"

def section(title: str):
    print(f"\n{SEP}")
    print(f"  TEST: {title}")
    print(SEP)

def send(label: str, msg: dict):
    print(f"  → SEND   {label:30s}  {msg}")

def result(label: str, robot_state):
    print(f"  {TICK} RESULT {label:30s}  {robot_state}")


# ── TRACK B — Button sequence test ───────────────────────────────────────────

async def button_sequence(ui: CDSControlServer):
    """
    Simulates every UI button/control in order, one per step.
    Loops indefinitely so you can watch state changes in the browser.
    """
    await asyncio.sleep(2.0)   # let server finish startup first

    loop_count = 0
    while True:
        loop_count += 1
        print(f"\n{'═' * 55}")
        print(f"  BUTTON SEQUENCE — loop {loop_count}")
        print(f"{'═' * 55}")

        # ── Speed limit ───────────────────────────────────────
        section("Speed Limit")
        for spd in [0.5, 1.5, 3.0, 1.5]:
            msg = {"type": "speed_limit", "speed": spd}
            send(f"speed={spd}", msg)
            await ui.controls.handle_ui_message(msg)
            result(f"speed_limit", f"{ui.robot.speed_limit} m/s")
            await asyncio.sleep(1.5)

        # ── Drive keys — press then release ───────────────────
        section("Drive Keys (press → release)")
        for key, label in [("w", "Forward"), ("s", "Reverse"),
                            ("a", "Turn Left"), ("d", "Turn Right")]:
            msg_press   = {"type": "key_command", "key": key, "state": "press"}
            msg_release = {"type": "key_command", "key": key, "state": "release"}

            send(f"{label} PRESS  (key={key})", msg_press)
            await ui.controls.handle_ui_message(msg_press)
            result("motors", f"L={ui.robot.cmd_left:.3f}  R={ui.robot.cmd_right:.3f}")
            await asyncio.sleep(1.5)

            send(f"{label} RELEASE (key={key})", msg_release)
            await ui.controls.handle_ui_message(msg_release)
            result("motors", f"L={ui.robot.cmd_left:.3f}  R={ui.robot.cmd_right:.3f}")
            await asyncio.sleep(1.0)

        # ── E-stop ────────────────────────────────────────────
        section("E-Stop")
        msg = {"type": "key_command", "key": "w", "state": "press"}
        send("drive forward first", msg)
        await ui.controls.handle_ui_message(msg)
        result("motors before estop", f"L={ui.robot.cmd_left:.3f}  R={ui.robot.cmd_right:.3f}")
        await asyncio.sleep(1.0)

        msg = {"type": "estop"}
        send("ESTOP ENGAGE", msg)
        await ui.controls.handle_ui_message(msg)
        result("motors after estop", f"L={ui.robot.cmd_left:.3f}  R={ui.robot.cmd_right:.3f}  estop={ui.robot._estop}")
        await asyncio.sleep(1.5)

        msg = {"type": "estop_reset"}
        send("ESTOP RESET", msg)
        await ui.controls.handle_ui_message(msg)
        result("estop released", f"estop={ui.robot._estop}")
        await asyncio.sleep(1.5)

        # ── Drive blocked by estop sanity check ───────────────
        section("Drive blocked by E-Stop (sanity check)")
        await ui.controls.handle_ui_message({"type": "estop"})
        msg = {"type": "key_command", "key": "w", "state": "press"}
        send("drive while estopped", msg)
        await ui.controls.handle_ui_message(msg)
        result("motors (should be 0)", f"L={ui.robot.cmd_left:.3f}  R={ui.robot.cmd_right:.3f}")
        await ui.controls.handle_ui_message({"type": "estop_reset"})
        await asyncio.sleep(1.5)

        # ── Robot Poses ───────────────────────────────────────
        section("Robot Poses")
        for pose in ["Pose1", "Pose2", "Pose3", "Pose4", "Pose5"]:
            msg = {"type": "pose_command", "pose": pose}
            send(f"pose={pose}", msg)
            await ui.controls.handle_ui_message(msg)
            result("active_pose", f"{ui.robot.active_pose}")
            await asyncio.sleep(1.5)

        # ── Servo angles ──────────────────────────────────────
        section("Servo Angles")
        for servo, angle in [("FrontLeft", 30.0), ("FrontRight", -20.0),
                              ("RearLeft", 15.0),  ("RearRight", -15.0),
                              ("Head", 45.0),      ("Head", 0.0)]:
            msg = {"type": "servo_angle", "servo": servo, "angle": angle}
            send(f"{servo}={angle}°", msg)
            await ui.controls.handle_ui_message(msg)
            result(f"servo[{servo}]", f"{ui.robot.servo_angles[servo]}°")
            await asyncio.sleep(1.5)

        # ── Audio ─────────────────────────────────────────────
        section("Audio")
        for sound in ["BarkShort", "BarkBurst", "Growl", "AggressiveBark"]:
            msg = {"type": "audio_play", "sound": sound}
            send(f"sound={sound}", msg)
            await ui.controls.handle_ui_message(msg)
            result("active_audio", f"{ui.robot.active_audio}")
            await asyncio.sleep(1.5)

        # ── MacroState commands ───────────────────────────────
        section("MacroState Commands")
        for state in ["Chase", "Alert", "Scan"]:
            msg = {"type": "macrostate_command", "state": state}
            send(f"state={state}", msg)
            await ui.controls.handle_ui_message(msg)
            result("sys_state", f"{ui.robot.sys_state}")
            await asyncio.sleep(1.5)

        # ── Navigation commands ───────────────────────────────
        section("Navigation — Move / Orient / SpeedCap")

        targets = [
            ("Move   to (60, 40)",     {"Id": "Move",     "Cmd": [60.0, 40.0, 0.0]}),
            ("Move   to (20, 80)",     {"Id": "Move",     "Cmd": [20.0, 80.0, 0.0]}),
            ("Orient to (50,50) 90°",  {"Id": "Orient",   "Cmd": [50.0, 50.0, 1.5708]}),
            ("Orient to (10,10) 180°", {"Id": "Orient",   "Cmd": [10.0, 10.0, 3.1416]}),
            ("SpeedCap 2.0 m/s",       {"Id": "SpeedCap", "Cmd": [2.0]}),
            ("SpeedCap 1.5 m/s",       {"Id": "SpeedCap", "Cmd": [1.5]}),
        ]

        for label, msg in targets:
            send(label, msg)
            await ui.map.Navigation_Receiver(msg)
            t = ui.robot.target
            result(
                "target + nav_type",
                f"type={ui.robot.nav_cmd_type}  "
                f"x={t['x']}  y={t['y']}  θ={t['theta']}  "
                f"speed={ui.robot.speed_limit} m/s"
            )
            await asyncio.sleep(1.5)

        # Verify nav_cmd_type clears correctly on SpeedCap (should stay at last Move/Orient)
        section("Nav type persistence check")
        send("SpeedCap only — nav_type should be unchanged", {"Id": "SpeedCap", "Cmd": [1.0]})
        await ui.map.Navigation_Receiver({"Id": "SpeedCap", "Cmd": [1.0]})
        result("nav_cmd_type (expect Orient)", f"{ui.robot.nav_cmd_type}")
        await asyncio.sleep(1.5)

        print(f"\n  ✓ Loop {loop_count} complete — restarting in 3s\n")
        await asyncio.sleep(3.0)


# ── TRACK A — Background fake data ───────────────────────────────────────────

async def fake_sensors(ui: CDSControlServer):
    """RobotPose / UWB / Accelerometer at 10 Hz — robot orbits field centre."""
    t = 0.0
    cx, cy, r, w = 50.0, 50.0, 10.0, 0.2
    while True:
        await asyncio.sleep(0.1)
        t += 0.1
        x     = cx + r * math.cos(w * t)
        y     = cy + r * math.sin(w * t)
        theta = w * t + math.pi / 2
        await ui.diagnostics.RobotPose_Receiver([x, y, theta])
        await ui.diagnostics.UWB_Sensors_Receiver([x + random.gauss(0, 0.05),
                                        y + random.gauss(0, 0.05)])
        await ui.diagnostics.Accelerometer_Receiver([theta + random.gauss(0, 0.01)])


async def fake_vision(ui: CDSControlServer):
    """VisionObject at 5 Hz — two orbiting turkeys, one stationary human."""
    t = 0.0
    while True:
        await asyncio.sleep(0.2)
        t += 0.2
        turkeys = [
            [50 + 20 * math.cos(0.15 * t),      50 + 20 * math.sin(0.15 * t),      0.0],
            [50 + 30 * math.cos(0.15 * t + 2.1), 50 + 30 * math.sin(0.15 * t + 2.1), 0.0],
        ]
        await ui.VisionObject_Receiver({"Turkeys": turkeys, "Humans": [[70.0, 30.0, 0.0]]})


async def fake_navigation(ui: CDSControlServer):
    """Navigation Move/Orient/SpeedCap cycling every 3s."""
    targets = [
        {"Id": "Move",     "Cmd": [60.0, 60.0, 0.0]},
        {"Id": "Orient",   "Cmd": [60.0, 60.0, 1.57]},
        {"Id": "Move",     "Cmd": [40.0, 40.0, 0.0]},
        {"Id": "SpeedCap", "Cmd": [2.0]},
        {"Id": "SpeedCap", "Cmd": [1.5]},
    ]
    i = 0
    while True:
        await asyncio.sleep(3.0)
        t = targets[i % len(targets)]
        print(f"[FAKE NAV]    Navigation_Receiver → {t}")
        await ui.map.Navigation_Receiver(t)
        i += 1


async def fake_macrostate(ui: CDSControlServer):
    """MacroState Scan/Chase/Alert cycling every 12s."""
    states = [{"Id": "Scan"}, {"Id": "Chase"}, {"Id": "Alert"}]
    i = 0
    while True:
        await asyncio.sleep(12.0)
        s = states[i % len(states)]
        print(f"[FAKE STATE]  MacroState_Receiver → {s}")
        await ui.diagnostics.MacroState_Receiver(s)
        i += 1


async def fake_canine(ui: CDSControlServer):
    """Random Canine Pose/Servo/Audio from system every 5s."""
    while True:
        await asyncio.sleep(5.0)
        choice = random.randint(0, 2)
        if choice == 0:
            pose = random.choice(["Pose1", "Pose2", "Pose3", "Pose4", "Pose5"])
            print(f"[FAKE CANINE] Canine_Receiver → Pose={pose}")
            await ui.controls.Canine_Receiver({"Id": "System", "Type": "Pose", "Data": pose})
        elif choice == 1:
            servo = random.choice(["FrontLeft", "FrontRight", "RearLeft", "RearRight", "Head"])
            angle = round(random.uniform(-30, 30), 1)
            print(f"[FAKE CANINE] Canine_Receiver → Servo {servo}={angle}°")
            await ui.controls.Canine_Receiver({"Id": "System", "Type": "Servo", "Name": servo, "Data": angle})
        else:
            audio = random.choice(["BarkShort", "BarkBurst", "Growl", "AggressiveBark"])
            print(f"[FAKE CANINE] Canine_Receiver → Audio={audio}")
            await ui.controls.Canine_Receiver({"Id": "System", "Type": "Audio", "Data": audio})


async def fake_video(ui: CDSControlServer):
    """Colour-shifting JPEG at 2 Hz."""
    t = 0.0
    while True:
        await asyncio.sleep(0.5)
        t += 0.5
        r = int(127 + 127 * math.sin(t * 0.3))
        g = int(127 + 127 * math.sin(t * 0.3 + 2.09))
        b = int(127 + 127 * math.sin(t * 0.3 + 4.19))
        await ui.VideoFeed_Receiver(_make_solid_jpeg(r, g, b))


def _make_solid_jpeg(r: int, g: int, b: int) -> bytes:
    try:
        from PIL import Image
        img = Image.new("RGB", (8, 8), (r, g, b))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
        return buf.getvalue()
    except ImportError:
        return (
            b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
            b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
            b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\x1e'
            b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'
            b'\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b'
            b'\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04'
            b'\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa'
            b'\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br'
            b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf5\x00\x00\x1f\xff\xd9'
        )


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    drivetrain_pub  = AsyncPublisher()
    canine_pub      = AsyncPublisher()
    macrostate_pub  = AsyncPublisher()
    navigation_pub  = AsyncPublisher()

    async def log_drivetrain(d):  print(f"  [DRIVETRAIN]  {d}")
    async def log_canine(d):      print(f"  [CANINE]      {d}")
    async def log_macrostate(d):  print(f"  [MACROSTATE]  {d}")
    async def log_navigation(d):  print(f"  [NAVIGATION]  {d}")

    drivetrain_pub.subscribe(log_drivetrain)
    canine_pub.subscribe(log_canine)
    macrostate_pub.subscribe(log_macrostate)
    navigation_pub.subscribe(log_navigation)

    ui = CDSControlServer(drivetrain_pub, canine_pub, macrostate_pub, navigation_pub)

    await asyncio.gather(
        ui.serve(),
        # fake_sensors(ui),
        # fake_vision(ui),
        # fake_navigation(ui),
        # fake_macrostate(ui),
        # fake_canine(ui),
        # fake_video(ui),
        # button_sequence(ui),
    )


if __name__ == "__main__":
    asyncio.run(main())