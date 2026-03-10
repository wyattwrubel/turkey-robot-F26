import DriveTrain.DriveTrainDriver as Drive
import asyncio
import numpy as np
import math
import matplotlib.pyplot as plt
import Navigation.NavigationDriver as Nav
import Canine.CanineDriver as K9
from UserInterface.app import CDSControlServer


# ═══════════════════════════════════════════════════════════════════════════════
# Async Publisher
# ═══════════════════════════════════════════════════════════════════════════════

class AsyncPublisher:
    def __init__(self):
        self._subscribers = []

    def subscribe(self, callback):
        self._subscribers.append(callback)

    async def publish(self, data):
        for callback in self._subscribers:
            try:
                await callback(data)
            except Exception as e:
                print(f"Subscriber error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Protocol Testers
# Each function exercises exactly one protocol channel.
# Run them individually or combine in test_all_protocols().
# ═══════════════════════════════════════════════════════════════════════════════

# ── Ch. 4: RobotPose ─────────────────────────────────────────────────────────
async def test_robot_pose(pose_publisher: AsyncPublisher):
    """
    Simulates robot driving a slow circle.
    Drives: UKF X, Y, Theta fields in the diagnostics panel.
    Protocol: 1x3 numpy array [X, Y, theta]
    """
    print("[TEST] RobotPose: starting circle trajectory")
    for i in range(60):
        t = i * 0.2
        x     = 20.0 + 5.0 * math.cos(t * 0.3)
        y     = 10.0 + 5.0 * math.sin(t * 0.3)
        theta = (t * 0.3) % (2 * math.pi)

        pose = np.array([x, y, theta])
        await pose_publisher.publish(pose.tolist())
        await asyncio.sleep(0.2)

    print("[TEST] RobotPose: complete")


# ── UncertaintyPose ──────────────────────────────────────────────────────────
async def test_uncertainty_pose(uncertainty_publisher: AsyncPublisher):
    """
    Ramps sigma values up then back down.
    Drives: σx, σy, σθ rows in the UKF diagnostics block.
    Protocol: 1x3 numpy array [sigma_x, sigma_y, sigma_theta]
    """
    print("[TEST] UncertaintyPose: ramping sigma values")
    steps = list(range(0, 21)) + list(range(20, -1, -1))
    for step in steps:
        t      = step / 20.0
        sigma  = np.array([
            0.02 + 0.18 * t,   # sigma_x:     0.02 → 0.20
            0.02 + 0.13 * t,   # sigma_y:     0.02 → 0.15
            0.005 + 0.095 * t, # sigma_theta: 0.005 → 0.10
        ])
        await uncertainty_publisher.publish(sigma.tolist())
        await asyncio.sleep(0.15)

    print("[TEST] UncertaintyPose: complete")


# ── Ch. 8.1: UWB_Values ──────────────────────────────────────────────────────
async def test_uwb(uwb_publisher: AsyncPublisher):
    """
    Sends UWB readings that drift slowly around a fixed point.
    Drives: UWB Raw X, Y and the 'last fix' age timer.
    Protocol: 1x2 numpy array [UWB_X, UWB_Y]
    """
    print("[TEST] UWB_Values: sending noisy position fixes")
    for i in range(40):
        uwb = np.array([
            15.0 + np.random.normal(0, 0.08),
            8.0  + np.random.normal(0, 0.08),
        ])
        await uwb_publisher.publish(uwb.tolist())
        await asyncio.sleep(0.3)   # intentionally slower — watch age timer climb between fixes

    print("[TEST] UWB_Values: complete")


# ── Ch. 8.2: Acceler_Values ──────────────────────────────────────────────────
async def test_accelerometer(accelerometer_publisher: AsyncPublisher):
    """
    Sweeps IMU theta through a full rotation.
    Drives: IMU Raw θ field.
    Protocol: 1x1 numpy array [theta (rad)]
    """
    print("[TEST] Acceler_Values: sweeping IMU theta")
    for i in range(63):
        theta = np.array([(i / 62.0) * 2 * math.pi])
        await accelerometer_publisher.publish(theta.tolist())
        await asyncio.sleep(0.1)

    print("[TEST] Acceler_Values: complete")


# ── Ch. 3: DriveTrain ────────────────────────────────────────────────────────
async def test_drivetrain(drivetrain_publisher: AsyncPublisher):
    """
    Cycles through the four drive commands from the protocol.
    Drives: L/R command chips and drivetrain source label.
    Protocol: {Id: "UI"|"System", Cmd: [left, right]}
    """
    print("[TEST] DriveTrain: cycling motor commands")
    moves = [
        ("System", [ 1.5,  1.5], "Forward"),
        ("System", [-1.5, -1.5], "Reverse"),
        ("System", [-1.0,  1.0], "Spin Left"),
        ("System", [ 1.0, -1.0], "Spin Right"),
        ("UI",     [ 0.8,  0.6], "UI Curve Right"),
        ("System", [ 0.0,  0.0], "Stop"),
    ]
    for source, cmd, label in moves:
        payload = {"Id": source, "Cmd": cmd}
        await drivetrain_publisher.publish(payload)
        print(f"  [DRIVE] {label:20s}  L={cmd[0]:+.1f}  R={cmd[1]:+.1f}  src={source}")
        await asyncio.sleep(1.5)

    print("[TEST] DriveTrain: complete")


# ── Ch. 6: Navigation (SpeedCap + Move + Orient + Chase + Scan + Alert) ───────
async def test_navigation(stateplanner_publisher: AsyncPublisher):
    """
    Exercises all Navigation channel message types.
    Drives: speed_ms display, wp_id, wp_dist, wp_bearing, target_status.
    """
    print("[TEST] Navigation: exercising all message types")

    # SpeedCap — ramp speed up then down
    for spd in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 2.0, 1.0, 1.5]:
        payload = {"Id": "SpeedCap", "Cmd": [spd]}
        await stateplanner_publisher.publish(payload)
        print(f"  [NAV] SpeedCap -> {spd:.1f} m/s")
        await asyncio.sleep(0.4)

    await asyncio.sleep(0.5)

    # Move — traverse a sequence of waypoints
    waypoints = [
        {"wp_id": "WP 1 / 5", "wp_dist": 18.2, "wp_bearing": 32.0},
        {"wp_id": "WP 2 / 5", "wp_dist": 12.7, "wp_bearing": 75.0},
        {"wp_id": "WP 3 / 5", "wp_dist":  7.1, "wp_bearing": 110.0},
        {"wp_id": "WP 4 / 5", "wp_dist":  3.4, "wp_bearing": 155.0},
        {"wp_id": "WP 5 / 5", "wp_dist":  0.9, "wp_bearing": 178.0},
    ]
    for wp in waypoints:
        payload = {"Id": "Move", "Cmd": [17, 5, 0], **wp}
        await stateplanner_publisher.publish(payload)
        print(f"  [NAV] Move -> {wp['wp_id']}  dist={wp['wp_dist']} m  bearing={wp['wp_bearing']}°")
        await asyncio.sleep(1.2)

    await asyncio.sleep(0.5)

    # Orient — small rotation adjustment
    payload = {"Id": "Orient", "Cmd": [17, 5, math.pi / 2], "wp_bearing": 90.0}
    await stateplanner_publisher.publish(payload)
    print("  [NAV] Orient -> adjusting heading to 90°")
    await asyncio.sleep(1.5)

    # Scan mode
    payload = {"Id": "Scan"}
    await stateplanner_publisher.publish(payload)
    print("  [NAV] Scan -> target_status=scanning")
    await asyncio.sleep(1.5)

    # Alert mode
    payload = {"Id": "Alert"}
    await stateplanner_publisher.publish(payload)
    print("  [NAV] Alert -> target_status=alert")
    await asyncio.sleep(1.5)

    # Chase mode — with a target distance
    payload = {"Id": "Chase", "Cmd": [12.0, 8.0, 0], "dist_target": 6.3}
    await stateplanner_publisher.publish(payload)
    print("  [NAV] Chase -> target_status=chasing  dist=6.3 m")
    await asyncio.sleep(1.5)

    # Relinquish
    payload = {"Id": "UnlockUI"}
    await stateplanner_publisher.publish(payload)
    print("  [NAV] UnlockUI -> returning to auto")

    print("[TEST] Navigation: complete")


# ── Ch. 2: Canine ────────────────────────────────────────────────────────────
async def test_canine(canine_publisher: AsyncPublisher):
    """
    Exercises all three Canine sub-channels: Pose, Servo, Audio.
    Drives: pose buttons, servo sliders, audio buttons in the Controls panel.
    Protocol: {Id, Type, Name?, Data}
    """
    print("[TEST] Canine: starting sequence")

    # Poses 1 → 5
    for pose in ["Pose1", "Pose2", "Pose3", "Pose4", "Pose5"]:
        payload = {"Id": "System", "Type": "Pose", "Data": pose}
        await canine_publisher.publish(payload)
        print(f"  [K9] Pose -> {pose}")
        await asyncio.sleep(1.5)

    await asyncio.sleep(0.5)

    # Servo sweep — all five servos
    servo_ranges = {
        "FrontLeft":  (0,   180),
        "FrontRight": (0,   180),
        "RearLeft":   (0,   270),
        "RearRight":  (-90,  90),
        "Head":       (0,   180),
    }
    print("  [K9] Servo sweep forward")
    for step in range(0, 21):
        t = step / 20.0
        for name, (lo, hi) in servo_ranges.items():
            angle = round(lo + (hi - lo) * t, 1)
            payload = {"Id": "System", "Type": "Servo", "Name": name, "Data": angle}
            await canine_publisher.publish(payload)
        await asyncio.sleep(0.2)

    print("  [K9] Servo sweep reverse")
    for step in range(20, -1, -1):
        t = step / 20.0
        for name, (lo, hi) in servo_ranges.items():
            angle = round(lo + (hi - lo) * t, 1)
            payload = {"Id": "System", "Type": "Servo", "Name": name, "Data": angle}
            await canine_publisher.publish(payload)
        await asyncio.sleep(0.2)

    await asyncio.sleep(0.5)

    # Audio clips
    for sound in ["Audio1", "Audio2", "Audio3", "Audio4", "Audio5"]:
        payload = {"Id": "System", "Type": "Audio", "Data": sound}
        await canine_publisher.publish(payload)
        print(f"  [K9] Audio -> {sound}")
        await asyncio.sleep(1.5)

    print("[TEST] Canine: complete")


# ── Ch. 1: VisionObject ──────────────────────────────────────────────────────
async def test_vision(visionobject_publisher: AsyncPublisher):
    """
    Sends a sequence of turkey and human detections.
    Drives: target_status and dist_target via Navigation's Chase handler.
    The Vision module itself doesn't write to robot state — it's consumed
    by Navigation — so this just confirms the receiver fires without error.
    Protocol: {Id: "DynamicObjects", Turkeys: Tx3, Humans: Px3}
    """
    print("[TEST] VisionObject: sending detection frames")
    for i in range(10):
        turkeys = np.array([[5.0 + i * 0.3, 5.0, 0.1]])
        humans  = np.array([[9.0, 7.0 + i * 0.2, -0.05]])
        payload = {
            "Id":      "DynamicObjects",
            "Turkeys": turkeys.tolist(),
            "Humans":  humans.tolist(),
        }
        await visionobject_publisher.publish(payload)
        print(f"  [VIS] Frame {i+1:2d}  turkey=({turkeys[0,0]:.1f}, {turkeys[0,1]:.1f})")
        await asyncio.sleep(0.5)

    print("[TEST] VisionObject: complete")


# ── Chase sequence ────────────────────────────────────────────────────────────
async def test_chase_sequence(
    pose_publisher:         AsyncPublisher,
    stateplanner_publisher: AsyncPublisher,
    visionobject_publisher: AsyncPublisher,
    drivetrain_publisher:   AsyncPublisher,
):
    """
    Full chase scenario: robot navigates to a waypoint, spots a turkey,
    enters Chase mode, closes in, then loses the target and returns to Scan.

    Drives (simultaneously):
      - Robot pose moving across the map            (map canvas, UKF panel)
      - Vision detections appearing / moving        (map canvas detection boxes)
      - Navigation Chase / Scan state transitions   (diagnostics sys-state)
      - Drivetrain source switching UI→System       (L/R command chips)
      - Speed ramping up during chase               (speed display)
    """
    print("[TEST] Chase sequence: starting")

    # ── Phase 1: Navigating to patrol waypoint (3 s) ─────────────────────────
    print("  [CHASE] Phase 1 — navigating to patrol point")
    await stateplanner_publisher.publish({"Id": "SpeedCap", "Cmd": [1.5]})
    for i in range(15):
        t     = i / 14.0
        x     = 10.0 + 20.0 * t
        y     = 10.0 + 10.0 * t
        theta = math.pi * 0.25
        await pose_publisher.publish([round(x, 2), round(y, 2), round(theta, 4)])
        await stateplanner_publisher.publish({
            "Id": "Move", "Cmd": [30, 20, 0],
            "wp_id": "WP 1 / 3", "wp_dist": round(18.0 * (1 - t), 1), "wp_bearing": 45.0,
        })
        await asyncio.sleep(0.2)

    # ── Phase 2: Arrived — 360° scan spin (2.5 s) ────────────────────────────
    print("  [CHASE] Phase 2 — arrived, scanning (360° spin)")
    await stateplanner_publisher.publish({"Id": "Scan"})
    await drivetrain_publisher.publish({"Id": "System", "Cmd": [-1.0, 1.0]})
    for i in range(25):
        theta = (i / 25) * 2 * math.pi
        await pose_publisher.publish([30.0, 20.0, round(theta, 4)])
        await asyncio.sleep(0.1)
    await drivetrain_publisher.publish({"Id": "System", "Cmd": [0.0, 0.0]})
    await asyncio.sleep(0.3)

    # ── Phase 3: Target spotted — detection box appears (2 s) ────────────────
    print("  [CHASE] Phase 3 — target spotted")
    for i in range(10):
        t = i / 9.0
        # Turkey starts near centre of field, drifting toward the far boundary
        await visionobject_publisher.publish({
            "Id": "DynamicObjects",
            "Turkeys": [[round(60.0 + t * 6, 2), round(38.0 + t * 3, 2)]],
            "Humans":  [],
        })
        await asyncio.sleep(0.2)

    # ── Phase 4: Chase engaged — robot closes in, turkey flees past boundary ──
    print("  [CHASE] Phase 4 — chase engaged")
    await stateplanner_publisher.publish({"Id": "SpeedCap", "Cmd": [3.0]})

    robot_x, robot_y   = 30.0, 20.0
    turkey_x, turkey_y = 66.0, 41.0   # turkey position at end of phase 3

    for i in range(25):
        t = i / 19.0
        # Compute heading toward turkey BEFORE moving the robot this step
        prev_rx, prev_ry = robot_x, robot_y
        theta   = math.atan2(turkey_y - prev_ry, turkey_x - prev_rx)

        # Now move robot toward turkey's current position
        robot_x = 30.0 + (turkey_x - 30.0) * t * 0.7
        robot_y = 20.0 + (turkey_y - 20.0) * t * 0.7
        dist    = round(math.hypot(turkey_x - robot_x, turkey_y - robot_y), 2)

        # Turkey accelerates away — crosses MAP boundary by the last few frames
        turkey_x = round(turkey_x + 0.7, 2)
        turkey_y = round(turkey_y + 2, 2)

        await pose_publisher.publish([round(robot_x, 2), round(robot_y, 2), round(theta, 4)])
        await stateplanner_publisher.publish({"Id": "Chase", "Cmd": [turkey_x, turkey_y, 0]})
        await visionobject_publisher.publish({
            "Id": "DynamicObjects",
            "Turkeys": [[turkey_x, turkey_y]],
            "Humans":  [],
        })
        await drivetrain_publisher.publish({"Id": "System", "Cmd": [1.0, 1.0]})
        print(f"  [CHASE]   closing  dist={dist:.1f} m  turkey=({turkey_x:.1f},{turkey_y:.1f})")
        await asyncio.sleep(0.2)

    # ── Phase 5: Turkey exits zone — target lost, return to Scan ─────────────
    print("  [CHASE] Phase 5 — turkey exited zone, returning to scan")
    await visionobject_publisher.publish({"Id": "DynamicObjects", "Turkeys": [], "Humans": []})
    await stateplanner_publisher.publish({"Id": "Scan"})
    await stateplanner_publisher.publish({"Id": "SpeedCap", "Cmd": [1.5]})
    await drivetrain_publisher.publish({"Id": "System", "Cmd": [0.0, 0.0]})
    await asyncio.sleep(5)

    print("[TEST] Chase sequence: complete")


# ── Run all testers concurrently ─────────────────────────────────────────────
async def test_all_protocols(
    pose_publisher,
    uncertainty_publisher,
    uwb_publisher,
    accelerometer_publisher,
    drivetrain_publisher,
    stateplanner_publisher,
    canine_publisher,
    visionobject_publisher,
):
    """
    Runs every protocol tester.
    Some run concurrently (sensors), others are staggered so the UI
    panels don't all update at once and you can see each change clearly.
    """
    print("\n[TEST] ══ Starting full protocol test ══\n")
    await asyncio.sleep(1.0)   # let server fully start

    await asyncio.gather(
        test_robot_pose(pose_publisher),
        test_uncertainty_pose(uncertainty_publisher),
        test_uwb(uwb_publisher),
        test_accelerometer(accelerometer_publisher),
    )

    await asyncio.sleep(0.5)
    await test_drivetrain(drivetrain_publisher)

    await asyncio.sleep(0.5)
    await test_navigation(stateplanner_publisher)

    await asyncio.sleep(0.5)
    await test_canine(canine_publisher)

    await asyncio.sleep(0.5)
    await test_vision(visionobject_publisher)

    print("\n[TEST] ══ Full protocol test complete ══\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

async def main():

    # ── Publishers ───────────────────────────────────────────────────────────
    pose_publisher          = AsyncPublisher()   # Ch. 4  RobotPose
    uncertainty_publisher   = AsyncPublisher()   # New    UncertaintyPose
    uwb_publisher           = AsyncPublisher()   # Ch. 8.1 UWB_Values
    accelerometer_publisher = AsyncPublisher()   # Ch. 8.2 Acceler_Values
    drivetrain_publisher    = AsyncPublisher()   # Ch. 3  DriveTrain
    stateplanner_publisher  = AsyncPublisher()   # Ch. 5/6 MacroState / Navigation
    canine_publisher        = AsyncPublisher()   # Ch. 2  Canine
    visionobject_publisher  = AsyncPublisher()   # Ch. 1  VisionObject
    waypoint_publisher      = AsyncPublisher()   # WaypointInfo — wp_id, wp_dist
    command_publisher       = AsyncPublisher()   # UI → drivetrain (outbound)
    TrajectoryCMD_Publisher = AsyncPublisher()   # High-level nav commands

    # ── UI server ─────────────────────────────────────────────────────────────
    ui = CDSControlServer(
        visionobject_publisher,   # publisher1 — VisionObject
        pose_publisher,           # publisher2 — RobotPose
        uwb_publisher,            # publisher3 — UWB / Accelerometer
        stateplanner_publisher,   # publisher4 — StatePlanner / Navigation
    )

    # ── Wire inbound publishers → UI subscriber methods ───────────────────────
    visionobject_publisher.subscribe(ui.VisionObject_Receiver)
    pose_publisher.subscribe(ui.RobotPose_Receiver)
    uncertainty_publisher.subscribe(ui.UncertaintyPose_Receiver)
    uwb_publisher.subscribe(ui.UWB_Sensors_Receiver)
    accelerometer_publisher.subscribe(ui.Accelerometer_Receiver)
    waypoint_publisher.subscribe(ui.WaypointInfo_Receiver)
    drivetrain_publisher.subscribe(ui.DriveTrain_Receiver)
    stateplanner_publisher.subscribe(ui.StatePlanner_Receiver)
    canine_publisher.subscribe(ui.Canine_Receiver)

    # ── Wire UI commands → drivetrain ─────────────────────────────────────────
    # ui.command_publisher.subscribe(driver.MotorCommandReceiver)

    # ── System components (uncomment as hardware comes online) ────────────────
    # driver    = Drive.MotorDriver(pose_publisher, command_publisher)
    # navigator = Nav.TrajectoryTracker(Verbose=True)
    # pose_publisher.subscribe(navigator.RobotPose_Receiver)
    # visionobject_publisher.subscribe(navigator.VisionObject_Receiver)
    # TrajectoryCMD_Publisher.subscribe(navigator.TrajectoryCMD_Receiver)

    # Canine = K9.CanineDriver()
    # canine_publisher.subscribe(Canine.CanineCommand_Receiver)

    await asyncio.gather(
        ui.serve(),

        # ── Run individual testers (comment/uncomment as needed) ──────────────
        # test_all_protocols(
        #     pose_publisher,
        #     uncertainty_publisher,
        #     uwb_publisher,
        #     accelerometer_publisher,
        #     drivetrain_publisher,
        #     stateplanner_publisher,
        #     canine_publisher,
        #     visionobject_publisher,
        # ),

        # Or run just one at a time:
        # test_robot_pose(pose_publisher),
        # test_uncertainty_pose(uncertainty_publisher),
        # test_uwb(uwb_publisher),
        # test_accelerometer(accelerometer_publisher),
        # test_drivetrain(drivetrain_publisher),
        # test_navigation(stateplanner_publisher),
        # test_canine(canine_publisher),
        # test_vision(visionobject_publisher),
        test_chase_sequence(pose_publisher, stateplanner_publisher, visionobject_publisher, drivetrain_publisher),
    )


if __name__ == "__main__":
    asyncio.run(main())