"""
tests.py — Test suite

Run:
    python main.py --test
    python tests.py          (directly)
"""

import asyncio
import json
import sys
import traceback

from logger import C, log_info
from robot  import MockRobotState


def run_tests() -> bool:
    print(f"\n{C.BOLD}{C.WHITE}{'='*55}")
    print(f"  CDS SERVER — TEST SUITE")
    print(f"{'='*55}{C.RESET}\n")

    results: list[tuple[bool, str]] = []

    def test(name: str, fn):
        try:
            fn()
            results.append((True, name))
            print(f"  {C.GREEN}OK{C.RESET}  {name}")
        except AssertionError as e:
            results.append((False, name))
            print(f"  {C.RED}FAIL{C.RESET}  {name}")
            print(f"       {C.DIM}{e}{C.RESET}")
        except Exception:
            results.append((False, name))
            print(f"  {C.RED}ERR{C.RESET}  {name}")
            traceback.print_exc()

    # ── MockRobotState ────────────────────────────────────────
    print(f"{C.DIM}-- MockRobotState --{C.RESET}")

    def t_initial_state():
        r = MockRobotState()
        assert r.x           == 2.0,  f"x={r.x}"
        assert r.y           == 2.0,  f"y={r.y}"
        assert r.theta       == 0.0,  f"theta={r.theta}"
        assert r.speed_limit == 1.5,  f"speed_limit={r.speed_limit}"
        assert r.det_count   == 0,    f"det_count={r.det_count}"
    test("Initial state values", t_initial_state)

    def t_servo_count():
        r = MockRobotState()
        assert len(r.servo_angles) == 5
        assert "SERVO_1" in r.servo_angles
        assert "SERVO_5" in r.servo_angles
    test("All 5 servos present", t_servo_count)

    def t_motor_command():
        r = MockRobotState()
        r.apply_motor_command(0.8, -0.5, 2.0)
        assert r.cmd_left    == 0.8,  f"cmd_left={r.cmd_left}"
        assert r.cmd_right   == -0.5, f"cmd_right={r.cmd_right}"
        assert r.drive_source == "UI"
    test("Motor command applied", t_motor_command)

    def t_speed_limit():
        r = MockRobotState()
        r.apply_speed_limit(2.5)
        assert r.speed_limit == 2.5
    test("Speed limit applied", t_speed_limit)

    def t_servo_angle():
        r = MockRobotState()
        r.apply_servo_angle("SERVO_3", 45.0)
        assert r.servo_angles["SERVO_3"] == 45.0
    test("Servo angle applied", t_servo_angle)

    def t_estop():
        r = MockRobotState()
        r.apply_motor_command(1.0, 1.0, 3.0)
        r.apply_estop()
        assert r.cmd_left    == 0.0
        assert r.cmd_right   == 0.0
        assert r.drive_source == "ESTOP"
    test("E-stop zeros motors", t_estop)

    def t_tick():
        r = MockRobotState()
        x0 = r.x
        for _ in range(3):
            r.tick(0.1)
        assert r.x != x0 or r.y != 2.0, "tick() did not move robot"
    test("tick() updates position", t_tick)

    # ── State message format ──────────────────────────────────
    print(f"\n{C.DIM}-- State message format --{C.RESET}")

    REQUIRED_KEYS = [
        "type", "uwb_x", "uwb_y", "uwb_age",
        "imu_theta", "cal_sys", "cal_gyro", "cal_mag",
        "x", "y", "theta",
        "sigma_x", "sigma_y", "sigma_theta",
        "dist_target", "target_status",
        "wp_id", "wp_dist", "wp_bearing",
        "cmd_left", "cmd_right", "drive_source",
        "det_status", "det_count", "det_last",
    ]

    def t_state_msg_keys():
        r = MockRobotState()
        r.tick(0.1)
        msg = r.to_state_msg()
        for k in REQUIRED_KEYS:
            assert k in msg, f"missing key: '{k}'"
        assert msg["type"] == "state"
    test("State message has all required keys", t_state_msg_keys)

    def t_state_msg_json():
        r = MockRobotState()
        r.tick(1.0)
        raw    = json.dumps(r.to_state_msg())
        parsed = json.loads(raw)
        assert parsed["type"] == "state"
    test("State message is JSON serialisable", t_state_msg_json)

    def t_servo_state_msg():
        r   = MockRobotState()
        msg = r.to_servo_state_msg()
        assert msg["type"] == "servo_state"
        assert len(msg["servos"]) == 5
        ids = {s["id"] for s in msg["servos"]}
        assert "SERVO_1" in ids
        assert "SERVO_5" in ids
    test("Servo state message format", t_servo_state_msg)

    # ── Message handler routing ───────────────────────────────
    print(f"\n{C.DIM}-- Message handler routing --{C.RESET}")

    # Import here to avoid circular import at module level during testing
    from handlers import handle_message

    class FakeWS:
        client = type("C", (), {"host": "test", "port": 0})()

    def run(coro):
        asyncio.run(coro)

    def t_handler_motor():
        r = MockRobotState()
        async def go():
            await handle_message(FakeWS(), {
                "type": "motor_command", "Id": "UI",
                "Cmd": [0.3, 0.7], "speed_ms": 1.5
            }, r)
        run(go())
        assert r.cmd_left  == 0.3, f"got {r.cmd_left}"
        assert r.cmd_right == 0.7, f"got {r.cmd_right}"
    test("handler: motor_command", t_handler_motor)

    def t_handler_speed():
        r = MockRobotState()
        async def go():
            await handle_message(FakeWS(), {"type": "speed_limit", "speed": 2.2}, r)
        run(go())
        assert r.speed_limit == 2.2, f"got {r.speed_limit}"
    test("handler: speed_limit", t_handler_speed)

    def t_handler_servo():
        r = MockRobotState()
        async def go():
            await handle_message(FakeWS(), {
                "type": "servo_angle", "servo": "SERVO_2", "angle": 55.0
            }, r)
        run(go())
        assert r.servo_angles["SERVO_2"] == 55.0
    test("handler: servo_angle", t_handler_servo)

    def t_handler_estop():
        r = MockRobotState()
        r.apply_motor_command(1.0, 1.0, 3.0)
        async def go():
            await handle_message(FakeWS(), {"type": "estop"}, r)
        run(go())
        assert r.cmd_left    == 0.0
        assert r.cmd_right   == 0.0
        assert r.drive_source == "ESTOP"
    test("handler: estop", t_handler_estop)

    def t_handler_lock():
        r = MockRobotState()
        r.apply_motor_command(0.5, 0.5, 1.5)
        async def go():
            await handle_message(FakeWS(), {"type": "lock_manual"}, r)
        run(go())
        assert r.cmd_left  == 0.0
        assert r.cmd_right == 0.0
    test("handler: lock_manual zeros motors", t_handler_lock)

    def t_handler_unknown():
        r = MockRobotState()
        async def go():
            await handle_message(FakeWS(), {"type": "completely_unknown"}, r)
        run(go())   # should not raise
    test("handler: unknown type does not raise", t_handler_unknown)

    # ── Summary ───────────────────────────────────────────────
    passed = sum(1 for ok, _ in results if ok)
    total  = len(results)
    colour = C.GREEN if passed == total else C.YELLOW

    print(f"\n{C.DIM}{'─'*55}{C.RESET}")
    print(f"  {colour}{C.BOLD}{passed} / {total} tests passed{C.RESET}\n")

    if passed < total:
        print(f"  {C.RED}Failed:{C.RESET}")
        for ok, name in results:
            if not ok:
                print(f"    - {name}")
        print()

    return passed == total


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
