#!/usr/bin/env python3
"""
motor_test.py  –  One-at-a-time open-loop motor test for RoboClaw (1 motor/driver)
TAB to cycle motors, W/S to spin fwd/back, SPACE to stop, Q to quit.

RoboClaw simple serial OL commands (no CRC, no ACK):
  Drive Forward  : 0x00  value 0-127
  Drive Backward : 0x01  value 0-127
  (These are the simple-serial single-channel commands)
"""

import curses
import time

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ── CONFIG ───────────────────────────────────────────────────────
PORTS = {
    "front_left":  "/dev/ttyMotor0",
    "front_right": "/dev/ttyMotor1",
    "rear_left":   "/dev/ttyMotor2",
    "rear_right":  "/dev/ttyMotor3",
}

MOTOR_DIRECTION = {
    "front_left":  -1,
    "front_right":  1,
    "rear_left":   -1,
    "rear_right":   1,
}

BAUDRATE   = 115200
TIMEOUT    = 0.1
OL_MAX     = 127
TEST_SPEED = 30   # 0-127

MOTOR_LABELS = ["front_left", "front_right", "rear_left", "rear_right"]

# ── PROTOCOL ─────────────────────────────────────────────────────
# Simple serial: just 2 bytes, no address, no CRC, no ACK
#   byte 0: command  (0 = forward, 1 = backward)
#   byte 1: value    (0-127)

def ol_send(ser, value: int, forward: bool) -> None:
    cmd = 0 if forward else 1
    ser.write(bytes([cmd, max(0, min(OL_MAX, value))]))

def ol_stop(ser) -> None:
    ol_send(ser, 0, True)

# ── MOTOR MANAGER ────────────────────────────────────────────────
class MotorManager:
    def __init__(self):
        self.serials   = {}
        self.connected = {}
        for label, port in PORTS.items():
            if not SERIAL_AVAILABLE:
                self.connected[label] = False
                continue
            try:
                ser = serial.Serial(port=port, baudrate=BAUDRATE, timeout=TIMEOUT)
                time.sleep(0.1)
                ser.reset_input_buffer()
                self.serials[label]   = ser
                self.connected[label] = True
            except Exception:
                self.connected[label] = False

    def send(self, label, value, forward):
        if label not in self.serials:
            return
        if MOTOR_DIRECTION[label] == -1:
            forward = not forward
        ol_send(self.serials[label], value, forward)

    def stop(self, label):
        if label in self.serials:
            ol_stop(self.serials[label])

    def stop_all(self):
        for label in self.serials:
            ol_stop(self.serials[label])

    def close(self):
        self.stop_all()
        for ser in self.serials.values():
            try: ser.close()
            except: pass

# ── TUI ──────────────────────────────────────────────────────────
def draw(stdscr, motors, selected, state, log):
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    def put(row, col, text, attr=0):
        if row >= h or col >= w:
            return
        try:
            stdscr.addstr(row, col, text[:w - col - 1], attr)
        except curses.error:
            pass

    HI  = curses.color_pair(1) | curses.A_BOLD
    DIM = curses.color_pair(2)
    SEL = curses.color_pair(3) | curses.A_BOLD
    OK  = curses.color_pair(4)
    ERR = curses.color_pair(5)

    put(0, 0, " MOTOR TEST – OL Simple Serial ".center(w - 1), HI)

    row = 2
    put(row, 2,  "Motor",       DIM)
    put(row, 20, "Port",        DIM)
    put(row, 38, "Conn",        DIM)
    put(row, 44, "State",       DIM)
    row += 1
    put(row, 2, "─" * min(52, w - 4), DIM)
    row += 1

    state_str = {"fwd": "▲ FWD", "bwd": "▼ BWD", "stop": "  ---"}

    for label in MOTOR_LABELS:
        is_sel     = (label == selected)
        conn       = motors.connected.get(label, False)
        attr       = SEL if is_sel else DIM
        prefix     = "►" if is_sel else " "
        conn_str   = "●" if conn else "○"
        conn_color = OK  if conn else ERR
        st         = state_str.get(state, "  ---") if is_sel else "  ---"

        put(row, 0,  prefix,       attr)
        put(row, 2,  label,        attr)
        put(row, 20, PORTS[label], DIM)
        put(row, 38, conn_str,     conn_color)
        put(row, 44, st,           attr)
        row += 1

    row += 1
    put(row, 2, f"Speed: {TEST_SPEED}/127  (no ACK – simple serial)", DIM)
    row += 2

    for key, desc in [("TAB",   "Next motor"),
                      ("W",     "Forward"),
                      ("S",     "Backward"),
                      ("SPACE", "Stop motor"),
                      ("Q/ESC", "Quit")]:
        put(row, 2, f"{key:<6}", HI)
        put(row, 9, desc, DIM)
        row += 1

    row += 1
    if log:
        put(row, 2, f"» {log}", DIM)

    sim = "" if SERIAL_AVAILABLE else "  [NO PYSERIAL – simulation]"
    put(h - 1, 0, f" Ready{sim} ".ljust(w - 1), HI)

    stdscr.refresh()

# ── MAIN ─────────────────────────────────────────────────────────
def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_BLACK,  curses.COLOR_CYAN)
    curses.init_pair(2, curses.COLOR_WHITE,  -1)
    curses.init_pair(3, curses.COLOR_CYAN,   -1)
    curses.init_pair(4, curses.COLOR_GREEN,  -1)
    curses.init_pair(5, curses.COLOR_YELLOW, -1)

    motors  = MotorManager()
    sel_idx = 0
    state   = "stop"
    log     = "TAB to pick motor, W/S to spin, SPACE to stop"

    try:
        while True:
            key      = stdscr.getch()
            selected = MOTOR_LABELS[sel_idx]

            if key in (ord('q'), ord('Q'), 27):
                break

            elif key == ord('\t'):
                motors.stop(selected)
                state   = "stop"
                sel_idx = (sel_idx + 1) % len(MOTOR_LABELS)
                selected = MOTOR_LABELS[sel_idx]
                log = f"Selected: {selected}"

            elif key in (ord('w'), ord('W')):
                motors.send(selected, TEST_SPEED, forward=True)
                state = "fwd"
                log = f"{selected}  FWD  OL={TEST_SPEED}"

            elif key in (ord('s'), ord('S')):
                motors.send(selected, TEST_SPEED, forward=False)
                state = "bwd"
                log = f"{selected}  BWD  OL={TEST_SPEED}"

            elif key == ord(' '):
                motors.stop(selected)
                state = "stop"
                log = f"{selected}  STOP"

            draw(stdscr, motors, selected, state, log)
            time.sleep(0.02)

    finally:
        motors.close()

if __name__ == "__main__":
    curses.wrapper(main)