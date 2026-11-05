"""
power_check.py: confirm whether the Pi 5's power supply is what cuts out
under load (ROB450).

It steps the CPU load up (idle -> 1 -> 2 -> 4 cores -> OpenVINO inference)
and, every 0.5 s, logs the supply voltage (EXT5V), the board's estimated
power draw, the under-voltage/throttle flags, temperature and clock speed.
Each line is flushed to disk immediately, so if the supply cuts out, the log
still holds the last readings right up to the moment it died.

Usage on the Pi (RunTimeTest venv has numpy + ultralytics):
  source ~/ME450/Sandbox/RunTimeTest/.venv/bin/activate
  python ~/ME450/MainDesign/bench/power_check.py              # run the load test
  python ~/ME450/MainDesign/bench/power_check.py --report     # after a crash/reboot: read the log

Paths are relative to this file: the log is written next to it, the test image comes from
MainDesign/Vision/, and the model from the Sandbox/ folder beside MainDesign/ (outside the repo).

How to read the result:
  On the battery pack: log stops mid-stage and/or the under-voltage flag
    trips with EXT5V sagging as load rises  -> the pack is the cause.
  On a 5V 5A wall supply: all stages finish, no under-voltage
    -> confirmed: the supply was the problem, not the software.
"""

import argparse
import csv
import multiprocessing as mp
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))          # MainDesign/bench
REPO = os.path.dirname(HERE)                                # MainDesign
ME450 = os.path.dirname(REPO)                               # parent of the repo, holds Sandbox/
LOG = os.path.join(HERE, "power_log.csv")
OV_MODEL = os.path.join(ME450, "Sandbox", "RunTimeTest", "640best_openvino_model")
TEST_IMG = os.path.join(REPO, "Vision", "turkey1.jpg")
STAGE_SECONDS = 25

FLAGS_NOW = {0: "UNDERVOLT_NOW", 1: "FREQ_CAPPED_NOW", 2: "THROTTLED_NOW", 3: "SOFT_TEMP_NOW"}
FLAGS_EVER = {16: "undervolt_since_boot", 17: "freq_capped_since_boot",
              18: "throttled_since_boot", 19: "soft_temp_since_boot"}


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        return ""


def decode_throttled(raw):
    try:
        v = int(raw.split("=")[1], 16)
    except Exception:
        return None, "?"
    names = [n for b, n in {**FLAGS_NOW, **FLAGS_EVER}.items() if v & (1 << b)]
    return v, ",".join(names) or "ok"


def read_pmic():
    """Pi 5 PMIC: EXT5V input voltage plus per-rail V and A -> estimated watts."""
    out = sh("vcgencmd", "pmic_read_adc")
    volts, amps = {}, {}
    for name, kind, val in re.findall(r"(\S+)\s+(current|volt)\(\d+\)=([\d.]+)", out):
        if kind == "volt":
            volts[name[:-2] if name.endswith("_V") else name] = float(val)
        else:
            amps[name[:-2] if name.endswith("_A") else name] = float(val)
    watts = sum(volts[k] * amps[k] for k in amps if k in volts)
    return volts.get("EXT5V", float("nan")), watts


def sample(stage):
    ext5v, watts = read_pmic()
    tv, tflags = decode_throttled(sh("vcgencmd", "get_throttled"))
    temp = sh("vcgencmd", "measure_temp").replace("temp=", "").replace("'C", "")
    clk = sh("vcgencmd", "measure_clock", "arm").split("=")[-1]
    try:
        mhz = int(clk) // 1_000_000
    except ValueError:
        mhz = ""
    return {"time": datetime.now().strftime("%H:%M:%S.%f")[:-4], "stage": stage,
            "ext5v_V": round(ext5v, 3), "est_W": round(watts, 2), "arm_MHz": mhz,
            "temp_C": temp, "throttled": hex(tv) if tv is not None else "?", "flags": tflags}


class Logger(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stage = "idle"
        self.stop = False
        self.min_v = 99.0
        self.max_w = 0.0
        self.undervolt_seen = False

    def run(self):
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        new = not os.path.exists(LOG)
        f = open(LOG, "a", newline="")
        w = None
        last_print = 0
        while not self.stop:
            row = sample(self.stage)
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())  # survive a sudden power cut
            if row["ext5v_V"] == row["ext5v_V"]:  # not NaN
                self.min_v = min(self.min_v, row["ext5v_V"])
            self.max_w = max(self.max_w, row["est_W"])
            if "UNDERVOLT_NOW" in row["flags"]:
                self.undervolt_seen = True
            if time.time() - last_print >= 2:
                print(f"  {row['time']}  {row['stage']:<10} {row['ext5v_V']:.3f} V  "
                      f"{row['est_W']:5.2f} W  {row['arm_MHz']} MHz  {row['temp_C']} C  {row['flags']}",
                      flush=True)
                last_print = time.time()
            time.sleep(0.5)
        f.close()


def burn(stop_at):
    """One core of heavy NEON math (matrix multiply), like inference does."""
    import numpy as np
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    a = np.random.rand(256, 256).astype(np.float32)
    while time.time() < stop_at:
        a = (a @ a) / 256.0


def cpu_stage(cores, seconds):
    stop_at = time.time() + seconds
    procs = [mp.Process(target=burn, args=(stop_at,)) for _ in range(cores)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


def openvino_stage(seconds):
    from ultralytics import YOLO
    import cv2
    model = YOLO(OV_MODEL, task="detect")
    img = cv2.imread(TEST_IMG)
    n, end = 0, time.time() + seconds
    while time.time() < end:
        model.predict(img, imgsz=640, device="cpu", verbose=False)
        n += 1
    return n / seconds


def supply_info():
    print("== Supply as the Pi sees it ==")
    print("  get_throttled:", decode_throttled(sh("vcgencmd", "get_throttled")))
    print("  usb_max_current_enable:", sh("vcgencmd", "get_config", "usb_max_current_enable") or "?")
    # Pi 5 firmware records the PSU's advertised max current (5000 = 5 A PD supply).
    p = "/proc/device-tree/chosen/power/max_current"
    if os.path.exists(p):
        with open(p, "rb") as fh:
            print("  PSU max current advertised:", int.from_bytes(fh.read()[:4], "big"), "mA")
    else:
        print("  PSU max current advertised: not reported")
    kern = sh("sh", "-c", "dmesg 2>/dev/null | grep -iE 'undervolt|under-volt|voltage' | tail -5")
    print("  kernel voltage messages this boot:", kern or "none (or dmesg needs sudo)")


def report():
    supply_info()
    prev = sh("sh", "-c", "journalctl -k -b -1 2>/dev/null | grep -iE 'undervolt|voltage' | tail -5")
    print("  previous boot voltage messages:", prev or "none / journal not persistent")
    if not os.path.exists(LOG):
        print("No log at", LOG)
        return
    with open(LOG) as f:
        rows = list(csv.DictReader(f))
    print(f"\n== Last 12 readings in {LOG} ==")
    for r in rows[-12:]:
        print(f"  {r['time']}  {r['stage']:<10} {r['ext5v_V']} V  {r['est_W']} W  "
              f"{r['arm_MHz']} MHz  {r['temp_C']} C  {r['flags']}")
    last = rows[-1]
    print(f"\nLog ended during stage '{last['stage']}' at {last['time']}.")
    if last["stage"] not in ("done",):
        print("It never reached 'done', so the Pi lost power mid-test. If EXT5V was falling and/or "
              "under-voltage flags were set just before, the supply cut out under load.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="read the log after a crash or reboot")
    ap.add_argument("--seconds", type=int, default=STAGE_SECONDS)
    ap.add_argument("--no-openvino", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
        return

    supply_info()
    print(f"\n== Load steps, {a.seconds}s each (logging every 0.5 s to {LOG}) ==")
    lg = Logger()
    lg.start()

    stages = [("idle", lambda: time.sleep(a.seconds)),
              ("cpu1", lambda: cpu_stage(1, a.seconds)),
              ("cpu2", lambda: cpu_stage(2, a.seconds)),
              ("cpu4", lambda: cpu_stage(4, a.seconds))]
    if not a.no_openvino and os.path.exists(OV_MODEL):
        stages.append(("openvino", lambda: print(f"  openvino: {openvino_stage(a.seconds):.2f} fps")))

    for name, fn in stages:
        print(f"-- stage: {name}", flush=True)
        lg.stage = name
        fn()

    lg.stage = "done"
    time.sleep(1)
    lg.stop = True
    lg.join()

    print("\n== Verdict ==")
    print(f"  lowest EXT5V: {lg.min_v:.3f} V   peak est. power: {lg.max_w:.2f} W")
    if lg.undervolt_seen or lg.min_v < 4.75:
        print("  UNDER-VOLTAGE under load: the supply cannot hold 5 V at this draw. "
              "Supply problem confirmed.")
    else:
        print("  Supply held up through every stage. If this was the battery pack, the pack is fine "
              "at this load; if it was a wall supply, rerun on the pack to compare.")


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
