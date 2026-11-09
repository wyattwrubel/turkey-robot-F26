"""
camera_char.py: P1 camera characterization for the ROB 450 robot (specs F1, F5).

Run on the Pi from the repo's bench/ folder with the RunTimeTest venv:
  source ~/ME450/Sandbox/RunTimeTest/.venv/bin/activate
  cd ~/ME450/MainDesign/bench

Steps, in the order you'd do them at the bench:
  python camera_char.py modes                     # formats, sizes, controls (v4l2-ctl)
  python camera_char.py fps                       # real fps per resolution, 300 frames each
  python camera_char.py snap                      # one frame per resolution (crop check, tape on wall)
  python camera_char.py fov --dist 2.0 --width 3.1 [--height 1.7]
                                                  # HFOV/VFOV from tape: camera square to a wall
                                                  # at --dist m; --width = measured span between
                                                  # the image's left and right edges (m)
  python camera_char.py calib-capture --res 640x480 --n 20
                                                  # saves checkerboard frames automatically when
                                                  # the board is found and has moved; move the
                                                  # board around the whole frame and the corners
  python camera_char.py calib --res 640x480 --square 0.025 [--board 9x6]
                                                  # intrinsics, distortion, reprojection error
  python camera_char.py table [--res 640x480]     # px on target at 5/10/15 m and bearing error
                                                  # of W26's linear 102 deg map vs. calibrated

Outputs go next to this script in camera_char/: CSV rows, frames, calib_<res>.json.
Board size is INNER corners (a 10x7-square board has 9x6 inner corners).
"""

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import time
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "camera_char")
CAMERA = "/dev/video_camera"
RESOLUTIONS = [(640, 480), (1280, 720), (1280, 960), (1920, 1080)]
W26_HFOV_DEG = 102.0          # constant in W26's position code (never measured)
BIRD_HEIGHTS_M = (0.6, 0.9)   # range of standing turkey heights used in DR1 sizing
RANGES_M = (5, 10, 15)
PX_FLOOR = 20                 # T17's minimum object height (from IEEE 2671-2022)


# ---------- helpers ----------

def parse_res(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def out_path(*parts):
    p = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def log_row(name, row):
    path = out_path(f"{name}.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def open_cam(w, h, fps=30):
    cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {CAMERA} (is the camera on USB port 3-2?)")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(10):  # let auto-exposure settle
        cap.read()
    return cap


def actual_size(cap):
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def stamp():
    return datetime.now().isoformat(timespec="seconds")


def fov_from_f(f_px, size_px):
    return math.degrees(2 * math.atan(size_px / (2 * f_px)))


def f_from_fov(fov_deg, size_px):
    return (size_px / 2) / math.tan(math.radians(fov_deg) / 2)


# ---------- subcommands ----------

def cmd_modes(_):
    for args in (["--list-formats-ext"], ["--list-ctrls"]):
        r = subprocess.run(["v4l2-ctl", "-d", CAMERA, *args], capture_output=True, text=True)
        print(r.stdout or r.stderr)
        with open(out_path(f"v4l2{args[0].replace('--', '_')}.txt"), "w") as f:
            f.write(r.stdout or r.stderr)


def cmd_fps(a):
    print(f"{'asked':>10} {'got':>10} {'fps':>6} {'read ms (median)':>17}")
    for w, h in RESOLUTIONS:
        try:
            cap = open_cam(w, h)
        except RuntimeError as e:
            print(e)
            return
        aw, ah = actual_size(cap)
        reads, ok = [], 0
        t0 = time.perf_counter()
        for _ in range(a.frames):
            t = time.perf_counter()
            good, _frame = cap.read()
            reads.append((time.perf_counter() - t) * 1000)
            ok += good
        dt = time.perf_counter() - t0
        cap.release()
        fps = ok / dt if dt else 0
        med = sorted(reads)[len(reads) // 2]
        print(f"{w}x{h:<5} {aw}x{ah:<5} {fps:6.1f} {med:17.1f}")
        log_row("fps", {"time": stamp(), "asked": f"{w}x{h}", "got": f"{aw}x{ah}",
                        "frames_ok": ok, "seconds": round(dt, 2), "fps": round(fps, 2),
                        "read_ms_median": round(med, 1), "note": a.note})


def cmd_snap(a):
    for w, h in RESOLUTIONS:
        cap = open_cam(w, h)
        good, frame = cap.read()
        aw, ah = actual_size(cap)
        cap.release()
        if good:
            p = out_path("snaps", f"{a.label}_{aw}x{ah}.jpg")
            cv2.imwrite(p, frame)
            print("saved", p)
    print("Compare the snaps: if a wider mode shows MORE of the scene, the narrower modes crop.")


def cmd_fov(a):
    hfov = math.degrees(2 * math.atan(a.width / (2 * a.dist)))
    vfov = math.degrees(2 * math.atan(a.height / (2 * a.dist))) if a.height else None
    print(f"HFOV = {hfov:.1f} deg" + (f", VFOV = {vfov:.1f} deg" if vfov else ""))
    print(f"(W26's code assumes {W26_HFOV_DEG:.0f} deg)")
    for w, _h in RESOLUTIONS:
        print(f"  f_px at {w} wide = {f_from_fov(hfov, w):.0f}")
    log_row("fov", {"time": stamp(), "res": a.res, "dist_m": a.dist, "width_m": a.width,
                    "height_m": a.height or "", "hfov_deg": round(hfov, 2),
                    "vfov_deg": round(vfov, 2) if vfov else ""})


def cmd_calib_capture(a):
    w, h = parse_res(a.res)
    board = parse_res(a.board)
    folder = out_path(f"calib_{a.res}", "x")[:-2]
    cap = open_cam(w, h)
    saved, last = 0, None
    print(f"Looking for a {board[0]}x{board[1]} inner-corner board. Ctrl-C to stop.")
    try:
        while saved < a.n:
            good, frame = cap.read()
            if not good:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, board, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
            if not found:
                continue
            c = corners.reshape(-1, 2).mean(axis=0)
            if last is not None and np.linalg.norm(c - last) < a.min_move * w:
                continue  # board hasn't moved enough; avoid near-duplicates
            p = os.path.join(folder, f"{saved:02d}.png")
            cv2.imwrite(p, frame)
            last, saved = c, saved + 1
            print(f"saved {saved}/{a.n} (board center at {c[0]:.0f},{c[1]:.0f}); move it")
            time.sleep(a.pause)
    except KeyboardInterrupt:
        pass
    cap.release()
    print(f"{saved} frames in {folder}")


def calibrate(folder, board, square):
    obj = np.zeros((board[0] * board[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square
    objpts, imgpts, size = [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    files = sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))
    for f in files:
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, board)
        if found:
            # refinement window must stay inside one square, or corners drift
            c2 = corners.reshape(-1, 2)
            sq_px = np.median(np.linalg.norm(np.diff(c2[:board[0]], axis=0), axis=1))
            win = int(max(2, min(11, sq_px // 3)))
            corners = cv2.cornerSubPix(gray, corners, (win, win), (-1, -1), crit)
            objpts.append(obj)
            imgpts.append(corners)
    if len(objpts) < 8:
        raise SystemExit(f"only {len(objpts)} usable frames of {len(files)}; need 8+ (aim for 15 to 20)")
    rms, K, dist, _r, _t = cv2.calibrateCamera(objpts, imgpts, size, None, None)
    return rms, K, dist, size, len(objpts), len(files)


def cmd_calib(a):
    folder = os.path.join(OUT, f"calib_{a.res}")
    rms, K, dist, (w, h), used, total = calibrate(folder, parse_res(a.board), a.square)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    res = {"time": stamp(), "res": f"{w}x{h}", "frames_used": used, "frames_total": total,
           "rms_reproj_px": rms, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
           "dist": dist.ravel().tolist(),
           "hfov_deg": fov_from_f(fx, w), "vfov_deg": fov_from_f(fy, h)}
    with open(out_path(f"calib_{w}x{h}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("Target: rms reprojection under 0.5 px. Above 1 px, recapture (blur, flat board?).")


def bearing_true(u, fx, cx, dist, K):
    """Calibrated bearing (deg, + = right of center) for pixel column u on the center row."""
    pt = np.array([[[u, K[1, 2]]]], np.float64)
    und = cv2.undistortPoints(pt, K, np.array(dist))  # normalized coords
    return math.degrees(math.atan(und[0, 0, 0]))


def cmd_table(a):
    path = os.path.join(OUT, f"calib_{a.res}.json")
    if not os.path.exists(path):
        raise SystemExit(f"no {path}; run calib first")
    c = json.load(open(path))
    w = int(c["res"].split("x")[0])
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]])
    print(f"Calibrated {c['res']}: HFOV {c['hfov_deg']:.1f} deg, fx {c['fx']:.0f} px, "
          f"rms {c['rms_reproj_px']:.2f} px\n")

    print("Pixels on target (bird height in px; floor is", PX_FLOOR, "px)")
    print(f"{'bird':>6} " + " ".join(f"{r:>6} m" for r in RANGES_M))
    for H in BIRD_HEIGHTS_M:
        cells = []
        for D in RANGES_M:
            px = c["fy"] * H / D
            cells.append(f"{px:6.1f}{'*' if px < PX_FLOOR else ' '} ")
            log_row("px_on_target", {"time": stamp(), "res": c["res"], "bird_m": H,
                                     "range_m": D, "px": round(px, 1)})
        print(f"{H:5.1f}m " + " ".join(cells))
    print("  * below the floor. Needed fy for a 0.6 m bird at 15 m:",
          f"{PX_FLOOR * 15 / 0.6:.0f} px\n")

    print("Bearing: W26 linear map (102 deg constant) vs. calibrated, center row")
    print(f"{'frac':>5} {'u px':>6} {'W26 deg':>8} {'calib deg':>10} {'error':>6}")
    worst = 0.0
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
        u = c["cx"] + frac * (w / 2)
        lin = frac * (W26_HFOV_DEG / 2)
        tru = bearing_true(u, c["fx"], c["cx"], c["dist"], K)
        err = lin - tru
        worst = max(worst, abs(err))
        print(f"{frac:5.2f} {u:6.0f} {lin:8.2f} {tru:10.2f} {err:6.2f}")
        log_row("bearing", {"time": stamp(), "res": c["res"], "frac": frac,
                            "w26_deg": round(lin, 2), "calib_deg": round(tru, 2),
                            "err_deg": round(err, 2)})
    print(f"Worst W26 bearing error: {worst:.1f} deg (spec F5: 2.5 deg)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("modes")
    p = sub.add_parser("fps"); p.add_argument("--frames", type=int, default=300); p.add_argument("--note", default="")
    p = sub.add_parser("snap"); p.add_argument("--label", default="wall")
    p = sub.add_parser("fov")
    p.add_argument("--dist", type=float, required=True); p.add_argument("--width", type=float, required=True)
    p.add_argument("--height", type=float); p.add_argument("--res", default="640x480")
    p = sub.add_parser("calib-capture")
    p.add_argument("--res", default="640x480"); p.add_argument("--n", type=int, default=20)
    p.add_argument("--board", default="9x6"); p.add_argument("--min-move", type=float, default=0.08)
    p.add_argument("--pause", type=float, default=1.0)
    p = sub.add_parser("calib")
    p.add_argument("--res", default="640x480"); p.add_argument("--board", default="9x6")
    p.add_argument("--square", type=float, required=True, help="square size in meters")
    p = sub.add_parser("table"); p.add_argument("--res", default="640x480")
    a = ap.parse_args()
    {"modes": cmd_modes, "fps": cmd_fps, "snap": cmd_snap, "fov": cmd_fov,
     "calib-capture": cmd_calib_capture, "calib": cmd_calib, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
