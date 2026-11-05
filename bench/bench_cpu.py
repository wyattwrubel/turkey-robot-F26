"""
bench_cpu.py: ROB 450 CV benchmark on the robot's Pi 5 (CPU paths).

Measures, separately:
  A. Camera capture: real frame rate and read time per resolution (MJPG).
  B. Inference: preprocess / inference / postprocess ms per model, using
     Ultralytics' own per-stage timers, on the T17 test images or live frames.
  C. End to end: live capture + inference loop, frames per second.

Run on the Pi with the RunTimeTest venv (it has ultralytics + openvino):
  source ~/ME450/Sandbox/RunTimeTest/.venv/bin/activate
  python ~/ME450/MainDesign/bench/bench_cpu.py                # everything
  python ~/ME450/MainDesign/bench/bench_cpu.py --skip-camera  # inference only, test images

Results print as tables and append to results.csv next to this file.
Paths are relative to this file: test images come from MainDesign/Vision/, models from the
Sandbox/ folder beside MainDesign/ (outside the repo).
"""

import argparse
import csv
import glob
import os
import statistics as st
import subprocess
import time
from datetime import datetime

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))          # MainDesign/bench
REPO = os.path.dirname(HERE)                                # MainDesign
RT = os.path.join(os.path.dirname(REPO), "Sandbox", "RunTimeTest")
MODELS = {
    "pt-640":       (f"{RT}/640best.pt", 640),
    "openvino-640": (f"{RT}/640best_openvino_model", 640),
    "pt-1280":      (f"{RT}/1280best.pt", 1280),
    "openvino-1280": (f"{RT}/1280best_openvino_model", 1280),
}
TEST_IMAGES = sorted(glob.glob(os.path.join(REPO, "Vision", "turkey*.jpg")))
CAMERA = "/dev/video_camera"
CAM_RES = [(640, 480), (1280, 720), (1280, 960), (1920, 1080)]
OUT_CSV = os.path.join(HERE, "results.csv")


def pi_state():
    """Temperature and throttle flags; throttling invalidates timings."""
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout.strip()
        except Exception:
            return "n/a"
    return f"{run(['vcgencmd', 'measure_temp'])} {run(['vcgencmd', 'get_throttled'])}"


def summarize(xs):
    xs = sorted(xs)
    p90 = xs[int(0.9 * (len(xs) - 1))]
    return st.mean(xs), st.median(xs), p90


def log_row(row):
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def open_cam(w, h, fps=30):
    cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {CAMERA}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def bench_camera(n):
    print(f"\n== A. Camera capture (MJPG, {n} frames each) ==")
    print(f"{'requested':>11} {'actual':>11} {'fps':>6} {'read ms (mean/p90)':>20}")
    for w, h in CAM_RES:
        cap = open_cam(w, h)
        for _ in range(10):  # let auto-exposure settle
            cap.read()
        times = []
        t_start = time.perf_counter()
        for _ in range(n):
            t0 = time.perf_counter()
            ok, frame = cap.read()
            times.append((time.perf_counter() - t0) * 1000)
            if not ok:
                break
        fps = len(times) / (time.perf_counter() - t_start)
        aw, ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        mean, _, p90 = summarize(times)
        print(f"{w:>5}x{h:<5} {aw:>5}x{ah:<5} {fps:6.1f} {mean:9.1f} / {p90:6.1f}")
        log_row({"time": datetime.now().isoformat(timespec="seconds"), "test": "camera",
                 "config": f"{w}x{h}", "actual": f"{aw}x{ah}", "fps": round(fps, 2),
                 "pre_ms": "", "inf_ms": round(mean, 2), "post_ms": "", "p90_ms": round(p90, 2),
                 "pi_state": pi_state()})


def bench_models(names, n, use_camera):
    from ultralytics import YOLO
    src = "live camera 640x480" if use_camera else f"{len(TEST_IMAGES)} T17 test images"
    print(f"\n== B. Inference ({n} runs each, source: {src}) ==")
    print(f"{'model':>14} {'pre':>6} {'infer (mean/med/p90)':>22} {'post':>6} {'fps*':>6}  pi")
    frames = []
    if use_camera:
        cap = open_cam(640, 480)
        for _ in range(10):
            cap.read()
        for _ in range(min(n, 30)):
            ok, f = cap.read()
            if ok:
                frames.append(f)
        cap.release()
    else:
        frames = [cv2.imread(p) for p in TEST_IMAGES]
    if not frames:
        raise RuntimeError("no input frames")

    for name in names:
        path, imgsz = MODELS[name]
        if not os.path.exists(path):
            print(f"{name:>14} missing: {path}")
            continue
        model = YOLO(path, task="detect")
        for i in range(5):  # warm-up
            model.predict(frames[i % len(frames)], imgsz=imgsz, device="cpu", verbose=False)
        pre, inf, post = [], [], []
        for i in range(n):
            r = model.predict(frames[i % len(frames)], imgsz=imgsz, device="cpu", verbose=False)[0]
            pre.append(r.speed["preprocess"])
            inf.append(r.speed["inference"])
            post.append(r.speed["postprocess"])
        m, med, p90 = summarize(inf)
        total = st.mean(pre) + m + st.mean(post)
        state = pi_state()
        print(f"{name:>14} {st.mean(pre):6.1f} {m:8.1f}/{med:6.1f}/{p90:6.1f} {st.mean(post):6.1f} "
              f"{1000 / total:6.2f}  {state}")
        log_row({"time": datetime.now().isoformat(timespec="seconds"), "test": "inference",
                 "config": name, "actual": f"imgsz={imgsz}", "fps": round(1000 / total, 2),
                 "pre_ms": round(st.mean(pre), 2), "inf_ms": round(m, 2),
                 "post_ms": round(st.mean(post), 2), "p90_ms": round(p90, 2), "pi_state": state})
    print("* fps from pre+infer+post only; capture is excluded here (see C).")


def bench_end_to_end(name, seconds, w, h):
    from ultralytics import YOLO
    path, imgsz = MODELS[name]
    print(f"\n== C. End to end: {name}, camera {w}x{h}, {seconds}s ==")
    model = YOLO(path, task="detect")
    cap = open_cam(w, h)
    for _ in range(10):
        cap.read()
    ok, f = cap.read()
    model.predict(f, imgsz=imgsz, device="cpu", verbose=False)  # warm-up
    n, dets, t_end = 0, 0, time.perf_counter() + seconds
    t_start = time.perf_counter()
    while time.perf_counter() < t_end:
        ok, f = cap.read()
        if not ok:
            continue
        r = model.predict(f, imgsz=imgsz, device="cpu", verbose=False)[0]
        dets += len(r.boxes)
        n += 1
    fps = n / (time.perf_counter() - t_start)
    cap.release()
    state = pi_state()
    print(f"{n} frames, {fps:.2f} fps, {dets} detections total  {state}")
    log_row({"time": datetime.now().isoformat(timespec="seconds"), "test": "end_to_end",
             "config": name, "actual": f"{w}x{h}", "fps": round(fps, 2), "pre_ms": "",
             "inf_ms": "", "post_ms": "", "p90_ms": "", "pi_state": state})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--skip-camera", action="store_true", help="no camera tests; test images only")
    ap.add_argument("--e2e-model", default="openvino-640", choices=list(MODELS))
    ap.add_argument("--e2e-seconds", type=int, default=20)
    a = ap.parse_args()

    print(f"Start: {pi_state()}  threads={cv2.getNumThreads()}")
    if not a.skip_camera:
        bench_camera(60)
    bench_models(a.models, a.runs, use_camera=False)
    if not a.skip_camera:
        bench_end_to_end(a.e2e_model, a.e2e_seconds, 640, 480)
    print(f"\nEnd: {pi_state()}\nResults appended to {OUT_CSV}")
