"""
Vision/config.py: one place for every perception setting (DR2 plan, contract C9).

Other modules read from here instead of keeping their own copies. The zig-zag
planner reads HFOV_DEG from this file.

Any value can be overridden with an environment variable of the same name,
prefixed VISION_ (for example VISION_BACKEND=pt), so laptop dev and the robot
use the same code without edits.
"""

import os
from pathlib import Path


def _env(name, default, cast=str):
    v = os.environ.get(f"VISION_{name}")
    return default if v is None else cast(v)


# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_DEVICE = _env("CAMERA_DEVICE", "/dev/video_camera")
CAMERA_WIDTH  = _env("CAMERA_WIDTH", 640, int)
CAMERA_HEIGHT = _env("CAMERA_HEIGHT", 480, int)
CAMERA_FPS    = _env("CAMERA_FPS", 30, int)      # the camera does MJPG 30 fps at every size (measured 11-13)
CAMERA_FOURCC = "MJPG"
JPEG_QUALITY  = 80                                # UI stream only; frames are never written to disk (C10)

# Horizontal field of view. PLACEHOLDER until the P1 measurement on the robot.
# W26 assumed 102 deg and never measured it. Replace with the measured value.
HFOV_DEG = _env("HFOV_DEG", 60.0, float)

# Checkerboard calibration (fx, fy, cx, cy, dist) written by bench/camera_char.py.
# If the file doesn't exist, geometry falls back to a pinhole model from HFOV_DEG.
CALIB_DIR  = Path(__file__).resolve().parent / "calib"
CALIB_PATH = CALIB_DIR / f"camera_{CAMERA_WIDTH}x{CAMERA_HEIGHT}.json"

# ── Detector ──────────────────────────────────────────────────────────────────
# "hailo": in-process on the Hailo-8L (robot, system Python with hailo_platform)
# "pt":    in-process with Ultralytics (laptop dev, recorded video; slow on the Pi CPU)
# "laptop": W26's path, boxes arrive from a laptop on ports 8765/8766 (fallback)
BACKEND = _env("BACKEND", "hailo")

MODEL_DIR = Path(_env("MODEL_DIR", str(Path.home() / "ME450" / "Sandbox" / "hef")))
# W26's turkey_640_v4.hef = turkey_640.pt compiled with split heads (box + class per stride).
# Checked on the robot 11-13 against turkey_640.pt: 100% of boxes matched, IoU 0.956, conf +0.003
# (3 images). The single-output HEFs (turkey_640.hef, best640_hailo8l.hef) lose every class score
# to uint8 quantization and detect nothing; don't use them.
# turkey_v1_split.hef = our fine-tune (ENA24 + Open Images) of turkey_640.pt, compiled 11-16 with
# Vision/training/hailo_kit.py + hailo_compile_split.ipynb (Colab CPU, DFC 3.34; loads on HailoRT 4.23).
# On the robot's Hailo, 67 held-out Open Images photos (94 turkeys): AP50 0.893 vs v4 0.562,
# recall@0.25 0.830 vs 0.436, precision 0.876 vs 0.932; live on the camera both ~28 fps, 0 false
# turkeys in 1,277 lab frames. Model files aren't in git: if v1 isn't on this machine, use v4.
HEF_V1 = MODEL_DIR / "turkey_v1_split.hef"
HEF_V4 = MODEL_DIR / "turkey_640_v4.hef"
HEF_PATH  = Path(_env("HEF_PATH", str(HEF_V1 if HEF_V1.exists() else HEF_V4)))
PT_PATH   = Path(_env("PT_PATH", str(MODEL_DIR / "turkey_1280.pt")))          # W26 baseline weights
PT_IMGSZ  = _env("PT_IMGSZ", 640, int)                                         # baseline runs 1280 weights at 640

# Which model class ids mean "turkey" and "person". W26's models: 0 turkey,
# 1 person, 2 vehicle. COCO models (dev only): 14 bird, 0 person.
MODEL_CLASSES = _env("MODEL_CLASSES", "w26")
CLASS_SETS = {
    "w26":  {"names": {0: "turkey", 1: "person", 2: "vehicle"}, "turkey": {0}, "person": {1}},
    "coco": {"names": {14: "turkey", 0: "person"}, "turkey": {14}, "person": {0}},
}

# Per-class confidence thresholds. W26 deployed turkey 0.30, person 0.80, vehicle 0.95.
CONF_THRESH = {
    "turkey": _env("CONF_TURKEY", 0.30, float),
    "person": _env("CONF_PERSON", 0.80, float),
    "vehicle": 0.95,
}
DECODE_CONF = min(CONF_THRESH.values())     # floor used inside the decoder before per-class filtering

# ── Confirmation (replaces W26's pass-through filter) ────────────────────────
# A track is confirmed when it was seen in at least CONFIRM_K of its last
# CONFIRM_N processed frames. Size these to the measured onboard fps so a
# confirm lands well inside 1 s (F4) and the planner's 1.5 s lost-bird window.
CONFIRM_K = _env("CONFIRM_K", 3, int)
CONFIRM_N = _env("CONFIRM_N", 5, int)
TRACK_GATE = 0.15        # max centroid jump between frames, as a fraction of image width
TRACK_DROP_MISSES = 5    # forget a track after this many frames without a match

# ── Camera-only range (fallback until the LiDAR gives range along the bearing) ─
TURKEY_HEIGHT_M = 0.75   # assumed standing height of a wild turkey (0.6 to 0.9 m)
