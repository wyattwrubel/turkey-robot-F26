"""Mac-side checks for cv/onboard-detector. Run from the repo root with the CV venv:
   VISION_BACKEND=pt VISION_MODEL_CLASSES=coco VISION_PT_PATH=path/to/yolov8n.pt python Vision/tests/check_vision.py path/to/coco128
   No hardware needed. Checks bearing sign and pinhole math, k-of-n confirm, the detector, and the DynamicObjects publish path (contract C2).
"""
import asyncio, math, os, sys, time
from types import SimpleNamespace as NS

import cv2
import numpy as np

sys.path.insert(0, os.getcwd())
from Vision import config
from Vision.Geometry import CameraModel
from Vision.Confirm import Confirmer

coco = sys.argv[1]
fails = 0
def check(name, ok, detail=""):
    global fails
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    fails += 0 if ok else 1

# 1. geometry: sign and pinhole numbers
cam = CameraModel(640, 480, hfov_deg=60.0, calib_path="/nonexistent")
check("bearing 0 at center", abs(cam.bearing(320)) < 1e-9)
check("left of center is +", cam.bearing(100) > 0, f"{math.degrees(cam.bearing(100)):.1f} deg")
check("right edge = -HFOV/2", abs(math.degrees(cam.bearing(640)) + 30) < 1e-6)
x, y, b, d = cam.locate((300, 200, 340, 260))       # 60 px tall box
f = 320 / math.tan(math.radians(30))
check("camera range = f*H/h", abs(d - f * 0.75 / 60) < 1e-6, f"{d:.2f} m")
check("x_cam = d*tan(b)", abs(x - d * math.tan(b)) < 1e-9 and y == d)

# 2. confirmer: k-of-n
c = Confirmer(k=3, n=5, gate=0.15, drop_misses=5)
det = NS(box=(100, 100, 140, 160), conf=0.8)
seq = [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]
out = []
for i, s in enumerate(seq):
    c.update([det] if s else [], 640, now=i * 0.1)
    out.append(len(c.confirmed_tracks()))
check("single frame does not confirm", out[0] == 0 and out[1] == 0)
check("confirms on 3rd hit (k=3)", out[3] == 1, str(out))
check("missed frame not published", out[2] == 0 and out[5] == 0)
check("track dropped after misses", len(c.tracks) == 0)
check("confirm latency recorded", abs(c.confirm_latencies[0] - 0.3) < 1e-9)
c2 = Confirmer(k=3, n=5, gate=0.15)
for i in range(5):  # one-frame flickers at random places never confirm
    c2.update([NS(box=(i * 150, 0, i * 150 + 20, 20), conf=0.5)], 640, now=i)
check("scattered flickers never confirm", not c2.confirmed_tracks())

# 3. detector (pt backend) on COCO bird images
from Vision.Detector import Detector
dt = Detector()
img = cv2.imread(os.path.join(coco, "images/train2017/000000000143.jpg"))
dets, info = dt(img)
names = [(d.name, round(d.conf, 2)) for d in dets]
check("detector finds the bird (as 'turkey' via coco map)", any(d.name == "turkey" for d in dets), str(names))
check("info has infer_ms/backend", info["backend"] == "pt" and info["infer_ms"] > 0, f"{info['infer_ms']:.0f} ms")

# 4. VisionObject end to end, no camera: feed frames, capture published payloads
from Vision.Vision import VisionObject

class Pub:
    def __init__(self): self.got = []
    async def publish(self, d): self.got.append(d)

async def run():
    vid, det_pub = Pub(), Pub()
    vo = VisionObject(vid, det_pub, Simulated=True, Backend="pt")
    vo._start_detector()
    loop = asyncio.get_running_loop()
    blank = np.full_like(img, 127)
    frames = [img] * 4 + [blank] * 3
    for fr in frames:
        await vo._detect_and_publish(fr, loop)
        await asyncio.sleep(0.05)
    return vo, det_pub.got

vo, got = asyncio.run(run())
time.sleep(0.5)
nt = [len(g["Turkeys"]) for g in got]
check("published every frame", len(got) == 7, str(len(got)))
check("confirmed from 3rd frame, empty on blanks", nt[:2] == [0, 0] and nt[2] >= 1 and nt[-1] == 0, str(nt))
g = got[3]
check("C2 keys", set(g) >= {"Id", "Turkeys", "Humans", "Ghosts", "Meta"} and g["Id"] == "DynamicObjects")
check("rows are [x, y, bearing]", all(len(r) == 3 for r in g["Turkeys"]))
m = g["Meta"]
check("Meta fields", set(m) >= {"t_capture", "fps", "confidence", "range_source", "backend"}, str(m))
check("UI overlay holds the last frame's boxes", isinstance(vo._latest_detections, list))
print(f"\n{fails} failure(s)")
sys.exit(1 if fails else 0)
