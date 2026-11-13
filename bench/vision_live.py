"""
vision_live.py: camera -> Detector -> k-of-n confirm -> bearing, on the robot, printing
fps and confirmed turkeys once a second. Frames stay in memory; nothing is saved (S4).

  /usr/bin/python3 bench/vision_live.py [--seconds 30] [--res 640x480]

Use it for the end-to-end fps number (Test B) and to check detection on the decoy
before the full Main.py path is wired.
"""
import argparse, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=30)
ap.add_argument("--res", default=None, help="e.g. 1280x720 (sets VISION_CAMERA_WIDTH/HEIGHT)")
a = ap.parse_args()
if a.res:
    w, h = a.res.split("x"); os.environ["VISION_CAMERA_WIDTH"], os.environ["VISION_CAMERA_HEIGHT"] = w, h

from Vision.UsbCamera import USBCamera
from Vision.Detector import Detector
from Vision.Confirm import Confirmer
from Vision.Geometry import CameraModel

cam = USBCamera(); cam.open(); cam.print_actual_settings()
det, conf, model = Detector(), Confirmer(), CameraModel()
print(f"backend={det.backend_name} bearing from {model.source} confirm {conf.k}-of-{conf.n}")
n, t0, last, inf = 0, time.perf_counter(), 0.0, []
try:
    while time.perf_counter() - t0 < a.seconds:
        ok, frame = cam.read()
        if not ok:
            continue
        dets, info = det(frame)
        inf.append(info["infer_ms"])
        conf.update([d for d in dets if d.name == "turkey"], frame.shape[1])
        n += 1
        if time.perf_counter() - last >= 1.0:
            last = time.perf_counter()
            fps = n / (last - t0)
            c = conf.confirmed_tracks()
            s = ", ".join(f"turkey {t.conf:.2f} bearing {math.degrees(model.locate(t.box)[2]):+.1f} deg "
                          f"h={t.box[3]-t.box[1]:.0f}px" for t in c) or "-"
            cand = sum(1 for d in dets if d.name == "turkey")
            print(f"{fps:5.1f} fps  infer {info['infer_ms']:5.1f} ms  candidates {cand}  confirmed: {s}")
finally:
    cam.release(); det.close()
el = time.perf_counter() - t0
print(f"END-TO-END {n / el:.1f} fps over {n} frames; mean infer {sum(inf)/max(len(inf),1):.1f} ms; "
      f"confirm latencies {[round(x, 2) for x in conf.confirm_latencies]}")
