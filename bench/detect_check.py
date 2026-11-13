"""
detect_check.py: run Vision/Detector.py on a folder of images and write the JSON that
`bench/hailo_run.py compare` reads. Used to check a HEF against its .pt model.

  /usr/bin/python3 bench/detect_check.py --backend hailo --hef H.hef --images Vision --out /tmp/hef.json
  <venv>/python    bench/detect_check.py --backend pt --pt M.pt --images Vision --out /tmp/pt.json
  python3 bench/hailo_run.py compare /tmp/pt.json /tmp/hef.json
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cv2
from Vision import config
from Vision.Detector import Detector

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["hailo", "pt"], required=True)
ap.add_argument("--hef"); ap.add_argument("--pt")
ap.add_argument("--images", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--conf", type=float, default=0.25, help="turkey/person threshold for this check")
a = ap.parse_args()
config.CONF_THRESH.update(turkey=a.conf, person=a.conf)
det = Detector(a.backend, hef_path=a.hef, pt_path=a.pt)
res = {}
for f in sorted(glob.glob(os.path.join(a.images, "*.jp*g")) + glob.glob(os.path.join(a.images, "*.png"))):
    img = cv2.imread(f)
    dets, info = det(img)
    res[os.path.basename(f)] = {"boxes": [[round(v, 1) for v in d.box] for d in dets],
                                "scores": [round(d.conf, 4) for d in dets],
                                "classes": [d.cls for d in dets]}
    print(f"{os.path.basename(f):24} {len(dets)} det {info['infer_ms']:6.1f} ms  "
          + " ".join(f"{d.name}:{d.conf:.2f} h={d.box[3]-d.box[1]:.0f}px" for d in dets[:4]))
det.close()
json.dump(res, open(a.out, "w"), indent=1)
print("wrote", a.out)
