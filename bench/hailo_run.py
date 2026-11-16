"""
hailo_run.py: run a YOLO .hef on the robot's Hailo-8L and decode its output correctly.

Why: W26's HEFs compiled and loaded fine (299 loads, Apr 4-10), but their Python
decoded the output with a YOLOv5 layout (objectness row), so real turkeys scored
near zero. This script reads the HEF's actual output shape and decodes it as one
of the layouts below, and checks itself against the .pt model on the same images.

Python environments on the Pi (the two backends live in different ones):
  HEF backend -> system Python (Raspberry Pi OS's hailo-all gives hailo_platform):
      /usr/bin/python3 bench/hailo_run.py ...
  PT backend  -> the RunTimeTest venv (has ultralytics, no hailo_platform):
      ~/ME450/Sandbox/RunTimeTest/.venv/bin/python bench/hailo_run.py ...
  Only numpy and cv2 are needed at import time; hailo_platform / ultralytics load lazily.

Commands (run from ~/ME450/MainDesign):
  python bench/hailo_run.py info  --hef H.hef            # input/output names, shapes, formats
  python bench/hailo_run.py bench --hef H.hef [-n 300]   # Hailo-only fps (random input, no camera)
  python bench/hailo_run.py detect --backend hef --hef H.hef --images DIR --out hef.json
  python bench/hailo_run.py detect --backend pt  --pt  M.pt  --images DIR --out pt.json
  python bench/hailo_run.py compare pt.json hef.json     # box-by-box agreement
  python bench/hailo_run.py live  --hef H.hef [--seconds 30]
                                                          # camera -> Hailo -> detections; prints fps.
                                                          # Frames stay in memory; nothing is saved (S4).

Output layouts the decoder handles (picked from the output shape, or forced with --layout):
  raw  : (4+nc, N) or (N, 4+nc), N anchors (8400 at 640): cx, cy, w, h, then one score per class.
         No objectness. Needs NMS (done here). YOLOv8 and YOLO26 one-to-many heads.
  e2e  : (K, 6) or (6, K), K <= 1000: x1, y1, x2, y2, conf, class. YOLO26 end-to-end (NMS-free).
  nms  : Hailo's on-chip NMS output (Model Zoo YOLOv8): per class, rows of
         ymin, xmin, ymax, xmax, score, normalized 0..1.
Several separate outputs (split heads with DFL not decoded on chip) are not decoded here:
`info` lists them, and the HEF should be recompiled with a single output or Hailo NMS.

Rules: open the Hailo once per process (never from two programs at once: W26 hit
HAILO_OUT_OF_PHYSICAL_DEVICES), and run bench/power_check.py first (W26 lost
/dev/hailo0 on 04-07, likely power).
"""

import argparse
import contextlib
import glob
import json
import os
import time

import cv2
import numpy as np

CAMERA = "/dev/video_camera"
W26_CLASSES = {0: "turkey", 1: "person", 2: "vehicle"}


# ---------------------------------------------------------------- preprocessing

def letterbox(img, size, pad=114):
    """Resize keeping aspect ratio and pad, as Ultralytics does. Returns image, scale, (padx, pady)."""
    h, w = img.shape[:2]
    th, tw = size
    r = min(th / h, tw / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padx, pady = (tw - nw) / 2, (th - nh) / 2
    top, left = int(round(pady - 0.1)), int(round(padx - 0.1))
    out = cv2.copyMakeBorder(resized, top, th - nh - top, left, tw - nw - left,
                             cv2.BORDER_CONSTANT, value=(pad, pad, pad))
    return out, r, (left, top)


def unletterbox(boxes, r, pad, orig_wh):
    """boxes (k,4) x1y1x2y2 in network pixels -> original image pixels, clipped."""
    if len(boxes) == 0:
        return boxes
    b = boxes.copy().astype(np.float32)
    b[:, [0, 2]] = (b[:, [0, 2]] - pad[0]) / r
    b[:, [1, 3]] = (b[:, [1, 3]] - pad[1]) / r
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, orig_wh[0])
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, orig_wh[1])
    return b


# ---------------------------------------------------------------- decoding

def nms(boxes, scores, classes, iou=0.45):
    """Class-aware NMS in numpy. boxes (k,4) x1y1x2y2."""
    keep = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        idx = idx[np.argsort(-scores[idx])]
        while len(idx):
            i = idx[0]
            keep.append(i)
            if len(idx) == 1:
                break
            rest = idx[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            idx = rest[inter / (a + b - inter + 1e-9) < iou]
    keep = np.array(sorted(keep, key=lambda k: -scores[k]), dtype=int)
    return keep


def guess_layout(arr):
    if isinstance(arr, (list, tuple)):
        return "nms"
    a = np.squeeze(np.asarray(arr))
    if a.ndim != 2:
        raise ValueError(f"can't decode output of shape {np.asarray(arr).shape}; run `info`")
    if 6 in a.shape and min(a.shape) == 6 and max(a.shape) <= 1000:
        return "e2e"
    return "raw"


def decode(out, layout, net_wh, conf):
    """Returns boxes (k,4) x1y1x2y2 in network pixels, scores (k,), classes (k,)."""
    nw, nh = net_wh
    if layout == "nms":
        # out: [batch][class] -> array (k,5) ymin,xmin,ymax,xmax,score normalized
        per_class = out[0] if len(out) and isinstance(out[0], (list, tuple)) else out
        B, S, C = [], [], []
        for c, rows in enumerate(per_class):
            rows = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
            rows = rows[rows[:, 4] >= conf]
            for y1, x1, y2, x2, s in rows:
                B.append([x1 * nw, y1 * nh, x2 * nw, y2 * nh]); S.append(s); C.append(c)
        return np.array(B, np.float32).reshape(-1, 4), np.array(S, np.float32), np.array(C, int)

    a = np.squeeze(np.asarray(out, dtype=np.float32))
    if layout == "e2e":
        if a.shape[0] == 6 and a.shape[1] != 6:
            a = a.T                                  # -> (K, 6)
        a = a[a[:, 4] >= conf]
        boxes = a[:, :4]
        if boxes.size and boxes.max() <= 1.5:        # normalized coordinates
            boxes = boxes * np.array([nw, nh, nw, nh], np.float32)
        return boxes, a[:, 4], a[:, 5].astype(int)

    # raw: (4+nc, N) -> (N, 4+nc); the anchor axis is the long one
    if a.shape[0] < a.shape[1]:
        a = a.T
    xywh, cls_scores = a[:, :4], a[:, 4:]
    classes = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(a)), classes]
    m = scores >= conf
    xywh, scores, classes = xywh[m], scores[m], classes[m]
    if xywh.size and xywh.max() <= 1.5:
        xywh = xywh * np.array([nw, nh, nw, nh], np.float32)
    boxes = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
    k = nms(boxes, scores, classes)
    return boxes[k], scores[k], classes[k]


# ---------------------------------------------------------------- backends

class HailoDetector:
    """Opens the Hailo once and keeps the inference pipeline open for the whole run."""

    def __init__(self, hef_path, layout="auto", conf=0.25):
        from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams,
                                    InputVStreamParams, OutputVStreamParams, FormatType,
                                    InferVStreams)
        self.conf, self.layout = conf, layout
        self.hef = HEF(hef_path)
        self._stack = contextlib.ExitStack()
        self.device = self._stack.enter_context(VDevice())
        cfg = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.ng = self.device.configure(self.hef, cfg)[0]
        inp = self.hef.get_input_vstream_infos()[0]
        self.input_name = inp.name
        self.in_h, self.in_w = inp.shape[0], inp.shape[1]
        outs = self.hef.get_output_vstream_infos()
        if len(outs) != 1:
            raise SystemExit(f"{len(outs)} outputs ({[o.name for o in outs]}); this script decodes "
                             "single-output HEFs. Recompile with one output or Hailo NMS.")
        self.output_name = outs[0].name
        ip = InputVStreamParams.make(self.ng, format_type=FormatType.UINT8)
        op = OutputVStreamParams.make(self.ng, format_type=FormatType.FLOAT32)
        self._stack.enter_context(self.ng.activate(self.ng.create_params()))
        self.pipe = self._stack.enter_context(InferVStreams(self.ng, ip, op))

    def raw(self, rgb_uint8):
        return self.pipe.infer({self.input_name: rgb_uint8[np.newaxis]})[self.output_name]

    def __call__(self, bgr):
        lb, r, pad = letterbox(bgr, (self.in_h, self.in_w))
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        t = time.perf_counter()
        out = self.raw(rgb)
        ms = (time.perf_counter() - t) * 1000
        layout = guess_layout(out) if self.layout == "auto" else self.layout
        boxes, scores, classes = decode(out, layout, (self.in_w, self.in_h), self.conf)
        boxes = unletterbox(boxes, r, pad, (bgr.shape[1], bgr.shape[0]))
        return boxes, scores, classes, {"layout": layout, "infer_ms": ms}

    def close(self):
        self._stack.close()


class PtDetector:
    def __init__(self, pt_path, imgsz=640, conf=0.25):
        from ultralytics import YOLO
        self.model, self.imgsz, self.conf = YOLO(pt_path), imgsz, conf

    def __call__(self, bgr):
        t = time.perf_counter()
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        ms = (time.perf_counter() - t) * 1000
        b = r.boxes
        return (b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int),
                {"layout": "pt", "infer_ms": ms})

    def close(self):
        pass


def make(a):
    if a.backend == "hef":
        return HailoDetector(a.hef, a.layout, a.conf)
    return PtDetector(a.pt, a.imgsz, a.conf)


# ---------------------------------------------------------------- commands

def cmd_info(a):
    from hailo_platform import HEF
    hef = HEF(a.hef)
    print("network groups:", hef.get_network_group_names())
    for kind, infos in (("input", hef.get_input_vstream_infos()), ("output", hef.get_output_vstream_infos())):
        for i in infos:
            print(f"{kind:6} {i.name:45} shape={tuple(i.shape)} format={i.format}")


def cmd_bench(a):
    det = HailoDetector(a.hef, a.layout, a.conf)
    x = np.random.randint(0, 255, (det.in_h, det.in_w, 3), np.uint8)
    for _ in range(10):
        det.raw(x)
    t = time.perf_counter()
    for _ in range(a.n):
        det.raw(x)
    dt = time.perf_counter() - t
    det.close()
    print(f"{a.hef}: {a.n / dt:.1f} fps ({dt / a.n * 1000:.1f} ms/frame), batch 1, Hailo only")


def cmd_detect(a):
    files = sorted(sum((glob.glob(os.path.join(a.images, e)) for e in ("*.jpg", "*.jpeg", "*.png")), []))
    if not files:
        raise SystemExit(f"no images in {a.images}")
    det = make(a)
    res = {}
    for f in files:
        img = cv2.imread(f)
        boxes, scores, classes, meta = det(img)
        res[os.path.basename(f)] = {"boxes": boxes.round(1).tolist(), "scores": scores.round(4).tolist(),
                                    "classes": classes.tolist(), **{k: (round(v, 2) if isinstance(v, float) else v)
                                                                    for k, v in meta.items()}}
        print(f"{os.path.basename(f):30} {len(scores):3} det  {meta['layout']:4} {meta['infer_ms']:7.1f} ms  "
              + " ".join(f"{c}:{s:.2f}" for c, s in zip(classes[:5], scores[:5])))
    det.close()
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", a.out)


def iou_1(b1, b2):
    ix = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    iy = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    inter = ix * iy
    u = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / u if u else 0.0


def cmd_compare(a):
    A, B = json.load(open(a.ref)), json.load(open(a.test))
    matched = total_ref = total_test = 0
    ious, dconf = [], []
    for name in sorted(set(A) & set(B)):
        ra, rb = A[name], B[name]
        used = set()
        total_ref += len(ra["scores"]); total_test += len(rb["scores"])
        for ba, sa, ca in zip(ra["boxes"], ra["scores"], ra["classes"]):
            best, bj = 0.0, None
            for j, (bb, cb) in enumerate(zip(rb["boxes"], rb["classes"])):
                if j in used or cb != ca:
                    continue
                v = iou_1(ba, bb)
                if v > best:
                    best, bj = v, j
            if bj is not None and best >= 0.5:
                used.add(bj); matched += 1; ious.append(best); dconf.append(rb["scores"][bj] - sa)
    print(f"images: {len(set(A) & set(B))}  ref boxes: {total_ref}  test boxes: {total_test}")
    print(f"matched (same class, IoU>=0.5): {matched} = {matched / max(total_ref, 1):.0%} of ref")
    if ious:
        print(f"mean IoU {np.mean(ious):.3f}, mean conf diff (test-ref) {np.mean(dconf):+.3f}")
    print("Pass bar (proposal): >=90% matched, mean IoU >=0.85, |conf diff| <=0.05. "
          "If nothing matches under any --layout, suspect input normalization in the HEF.")


def cmd_live(a):
    det = HailoDetector(a.hef, a.layout, a.conf)
    cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    w, h = (int(v) for v in a.res.split("x"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    n, t0, last = 0, time.perf_counter(), 0.0
    names = W26_CLASSES if a.w26_classes else {}
    try:
        while time.perf_counter() - t0 < a.seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            boxes, scores, classes, meta = det(frame)   # frame stays in memory; never written (S4)
            n += 1
            if time.perf_counter() - last > 1.0:
                last = time.perf_counter()
                fps = n / (last - t0)
                dets = ", ".join(f"{names.get(c, c)} {s:.2f} h={b[3] - b[1]:.0f}px"
                                 for b, s, c in zip(boxes[:3], scores[:3], classes[:3]))
                print(f"{fps:5.1f} fps  infer {meta['infer_ms']:5.1f} ms  {dets or '-'}")
    finally:
        cap.release()
        det.close()
    print(f"end-to-end: {n / (time.perf_counter() - t0):.1f} fps over {n} frames")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, hef_required=True):
        p.add_argument("--hef", required=hef_required)
        p.add_argument("--layout", default="auto", choices=["auto", "raw", "e2e", "nms"])
        p.add_argument("--conf", type=float, default=0.25)

    common(sub.add_parser("info"))
    p = sub.add_parser("bench"); common(p); p.add_argument("-n", type=int, default=300)
    p = sub.add_parser("detect"); common(p, hef_required=False)
    p.add_argument("--backend", choices=["hef", "pt"], required=True)
    p.add_argument("--pt"); p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--images", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("compare"); p.add_argument("ref"); p.add_argument("test")
    p = sub.add_parser("live"); common(p)
    p.add_argument("--seconds", type=int, default=30); p.add_argument("--res", default="640x480")
    p.add_argument("--w26-classes", action="store_true", help="label classes turkey/person/vehicle")
    a = ap.parse_args()
    {"info": cmd_info, "bench": cmd_bench, "detect": cmd_detect, "compare": cmd_compare,
     "live": cmd_live}[a.cmd](a)


if __name__ == "__main__":
    main()
