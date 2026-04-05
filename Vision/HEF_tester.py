"""
compare_models.py — PT vs HEF detection comparison
Usage: python compare_models.py <image_path>
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams,
    HailoSchedulingAlgorithm
)

# ── Config ──────────────────────────────────────────────────────
PT_PATH  = "/home/turkeysquad/ME450/MainDesign/Vision/weights/turkey_640.pt"
HEF_PATH = "/home/turkeysquad/ME450/MainDesign/Vision/weights/best640_hailo8l.hef"
CONF     = 0.25
IMGSZ    = 640
CLASSES  = {0: "turkey", 1: "person", 2: "vehicle"}
OUTPUT_KEY = 'best640/format_conversion13'

# ────────────────────────────────────────────────────────────────


def run_pt(image):
    results = YOLO(PT_PATH).predict(source=image, conf=CONF, imgsz=IMGSZ, verbose=False)
    dets = []
    for r in results:
        for i in range(len(r.boxes)):
            dets.append((
                CLASSES.get(int(r.boxes.cls[i].item()), "unknown"),
                float(r.boxes.conf[i].item()),
                tuple(map(int, r.boxes.xyxy[i].tolist())),
            ))
    return dets


def run_hef(image):
    orig_h, orig_w = image.shape[:2]
    input_data = cv2.resize(image, (IMGSZ, IMGSZ))[np.newaxis].astype(np.uint8)

    hef    = HEF(HEF_PATH)
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE
    target = VDevice(params)

    cfg_params     = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    ng             = target.configure(hef, cfg_params)[0]
    input_name     = hef.get_input_vstream_infos()[0].name
    input_params   = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    output_params  = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

    with ng.activate(ng.create_params()):
        with InferVStreams(ng, input_params, output_params) as pipeline:
            raw = pipeline.infer({input_name: input_data})

    return _decode(raw, orig_w, orig_h)


def _decode(raw, orig_w, orig_h):
    data = raw[OUTPUT_KEY].squeeze()
    print("full row shapes:", [data[i].shape for i in range(6)])
    for i in range(6):
        nonzero = data[i][data[i] != 0]
        print(f"row[{i}] nonzero sample: {nonzero[:5]}")
    x1s, y1s, x2s, y2s = data[0], data[1], data[2], data[3]

    sx, sy = orig_w / IMGSZ, orig_h / IMGSZ
    dets = []
    for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s):
        if x2 <= x1 or y2 <= y1:
            continue
        dets.append((
            "turkey", 1.0,
            (max(0, int(x1 * sx)), max(0, int(y1 * sy)),
             min(orig_w, int(x2 * sx)), min(orig_h, int(y2 * sy))),
        ))
    return _nms(dets)


def _nms(dets, iou_thresh=0.45):
    dets = sorted(dets, key=lambda d: d[1], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if _iou(best[2], d[2]) < iou_thresh]
    return keep


def _iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / union if union else 0.0


def draw(image, dets, label):
    out = image.copy()
    for cls, conf, (x1, y1, x2, y2) in dets:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{cls} {conf:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return out


# ── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/turkeysquad/ME450/MainDesign/Vision/turkey1.jpg"

    image = cv2.imread(img_path)
    if image is None:
        print(f"ERROR: could not load {img_path}")
        sys.exit(1)

    print("Running PT model...")
    pt_dets  = run_pt(image)
    print(f"  {len(pt_dets)} detections")

    print("Running HEF model...")
    hef_dets = run_hef(image)
    print(f"  {len(hef_dets)} detections")

    print(f"\n{'─'*40}")
    print(f"PT  ({len(pt_dets)}):")
    for cls, conf, bbox in pt_dets:
        print(f"  {cls:<10} conf={conf:.3f}  bbox={bbox}")
    print(f"HEF ({len(hef_dets)}):")
    for cls, conf, bbox in hef_dets:
        print(f"  {cls:<10} conf={conf:.3f}  bbox={bbox}")

    out = np.hstack([draw(image, pt_dets, "PT (CPU)"), draw(image, hef_dets, "HEF (Hailo)")])
    cv2.imwrite("comparison.jpg", out)
    print("\nSaved comparison.jpg")