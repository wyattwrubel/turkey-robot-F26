import cv2
import numpy as np
from ultralytics import YOLO
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams,
    HailoSchedulingAlgorithm
)

# ── Config ──────────────────────────────────────────────────────
PT_PATH     = "/home/turkeysquad/ME450/MainDesign/Vision/weights/turkey_640.pt"
HEF_PATH    = "/home/turkeysquad/ME450/MainDesign/Vision/weights/turkey_640_v4.hef"
IMG_PATH    = "/home/turkeysquad/ME450/MainDesign/Vision/turkey1.jpg"
CONF        = 0.25
IMGSZ       = 640
CLASS_MAP   = {0: "turkey", 1: "person", 2: "vehicle"}
STRIDES     = [8, 16, 32]

# Exact quantization params from HEF (scale, zero_point)
QUANT = {
    "turkey_640/conv61": (0.031482595950365067, 16.0),
    "turkey_640/conv64": (0.13635796308517456,  255.0),
    "turkey_640/conv77": (0.05225154012441635,  2.0),
    "turkey_640/conv80": (0.3352046012878418,   247.0),
    "turkey_640/conv91": (0.05027148500084877,  0.0),
    "turkey_640/conv94": (0.2854524254798889,   246.0),
}
# ────────────────────────────────────────────────────────────────


def run_pt(image):
    model = YOLO(PT_PATH)
    results = model.predict(source=image, conf=CONF, imgsz=IMGSZ, verbose=False)
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i].item())
            conf   = float(r.boxes.conf[i].item())
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].tolist())
            detections.append((CLASS_MAP.get(cls_id, str(cls_id)), conf, (x1, y1, x2, y2)))
    return detections


def _dequantize(tensor, name):
    scale, zp = QUANT[name]
    return (tensor.astype(np.float32) - zp) * scale


def _infer_hailo(image):
    orig_h, orig_w = image.shape[:2]
    resized    = cv2.resize(image, (IMGSZ, IMGSZ))
    input_data = resized[np.newaxis].astype(np.uint8)

    hef    = HEF(HEF_PATH)
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE
    target = VDevice(params)

    configure_params = ConfigureParams.create_from_hef(
        hef=hef, interface=HailoStreamInterface.PCIe
    )
    network_groups = target.configure(hef, configure_params)
    ng             = network_groups[0]
    ng_params      = ng.create_params()

    input_params  = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
    output_params = OutputVStreamParams.make(ng, format_type=FormatType.UINT8)
    input_name    = hef.get_input_vstream_infos()[0].name

    with ng.activate(ng_params):
        with InferVStreams(ng, input_params, output_params) as pipeline:
            raw = pipeline.infer({input_name: input_data})

    return raw, orig_w, orig_h


def _decode(raw, orig_w, orig_h):
    box_keys = ["turkey_640/conv61", "turkey_640/conv77", "turkey_640/conv91"]
    cls_keys = ["turkey_640/conv64", "turkey_640/conv80", "turkey_640/conv94"]

    detections = []

    for stride, bk, ck in zip(STRIDES, box_keys, cls_keys):
        box_tensor = _dequantize(raw[bk].squeeze(), bk)    # (H, W, 4) float
        cls_tensor = _dequantize(raw[ck].squeeze(), ck)    # (H, W, 3) float

        H, W, _ = box_tensor.shape

        # Sigmoid on dequantized class logits
        cls_scores = 1.0 / (1.0 + np.exp(-cls_tensor))    # (H, W, 3)

        # Best class per cell
        cls_ids   = np.argmax(cls_scores, axis=2)          # (H, W)
        cls_confs = cls_scores[
            np.arange(H)[:, None],
            np.arange(W)[None, :],
            cls_ids
        ]                                                   # (H, W)

        mask = cls_confs >= CONF
        gy_list, gx_list = np.where(mask)

        for gy, gx in zip(gy_list, gx_list):
            cls_id   = int(cls_ids[gy, gx])
            cls_conf = float(cls_confs[gy, gx])

            box = box_tensor[gy, gx]
            x1  = float(box[0])
            y1  = float(box[1])
            x2  = float(box[2])
            y2  = float(box[3])

            sx = orig_w / IMGSZ
            sy = orig_h / IMGSZ
            x1 = max(0, min(int(x1 * sx), orig_w))
            y1 = max(0, min(int(y1 * sy), orig_h))
            x2 = max(0, min(int(x2 * sx), orig_w))
            y2 = max(0, min(int(y2 * sy), orig_h))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append((CLASS_MAP.get(cls_id, str(cls_id)), cls_conf, (x1, y1, x2, y2)))

    return _nms(detections, iou_thresh=0.45)


def _nms(detections, iou_thresh=0.45):
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d[1], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [
            d for d in detections
            if d[0] != best[0] or _iou(best[2], d[2]) < iou_thresh
        ]
    return keep


def _iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def run_hef(image):
    raw, orig_w, orig_h = _infer_hailo(image)
    return _decode(raw, orig_w, orig_h)


def run_hef_debug(image):
    print("\n" + "="*50)
    print("HEF DEBUG")
    print("="*50)
    raw, orig_w, orig_h = _infer_hailo(image)

    print("Dequantized output stats:")
    for name in sorted(raw.keys()):
        t = _dequantize(raw[name].squeeze(), name)
        print(f"  {name}: shape={t.shape}  min={t.min():.4f}  max={t.max():.4f}")

    dets = _decode(raw, orig_w, orig_h)
    print(f"\nDecoded {len(dets)} detections:")
    for cls, conf, bbox in dets:
        print(f"  {cls} conf={conf:.3f} bbox={bbox}")
    print("="*50 + "\n")
    return dets


def draw_detections(image, detections, label):
    out = image.copy()
    for cls_name, conf, (x1, y1, x2, y2) in detections:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{cls_name} {conf:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return out


# ── Main ────────────────────────────────────────────────────────
image = cv2.imread(IMG_PATH)
if image is None:
    print(f"ERROR: Could not load image: {IMG_PATH}")
    exit(1)

print("Running .pt model...")
pt_dets = run_pt(image)

print("Running .hef model...")
hef_dets = run_hef_debug(image)

print("="*50)
print(f"PT  model — {len(pt_dets)} detections:")
for cls, conf, bbox in pt_dets:
    print(f"  {cls} conf={conf:.3f} bbox={bbox}")

print(f"\nHEF model — {len(hef_dets)} detections:")
for cls, conf, bbox in hef_dets:
    print(f"  {cls} conf={conf:.3f} bbox={bbox}")
print("="*50)

pt_img   = draw_detections(image, pt_dets,  "PT (CPU)")
hef_img  = draw_detections(image, hef_dets, "HEF (Hailo)")
combined = np.hstack([pt_img, hef_img])
cv2.imwrite("comparison.jpg", combined)
print("\nSaved comparison.jpg")