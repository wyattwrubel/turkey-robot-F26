import cv2
import numpy as np
from ultralytics import YOLO
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams
)

# ── Config ──────────────────────────────────────────────────────
PT_PATH  = "/home/turkeysquad/ME450/MainDesign/Vision/weights/turkey_640.pt"
HEF_PATH = "/home/turkeysquad/ME450/MainDesign/Vision/weights/turkey_640.hef"
IMG_PATH  = "/home/turkeysquad/ME450/MainDesign/Vision/human_test.jpg"   # path to a test image with a person/turkey in it
CONF      = 0.05               # low threshold to catch anything
IMGSZ     = 640
CLASS_MAP = {0: "turkey", 1: "person", 2: "vehicle"}
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

def run_hef(image):
    orig_h, orig_w = image.shape[:2]
    resized = cv2.resize(image, (IMGSZ, IMGSZ))
    input_data = (resized.astype(np.float32) / 255.0)[np.newaxis]

    hef = HEF(HEF_PATH)
    target = VDevice()
    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    ng = network_groups[0]
    ng_params = ng.create_params()
    input_params  = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    output_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

    with InferVStreams(ng, input_params, output_params) as pipeline:
        with ng.activate(ng_params):
            input_name = hef.get_input_vstream_infos()[0].name
            raw = pipeline.infer({input_name: input_data})

    predictions = list(raw.values())[0].squeeze().T  # (8400, 7)

    detections = []
    for pred in predictions:
        x1, y1, x2, y2 = pred[:4]
        obj_conf     = pred[4]
        class_scores = pred[5:]
        cls_id = int(np.argmax(class_scores))
        conf   = float(obj_conf * class_scores[cls_id])

        if conf < CONF:
            continue

        x1 = max(0, min(int(x1 * orig_w / IMGSZ), orig_w))
        y1 = max(0, min(int(y1 * orig_h / IMGSZ), orig_h))
        x2 = max(0, min(int(x2 * orig_w / IMGSZ), orig_w))
        y2 = max(0, min(int(y2 * orig_h / IMGSZ), orig_h))

        detections.append((CLASS_MAP.get(cls_id, str(cls_id)), conf, (x1, y1, x2, y2)))

    return detections

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
pt_dets  = run_pt(image)

print("Running .hef model...")
hef_dets = run_hef(image)

# Print comparison
print("\n" + "="*50)
print(f"PT  model — {len(pt_dets)} detections:")
for cls, conf, bbox in pt_dets:
    print(f"  {cls} conf={conf:.3f} bbox={bbox}")

print(f"\nHEF model — {len(hef_dets)} detections:")
for cls, conf, bbox in hef_dets:
    print(f"  {cls} conf={conf:.3f} bbox={bbox}")
print("="*50)

# Side by side visual
pt_img  = draw_detections(image, pt_dets,  "PT (CPU)")
hef_img = draw_detections(image, hef_dets, "HEF (Hailo)")
combined = np.hstack([pt_img, hef_img])
cv2.imwrite("comparison.jpg", combined)
print("\nSaved side-by-side comparison to comparison.jpg")