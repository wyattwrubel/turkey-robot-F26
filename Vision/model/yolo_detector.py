from typing import List
from ultralytics import YOLO
from utils.types import Detection
from config import CONFIG


class YOLODetector:
    def __init__(self):
        self.model_path = CONFIG.model.model_path
        self.conf_thresh = CONFIG.model.conf_thresh
        self.iou_thresh = CONFIG.model.iou_thresh
        self.imgsz = CONFIG.model.imgsz
        self.max_det = CONFIG.model.max_det
        self.device = CONFIG.model.device
        self.half = CONFIG.model.half

        self.class_name_map = CONFIG.class_cfg.class_name_map
        self.model = YOLO(self.model_path)

        print(f"[YOLO] Model loaded from: {self.model_path}")
        print(f"[YOLO] Device: {self.device}, half: {self.half}")

    def detect(self, frame) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            half=self.half,
            verbose=False
        )

        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

                class_name = self.class_name_map.get(cls_id, str(cls_id))
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        return detections