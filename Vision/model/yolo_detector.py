import numpy as np
import cv2
from typing import List
from utils.types import Detection
from config import CONFIG


class YOLODetector:
    def __init__(self):
        self.model_path     = CONFIG.model.model_path
        self.conf_thresh    = CONFIG.model.conf_thresh
        self.iou_thresh     = CONFIG.model.iou_thresh
        self.imgsz          = CONFIG.model.imgsz
        self.max_det        = CONFIG.model.max_det
        self.class_name_map = CONFIG.class_cfg.class_name_map
        self.is_hailo       = self.model_path.endswith(".hef")

        if self.is_hailo:
            self._init_hailo()
        else:
            self._init_ultralytics()

        print(f"[YOLO] Model loaded: {self.model_path}")
        print(f"[YOLO] Hailo mode:   {self.is_hailo}")

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_ultralytics(self):
        import os
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["OPENBLAS_NUM_THREADS"] = "4"
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)

    def _init_hailo(self):
        from hailo_platform import (
            HEF, VDevice, HailoStreamInterface,
            ConfigureParams, InputVStreamParams,
            OutputVStreamParams, FormatType, InferVStreams,
        )
        self._InferVStreams         = InferVStreams
        self._InputVStreamParams    = InputVStreamParams
        self._OutputVStreamParams   = OutputVStreamParams
        self._FormatType            = FormatType

        self.hef    = HEF(self.model_path)
        self.target = VDevice()

        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef,
            interface=HailoStreamInterface.PCIe,
        )
        self.network_groups       = self.target.configure(self.hef, configure_params)
        self.network_group        = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstream_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8,
        )
        self.output_vstream_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32,
        )

        input_info        = self.hef.get_input_vstream_infos()[0]
        self.input_name   = input_info.name
        self.input_height = input_info.shape[0]   # 640
        self.input_width  = input_info.shape[1]   # 640

        # output name — grab dynamically so we never hardcode it
        self.output_name  = self.hef.get_output_vstream_infos()[0].name

        print(f"[YOLO][Hailo] Input  : {self.input_name} {input_info.shape}")
        print(f"[YOLO][Hailo] Output : {self.output_name}")

    # ------------------------------------------------------------------
    # Hailo inference
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR → uint8 RGB, resized to HEF input size, shape (1, H, W, 3)."""
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb[np.newaxis].astype(np.uint8)   # (1, 640, 640, 3)

    def _detect_hailo(self, frame: np.ndarray) -> List[Detection]:
        orig_h, orig_w = frame.shape[:2]
        inp = self._preprocess(frame)

        with self._InferVStreams(
            self.network_group,
            self.input_vstream_params,
            self.output_vstream_params,
        ) as pipeline:
            with self.network_group.activate(self.network_group_params):
                raw = pipeline.infer({self.input_name: inp})

        # Output shape: (1, 1, 7, 8400)
        # Squeeze to (7, 8400) then transpose to (8400, 7)
        tensor = np.array(raw[self.output_name])
        tensor = tensor.squeeze()           # (7, 8400)
        tensor = tensor.T                   # (8400, 7)

        return self._parse(tensor, orig_w, orig_h)

    def _parse(self, tensor: np.ndarray, orig_w: int, orig_h: int) -> List[Detection]:
        """
        Each row: [x1, y1, x2, y2, conf, cls_score_0, cls_score_1, ...]
        Coords are in HEF input pixel space (0–640). Scale to original frame.
        """
        detections: List[Detection] = []

        scale_x = orig_w / self.input_width
        scale_y = orig_h / self.input_height

        # Column layout
        # [0:4] = x1, y1, x2, y2   (pixel coords in 640x640 space)
        # [4]   = objectness / box confidence
        # [5:]  = per-class scores
        x1s  = tensor[:, 0]
        y1s  = tensor[:, 1]
        x2s  = tensor[:, 2]
        y2s  = tensor[:, 3]
        confs = tensor[:, 4]

        if tensor.shape[1] > 5:
            cls_scores = tensor[:, 5:]                      # (8400, num_classes)
            cls_ids    = np.argmax(cls_scores, axis=1)      # (8400,)
            final_conf = confs * cls_scores[np.arange(len(cls_ids)), cls_ids]
        else:
            # No separate class scores — conf IS the final score, cls=0
            cls_ids    = np.zeros(len(confs), dtype=int)
            final_conf = confs

        # Filter by confidence threshold
        mask = final_conf >= self.conf_thresh
        if not np.any(mask):
            return []

        x1s        = x1s[mask];   y1s  = y1s[mask]
        x2s        = x2s[mask];   y2s  = y2s[mask]
        final_conf = final_conf[mask]
        cls_ids    = cls_ids[mask]

        for i in range(len(x1s)):
            x1 = int(x1s[i] * scale_x)
            y1 = int(y1s[i] * scale_y)
            x2 = int(x2s[i] * scale_x)
            y2 = int(y2s[i] * scale_y)

            if x2 <= x1 or y2 <= y1:
                continue

            cls_id     = int(cls_ids[i])
            class_name = self.class_name_map.get(cls_id, str(cls_id))

            detections.append(Detection(
                class_id=cls_id,
                class_name=class_name,
                confidence=float(final_conf[i]),
                bbox=(x1, y1, x2, y2),
            ))

        detections = _nms(detections, self.iou_thresh)
        return detections[: self.max_det]

    # ------------------------------------------------------------------
    # Ultralytics inference
    # ------------------------------------------------------------------

    def _detect_ultralytics(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            max_det=self.max_det,
            verbose=False,
        )
        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i].item())
                conf   = float(result.boxes.conf[i].item())
                x1, y1, x2, y2 = map(int, result.boxes.xyxy[i].tolist())
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=self.class_name_map.get(cls_id, str(cls_id)),
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                ))
        return detections

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.is_hailo:
            return self._detect_hailo(frame)
        return self._detect_ultralytics(frame)


# ------------------------------------------------------------------
# NMS + IOU helpers
# ------------------------------------------------------------------

def _nms(detections: List[Detection], iou_thresh: float) -> List[Detection]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [
            d for d in detections
            if d.class_name != best.class_name or _iou(best.bbox, d.bbox) < iou_thresh
        ]
    return keep


def _iou(b1, b2) -> float:
    ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1    = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2    = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0