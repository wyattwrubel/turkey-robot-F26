import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from config import CONFIG
from utils.types import TrackedTarget, PositionResult


class PositionEstimator:
    def __init__(self):
        self.hfov_deg = CONFIG.position.hfov_deg

        self.human_model_path = CONFIG.position.human_model_path
        self.turkey_model_path = CONFIG.position.turkey_model_path

        self.human_model = self._load_model(self.human_model_path)
        self.turkey_model = self._load_model(self.turkey_model_path)

    def _load_model(self, model_path: str):
        """Load a sklearn/joblib distance model."""
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Distance model not found: {model_path}")
        model = joblib.load(model_path)
        print(f"[PositionEstimator] Loaded model: {model_path}")
        return model

    def _compute_angle_deg(self, center_x: float, image_width: int) -> float:
        """
        Convert bbox center x-coordinate to horizontal angle in degrees.
        Negative = left, positive = right.
        """
        image_center_x = image_width / 2.0
        normalized_offset = (center_x - image_center_x) / image_center_x
        angle_deg = normalized_offset * (self.hfov_deg / 2.0)
        return angle_deg

    def _extract_bbox_geometry(self, target: TrackedTarget):
        """
        Extract geometry from bbox in xyxy format:
            bbox = (x1, y1, x2, y2)
        """
        if target.bbox is None or len(target.bbox) != 4:
            raise ValueError(f"Invalid bbox for target {target.target_id}: {target.bbox}")

        x1, y1, x2, y2 = target.bbox

        width = float(x2 - x1)
        height = float(y2 - y1)

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Non-positive bbox size for target {target.target_id}: bbox={target.bbox}"
            )

        area = width * height
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        aspect_ratio = width / height

        return {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "width": width,
            "height": height,
            "area": area,
            "center_x": center_x,
            "center_y": center_y,
            "aspect_ratio": aspect_ratio,
        }

    def _build_features(self, target: TrackedTarget) -> pd.DataFrame:
        """
        Build model features exactly matching training-time feature columns:
            - inv_height
            - inv_sqrt_area
            - aspect_ratio
        """
        geom = self._extract_bbox_geometry(target)

        height = geom["height"]
        area = geom["area"]
        aspect_ratio = geom["aspect_ratio"]

        inv_height = 1.0 / height
        inv_sqrt_area = 1.0 / np.sqrt(area)

        X = pd.DataFrame([{
            "inv_height": inv_height,
            "inv_sqrt_area": inv_sqrt_area,
            "aspect_ratio": aspect_ratio,
        }])

        return X

    def _select_model(self, class_name: str):
        """
        Select model by class name.
        Returns None if the target should be ignored.
        """
        cls = class_name.lower().strip()

        if cls == "vehicle":
            return None
        if cls == "turkey":
            return self.turkey_model
        if cls in {"person", "human"}:
            return self.human_model

        return None

    def _predict_distance(self, target: TrackedTarget) -> Optional[float]:
        """
        Predict distance/label using the corresponding model.
        Returns None if this class should be ignored.
        """
        model = self._select_model(target.class_name)
        if model is None:
            return None

        X = self._build_features(target)
        pred = float(model.predict(X)[0])
        return pred

    def estimate_one(
        self,
        target: TrackedTarget,
        image_width: int,
        image_height: int,
    ) -> Optional[PositionResult]:
        """
        Estimate angle + distance for one confirmed target.
        Returns None for ignored classes such as vehicle.
        """
        if not target.confirmed:
            return None

        geom = self._extract_bbox_geometry(target)
        angle_deg = self._compute_angle_deg(geom["center_x"], image_width)

        try:
            distance_m = self._predict_distance(target)
        except Exception as e:
            print(
                f"[PositionEstimator] Distance prediction failed "
                f"for target {target.target_id} ({target.class_name}): {e}"
            )
            return None

        if distance_m is None:
            return None

        image_area = image_width * image_height
        area_ratio = geom["area"] / image_area if image_area > 0 else 0.0

        return PositionResult(
            target_id=target.target_id,
            class_name=target.class_name,
            angle_deg=angle_deg,
            distance_m=distance_m,
            bbox=target.bbox,
            center=(geom["center_x"], geom["center_y"]),
            confidence=target.confidence,
            area_ratio=area_ratio,
        )

    def estimate_all(
        self,
        targets: List[TrackedTarget],
        image_width: int,
        image_height: int,
    ) -> List[PositionResult]:
        """
        Estimate positions for all confirmed targets.
        vehicle and unsupported classes are ignored.
        """
        results: List[PositionResult] = []

        for target in targets:
            if not target.confirmed:
                continue

            # Ignore vehicle directly
            if target.class_name.lower().strip() == "vehicle":
                continue

            result = self.estimate_one(target, image_width, image_height)
            if result is not None:
                results.append(result)

        return results