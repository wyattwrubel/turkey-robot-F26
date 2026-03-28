from typing import List, Optional
from utils.types import Detection, TrackedTarget


def compute_iou(box1, box2) -> float:
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    inter_x1 = max(x11, x21)
    inter_y1 = max(y11, y21)
    inter_x2 = min(x12, x22)
    inter_y2 = min(y12, y22)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0, x12 - x11) * max(0, y12 - y11)
    area2 = max(0, x22 - x21) * max(0, y22 - y21)

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


class TemporalManager:
    def __init__(
        self,
        init_window: int = 10,
        init_min_hits: int = 7,
        # delete_missing_frames: int = 30,
        delete_missing_frames: int = 0,
    ):
        self.init_window = init_window
        self.init_min_hits = init_min_hits
        self.delete_missing_frames = delete_missing_frames

        self.targets: List[TrackedTarget] = []
        self.next_target_id = 0

    def _match_target(
        self,
        detection: Detection,
        candidates: List[TrackedTarget],
    ) -> Optional[TrackedTarget]:
        """
        Match by:
        1. same class
        2. IoU > 0
        Return the best IoU match if exists.
        """
        best_target = None
        best_iou = 0.0

        for target in candidates:
            if target.class_name != detection.class_name:
                continue

            iou = compute_iou(detection.bbox, target.bbox)
            if iou > 0 and iou > best_iou:
                best_iou = iou
                best_target = target

        return best_target

    def _create_target(self, detection: Detection) -> TrackedTarget:
        target = TrackedTarget(
            target_id=self.next_target_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            bbox=detection.bbox,
            confidence=detection.confidence,
            confirmed=False,
            frame_history=[True],
            missing_streak=0,
        )
        self.next_target_id += 1
        return target

    def update(self, detections: List[Detection]) -> List[TrackedTarget]:
        """
        Update temporal state with detections from current frame.

        Returns:
            confirmed targets only
        """
        matched_target_ids = set()

        # 1. match current detections to existing targets
        for det in detections:
            matched = self._match_target(det, self.targets)

            if matched is not None:
                matched.bbox = det.bbox
                matched.confidence = det.confidence
                matched.missing_streak = 0
                matched_target_ids.add(matched.target_id)

                if not matched.confirmed:
                    matched.frame_history.append(True)
                    if len(matched.frame_history) > self.init_window:
                        matched.frame_history.pop(0)

                    if (
                        len(matched.frame_history) == self.init_window
                        and sum(matched.frame_history) >= self.init_min_hits
                    ):
                        matched.confirmed = True
            else:
                # create a new candidate target
                new_target = self._create_target(det)
                self.targets.append(new_target)
                matched_target_ids.add(new_target.target_id)

        # 2. update unmatched targets
        remaining_targets = []
        for target in self.targets:
            if target.target_id not in matched_target_ids:
                if target.confirmed:
                    target.missing_streak += 1
                    if target.missing_streak < self.delete_missing_frames:
                        remaining_targets.append(target)
                else:
                    target.frame_history.append(False)
                    if len(target.frame_history) > self.init_window:
                        target.frame_history.pop(0)

                    # if candidate cannot be confirmed within window, drop it
                    if len(target.frame_history) < self.init_window:
                        remaining_targets.append(target)
                    else:
                        if sum(target.frame_history) >= self.init_min_hits:
                            target.confirmed = True
                            remaining_targets.append(target)
                        # else: drop this candidate
            else:
                remaining_targets.append(target)

        self.targets = remaining_targets

        return [t for t in self.targets if t.confirmed]