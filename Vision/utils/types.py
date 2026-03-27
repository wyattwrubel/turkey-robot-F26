from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class TrackedTarget:
    target_id: int
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float

    confirmed: bool = False
    frame_history: List[bool] = field(default_factory=list)
    missing_streak: int = 0

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass
class PositionResult:
    target_id: int
    class_name: str
    angle_deg: float
    distance_m: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    confidence: float
    area_ratio: float

@dataclass
class CoordinateResult:
    target_id: int
    class_name: str
    angle_deg: float
    distance_m: float
    x_m: float
    y_m: float
    bbox: Tuple[int, int, int, int]
    confidence: float