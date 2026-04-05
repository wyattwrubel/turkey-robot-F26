import math
from typing import List
from utils.types import PositionResult, CoordinateResult


class CoordinateOutput:
    def __init__(self):
        pass

    def convert_one(self, result: PositionResult) -> CoordinateResult:
        """
        Convert angle + distance into camera-relative Cartesian coordinates.

        Coordinate convention:
            right  = +x
            front  = +y

        Polar-to-Cartesian:
            x = d * sin(theta)
            y = d * cos(theta)
        """
        theta_rad = math.radians(result.angle_deg)
        d = result.distance_m

        x_m = d * math.sin(theta_rad)
        y_m = d * math.cos(theta_rad)

        return CoordinateResult(
            target_id=result.target_id,
            class_name=result.class_name,
            angle_deg=result.angle_deg,
            distance_m=result.distance_m,
            x_m=x_m,
            y_m=y_m,
            bbox=result.bbox,
            confidence=result.confidence,
        )

    def convert_all(self, results: List[PositionResult]) -> List[CoordinateResult]:
        outputs = []
        for result in results:
            outputs.append(self.convert_one(result))
        return outputs

    def print_all(self, outputs: List[CoordinateResult]) -> None:
        if not outputs:
            print("[Output] No targets")
            return

        for out in outputs:
            print(
                f"[Output] ID={out.target_id}, "
                f"class={out.class_name}, "
                f"angle={out.angle_deg:.2f} deg, "
                f"distance={out.distance_m:.2f} m, "
                f"x={out.x_m:.2f} m, "
                f"y={out.y_m:.2f} m"
            )