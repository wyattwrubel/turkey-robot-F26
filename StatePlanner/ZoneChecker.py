"""
ZoneChecker.py

Loads a convex zone boundary from Boundary.csv and provides a
point-in-zone test using the cross product method.

Expected CSV format:
    # optional comment lines
    x,y
    1.0,0.0
    4.0,0.0
    4.0,3.0
    1.0,3.0
    ...

Vertices must be ordered consistently (clockwise or counter-clockwise).
Raises RuntimeError if the file is missing or has fewer than 3 valid vertices.
"""

import csv
import numpy as np
from pathlib import Path


class ZoneChecker:

    BOUNDARY_PATH = Path('/home/turkeysquad/ME450/MainDesign/Navigation/Boundary.csv')

    def __init__(self, boundary_path: Path | None = None):
        self._path     = boundary_path or self.BOUNDARY_PATH
        self._vertices = []
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            raise RuntimeError(f"[ZoneChecker] Boundary file not found: {self._path}")

        vertices = []
        with open(self._path, newline="") as f:
            for row in csv.reader(f):
                row = [c.strip() for c in row]
                if not row or row[0].startswith("#"):
                    continue
                if row[0].lower() in ("x", "label"):
                    continue
                try:
                    x = float(row[0])
                    y = float(row[1])
                    vertices.append(np.array([x, y]))
                except (ValueError, IndexError):
                    continue

        if len(vertices) < 3:
            raise RuntimeError(
                f"[ZoneChecker] Boundary file must contain at least 3 valid vertices, "
                f"got {len(vertices)}: {self._path}"
            )

        self._vertices = vertices
        print(f"[ZoneChecker] Loaded {len(vertices)}-vertex boundary polygon")

    # ------------------------------------------------------------------
    # Point-in-Zone Test (convex cross product method)
    # ------------------------------------------------------------------

    def is_inside(self, x: float, y: float) -> bool:
        """
        Returns True if (x, y) is inside or on the convex polygon boundary.

        For a convex polygon with consistently ordered vertices, a point is
        inside if the 2D cross product of each edge with the vector to the
        point all share the same sign.
        """
        point = np.array([x, y])
        sign  = None

        n = len(self._vertices)
        for i in range(n):
            a = self._vertices[i]
            b = self._vertices[(i + 1) % n]

            edge  = b - a
            to_pt = point - a
            cross = float(edge[0] * to_pt[1] - edge[1] * to_pt[0])

            if abs(cross) < 1e-9:
                continue  # point lies on this edge -- inside

            current_sign = cross > 0
            if sign is None:
                sign = current_sign
            elif sign != current_sign:
                return False

        return True

    def is_outside(self, x: float, y: float) -> bool:
        return not self.is_inside(x, y)