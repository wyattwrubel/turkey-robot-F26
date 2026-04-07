import csv
import numpy as np

class ZoneChecker:

    CSV_PATH = "/home/turkeysquad/ME450/MainDesign/Navigation/Zone.csv"

    def __init__(self):
        self._grid, self._n_rows, self._n_cols = self._load_grid(self.CSV_PATH)

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    def _load_grid(self, path: str):
        r"""
        CSV format:
          Header row:  x\y, 0, 1, 2, ..., N   (y column indices)
          Data rows:   x_idx, val, val, ...
          Cell value:  0 = inside zone, 1 = outside zone
        Returns grid[x][y] as a 2-D numpy int array.
        """
        with open(path, newline='') as f:
            reader = csv.reader(f)
            rows   = [r for r in reader]

        # First row is header (x\y, 0, 1, ...) — skip it
        data_rows = rows[1:]
        n_rows    = len(data_rows)
        n_cols    = len(data_rows[0]) - 1   # first column is the x index label

        grid = np.zeros((n_rows, n_cols), dtype=int)
        for row in data_rows:
            x_idx = int(row[0])
            for col_idx, val in enumerate(row[1:]):
                grid[x_idx][col_idx] = int(val)

        return grid, n_rows, n_cols

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _to_tile(self, x: float, y: float):
        """Round world coordinates to nearest tile indices, clamped to grid bounds."""
        xi = int(round(x))
        yi = int(round(y))
        xi = max(0, min(xi, self._n_rows - 1))
        yi = max(0, min(yi, self._n_cols - 1))
        return xi, yi

    # ------------------------------------------------------------------
    # Public API  (same signatures as before)
    # ------------------------------------------------------------------

    def is_inside(self, x: float, y: float) -> bool:
        """Return True if the tile at (x, y) is inside the zone (grid value == 0)."""
        xi, yi = self._to_tile(x, y)
        return self._grid[xi][yi] == 0

    def is_outside(self, x: float, y: float) -> bool:
        return not self.is_inside(x, y)

    def border_distance(self, x: float, y: float) -> float:
        """
        Euclidean distance (in tiles) from (x, y) to the nearest outside tile.
        Returns 0.0 if the point is already outside the zone.
        """
        xi, yi = self._to_tile(x, y)
        if self._grid[xi][yi] != 0:
            return 0.0
        # Find all outside tiles and return distance to closest
        outside = np.argwhere(self._grid == 1)
        if outside.size == 0:
            return float('inf')
        dists = np.linalg.norm(outside - np.array([xi, yi]), axis=1)
        return float(np.min(dists))