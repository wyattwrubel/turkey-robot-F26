import csv
import numpy as np

from config.paths import ZONE_CSV

class ZoneChecker:

    CSV_PATH = str(ZONE_CSV)

    def __init__(self):
        with open(self.CSV_PATH, newline='') as f:
            rows = list(csv.reader(f))
        
        data_rows = rows[1:]  # skip header
        n_rows = len(data_rows)
        n_cols = len(data_rows[0]) - 1

        self._grid = np.zeros((n_rows, n_cols), dtype=int)
        for row in data_rows:
            x = int(row[0])
            for y, val in enumerate(row[1:]):
                self._grid[x][y] = int(val)

    def is_inside(self, x: float, y: float) -> bool:
        xi = max(0, min(int(round(y)), self._grid.shape[0] - 1))  # world Y → row
        yi = max(0, min(int(round(x)), self._grid.shape[1] - 1))  # world X → col

        return self._grid[xi][yi] != -1

    def is_outside(self, x: float, y: float) -> bool:
        return not self.is_inside(x, y)