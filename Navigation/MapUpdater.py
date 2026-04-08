import numpy as np
import csv


class Map:
    # DEFINITIONS
    # 0   Free space
    # 1   Fixed obstacles (walls, trees, rocks)
    # 2   Dynamic obstacles (humans, turkeys — written by vision system)
    # >2  Penalised free space (near humans)

    _Map_File = "/home/turkeysquad/ME450/MainDesign/Navigation/Zone.csv"

    _Human_Obstacle_Width  = 1
    _Human_Penalty_Width   = 2
    _Turkey_Obstacle_Width = 1

    # -------------------------------------------------------------------------
    # Hardcoded obstacle rectangle decomposition for the current placeholder map.
    # Format: (x_min, x_max, y_min, y_max) in grid cells (== metres).
    #
    # TODO: replace with automated decomposition once the final map is received.
    # -------------------------------------------------------------------------
    OBSTACLE_RECTS = [
        (36, 40, 12, 58),
        ( 0,  1,  0, 77),
        ( 2, 40,  0,  1),
        ( 2, 40, 76, 77),
        ( 8, 14, 52, 58),
        (39, 40, 59, 75),
        (39, 40,  2, 11),
        (12, 14, 20, 22),
    ]

    def __init__(self):
        self._Human_List  = []
        self._Turkey_List = []

        self.Zone_Map       = np.zeros((100, 100), dtype=int)
        self.Obstacle_Rects = self.OBSTACLE_RECTS

        self.LoadGrid()

    # -------------------------------------------------------------------------
    # Dynamic obstacle management
    # -------------------------------------------------------------------------

    def ClearDynamicObstacles(self):
        for h_idx in self._Human_List:
            self.UpdateAroundObject(h_idx[0], h_idx[1],
                                    width=self._Human_Penalty_Width, data=0)
        for t_idx in self._Turkey_List:
            self.UpdateAroundObject(t_idx[0], t_idx[1],
                                    width=self._Turkey_Obstacle_Width, data=0)
        self._Human_List  = []
        self._Turkey_List = []

    def UpdateHuman(self, xcord, ycord):
        self._Human_List.append([xcord, ycord])
        self.UpdateAroundObject(xcord, ycord, width=self._Human_Penalty_Width,  data=3)
        self.UpdateAroundObject(xcord, ycord, width=self._Human_Obstacle_Width, data=2)

    def UpdateTurkey(self, xcord, ycord):
        self._Turkey_List.append([xcord, ycord])
        self.UpdateAroundObject(xcord, ycord, width=self._Turkey_Obstacle_Width, data=2)

    # -------------------------------------------------------------------------
    # Grid I/O
    # -------------------------------------------------------------------------

    def LoadGrid(self):
        grid = []
        with open(self._Map_File, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                data_cells  = row[1:]
                numeric_row = [int(cell) if cell.strip() != '' else 0
                               for cell in data_cells]
                grid.append(numeric_row)
        self.Zone_Map = np.array(grid, dtype=int)

    def UpdateAroundObject(self, xcord, ycord, width, data):
        """
        Write `data` into all cells within `width` steps of (xcord, ycord),
        forming a (2*width+1) x (2*width+1) square.

        Fixed: upper bounds now correctly include the +width cell so that
        a width of 1 produces a full 3x3 box as intended.
        Static obstacle cells (== 1) are never overwritten.
        When clearing (data == 0), dynamic obstacle cells (== 2) are also
        cleared; penalty cells (> 2) are cleared by the outer penalty pass.
        """
        h, w = self.Zone_Map.shape

        xcord_low  = max(xcord - width, 0)
        xcord_high = min(xcord + width + 1, w)   # was: shape[1] - 1 (off by one)
        ycord_low  = max(ycord - width, 0)
        ycord_high = min(ycord + width + 1, h)   # was: shape[0] - 1 (off by one)

        for x in range(xcord_low, xcord_high):
            for y in range(ycord_low, ycord_high):
                if self.Zone_Map[y, x] != 1 and (data == 0 or self.Zone_Map[y, x] != 2):
                    self.Zone_Map[y, x] = data

    # -------------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------------

    def min_rect_dist(self, x: float, y: float) -> float:
        """Minimum distance from (x, y) to any fixed obstacle rectangle."""
        min_d = np.inf
        for (x0, x1, y0, y1) in self.Obstacle_Rects:
            dx = max(x0 - x, 0.0, x - x1)
            dy = max(y0 - y, 0.0, y - y1)
            d  = np.sqrt(dx**2 + dy**2)
            if d < min_d:
                min_d = d
        return float(min_d)