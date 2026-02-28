import numpy as np
import csv

class Map:
    Zone_Map = []
    _Human_List = []
    _Turkey_List = []
    _Map_File = "/home/meathead/ME450/MainDesign/Navigation/Zone.csv"

    _Human_Obstacle_Width = 1
    _Human_Penalty_width = 2
    _Turkey_Obstacle_Width = 1

    # DEFINITIONS
    # 1 Fixed obstacles
    # 2 Dynamic Obstacles
    # >2 penalized weight not default 0
    
    def __init__(self):
        self.LoadGrid()

    def ClearDynamicObstacles(self):
        for h_idx in self._Human_List:
            self.UpdateAroundObject(h_idx[0], h_idx[1], width=self._Human_Penalty_width, data=0)

        for t_idx in self._Turkey_List:
            self.UpdateAroundObject(t_idx[0], t_idx[1], width=self._Turkey_Obstacle_Width, data=0)

        self._Human_List = []
        self._Turkey_List = []

    def UpdateHuman(self, xcord, ycord):
        self._Human_List.append([xcord, ycord])

        # Outer Box Near Human Allowed But Penalized With Weight 3
        self.UpdateAroundObject(xcord, ycord, width=2, data=3)
        # 1m radius Around Human Considered Obstacle
        self.UpdateAroundObject(xcord, ycord, width=1, data=2)

    def UpdateTurkey(self, xcord, ycord):
        self._Turkey_List.append([xcord, ycord])

        # 1m radius Around Turkey Considered Obstacle
        self.UpdateAroundObject(xcord, ycord, width=1, data=2)

    def LoadGrid(self):
        grid = []
        with open(self._Map_File, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # skip header row (x-axis labels)

            for row in reader:
                data_cells = row[1:]  # skip y-axis label (first column)
                numeric_row = [int(cell) if cell.strip() != '' else 0
                            for cell in data_cells]
                grid.append(numeric_row)

        self.Zone_Map = np.array(grid, dtype=int)

    def UpdateAroundObject(self, xcord, ycord, width, data):
        xcord_low = max(xcord - width, 0)
        xcord_high = min(xcord + width+1, self.Zone_Map.shape[1]-1)
        ycord_low = max(ycord - width, 0)
        ycord_high = min(ycord + width+1, self.Zone_Map.shape[0]-1)
        
        for x in range(xcord_low, xcord_high):
            for y in range(ycord_low, ycord_high):
                if(self.Zone_Map[y,x] != 1 and (data == 0 or self.Zone_Map[y, x] != 2)):
                    self.Zone_Map[y, x] = data 