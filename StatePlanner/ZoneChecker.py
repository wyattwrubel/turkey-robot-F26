class ZoneChecker:

    TOP    = 0
    BOTTOM = 90
    LEFT   = 0
    RIGHT  = 50

    def is_inside(self, x: float, y: float) -> bool:
        return self.LEFT <= x <= self.RIGHT and self.TOP <= y <= self.BOTTOM

    def is_outside(self, x: float, y: float) -> bool:
        return not self.is_inside(x, y)

    def border_distance(self, x:float, y:float) -> bool:
        return min(abs(x - self.LEFT), abs(x - self.RIGHT), abs(y - self.TOP), abs(y - self.BOTTOM))