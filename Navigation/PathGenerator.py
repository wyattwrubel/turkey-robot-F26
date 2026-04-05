import heapq
import numpy as np
from collections import deque


# =============================================================================
# Config
# =============================================================================

# Cells within this distance of a fixed obstacle are treated as blocked during
# planning.  1.5m keeps the planner naturally away from walls, accounting for
# mapping uncertainty and robot footprint.
PLANNING_CLEARANCE = 1.5   # metres (== grid cells, 1 cell = 1 m)


# =============================================================================
# Clearance check using rectangle list
# =============================================================================

def _min_rect_dist(obstacle_rects, x, y):
    """
    Minimum distance from grid cell (x, y) to any fixed obstacle rectangle.
    Cheap NumPy arithmetic — no grid array lookup needed.
    """
    min_d = np.inf
    for (x0, x1, y0, y1) in obstacle_rects:
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        d  = np.sqrt(dx*dx + dy*dy)
        if d < min_d:
            min_d = d
    return min_d


def _is_blocked(map_grid, obstacle_rects, x, y):
    """True if cell (x, y) is an obstacle, dynamic obstacle, or too close to a
    fixed obstacle rectangle."""
    h, w = map_grid.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return True
    if map_grid[y, x] == 1 or map_grid[y, x] == 2:
        return True
    if _min_rect_dist(obstacle_rects, x, y) < PLANNING_CLEARANCE:
        return True
    return False


def _line_of_sight(map_grid, obstacle_rects, a, b):
    """
    Bresenham line-of-sight check between grid cells a and b.
    Returns True iff every cell along the line is free and has sufficient
    clearance from fixed obstacle rectangles.
    """
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0);  dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    h, w = map_grid.shape

    while True:
        if x0 < 0 or x0 >= w or y0 < 0 or y0 >= h:
            return False
        if map_grid[y0, x0] == 1 or map_grid[y0, x0] == 2:
            return False
        if _min_rect_dist(obstacle_rects, x0, y0) < PLANNING_CLEARANCE:
            return False
        if x0 == x1 and y0 == y1:
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy;  x0 += sx
        if e2 <  dx:
            err += dx;  y0 += sy


# =============================================================================
# Theta*
# =============================================================================

def _theta_star(map_grid, obstacle_rects, start, goal):
    """
    Theta* on the Zone_Map grid.  Produces sparse, geometrically smooth
    waypoints by allowing any-angle edges via line-of-sight checks.
    Falls back to closest node reached if the goal is unreachable.
    """
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                 (-1,-1),(-1,1),(1,-1),(1,1)]

    came_from = {start: start}
    gscore    = {start: 0.0}
    open_heap = []
    closed    = set()
    heapq.heappush(open_heap, (np.hypot(start[0]-goal[0], start[1]-goal[1]), start))

    best_node = start
    best_h    = np.hypot(start[0]-goal[0], start[1]-goal[1])

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            best_node = goal
            break

        if current in closed:
            continue
        closed.add(current)

        h = np.hypot(current[0]-goal[0], current[1]-goal[1])
        if h < best_h:
            best_h    = h
            best_node = current

        parent = came_from[current]

        for dx, dy in neighbors:
            nx, ny   = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)

            if _is_blocked(map_grid, obstacle_rects, nx, ny):
                continue
            if neighbor in closed:
                continue

            if _line_of_sight(map_grid, obstacle_rects, parent, neighbor):
                tentative_g = gscore[parent] + np.hypot(
                    parent[0]-neighbor[0], parent[1]-neighbor[1])
                use_parent  = parent
            else:
                tentative_g = gscore[current] + np.hypot(dx, dy)
                use_parent  = current

            if neighbor not in gscore or tentative_g < gscore[neighbor]:
                came_from[neighbor] = use_parent
                gscore[neighbor]    = tentative_g
                fscore              = tentative_g + np.hypot(
                    neighbor[0]-goal[0], neighbor[1]-goal[1])
                heapq.heappush(open_heap, (fscore, neighbor))

    # Reconstruct
    path    = []
    node    = best_node
    visited = set()
    while node != came_from[node]:
        if node in visited:
            break
        visited.add(node)
        path.append(node)
        node = came_from[node]
    path.append(node)
    path.reverse()
    return path


# =============================================================================
# Public interface
# =============================================================================

def GenerateTrajectory(map_obj, current_state, desired_state):
    """
    Theta* path planner.  Produces a sparse list of (x, y) grid-cell waypoints.

    Parameters
    ----------
    map_obj       : MapUpdater.Map instance (has .Zone_Map and .Obstacle_Rects)
    current_state : array-like [x, y, ...]
    desired_state : array-like [x, y, ...]

    Returns
    -------
    List of (int x, int y) waypoints — typically 2-6 points.
    """
    map_grid      = map_obj.Zone_Map
    obstacle_rects = map_obj.Obstacle_Rects

    start = tuple(np.array(current_state[0:2]).round().astype(int))
    goal  = tuple(np.array(desired_state[0:2]).round().astype(int))

    h, w  = map_grid.shape
    start = (int(np.clip(start[0], 0, w-1)), int(np.clip(start[1], 0, h-1)))
    goal  = (int(np.clip(goal[0],  0, w-1)), int(np.clip(goal[1],  0, h-1)))

    if _is_blocked(map_grid, obstacle_rects, *start):
        start = _nearest_free(map_grid, obstacle_rects, start, h, w)
    if _is_blocked(map_grid, obstacle_rects, *goal):
        goal  = _nearest_free(map_grid, obstacle_rects, goal,  h, w)

    path = _theta_star(map_grid, obstacle_rects, start, goal)
    return [(int(x), int(y)) for x, y in path]


def _nearest_free(map_grid, obstacle_rects, cell, h, w):
    """BFS outward from cell until a free, clear cell is found."""
    q    = deque([cell])
    seen = {cell}
    while q:
        cx, cy = q.popleft()
        if not _is_blocked(map_grid, obstacle_rects, cx, cy):
            return (cx, cy)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = cx+dx, cy+dy
            if (nx, ny) not in seen and 0 <= nx < w and 0 <= ny < h:
                seen.add((nx, ny))
                q.append((nx, ny))
    return cell