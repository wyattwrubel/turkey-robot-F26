import heapq
import numpy as np
from collections import deque
from scipy.ndimage import distance_transform_edt

# =============================================================================
# Config
# =============================================================================

EDT_WEIGHT      = 2.0   # how strongly A* favours corridor centres
STRAIGHT_WEIGHT = 0.3   # bonus cost for direction changes (favours straight lines)

# Cells within this EDT distance of a static obstacle are treated as blocked.
# This must be >= MPC STATIC_MARGIN (0.5m) so A* never routes through gaps
# that MPC cannot physically execute.  Set to 0.7m to give a small buffer
# above the 0.5m MPC hard constraint margin.
PLANNING_CLEARANCE = 0.7   # metres (== grid cells, 1 cell = 1 m)

# =============================================================================
# EDT cache — recomputed only when the static obstacle layer changes
# =============================================================================

_edt_cache   = None
_static_hash = None


def _get_edt(map_grid):
    """
    Return the EDT of the static-obstacle-free space.
    Cached — only recomputed when the static layer (cells == 1) changes.
    """
    global _edt_cache, _static_hash

    static_mask = (map_grid == 1).astype(np.uint8)
    h           = hash(static_mask.tobytes())

    if h != _static_hash:
        free         = (static_mask == 0).astype(np.uint8)
        _edt_cache   = distance_transform_edt(free).astype(np.float32)
        _static_hash = h

    return _edt_cache


# =============================================================================
# Helpers
# =============================================================================

def _is_blocked(map_grid, edt, x, y):
    """
    True if cell (x, y) is:
      - out of bounds
      - a static obstacle  (map == 1)
      - a dynamic obstacle (map == 2)
      - within PLANNING_CLEARANCE of a static obstacle (EDT check)

    The clearance rejection is what prevents A* from routing through gaps
    narrower than the robot can physically traverse under MPC.
    """
    h, w = map_grid.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return True
    cell = map_grid[y, x]
    if cell == 1 or cell == 2:
        return True
    if float(edt[y, x]) < PLANNING_CLEARANCE:
        return True
    return False


def _diagonal_cuts_corner(map_grid, edt, cx, cy, dx, dy):
    """True when a diagonal step would clip through an obstacle corner."""
    if abs(dx) == 1 and abs(dy) == 1:
        return (_is_blocked(map_grid, edt, cx + dx, cy) or
                _is_blocked(map_grid, edt, cx,       cy + dy))
    return False


def _nearest_free(map_grid, edt, cell):
    """BFS outward from cell until a free, sufficiently clear cell is found."""
    h, w = map_grid.shape
    q    = deque([cell])
    seen = {cell}
    while q:
        cx, cy = q.popleft()
        if not _is_blocked(map_grid, edt, cx, cy):
            return (cx, cy)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in seen and 0 <= nx < w and 0 <= ny < h:
                seen.add((nx, ny))
                q.append((nx, ny))
    return cell


# =============================================================================
# A* with EDT centering, planning clearance, and straight-line preference
# =============================================================================

NEIGHBORS = [
    (-1,  0, 1.0),   ( 1,  0, 1.0),   ( 0, -1, 1.0),   ( 0,  1, 1.0),
    (-1, -1, 1.414), (-1,  1, 1.414), ( 1, -1, 1.414),  ( 1,  1, 1.414),
]


def _astar(map_grid, start, goal):
    """
    A* on the Zone_Map grid.

    Edge cost = move_distance
              + EDT_WEIGHT / (edt(neighbour) + 0.5)   ← corridor centering
              + STRAIGHT_WEIGHT * direction_changed     ← straight-line bias

    Cells with EDT < PLANNING_CLEARANCE are treated as blocked (see _is_blocked),
    so the planner naturally routes through the centre of corridors and never
    produces a path through a gap too tight for MPC to execute.

    Falls back to closest reachable node if goal is unreachable.
    """
    edt  = _get_edt(map_grid)
    h, w = map_grid.shape

    def edt_penalty(x, y):
        col = int(np.clip(x, 0, w - 1))
        row = int(np.clip(y, 0, h - 1))
        return EDT_WEIGHT / (float(edt[row, col]) + 0.5)

    came_from = {start: (start, None)}
    gscore    = {start: 0.0}
    open_heap = []
    closed    = set()
    heapq.heappush(open_heap,
                   (np.hypot(start[0]-goal[0], start[1]-goal[1]), start))

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

        h_val = np.hypot(current[0]-goal[0], current[1]-goal[1])
        if h_val < best_h:
            best_h    = h_val
            best_node = current

        parent_node, parent_dir = came_from[current]

        for dx, dy, move_cost in NEIGHBORS:
            nx, ny   = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)

            if _is_blocked(map_grid, edt, nx, ny):
                continue
            if _diagonal_cuts_corner(map_grid, edt, current[0], current[1], dx, dy):
                continue
            if neighbor in closed:
                continue

            direction_cost = (STRAIGHT_WEIGHT
                              if parent_dir is not None and (dx, dy) != parent_dir
                              else 0.0)

            tentative_g = (gscore[current]
                           + move_cost
                           + edt_penalty(nx, ny)
                           + direction_cost)

            if neighbor not in gscore or tentative_g < gscore[neighbor]:
                came_from[neighbor] = (current, (dx, dy))
                gscore[neighbor]    = tentative_g
                fscore              = tentative_g + np.hypot(nx-goal[0], ny-goal[1])
                heapq.heappush(open_heap, (fscore, neighbor))

    # Reconstruct path
    path    = []
    node    = best_node
    visited = set()
    while True:
        if node in visited:
            break
        visited.add(node)
        path.append(node)
        parent_node, _ = came_from[node]
        if parent_node == node:
            break
        node = parent_node

    path.reverse()
    return path


# =============================================================================
# Public interface
# =============================================================================

def GenerateTrajectory(map_obj, current_state, desired_state):
    """
    A* path planner with EDT corridor-centering, planning clearance, and
    straight-line bias.

    Parameters
    ----------
    map_obj       : MapUpdater.Map instance  (.Zone_Map)
    current_state : array-like [x, y, ...]
    desired_state : array-like [x, y, ...]

    Returns
    -------
    List of (int x, int y) waypoints.
    If the goal is unreachable the path terminates at the closest reachable cell.
    """
    map_grid = map_obj.Zone_Map
    h, w     = map_grid.shape
    edt      = _get_edt(map_grid)

    start = tuple(np.array(current_state[:2]).round().astype(int))
    goal  = tuple(np.array(desired_state[:2]).round().astype(int))

    start = (int(np.clip(start[0], 0, w-1)), int(np.clip(start[1], 0, h-1)))
    goal  = (int(np.clip(goal[0],  0, w-1)), int(np.clip(goal[1],  0, h-1)))

    if _is_blocked(map_grid, edt, *start):
        start = _nearest_free(map_grid, edt, start)
    if _is_blocked(map_grid, edt, *goal):
        goal  = _nearest_free(map_grid, edt, goal)

    path = _astar(map_grid, start, goal)
    return [(int(x), int(y)) for x, y in path]