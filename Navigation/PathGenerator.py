import heapq
import numpy as np
from collections import deque
from scipy.ndimage import distance_transform_edt

# =============================================================================
# Config
# =============================================================================

EDT_WEIGHT      = 2.5   # how strongly A* favours corridor centres
                        # (raised from 2.0 for stronger centering)
STRAIGHT_WEIGHT = 0.3   # bonus cost for direction changes (favours straight lines)

# Hard block: cells within this distance of a static obstacle are completely
# impassable.  Must be >= MPC physical margin so A* never routes through gaps
# the robot cannot physically execute.
PLANNING_CLEARANCE = 0.7   # metres

# Soft penalty zone: cells between PLANNING_CLEARANCE and SOFT_CLEARANCE are
# passable but incur SOFT_CLEARANCE_PENALTY added to their edge cost.
# This strongly discourages tight-margin routing while still allowing the
# planner to find a path if no wider alternative exists (e.g. turkey near wall).
SOFT_CLEARANCE         = 1.0   # metres — desired minimum separation
SOFT_CLEARANCE_PENALTY = 4.0   # flat extra cost per step inside the soft zone

# When replanning in chase mode, cells that lie on (or near) the previous
# A* path receive a cost discount.  Cells farther than PATH_BIAS_RADIUS from
# any old waypoint receive PATH_BIAS_PENALTY added to their edge cost.
PATH_BIAS_RADIUS  = 1.5   # metres
PATH_BIAS_PENALTY = 1.2   # extra cost per step when leaving the old-path corridor

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
      - within PLANNING_CLEARANCE (0.7m) of a static obstacle — hard block

    Cells between 0.7m and 1.0m are NOT blocked here; they receive
    SOFT_CLEARANCE_PENALTY in the edge cost instead (see _soft_penalty).
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


def _soft_penalty(edt, x, y):
    """
    Returns SOFT_CLEARANCE_PENALTY if the cell is in the soft zone
    (between PLANNING_CLEARANCE and SOFT_CLEARANCE), otherwise 0.
    Applied on top of the normal EDT centering cost.
    """
    d = float(edt[int(y), int(x)])
    if d < SOFT_CLEARANCE:   # PLANNING_CLEARANCE <= d < SOFT_CLEARANCE
        return SOFT_CLEARANCE_PENALTY
    return 0.0


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


def _build_bias_set(old_waypoints):
    """
    Return a set of (x, y) integer cells that are within PATH_BIAS_RADIUS of
    any waypoint on the previous path.  Cells in this set receive no extra
    penalty; cells outside it pay PATH_BIAS_PENALTY per step.
    """
    if not old_waypoints:
        return None
    radius_sq = PATH_BIAS_RADIUS ** 2
    bias = set()
    for wx, wy in old_waypoints:
        r = int(np.ceil(PATH_BIAS_RADIUS))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= radius_sq:
                    bias.add((wx + dx, wy + dy))
    return bias


# =============================================================================
# A* with EDT centering, hard clearance, soft clearance penalty,
# straight-line preference, and optional old-path bias for smooth replanning
# =============================================================================

NEIGHBORS = [
    (-1,  0, 1.0),   ( 1,  0, 1.0),   ( 0, -1, 1.0),   ( 0,  1, 1.0),
    (-1, -1, 1.414), (-1,  1, 1.414), ( 1, -1, 1.414),  ( 1,  1, 1.414),
]


def _astar(map_grid, start, goal, bias_set=None):
    """
    A* on the Zone_Map grid.

    Edge cost = move_distance
              + EDT_WEIGHT / (edt(neighbour) + 0.5)       ← corridor centering
              + SOFT_CLEARANCE_PENALTY (if EDT < 1.0m)    ← soft 1m margin
              + STRAIGHT_WEIGHT * direction_changed        ← straight-line bias
              + PATH_BIAS_PENALTY * (not in bias_set)      ← old-path continuity

    Hard block at 0.7m (PLANNING_CLEARANCE) — physically impassable.
    Soft penalty at 1.0m (SOFT_CLEARANCE) — passable but strongly discouraged.
    Falls back to closest reachable node if goal is unreachable.
    """
    edt  = _get_edt(map_grid)
    h, w = map_grid.shape
    use_bias = bias_set is not None

    def edt_cost(x, y):
        col = int(np.clip(x, 0, w - 1))
        row = int(np.clip(y, 0, h - 1))
        centering = EDT_WEIGHT / (float(edt[row, col]) + 0.5)
        soft      = _soft_penalty(edt, col, row)
        return centering + soft

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

            bias_cost = (0.0 if (not use_bias or neighbor in bias_set)
                         else PATH_BIAS_PENALTY)

            tentative_g = (gscore[current]
                           + move_cost
                           + edt_cost(nx, ny)
                           + direction_cost
                           + bias_cost)

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

def GenerateTrajectory(map_obj, current_state, desired_state,
                       old_waypoints=None):
    """
    A* path planner with EDT corridor-centering, hard + soft clearance,
    straight-line bias, and optional old-path continuity bias for replanning.

    Parameters
    ----------
    map_obj       : MapUpdater.Map instance  (.Zone_Map)
    current_state : array-like [x, y, ...]
    desired_state : array-like [x, y, ...]
    old_waypoints : list of (int x, int y) or None
        Previous A* path.  When provided, cells near the old path receive a
        lower edge cost so the replanned path stays close to the original
        where possible (used in chase-mode replanning for continuity).

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

    bias_set = _build_bias_set(old_waypoints) if old_waypoints else None
    path     = _astar(map_grid, start, goal, bias_set=bias_set)
    return [(int(x), int(y)) for x, y in path]