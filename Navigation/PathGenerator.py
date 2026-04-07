import heapq
import numpy as np
from collections import deque
from scipy.ndimage import distance_transform_edt

# =============================================================================
# Config
# =============================================================================

EDT_WEIGHT      = 2.0   # how strongly A* favours corridor centres
STRAIGHT_WEIGHT = 0.3   # bonus cost for direction changes (favours straight lines)

# Hard block: cells within this distance of any static obstacle are completely
# impassable.  A* will never plan through here — guaranteed.
PLANNING_CLEARANCE = 1.0   # metres

# Soft penalty zone: cells between PLANNING_CLEARANCE (1.0m) and
# SOFT_CLEARANCE (2.0m) are passable but incur SOFT_CLEARANCE_PENALTY per
# step, strongly discouraging tight-margin routing while still allowing a path
# if no wider route exists.
SOFT_CLEARANCE         = 2.0   # metres
SOFT_CLEARANCE_PENALTY = 6.0   # flat extra cost per step inside soft zone

# When replanning in chase mode, cells near the previous A* path get a cost
# discount.  Cells farther than PATH_BIAS_RADIUS pay PATH_BIAS_PENALTY extra.
PATH_BIAS_RADIUS  = 1.5   # metres
PATH_BIAS_PENALTY = 1.2   # extra cost per step outside the old-path corridor

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
    True if cell (x, y) is out of bounds, is an obstacle cell, or is within
    PLANNING_CLEARANCE (1.0m) of a static obstacle.
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
    (PLANNING_CLEARANCE <= EDT < SOFT_CLEARANCE), otherwise 0.
    """
    d = float(edt[int(y), int(x)])
    if d < SOFT_CLEARANCE:
        return SOFT_CLEARANCE_PENALTY
    return 0.0


def _diagonal_cuts_corner(map_grid, edt, cx, cy, dx, dy):
    """True when a diagonal step would clip through an obstacle corner."""
    if abs(dx) == 1 and abs(dy) == 1:
        return (_is_blocked(map_grid, edt, cx + dx, cy) or
                _is_blocked(map_grid, edt, cx,       cy + dy))
    return False


def _nearest_free(map_grid, edt, cell):
    """
    BFS outward from cell until a free cell (EDT >= PLANNING_CLEARANCE) is found.
    Used for snapping the robot's start position if it's somehow in a tight spot.
    """
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


def _snap_goal_toward_robot(map_grid, edt, goal, robot_xy):
    """
    Snap a blocked goal to the nearest free cell along the robot→goal vector.

    Instead of a plain BFS (which finds the nearest free cell geometrically,
    potentially in any direction), this walks from the goal back toward the
    robot along the straight line until it finds a free cell.  This ensures
    the snapped goal is on the approach side of the obstacle — the side the
    robot is actually coming from — and avoids routing the robot around to the
    far side of a wall to reach a geometrically-near-but-approach-wrong cell.

    If the directional walk finds no free cell, falls back to BFS.
    """
    if not _is_blocked(map_grid, edt, *goal):
        return goal

    gx, gy   = float(goal[0]), float(goal[1])
    rx, ry   = float(robot_xy[0]), float(robot_xy[1])
    dx, dy   = rx - gx, ry - gy
    dist     = np.hypot(dx, dy)

    if dist < 1e-6:
        return _nearest_free(map_grid, edt, goal)

    # Step along the goal→robot direction in 0.5-cell increments
    ux, uy   = dx / dist, dy / dist
    steps    = int(dist / 0.5) + 2
    h, w     = map_grid.shape

    for i in range(1, steps + 1):
        cx = int(round(gx + ux * i * 0.5))
        cy = int(round(gy + uy * i * 0.5))
        cx = int(np.clip(cx, 0, w - 1))
        cy = int(np.clip(cy, 0, h - 1))
        if not _is_blocked(map_grid, edt, cx, cy):
            return (cx, cy)

    # Fallback: BFS
    return _nearest_free(map_grid, edt, goal)


def _build_bias_set(old_waypoints):
    """
    Return a set of (x, y) integer cells within PATH_BIAS_RADIUS of any
    waypoint on the previous path (for chase-mode replan continuity).
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
# A* with EDT centering, hard + soft clearance, straight-line preference,
# and optional old-path bias for smooth replanning
# =============================================================================

NEIGHBORS = [
    (-1,  0, 1.0),   ( 1,  0, 1.0),   ( 0, -1, 1.0),   ( 0,  1, 1.0),
    (-1, -1, 1.414), (-1,  1, 1.414), ( 1, -1, 1.414),  ( 1,  1, 1.414),
]


def _astar(map_grid, start, goal, bias_set=None):
    """
    A* on the Zone_Map grid.

    Edge cost = move_distance
              + EDT_WEIGHT / (edt(neighbour) + 0.5)          ← corridor centering
              + SOFT_CLEARANCE_PENALTY (if EDT < 2.0m)       ← soft 2m margin
              + STRAIGHT_WEIGHT * direction_changed           ← straight-line bias
              + PATH_BIAS_PENALTY * (not in bias_set)         ← old-path continuity

    Hard block at 1.0m — guaranteed by _is_blocked.
    Soft penalty between 1.0m and 2.0m — passable but heavily penalised.

    If the goal is unreachable (e.g. turkey inside an obstacle, goal snapped
    to the nearest clear cell on the approach side), A* terminates at
    best_node — the closest reachable cell to the goal.  The MPC will drive
    to that point, orient toward the turkey, and go idle cleanly without
    replanning into the same blocked goal.
    """
    edt  = _get_edt(map_grid)
    h, w = map_grid.shape
    use_bias = bias_set is not None

    def edt_cost(x, y):
        col = int(np.clip(x, 0, w - 1))
        row = int(np.clip(y, 0, h - 1))
        return EDT_WEIGHT / (float(edt[row, col]) + 0.5) + _soft_penalty(edt, col, row)

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

    # Reconstruct path from best_node (== goal if reachable, else closest cell)
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
    A* path planner with 1m hard clearance, EDT corridor-centering, soft
    2m margin, straight-line bias, and optional old-path continuity bias.

    When the goal (e.g. turkey position) is inside or too close to an obstacle,
    the goal is snapped toward the robot along the approach vector — ensuring
    the snapped endpoint is on the robot's side of the wall.  A* then plans
    to that snapped goal; if even that is unreachable it falls back to the
    closest reachable cell.  In both cases the MPC drives to the path end,
    orients toward the original target, and terminates cleanly.

    Parameters
    ----------
    map_obj       : MapUpdater.Map instance  (.Zone_Map)
    current_state : array-like [x, y, ...]
    desired_state : array-like [x, y, ...]  (may be inside an obstacle)
    old_waypoints : list of (int x, int y) or None

    Returns
    -------
    List of (int x, int y) waypoints, never empty (at minimum contains start).
    """
    map_grid = map_obj.Zone_Map
    h, w     = map_grid.shape
    _get_edt(map_grid)   # ensure cache is warm

    start = tuple(np.array(current_state[:2]).round().astype(int))
    goal  = tuple(np.array(desired_state[:2]).round().astype(int))

    start = (int(np.clip(start[0], 0, w-1)), int(np.clip(start[1], 0, h-1)))
    goal  = (int(np.clip(goal[0],  0, w-1)), int(np.clip(goal[1],  0, h-1)))

    # Snap start with plain BFS (robot should always be in free space)
    edt = _get_edt(map_grid)
    if _is_blocked(map_grid, edt, *start):
        start = _nearest_free(map_grid, edt, start)

    # Snap goal toward robot — keeps the endpoint on the approach side of any wall
    if _is_blocked(map_grid, edt, *goal):
        goal = _snap_goal_toward_robot(map_grid, edt, goal,
                                       robot_xy=(start[0], start[1]))

    bias_set = _build_bias_set(old_waypoints) if old_waypoints else None
    path     = _astar(map_grid, start, goal, bias_set=bias_set)
    return [(int(x), int(y)) for x, y in path]