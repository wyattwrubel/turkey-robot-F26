import heapq
import numpy as np
from collections import deque
from scipy.ndimage import distance_transform_edt

# =============================================================================
# Config
# =============================================================================

EDT_WEIGHT      = 2.0   # how strongly A* favours corridor centres
STRAIGHT_WEIGHT = 0.3   # bonus cost for direction changes (favours straight lines)

# Soft penalty zone: cells within SOFT_CLEARANCE metres of a static obstacle
# incur a small extra cost per step.
SOFT_CLEARANCE         = 3.0   # metres — zone within which penalty applies
SOFT_CLEARANCE_PENALTY = 1.5   # additive cost per step

PLANNING_CLEARANCE = 0.0

# Old-path bias on replan.
PATH_BIAS_RADIUS  = 2.5   # metres
PATH_BIAS_PENALTY = 1.5   # max extra cost per step at the corridor edge

# Turkey proximity early-exit: A* terminates as soon as a node is within this
# distance of the nearest turkey (closest to the robot), so the robot stops
# in front of the turkey rather than on top of it.
TURKEY_PROXIMITY_DIST = 4.0   # metres

# =============================================================================
# EDT cache — recomputed only when the static obstacle layer changes
# =============================================================================

_edt_cache   = None
_static_hash = None


def _get_edt(map_grid):
    """
    Return the EDT of static-obstacle-free space.
    Cached — only recomputed when the static layer (cells == 1 or -1) changes.
    """
    global _edt_cache, _static_hash

    static_mask = ((map_grid == 1) | (map_grid == -1)).astype(np.uint8)
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
    True if cell (x, y) is out of bounds, a static obstacle (1 or -1),
    or a dynamic obstacle (2).
    """
    h, w = map_grid.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return True
    cell = map_grid[y, x]
    return cell == 1 or cell == -1 or cell == 2


def _soft_penalty(edt, x, y):
    """
    Extra cost for cells near static obstacles, fading linearly from full
    penalty at the wall face to zero at SOFT_CLEARANCE.
    """
    d = float(edt[int(y), int(x)])
    if d >= SOFT_CLEARANCE:
        return 0.0
    if SOFT_CLEARANCE <= 0:
        return 0.0
    t = (SOFT_CLEARANCE - d) / SOFT_CLEARANCE
    return SOFT_CLEARANCE_PENALTY * float(np.clip(t, 0.0, 1.0))


def _diagonal_cuts_corner(map_grid, edt, cx, cy, dx, dy):
    """True when a diagonal step would clip through an obstacle corner."""
    if abs(dx) == 1 and abs(dy) == 1:
        return (_is_blocked(map_grid, edt, cx + dx, cy) or
                _is_blocked(map_grid, edt, cx,       cy + dy))
    return False


def _nearest_free(map_grid, edt, cell):
    """
    BFS outward from cell until a non-obstacle cell is found.
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


def _snap_goal_to_closest_free(map_grid, edt, goal, verbose=False):
    """
    Find the free cell (non-obstacle) closest to `goal` by Euclidean distance.
    """
    if not _is_blocked(map_grid, edt, *goal):
        return goal

    h, w        = map_grid.shape
    gx, gy      = goal
    max_radius  = 8
    q           = deque([(gx, gy, 0)])
    seen        = {(gx, gy)}
    candidates  = []
    found_depth = None

    while q:
        cx, cy, depth = q.popleft()

        if found_depth is not None and depth > found_depth + 1:
            break
        if depth > max_radius:
            break

        if not _is_blocked(map_grid, edt, cx, cy):
            dist = np.hypot(cx - gx, cy - gy)
            candidates.append((dist, (cx, cy)))
            if found_depth is None:
                found_depth = depth

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in seen and 0 <= nx < w and 0 <= ny < h:
                seen.add((nx, ny))
                q.append((nx, ny, depth + 1))

    if not candidates:
        if verbose:
            print(f"[PathGenerator] No free cell found near {goal} within radius {max_radius}")
        return _nearest_free(map_grid, edt, goal)

    best = min(candidates, key=lambda c: c[0])[1]
    if verbose and best != goal:
        print(f"[PathGenerator] Goal {goal} blocked — snapped to {best} "
              f"(dist={np.hypot(best[0]-gx, best[1]-gy):.2f}m)")
    return best


def _build_bias_map(old_waypoints):
    """
    Return a dict (x, y) → penalty for cells near the previous A* path.
    """
    if not old_waypoints:
        return None
    r    = int(np.ceil(PATH_BIAS_RADIUS))
    r_sq = PATH_BIAS_RADIUS ** 2
    bias = {}
    for wx, wy in old_waypoints:
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                dist_sq = dx * dx + dy * dy
                if dist_sq <= r_sq:
                    dist    = np.sqrt(dist_sq)
                    penalty = PATH_BIAS_PENALTY * (dist / PATH_BIAS_RADIUS)
                    cell    = (wx + dx, wy + dy)
                    if cell not in bias or penalty < bias[cell]:
                        bias[cell] = penalty
    return bias


# =============================================================================
# A* with EDT centering, soft clearance, straight-line preference,
# optional old-path bias, and optional turkey proximity early-exit
# =============================================================================

NEIGHBORS = [
    (-1,  0, 1.0),   ( 1,  0, 1.0),   ( 0, -1, 1.0),   ( 0,  1, 1.0),
    (-1, -1, 1.414), (-1,  1, 1.414), ( 1, -1, 1.414),  ( 1,  1, 1.414),
]


def _astar(map_grid, start, goal, bias_map=None,
           turkey_target_xy=None, verbose=False):
    """
    A* on the Zone_Map grid.

    Edge cost = move_distance
              + EDT_WEIGHT / (edt(neighbour) + 0.5)
              + _soft_penalty(edt, x, y)
              + STRAIGHT_WEIGHT * direction_changed
              + bias_map.get(neighbour, PATH_BIAS_PENALTY)

    turkey_target_xy : array-like [x, y] or None
        If provided, A* terminates early as soon as the current node is within
        TURKEY_PROXIMITY_DIST metres of this position (Euclidean).  This stops
        the robot in front of the turkey rather than planning all the way to
        its exact grid cell.  The early-exit uses Euclidean distance only —
        no line-of-sight check.

    If the goal is unreachable A* terminates at the closest reachable cell.
    """
    edt      = _get_edt(map_grid)
    h, w     = map_grid.shape
    use_bias = bias_map is not None

    # Pre-compute turkey proximity check components.
    use_turkey_exit = turkey_target_xy is not None
    if use_turkey_exit:
        tx = float(turkey_target_xy[0])
        ty = float(turkey_target_xy[1])
        turkey_dist_sq = TURKEY_PROXIMITY_DIST ** 2

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

        # ── Standard goal check ───────────────────────────────────────────────
        if current == goal:
            best_node = goal
            break

        # ── Turkey proximity early-exit ───────────────────────────────────────
        # Terminate as soon as we pop a node within TURKEY_PROXIMITY_DIST of
        # the turkey.  This is checked after popping (not on push) so the path
        # to this node is guaranteed to be the lowest-cost path reaching it.
        if use_turkey_exit:
            dx_t = current[0] - tx
            dy_t = current[1] - ty
            if dx_t * dx_t + dy_t * dy_t <= turkey_dist_sq:
                best_node = current
                if verbose:
                    dist_to_turkey = np.hypot(dx_t, dy_t)
                    print(f"[PathGenerator] Turkey proximity exit at {current} "
                          f"(dist={dist_to_turkey:.2f}m <= {TURKEY_PROXIMITY_DIST}m)")
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

            bias_cost = (bias_map.get(neighbor, PATH_BIAS_PENALTY)
                         if use_bias else 0.0)

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

    if verbose and best_node != goal and not (use_turkey_exit and best_node != start):
        print(f"[PathGenerator] Goal {goal} unreachable — terminating at {best_node}")

    # Reconstruct path from best_node
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
                       old_waypoints=None, turkey_target_xy=None,
                       verbose=False):
    """
    A* path planner with optional turkey proximity early-exit.

    Parameters
    ----------
    map_obj          : MapUpdater.Map instance  (.Zone_Map)
    current_state    : array-like [x, y, ...]
    desired_state    : array-like [x, y, ...]  (may be inside an obstacle)
    old_waypoints    : list of (int x, int y) or None
    turkey_target_xy : array-like [x, y] or None
        When provided (i.e. the target is a turkey), A* exits as soon as it
        reaches a node within TURKEY_PROXIMITY_DIST metres of this position.
        Should be the turkey closest to the robot (in world coordinates).
    verbose          : bool

    Returns
    -------
    List of (int x, int y) waypoints, never empty.
    """
    map_grid = map_obj.Zone_Map
    h, w     = map_grid.shape
    _get_edt(map_grid)   # warm the cache

    start = tuple(np.array(current_state[:2]).round().astype(int))
    goal  = tuple(np.array(desired_state[:2]).round().astype(int))

    start = (int(np.clip(start[0], 0, w-1)), int(np.clip(start[1], 0, h-1)))
    goal  = (int(np.clip(goal[0],  0, w-1)), int(np.clip(goal[1],  0, h-1)))

    edt = _get_edt(map_grid)

    if _is_blocked(map_grid, edt, *start):
        snapped_start = _nearest_free(map_grid, edt, start)
        if verbose:
            print(f"[PathGenerator] Start {start} blocked — snapped to {snapped_start}")
        start = snapped_start

    if _is_blocked(map_grid, edt, *goal):
        goal = _snap_goal_to_closest_free(map_grid, edt, goal, verbose=verbose)

    bias_map = _build_bias_map(old_waypoints) if old_waypoints else None
    path     = _astar(map_grid, start, goal, bias_map=bias_map,
                      turkey_target_xy=turkey_target_xy, verbose=verbose)

    if verbose:
        print(f"[PathGenerator] Path: {len(path)} waypoints, "
              f"start={path[0]}, end={path[-1]}")

    return [(int(x), int(y)) for x, y in path]