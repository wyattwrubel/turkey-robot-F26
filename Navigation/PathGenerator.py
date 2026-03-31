import numpy as np
import heapq

def heuristic(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

_prev_path_set = set()  # nodes from the last plan, used to bias replanning

def GenerateTrajectory(map, current_state, desired_state):
    global _prev_path_set

    neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                 (-1,-1),(-1,1),(1,-1),(1,1)]

    CONSISTENCY_BONUS = 0.6   # cost discount for staying on previous path
                               # higher = smoother but slower to replan around obstacles

    desired_state = np.array(desired_state)
    current_state = tuple(np.array(current_state[0:2]).round().astype(int))
    desired_state = tuple(np.array(desired_state[0:2]).round().astype(int))

    width  = map.shape[1]
    height = map.shape[0]

    close_set = set()
    came_from = {}
    gscore    = {current_state: 0}
    fscore    = {current_state: heuristic(current_state, desired_state)}
    direction = {current_state: None}
    open_heap = []
    heapq.heappush(open_heap, (fscore[current_state], current_state))

    best_node = current_state
    best_h    = heuristic(current_state, desired_state)

    while open_heap:
        current   = heapq.heappop(open_heap)[1]
        current_h = heuristic(current, desired_state)

        if current_h < best_h:
            best_h    = current_h
            best_node = current
        if current == desired_state:
            best_node = desired_state
            break

        close_set.add(current)
        current_dir = direction[current]

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            x, y     = neighbor

            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            if map[y, x] == 1 or map[y, x] == 2:
                continue
            if neighbor in close_set:
                continue

            if dx != 0 and dy != 0:
                if map[current[1] + dy, current[0]] == 1 or \
                   map[current[1], current[0] + dx] == 1:
                    continue

            tentative_g = gscore[current] + np.hypot(dx, dy)

            # Penalise direction changes to prefer straight lines
            new_dir = (dx, dy)
            if current_dir is not None and new_dir != current_dir:
                tentative_g += 0.3
                curr_is_diag = (current_dir[0] != 0 and current_dir[1] != 0)
                new_is_diag  = (dx != 0 and dy != 0)
                if curr_is_diag != new_is_diag:
                    tentative_g += 0.5

            # Consistency bonus — prefer nodes that were on the previous path
            if neighbor in _prev_path_set:
                tentative_g -= CONSISTENCY_BONUS

            if neighbor not in gscore or tentative_g < gscore[neighbor]:
                came_from[neighbor] = current
                gscore[neighbor]    = tentative_g
                fscore[neighbor]    = tentative_g + heuristic(neighbor, desired_state)
                direction[neighbor] = new_dir
                heapq.heappush(open_heap, (fscore[neighbor], neighbor))

    if best_node not in came_from and best_node != current_state:
        _prev_path_set = set()
        return None

    path = [best_node]
    while best_node in came_from:
        best_node = came_from[best_node]
        path.append(best_node)

    path.reverse()
    path = [(int(x), int(y)) for (x, y) in path]

    # Store this plan for next call
    _prev_path_set = set(path)

    return path