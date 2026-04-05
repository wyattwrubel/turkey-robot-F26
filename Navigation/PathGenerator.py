import heapq
import numpy as np


def heuristic(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def GenerateTrajectory(map, current_state, desired_state):
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                 (-1,-1),(-1,1),(1,-1),(1,1)]

    start = tuple(np.array(current_state[0:2]).round().astype(int))
    goal  = tuple(np.array(desired_state[0:2]).round().astype(int))

    height, width = map.shape

    closed    = set()
    came_from = {}
    gscore    = {start: 0}
    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), start))

    best_node = start
    best_h    = heuristic(start, goal)

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            best_node = goal
            break

        closed.add(current)

        current_h = heuristic(current, goal)
        if current_h < best_h:
            best_h    = current_h
            best_node = current

        for dx, dy in neighbors:
            nx, ny = current[0] + dx, current[1] + dy

            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if map[ny, nx] == 1 or map[ny, nx] == 2:
                continue
            if (nx, ny) in closed:
                continue

            # Block diagonal moves through obstacle corners
            if dx != 0 and dy != 0:
                if map[current[1] + dy, current[0]] == 1 or \
                   map[current[1], current[0] + dx] == 1:
                    continue

            neighbor    = (nx, ny)
            tentative_g = gscore[current] + np.hypot(dx, dy)

            if neighbor not in gscore or tentative_g < gscore[neighbor]:
                came_from[neighbor] = current
                gscore[neighbor]    = tentative_g
                fscore              = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (fscore, neighbor))

    # Reconstruct path from best node reached
    path = []
    node = best_node
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.append(node)
    path.reverse()

    return [(int(x), int(y)) for x, y in path]