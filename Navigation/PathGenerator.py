import numpy as np
import heapq

# Library to interface with TrajectoryTracker -> High Level Function
def GenerateTrajectory(map, current_state, desired_state):
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                (-1,-1),(-1,1),(1,-1),(1,1)]
  
    desired_state = np.array(desired_state)
    current_state = tuple(np.array(current_state[0:2]).astype(int))
    desired_state = tuple(np.array(desired_state[0:2]).astype(int))

    width = map.shape[1]
    height = map.shape[0]

    close_set = set()
    came_from = {}
    gscore = {current_state: 0}
    fscore = {current_state: heuristic(current_state, desired_state)}
    open_heap = []
    heapq.heappush(open_heap, (fscore[current_state], current_state))

    best_node = current_state
    best_h = heuristic(current_state, desired_state)

    while open_heap:
        current = heapq.heappop(open_heap)[1]
        current_h = heuristic(current, desired_state)

        if(current_h < best_h):
            best_h = current_h
            best_node = current
        if(current == desired_state):
            best_node = desired_state
            break

        close_set.add(current)

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            x,y, = neighbor

            if x < 0 or x >= width or y < 0 or y >= height: 
                continue
            if map[y, x] == 1 or map[y,x] == 2:
                continue
            if neighbor in close_set: 
                continue
            # TODO ANY OTHER PRUNING TACTICS GOES HERE

            tentative_g = gscore[current] + np.hypot(dx, dy)
            if (map[y, x] >= 2):
                tentative_g = tentative_g + map[y,x]

            if neighbor not in gscore or tentative_g < gscore[neighbor]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g
                fscore[neighbor] = tentative_g + heuristic(neighbor, desired_state)
                heapq.heappush(open_heap, (fscore[neighbor], neighbor))

    if best_node not in came_from and best_node != current_state:
        return None
            
    path = [best_node]
    while best_node in came_from:
        best_node = came_from[best_node]
        path.append(best_node)
    
    path.reverse()
    path = [(int(x), int(y)) for (x, y) in path]
    return path

def heuristic(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

