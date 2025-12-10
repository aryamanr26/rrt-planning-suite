import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
### YOUR IMPORTS HERE ###
from queue import PriorityQueue
#########################

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2doorway.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]
    print(base_joints, len(base_joints))

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, -1.3, -np.pi/2)))
    # print("Robot colliding? ", collision_fn((-3.4, -1.4, 0)))

    # Example use of setting body poses
    # set_pose(obstacles['ikeatable6'], ((0, 0, 0), (1, 0, 0, 0)))

    # Example of draw 
    # draw_sphere_marker((0, 0, 1), 0.1, (1, 0, 0, 1))
    
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    # print(start_config)
    goal_config = (2.6, -1.3, -np.pi/2) # (X, Y, Theta)
    path = []
    start_time = time.time()
    ### YOUR CODE HERE ###

    def get_neighbors_4connected(config):
        x, y, theta = config
        step = 0.15
        dtheta = np.pi / 2
        directions = [
            ( step, 0, 0),
            (-step, 0, 0),
            (0,  step, 0),
            (0, -step, 0),
            (0, 0,  dtheta),
            (0, 0, -dtheta)
        ]
        neighbors = []
        for dx, dy, dth in directions:
            new_config = (x + dx, y + dy, (theta + dth + np.pi) % (2*np.pi) - np.pi)
            line_start = (x, y, 0.2)           # z offset for visibility
            line_end   = (x + dx, y + dy, 0.2)
            if not collision_fn(new_config):
                neighbors.append(new_config)
                draw_line(line_start, line_end, width=1, color=(0, 0, 1))
            else:
                draw_line(line_start, line_end, width=1, color=(1, 0, 0))

        return neighbors

    def get_neighbors_8connected(config):
        x, y, theta = config
        step = 0.15  # translational step in meters
        dtheta = np.pi / 2  # small rotation (optional if you want to allow turning)

        # 8 directions: N, S, E, W, NE, NW, SE, SW
        directions = [
            ( step,  0, 0),   # +x
            (-step,  0, 0),   # -x
            ( 0,  step, 0),   # +y
            ( 0, -step, 0),   # -y
            ( step,  step, 0),  # +x, +y
            ( step, -step, 0),  # +x, -y
            (-step,  step, 0),  # -x, +y
            (-step, -step, 0),   # -x, -y
            (0, 0, dtheta),
            (0, 0, -dtheta),
        ]

        neighbors = []
        for dx, dy, dth in directions:
            new_config = (x + dx, y + dy, (theta + dth + np.pi) % (2*np.pi) - np.pi)
            line_start = (x, y, 0.2)           # z offset for visibility
            line_end   = (x + dx, y + dy, 0.2)

            if collision_fn(new_config):
                # Colliding configuration → draw RED
                draw_line(line_start, line_end, width=1, color=(1, 0, 0))
            else:
                # Collision-free configuration → draw BLUE
                draw_line(line_start, line_end, width=1, color=(0, 0, 1))
                neighbors.append(new_config)
        
        return neighbors

    
    def heuristic(pos, goal) -> float:
        return np.sqrt((pos[0]-goal[0])**2 + (pos[1]-goal[1])**2 + min(abs(pos[2] - goal[2]), 2*np.pi - abs(pos[2] - goal[2])))
    
    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    print("Start colliding?", collision_fn(start_config))
    print("Goal colliding?", collision_fn(goal_config))

    g_score = dict()
    f_score = dict()
    came_from = dict()

    g_score[start_config] = 0
    f_score[start_config] = heuristic(start_config, goal_config)

    open_set = PriorityQueue()
    open_set.put((f_score[start_config], start_config))

    action_cost = 0
    
    while not open_set.empty():
        _, current = open_set.get()
        # print("Current : ", current)

        if heuristic(current, goal_config) < 0.1:
            print("Goal reached!")
            path = reconstruct_path(came_from, current)
            for i in range(len(path) - 1):
                x1, y1, _ = path[i]
                x2, y2, _ = path[i + 1]
                line_start = (x1, y1, 1.0)
                line_end   = (x2, y2, 1.0)
                draw_line(line_start, line_end, width=2, color=(0, 0, 0))
            break

        for neighbour in get_neighbors_4connected(current):
            # print("Neighbor:", neighbour)
            temp_g = g_score[current] + heuristic(current, neighbour)
            if neighbour not in g_score or temp_g < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = temp_g
                f_score[neighbour] = g_score[neighbour] + heuristic(neighbour, goal_config)
                
                # if neighbour not in open_set:
                open_set.put((f_score[neighbour], neighbour))

    def action_cost(start_config: tuple, path: list)-> float:
        total = 0
        current = start_config
        for neighbour in path:
            action = heuristic(current,neighbour)
            total += action
            current = neighbour
        return total
    
    print("Path Travelled: ", path)
    print("Action Cost of the path: ", action_cost(start_config, path))
    print("Length of the path: ", len(path))
    ######################
    print("Planner run time: ", time.time() - start_time)
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()