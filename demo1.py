import numpy as np
from utils import load_env, get_collision_fn_PR2, execute_trajectory
from pybullet_tools.utils import connect, disconnect, wait_if_gui, joint_from_name, get_joint_positions, set_joint_positions, get_joint_info, get_link_pose, link_from_name
import random
import time
from utils import draw_sphere_marker
import heapq

joint_names = ('l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_elbow_flex_joint', 
               'l_upper_arm_roll_joint', 'l_forearm_roll_joint', 'l_wrist_flex_joint')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_nearest_node(tree, q_sample):
    """Find nearest node in tree to sample"""
    min_dist = float('inf')
    nearest_node = None
    for node in tree:
        dist = np.linalg.norm(np.array(node) - np.array(q_sample))
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    return nearest_node

def steer(q_from, q_to, step_size):
    """Steer from q_from toward q_to with maximum step_size"""
    q_from = np.array(q_from)
    q_to = np.array(q_to)
    dist = np.linalg.norm(q_to - q_from)
    
    if dist < step_size:
        return tuple(float(x) for x in q_to)
    
    vec = q_to - q_from
    unit = vec / dist
    q_new = q_from + unit * step_size
    return tuple(float(x) for x in q_new)

def sample_random_node(goal, limits, bias):
    """Sample random configuration with goal bias"""
    if random.random() < bias:
        return goal
    sample = []
    for (min_val, max_val) in limits:
        sample.append(random.uniform(min_val, max_val))
    return tuple(sample)

def distance(q1, q2):
    """Euclidean distance between configurations"""
    return np.linalg.norm(np.array(q1) - np.array(q2))

def reconstruct_path(parents, goal):
    """Reconstruct path from start to goal using parent dict"""
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = parents.get(curr)
    return path[::-1]

def is_path_clear(q1, q2, collision_fn, step_size=0.05):
    """Check if straight-line path between q1 and q2 is collision-free"""
    dist = np.linalg.norm(np.array(q1) - np.array(q2))
    if dist == 0:
        return True
    
    num_steps = int(np.ceil(dist / step_size))
    q1_arr = np.array(q1)
    q2_arr = np.array(q2)
    
    for i in range(1, num_steps + 1):
        t = i / num_steps
        q_interp = tuple(q1_arr + t * (q2_arr - q1_arr))
        if collision_fn(q_interp):
            return False
    return True

# ============================================================================
# ALGORITHM 1: RRT-CONNECT
# ============================================================================

def RRTConnect(start, goal, limits, collision_fn, step_size=0.05, goal_bias=0.25, max_iter=3000):
    """RRT-Connect: Bidirectional RRT with aggressive extension"""
    T_start = {start}
    T_goal = {goal}
    parents_A = {start: None}
    parents_B = {goal: None}

    for iteration in range(max_iter):
        q_rand = sample_random_node(goal, limits, goal_bias)
        
        q_near_start = get_nearest_node(T_start, q_rand)
        q_new_start = steer(q_near_start, q_rand, step_size)
        
        if not collision_fn(q_new_start):
            T_start.add(q_new_start)
            parents_A[q_new_start] = q_near_start
            
            q_near_goal = get_nearest_node(T_goal, q_new_start)
            q_new_goal = steer(q_near_goal, q_new_start, step_size)
            
            if not collision_fn(q_new_goal):
                T_goal.add(q_new_goal)
                parents_B[q_new_goal] = q_near_goal
                
                if np.allclose(q_new_goal, q_new_start, atol=1e-5):
                    print(f"RRT-Connect: Connected at iteration {iteration}")
                    parents_B[q_new_start] = q_near_goal
                    
                    # Reconstruct path
                    path_start = []
                    curr = q_new_start
                    while curr:
                        path_start.append(curr)
                        curr = parents_A[curr]
                    
                    path_goal = []
                    curr = parents_B[q_new_start]
                    while curr:
                        path_goal.append(curr)
                        curr = parents_B[curr]
                    
                    return path_start[::-1] + path_goal
        
        T_start, T_goal = T_goal, T_start
        parents_A, parents_B = parents_B, parents_A
        start, goal = goal, start
    
    print("RRT-Connect: Failed to find a path")
    return None

# ============================================================================
# ALGORITHM 2: RRT* (Optimal RRT)
# ============================================================================

def RRTStar(start, goal, limits, collision_fn, step_size=0.05, goal_bias=0.25,
            max_iter=6000, neighbor_radius=1.2, goal_threshold=0.25):

    tree = {start}
    parents = {start: None}
    costs = {start: 0.0}

    for iteration in range(max_iter):

        # Force goal extension
        if iteration % 20 == 0:
            q_rand = goal
        else:
            q_rand = sample_random_node(goal, limits, goal_bias)

        q_nearest = get_nearest_node(tree, q_rand)
        q_new = steer(q_nearest, q_rand, step_size)

        if collision_fn(q_new):
            continue

        # Neighbor radius is larger now
        neighbors = [n for n in tree if distance(n, q_new) < neighbor_radius]

        best_parent = None
        best_cost = float('inf')

        for n in neighbors:
            c = costs[n] + distance(n, q_new)
            if c < best_cost and is_path_clear(n, q_new, collision_fn, step_size):
                best_parent = n
                best_cost = c

        if best_parent is None:
            best_parent = q_nearest
            best_cost = costs[q_nearest] + distance(q_nearest, q_new)

        tree.add(q_new)
        parents[q_new] = best_parent
        costs[q_new] = best_cost

        # Rewire neighbors
        for n in neighbors:
            new_cost = costs[q_new] + distance(q_new, n)
            if new_cost < costs[n] and is_path_clear(q_new, n, collision_fn, step_size):
                parents[n] = q_new
                costs[n] = new_cost

        # Goal reached
        if distance(q_new, goal) < goal_threshold:
            parents[goal] = q_new
            return reconstruct_path(parents, goal)

    return None


# ============================================================================
# ALGORITHM 3: BIDIRECTIONAL RRT*
# ============================================================================

def BiRRTStar(start, goal, limits, collision_fn, step_size=0.05, goal_bias=0.2,
              max_iter=8000, neighbor_radius=1.0):
    """
    Bidirectional RRT* with stronger connections between the trees.
    """
    T_start = {start}
    T_goal = {goal}
    parents_start = {start: None}
    parents_goal = {goal: None}
    costs_start = {start: 0.0}
    costs_goal = {goal: 0.0}

    best_path = None
    best_cost = float('inf')

    for iteration in range(max_iter):
        # Expand start tree
        if iteration % 25 == 0:
            q_rand = goal
        else:
            q_rand = sample_random_node(goal, limits, goal_bias)

        q_nearest = get_nearest_node(T_start, q_rand)
        q_new = steer(q_nearest, q_rand, step_size)

        if not collision_fn(q_new):
            neighbors = [n for n in T_start if distance(n, q_new) < neighbor_radius]

            best_parent = None
            best_cost_here = float('inf')
            for n in neighbors:
                c = costs_start[n] + distance(n, q_new)
                if c < best_cost_here and is_path_clear(n, q_new, collision_fn, step_size=0.05):
                    best_parent = n
                    best_cost_here = c

            if best_parent is None:
                best_parent = q_nearest
                best_cost_here = costs_start[q_nearest] + distance(q_nearest, q_new)

            T_start.add(q_new)
            parents_start[q_new] = best_parent
            costs_start[q_new] = best_cost_here

            # Rewire
            for n in neighbors:
                new_cost = costs_start[q_new] + distance(q_new, n)
                if new_cost < costs_start[n] and is_path_clear(q_new, n, collision_fn, step_size=0.05):
                    parents_start[n] = q_new
                    costs_start[n] = new_cost

            # Try to connect to goal tree
            q_near_goal = get_nearest_node(T_goal, q_new)
            if is_path_clear(q_new, q_near_goal, collision_fn, step_size=0.05):
                total_cost = (costs_start[q_new] + distance(q_new, q_near_goal) +
                              costs_goal[q_near_goal])
                if total_cost < best_cost:
                    best_cost = total_cost

                    # Reconstruct path start → connection
                    path_start = []
                    curr = q_new
                    while curr is not None:
                        path_start.append(curr)
                        curr = parents_start[curr]

                    # Reconstruct path connection → goal
                    path_goal = []
                    curr = q_near_goal
                    while curr is not None:
                        path_goal.append(curr)
                        curr = parents_goal[curr]

                    best_path = path_start[::-1] + path_goal
                    print(f"BiRRT*: Found path with cost {best_cost:.3f} at iteration {iteration}")

        # Swap roles of the trees
        T_start, T_goal = T_goal, T_start
        parents_start, parents_goal = parents_goal, parents_start
        costs_start, costs_goal = costs_goal, costs_start
        start, goal = goal, start

    if best_path:
        print(f"BiRRT*: Returning best path with cost {best_cost:.3f}")
        return best_path

    print("BiRRT*: Failed to find a path")
    return None

# ============================================================================
# ALGORITHM 4: INFORMED RRT*
# ============================================================================

def sample_informed(start, goal, c_best, c_min, limits):
    """
    Simple informed sampler: sample within joint limits but reject if outside
    the prolate ellipsoid defined by start, goal and c_best.
    """
    if c_best == float('inf'):
        return sample_random_node(goal, limits, 0.0)

    max_attempts = 50
    for _ in range(max_attempts):
        q_rand = sample_random_node(goal, limits, 0.0)
        if distance(start, q_rand) + distance(q_rand, goal) <= c_best:
            return q_rand

    # Fallback if rejection sampling fails
    return sample_random_node(goal, limits, 0.0)


def InformedRRTStar(start, goal, limits, collision_fn, step_size=0.05, goal_bias=0.3,
                    max_iter=8000, neighbor_radius=1.0, goal_threshold=0.25):
    """
    Informed RRT*: behaves like RRT* until an initial solution is found,
    then focuses sampling within an ellipsoid between start and goal.
    """
    tree = {start}
    parents = {start: None}
    costs = {start: 0.0}

    best_path = None
    best_cost = float('inf')

    c_min = distance(start, goal)  # direct line distance

    for iteration in range(max_iter):
        # Sampling strategy
        if best_path is None:
            # No solution yet -> behave like RRT*
            if iteration % 25 == 0:
                q_rand = goal
            else:
                q_rand = sample_random_node(goal, limits, goal_bias)
        else:
            # We have a solution -> mostly informed sampling
            if random.random() < 0.3:
                q_rand = sample_random_node(goal, limits, goal_bias)
            else:
                q_rand = sample_informed(start, goal, best_cost, c_min, limits)

        q_nearest = get_nearest_node(tree, q_rand)
        q_new = steer(q_nearest, q_rand, step_size)

        if collision_fn(q_new):
            continue

        neighbors = [n for n in tree if distance(n, q_new) < neighbor_radius]

        best_parent = None
        best_cost_here = float('inf')
        for n in neighbors:
            c = costs[n] + distance(n, q_new)
            if c < best_cost_here and is_path_clear(n, q_new, collision_fn, step_size=0.05):
                best_parent = n
                best_cost_here = c

        if best_parent is None:
            best_parent = q_nearest
            best_cost_here = costs[q_nearest] + distance(q_nearest, q_new)

        tree.add(q_new)
        parents[q_new] = best_parent
        costs[q_new] = best_cost_here

        # Rewire neighbors
        for n in neighbors:
            new_cost = costs[q_new] + distance(q_new, n)
            if new_cost < costs[n] and is_path_clear(q_new, n, collision_fn, step_size=0.05):
                parents[n] = q_new
                costs[n] = new_cost

        # Try to connect to goal
        if distance(q_new, goal) < goal_threshold and is_path_clear(q_new, goal, collision_fn, step_size=0.05):
            total_cost = costs[q_new] + distance(q_new, goal)
            if total_cost < best_cost:
                parents[goal] = q_new
                costs[goal] = total_cost
                best_cost = total_cost
                best_path = reconstruct_path(parents, goal)
                print(f"Informed RRT*: Improved path, cost={best_cost:.3f} at iteration {iteration}")

    if best_path:
        return best_path

    print("Informed RRT*: Failed to find a path")
    return None

# ============================================================================
# ALGORITHM 5: RRT WITH GOAL REGION
# ============================================================================

def RRT(start, goal, limits, collision_fn, step_size=0.05, goal_bias=0.25, 
        max_iter=4000, goal_threshold=0.25):

    tree = {start}
    parents = {start: None}

    for iteration in range(max_iter):

        # Force goal extension sometimes
        if iteration % 20 == 0:
            q_rand = goal
        else:
            q_rand = sample_random_node(goal, limits, goal_bias)

        q_nearest = get_nearest_node(tree, q_rand)
        q_new = steer(q_nearest, q_rand, step_size)

        if not collision_fn(q_new):
            tree.add(q_new)
            parents[q_new] = q_nearest

            if distance(q_new, goal) < goal_threshold:
                parents[goal] = q_new
                return reconstruct_path(parents, goal)

    return None

# ============================================================================
# SMOOTHING
# ============================================================================

def shortcut_smooth(path, collision_fn, iterations, step_size=0.05):
    """Shortcut smoothing algorithm"""
    if path is None or len(path) <= 2:
        return path
    
    smoothed_path = list(path)
    for _ in range(iterations):
        if len(smoothed_path) <= 2:
            break
        
        idx1 = random.randint(0, len(smoothed_path) - 2)
        idx2 = random.randint(idx1 + 1, len(smoothed_path) - 1)
        
        q_a = smoothed_path[idx1]
        q_b = smoothed_path[idx2]
        
        if is_path_clear(q_a, q_b, collision_fn, step_size):
            smoothed_path = smoothed_path[:idx1+1] + smoothed_path[idx2:]
    
    return smoothed_path

# ============================================================================
# VISUALIZATION
# ============================================================================

def get_ee_path(robot, joints, link_id, path, interp_step_size=0.01):
    """Get end-effector trajectory from joint space path"""
    ee_path = []
    if not path:
        return ee_path
    
    current_config = get_joint_positions(robot, joints)
    
    set_joint_positions(robot, joints, path[0])
    ee_path.append(get_link_pose(robot, link_id)[0])
    
    for i in range(len(path) - 1):
        q1 = np.array(path[i])
        q2 = np.array(path[i+1])
        dist = np.linalg.norm(q2 - q1)
        
        if dist == 0:
            continue
        
        num_steps = int(np.ceil(dist / interp_step_size))
        for j in range(1, num_steps + 1):
            t = j / num_steps
            q_interp = tuple(q1 + t * (q2 - q1))
            set_joint_positions(robot, joints, q_interp)
            ee_path.append(get_link_pose(robot, link_id)[0])
    
    set_joint_positions(robot, joints, current_config)
    return ee_path

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(screenshot=False):
    connect(use_gui=True)
    robots, obstacles = load_env('pr2table.json')
    
    joint_idx = [joint_from_name(robots['pr2'], jn) for jn in joint_names]
    joint_limits = {joint_names[i]: (get_joint_info(robots['pr2'], joint_idx[i]).jointLowerLimit,
                                      get_joint_info(robots['pr2'], joint_idx[i]).jointUpperLimit)
                    for i in range(len(joint_idx))}
    
    collision_fn = get_collision_fn_PR2(robots['pr2'], joint_idx, list(obstacles.values()))
    
    start_config = tuple(float(x) for x in get_joint_positions(robots['pr2'], joint_idx))
    goal_config = tuple(float(x) for x in (0.5, 0.33, -1.548, 1.557, -1.32, -0.1928))
    
    joint_limits_list = [joint_limits[jn] for jn in joint_names]
    
    # ========================================================================
    # CHOOSE YOUR ALGORITHM HERE - Just change this one line!
    # ========================================================================
    
    print("Running motion planning algorithm...")
    start_time = time.time()
    
    # Option 1: RRT-Connect (Fast, bidirectional) works 
    # path = RRTConnect(start_config, goal_config, joint_limits_list, collision_fn)
    
    # Option 2: RRT* (Optimal, slower) not good
    # path = RRTStar(start_config, goal_config, joint_limits_list, collision_fn)
    
    # Option 3: Bidirectional RRT* (Optimal, bidirectional) works
    # path = BiRRTStar(start_config, goal_config, joint_limits_list, collision_fn)
    
    # Option 4: Informed RRT* (Optimal with informed sampling) not works 
    path = InformedRRTStar(start_config, goal_config, joint_limits_list, collision_fn)
    
    # Option 5: Standard RRT (Basic, single tree) not works
    # path = RRT(start_config, goal_config, joint_limits_list, collision_fn)
    
    # ========================================================================
    
    print(f"Planning finished in {time.time() - start_time:.2f}s")
    
    if path:
        print(f"Path found with {len(path)} waypoints.")
        
        # Smoothing
        smooth_iterations = 150
        print(f"Running smoothing ({smooth_iterations} iterations)...")
        start_time = time.time()
        smoothed_path = shortcut_smooth(path, collision_fn, smooth_iterations)
        print(f"Smoothing finished in {time.time() - start_time:.2f}s")
        print(f"Smoothed path has {len(smoothed_path)} waypoints.")
        
        # Visualization
        print("Drawing paths...")
        ee_link_id = link_from_name(robots['pr2'], 'l_gripper_tool_frame')
        
        ee_path_original = get_ee_path(robots['pr2'], joint_idx, ee_link_id, path, interp_step_size=0.01)
        for p in ee_path_original:
            draw_sphere_marker(p, 0.005, (1, 0, 0, 0.8))  # Red
        
        ee_path_smooth = get_ee_path(robots['pr2'], joint_idx, ee_link_id, smoothed_path, interp_step_size=0.01)
        for p in ee_path_smooth:
            draw_sphere_marker(p, 0.007, (0, 0, 1, 0.8))  # Blue
        
        path = smoothed_path
    else:
        print("No path found.")
        path = []
    
    execute_trajectory(robots['pr2'], joint_idx, path, sleep=0.1)
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()