import numpy as np
from utils import load_env, get_collision_fn_PR2, execute_trajectory
from pybullet_tools.utils import connect, disconnect, wait_if_gui, joint_from_name, get_joint_positions, set_joint_positions, get_joint_info, get_link_pose, link_from_name
import random
### YOUR IMPORTS HERE ###
import time
from utils import draw_sphere_marker
#########################


joint_names =('l_shoulder_pan_joint','l_shoulder_lift_joint','l_elbow_flex_joint','l_upper_arm_roll_joint','l_forearm_roll_joint','l_wrist_flex_joint')

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2table.json')

    # define active DoFs
    joint_names =('l_shoulder_pan_joint','l_shoulder_lift_joint','l_elbow_flex_joint','l_upper_arm_roll_joint','l_forearm_roll_joint','l_wrist_flex_joint')
    joint_idx = [joint_from_name(robots['pr2'], jn) for jn in joint_names]

    # parse active DoF joint limits
    joint_limits = {joint_names[i] : (get_joint_info(robots['pr2'], joint_idx[i]).jointLowerLimit, get_joint_info(robots['pr2'], joint_idx[i]).jointUpperLimit) for i in range(len(joint_idx))}

    collision_fn = get_collision_fn_PR2(robots['pr2'], joint_idx, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, 1.19, -1.548, 1.557, -1.32, -0.1928)))

    start_config = tuple(get_joint_positions(robots['pr2'], joint_idx))
    goal_config = (0.5, 0.33, -1.548, 1.557, -1.32, -0.1928)
    path = []
    ### YOUR CODE HERE ###
    
    # --- RRT Parameters ---
    STEP_SIZE = 0.05 #   0.15 for BiRRT*
    GOAL_BIAS = 0.1 #   0.3 for BiRRT*
    MAX_ITERATIONS = 5000
    SMOOTH_ITERATIONS = 200 # 

    # Get joint limits as a list of (min, max) tuples
    joint_limits_list = [joint_limits[jn] for jn in joint_names]

    # --- Helper Functions ---
    
    def get_distance(q1, q2):
        """Calculates Euclidean distance between two 6-DOF configurations."""
        return np.linalg.norm(np.array(q1) - np.array(q2))

    def get_nearest_node(tree, q_sample):
        """Finds the node in the tree (a set of tuples) closest to q_sample."""
        min_dist = float('inf')
        nearest_node = None
        for node in tree:
            dist = get_distance(node, q_sample)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        return nearest_node

    def sample_config(goal, limits, bias):
        """Samples a random 6-DOF config, with goal_bias."""
        if random.random() < bias:
            return goal
        
        sample = []
        for (min_val, max_val) in limits:
            sample.append(random.uniform(min_val, max_val))
        return tuple(sample)

    def steer(q_from, q_to, step_size):
        """Steers from q_from towards q_to by step_size."""
        q_from_arr = np.array(q_from)
        q_to_arr = np.array(q_to)
        
        vec = q_to_arr - q_from_arr
        dist = np.linalg.norm(vec)
        
        if dist < step_size:
            return q_to # We can reach the target
            
        unit_vec = vec / dist
        q_new_arr = q_from_arr + unit_vec * step_size
        return tuple(q_new_arr)

    def reconstruct_path(parents_a, parents_b, connection_node):
        """Reconstructs the path after trees connect."""
        path_a = []
        curr = connection_node
        while curr is not None:
            path_a.append(curr)
            curr = parents_a[curr]
        
        path_b = []
        curr = parents_b[connection_node] # Start from connection's parent in B
        while curr is not None:
            path_b.append(curr)
            curr = parents_b[curr]
            
        return path_a[::-1] + path_b # path_a (start->conn) + path_b (conn->goal)

    def rrt_connect(start, goal, limits, collision_fn):
        """Implements RRT-Connect."""
        T_a = {start} # Tree from start
        T_b = {goal}  # Tree from goal
        parents_a = {start: None}
        parents_b = {goal: None}
        
        for _ in range(MAX_ITERATIONS):
            q_rand = sample_config(goal, limits, GOAL_BIAS)
            
            # --- Grow Tree A ---
            q_near_a = get_nearest_node(T_a, q_rand)
            q_new_a = steer(q_near_a, q_rand, STEP_SIZE)
            
            if not collision_fn(q_new_a):
                T_a.add(q_new_a)
                parents_a[q_new_a] = q_near_a
                
                # --- Connect Tree B ---
                q_near_b = get_nearest_node(T_b, q_new_a)
                q_new_b = steer(q_near_b, q_new_a, STEP_SIZE)
                
                if not collision_fn(q_new_b):
                    T_b.add(q_new_b)
                    parents_b[q_new_b] = q_near_b
                    
                    # Check for connection
                    if np.allclose(q_new_b, q_new_a, atol=1e-5):
                        print("RRT trees connected!")
                        # We need to link the parent maps for reconstruction
                        parents_b[q_new_a] = q_near_b
                        return reconstruct_path(parents_a, parents_b, q_new_a)
            
            # Swap trees for next iteration
            T_a, T_b = T_b, T_a
            parents_a, parents_b = parents_b, parents_a
            start, goal = goal, start # Swap goal for bias
            
        print("RRT failed to find a path.")
        return None

    def is_path_clear(q1, q2, collision_fn, step_size=0.05):
            """Checks if a straight line path is collision-free at a resolution 
            consistent with the RRT step size."""
            
            dist = get_distance(q1, q2)
            if dist == 0:
                return True
                
            # Calculate number of steps needed based on distance and step_size
            num_steps = int(np.ceil(dist / step_size))
            
            q1_arr = np.array(q1)
            q2_arr = np.array(q2)
            
            for i in range(1, num_steps + 1):
                t = i / num_steps
                # Linearly interpolate between q1 and q2
                q_interp = tuple(q1_arr + t * (q2_arr - q1_arr))
                if collision_fn(q_interp):
                    return False
            return True

    def shortcut_smooth(path, collision_fn, iterations):
        """Performs shortcut smoothing on a given path."""
        if path is None:
            return None
            
        smoothed_path = list(path)
        for _ in range(iterations):
            if len(smoothed_path) <= 2:
                break # Can't smooth
                
            idx1 = random.randint(0, len(smoothed_path) - 2)
            idx2 = random.randint(idx1 + 1, len(smoothed_path) - 1)
            
            q_a = smoothed_path[idx1]
            q_b = smoothed_path[idx2]
            
            if is_path_clear(q_a, q_b, collision_fn, STEP_SIZE):
                # Create a new path by removing intermediate nodes
                smoothed_path = smoothed_path[:idx1+1] + smoothed_path[idx2:]
                
        return smoothed_path

    def get_ee_path(robot, joints, link_id, path, interp_step_size=0.01):
            """Gets a finely interpolated end-effector (x,y,z) path."""
            ee_path = []
            if not path:
                return ee_path
                
            # Save current config to restore later
            current_config = get_joint_positions(robot, joints)
            
            # Add the very first point
            set_joint_positions(robot, joints, path[0])
            ee_path.append(get_link_pose(robot, link_id)[0])

            # Iterate through path segments
            for i in range(len(path) - 1):
                q1 = path[i]
                q2 = path[i+1]
                
                q1_arr = np.array(q1)
                q2_arr = np.array(q2)
                dist = get_distance(q1, q2)
                if dist == 0:
                    continue

                # Calculate number of steps for this segment
                num_steps = int(np.ceil(dist / interp_step_size))

                for j in range(1, num_steps + 1):
                    t = j / num_steps
                    q_interp = tuple(q1_arr + t * (q2_arr - q1_arr))
                    
                    # Set joint pos and get ee pose
                    set_joint_positions(robot, joints, q_interp)
                    pose = get_link_pose(robot, link_id)
                    ee_path.append(pose[0]) # pose[0] is the (x,y,z) position
                
            # Restore original config
            set_joint_positions(robot, joints, current_config)
            return ee_path

    # --- Additional planners implementations: RRT, RRT*, Bi-directional RRT*, Informed RRT* ---

    # def rrt_basic(start, goal, limits, collision_fn):
    #     """Simple single-tree RRT returning a path from start to goal if found."""
    #     nodes = [start]
    #     parents = {start: None}
    #     for it in range(MAX_ITERATIONS):
    #         q_rand = sample_config(goal, limits, GOAL_BIAS)
    #         q_near = get_nearest_node(nodes, q_rand)
    #         q_new = steer(q_near, q_rand, STEP_SIZE)
    #         if collision_fn(q_new):
    #             continue
    #         # try to connect to q_near via collision-free interpolation
    #         if not is_path_clear(q_near, q_new, collision_fn, STEP_SIZE):
    #             continue
    #         nodes.append(q_new)
    #         parents[q_new] = q_near
    #         # check if we can connect q_new to goal
    #         if get_distance(q_new, goal) <= STEP_SIZE and is_path_clear(q_new, goal, collision_fn, STEP_SIZE):
    #             parents[goal] = q_new
    #             # reconstruct
    #             path = []
    #             cur = goal
    #             while cur is not None:
    #                 path.append(cur)
    #                 cur = parents[cur]
    #             return path[::-1]
    #     print("RRT basic failed to find a path.")
    #     return None

    def rrt_basic(start, goal, limits, collision_fn):
        """Simple single-tree RRT with multi-step extension toward random samples."""
        nodes = [start]
        parents = {start: None}

        def extend_towards(q_near, q_rand):
            """Multi-step extension toward q_rand."""
            path_ext = []
            q_curr = q_near

            while True:
                q_next = steer(q_curr, q_rand, STEP_SIZE)

                # Stop if next step collides
                if collision_fn(q_next) or not is_path_clear(q_curr, q_next, collision_fn, STEP_SIZE):
                    break

                path_ext.append(q_next)
                q_curr = q_next

                # Stop if reached sample
                if get_distance(q_curr, q_rand) < STEP_SIZE:
                    break

            return path_ext

        for it in range(MAX_ITERATIONS):

            # --- sample ---
            q_rand = sample_config(goal, limits, GOAL_BIAS)

            # --- nearest ---
            q_near = get_nearest_node(nodes, q_rand)

            # --- extend multiple steps ---
            extension = extend_towards(q_near, q_rand)
            if not extension:
                continue

            # add all extension nodes sequentially
            for q_new in extension:
                nodes.append(q_new)
                parents[q_new] = q_near
                q_near = q_new  # chain extension

            # --- last node in extension ---
            q_last = extension[-1]

            # --- connect to goal if possible ---
            if is_path_clear(q_last, goal, collision_fn, STEP_SIZE):

                parents[goal] = q_last

                # reconstruct path
                path = []
                cur = goal
                while cur is not None:
                    path.append(cur)
                    cur = parents[cur]
                return path[::-1]

        print("RRT basic failed to find a path.")
        return None

    # def rrt_star(start, goal, limits, collision_fn):
    #     """Simple RRT* (single-tree) with rewiring."""
    #     nodes = [start]
    #     parents = {start: None}
    #     cost = {start: 0.0}
    #     # neighbor radius (static, could be improved)
    #     neighbor_radius = 0.4

    #     for it in range(1, MAX_ITERATIONS+1):
    #         q_rand = sample_config(goal, limits, GOAL_BIAS)
    #         q_near = get_nearest_node(nodes, q_rand)
    #         q_new = steer(q_near, q_rand, STEP_SIZE)
    #         if collision_fn(q_new) or not is_path_clear(q_near, q_new, collision_fn, STEP_SIZE):
    #             continue
    #         # find neighbors within radius
    #         neighbors = [n for n in nodes if get_distance(n, q_new) <= neighbor_radius]
    #         # choose parent that gives min cost
    #         best_parent = q_near
    #         best_cost = cost[q_near] + get_distance(q_near, q_new)
    #         for n in neighbors:
    #             if is_path_clear(n, q_new, collision_fn, STEP_SIZE):
    #                 c = cost[n] + get_distance(n, q_new)
    #                 if c < best_cost:
    #                     best_cost = c
    #                     best_parent = n
    #         nodes.append(q_new)
    #         parents[q_new] = best_parent
    #         cost[q_new] = best_cost
    #         # rewire neighbors
    #         for n in neighbors:
    #             if n == best_parent:
    #                 continue
    #             if is_path_clear(q_new, n, collision_fn, STEP_SIZE):
    #                 new_cost = cost[q_new] + get_distance(q_new, n)
    #                 if new_cost < cost.get(n, float('inf')):
    #                     parents[n] = q_new
    #                     cost[n] = new_cost
    #         # try connecting to goal if close enough
    #         if get_distance(q_new, goal) <= STEP_SIZE and is_path_clear(q_new, goal, collision_fn, STEP_SIZE):
    #             parents[goal] = q_new
    #             # reconstruct path
    #             path = []
    #             cur = goal
    #             while cur is not None:
    #                 path.append(cur)
    #                 cur = parents[cur]
    #             return path[::-1]
    #     print("RRT* failed to find a path.")
    #     return None

    def rrt_star(start, goal, limits, collision_fn):
        """Corrected RRT* with multi-step extension + proper goal connection."""
        nodes = [start]
        parents = {start: None}
        cost = {start: 0.0}
        neighbor_radius = 0.2

        def extend_towards(q_near, q_rand):
            """Multi-step extension toward q_rand."""
            path_ext = []
            q_curr = q_near

            while True:
                q_next = steer(q_curr, q_rand, STEP_SIZE)
                if collision_fn(q_next) or not is_path_clear(q_curr, q_next, collision_fn, STEP_SIZE):
                    break
                path_ext.append(q_next)
                q_curr = q_next
                # stop if reached sample
                if get_distance(q_curr, q_rand) < STEP_SIZE:
                    break
            return path_ext

        for it in range(1, MAX_ITERATIONS + 1):

            # 1. Sample
            q_rand = sample_config(goal, limits, GOAL_BIAS)

            # 2. Nearest node
            q_near = get_nearest_node(nodes, q_rand)

            # 3. Extend multiple steps
            extension = extend_towards(q_near, q_rand)
            if not extension:
                continue

            # 4. Last reachable node in this extension
            q_new = extension[-1]

            # ----- RRT* parent selection -----
            neighbors = [n for n in nodes if get_distance(n, q_new) <= neighbor_radius]

            best_parent = q_near
            best_cost = cost[q_near] + get_distance(q_near, q_new)

            for n in neighbors:
                if is_path_clear(n, q_new, collision_fn, STEP_SIZE):
                    c = cost[n] + get_distance(n, q_new)
                    if c < best_cost:
                        best_cost = c
                        best_parent = n

            nodes.append(q_new)
            parents[q_new] = best_parent
            cost[q_new] = best_cost

            # ----- Rewire neighbors -----
            for n in neighbors:
                if n == best_parent:
                    continue
                if is_path_clear(q_new, n, collision_fn, STEP_SIZE):
                    new_cost = cost[q_new] + get_distance(q_new, n)
                    if new_cost < cost.get(n, float('inf')):
                        parents[n] = q_new
                        cost[n] = new_cost

            # ----- NEW: Proper goal connect -----
            if is_path_clear(q_new, goal, collision_fn, STEP_SIZE):
                parents[goal] = q_new
                cost[goal] = cost[q_new] + get_distance(q_new, goal)

                # reconstruct full path
                path = []
                curr = goal
                while curr is not None:
                    path.append(curr)
                    curr = parents[curr]
                return path[::-1]

        print("RRT* failed to find a path.")
        return None

    # def birrt_star(start, goal, limits, collision_fn):
    #     """Bidirectional RRT* - simplified: run two RRT* trees and try to connect."""
    #     # Each tree: nodes list, parents dict, cost dict
    #     nodes_a = [start]
    #     parents_a = {start: None}
    #     cost_a = {start: 0.0}

    #     nodes_b = [goal]
    #     parents_b = {goal: None}
    #     cost_b = {goal: 0.0}

    #     neighbor_radius = 0.3

    #     def try_extend(nodes_from, parents_from, cost_from, nodes_to, parents_to):
    #         q_rand = sample_config(goal, limits, GOAL_BIAS) if nodes_from is nodes_a else sample_config(start, limits, GOAL_BIAS)
    #         q_near = get_nearest_node(nodes_from, q_rand)
    #         q_new = steer(q_near, q_rand, STEP_SIZE)
    #         if collision_fn(q_new) or not is_path_clear(q_near, q_new, collision_fn, STEP_SIZE):
    #             return None
    #         # rewire logic like RRT*
    #         neighbors = [n for n in nodes_from if get_distance(n, q_new) <= neighbor_radius]
    #         best_parent = q_near
    #         best_cost = cost_from[q_near] + get_distance(q_near, q_new)
    #         for n in neighbors:
    #             if is_path_clear(n, q_new, collision_fn, STEP_SIZE):
    #                 c = cost_from[n] + get_distance(n, q_new)
    #                 if c < best_cost:
    #                     best_cost = c
    #                     best_parent = n
    #         nodes_from.append(q_new)
    #         parents_from[q_new] = best_parent
    #         cost_from[q_new] = best_cost
    #         for n in neighbors:
    #             if n == best_parent:
    #                 continue
    #             if is_path_clear(q_new, n, collision_fn, STEP_SIZE):
    #                 new_cost = cost_from[q_new] + get_distance(q_new, n)
    #                 if new_cost < cost_from.get(n, float('inf')):
    #                     parents_from[n] = q_new
    #                     cost_from[n] = new_cost
    #         # Now try to connect to nearest node in other tree
    #         q_near_other = get_nearest_node(nodes_to, q_new)
    #         if get_distance(q_new, q_near_other) <= STEP_SIZE and is_path_clear(q_new, q_near_other, collision_fn, STEP_SIZE):
    #             # found connection between q_new (in from) and q_near_other (in to)
    #             # reconstruct path from start->q_new and q_near_other->goal
    #             # Use parents_from and parents_to
    #             if nodes_from is nodes_a:
    #                 # connection node is q_new, corresponding other parent is q_near_other
    #                 parents_b_copy = parents_b
    #                 parents_a_copy = parents_a
    #                 # link for reconstruction: set parents_b[q_new] = q_near_other
    #                 parents_to[q_new] = q_near_other
    #                 # build path: start->...->q_new, then q_near_other->...->goal
    #                 path_a = []
    #                 cur = q_new
    #                 while cur is not None:
    #                     path_a.append(cur)
    #                     cur = parents_from[cur]
    #                 path_b = []
    #                 cur = q_near_other
    #                 while cur is not None:
    #                     path_b.append(cur)
    #                     cur = parents_to[cur]
    #                 return path_a[::-1] + path_b
    #             else:
    #                 # symmetric case
    #                 parents_to[q_new] = q_near_other
    #                 path_a = []
    #                 cur = q_near_other
    #                 while cur is not None:
    #                     path_a.append(cur)
    #                     cur = parents_from[cur]
    #                 path_b = []
    #                 cur = q_new
    #                 while cur is not None:
    #                     path_b.append(cur)
    #                     cur = parents_to[cur]
    #                 return path_a[::-1] + path_b
    #         return None

    #     for it in range(MAX_ITERATIONS):
    #         # alternate extending both trees
    #         res = try_extend(nodes_a, parents_a, cost_a, nodes_b, parents_b)
    #         if res:
    #             return res
    #         res = try_extend(nodes_b, parents_b, cost_b, nodes_a, parents_a)
    #         if res:
    #             return res
    #     print("BiRRT* failed to find a path.")
    #     return None

    def birrt_star(start, goal, limits, collision_fn):
        """Bidirectional RRT* with multi-step extension + proper rewiring and tree connection."""

        # Tree A = forward tree (from start)
        nodes_a = [start]
        parents_a = {start: None}
        cost_a = {start: 0.0}

        # Tree B = backward tree (from goal)
        nodes_b = [goal]
        parents_b = {goal: None}
        cost_b = {goal: 0.0}

        neighbor_radius = 0.2

        # -------------------------------------------------------
        # Multi-step extension (same as RRT*)
        # -------------------------------------------------------
        def extend_towards(q_near, q_rand):
            path_ext = []
            q_curr = q_near

            while True:
                q_next = steer(q_curr, q_rand, STEP_SIZE)

                if collision_fn(q_next) or not is_path_clear(q_curr, q_next, collision_fn, STEP_SIZE):
                    break

                path_ext.append(q_next)
                q_curr = q_next

                if get_distance(q_curr, q_rand) < STEP_SIZE:
                    break

            return path_ext

        # -------------------------------------------------------
        # Try to extend one tree toward random sample, then connect to the other tree
        # -------------------------------------------------------
        def try_extend(nodes_from, parents_from, cost_from, nodes_to, parents_to, cost_to):
            # Use correct bias direction
            q_rand = sample_config(goal, limits, GOAL_BIAS) if nodes_from is nodes_a else sample_config(start, limits, GOAL_BIAS)

            # nearest node in the active tree
            q_near = get_nearest_node(nodes_from, q_rand)

            # multi-step extension
            extension = extend_towards(q_near, q_rand)
            if not extension:
                return None

            q_new = extension[-1]

            # ---------------- Parent selection (RRT*) ----------------
            neighbors = [n for n in nodes_from if get_distance(n, q_new) <= neighbor_radius]

            best_parent = q_near
            best_cost = cost_from[q_near] + get_distance(q_near, q_new)

            for n in neighbors:
                if is_path_clear(n, q_new, collision_fn, STEP_SIZE):
                    c = cost_from[n] + get_distance(n, q_new)
                    if c < best_cost:
                        best_cost = c
                        best_parent = n

            nodes_from.append(q_new)
            parents_from[q_new] = best_parent
            cost_from[q_new] = best_cost

            # ---------------- Rewiring ----------------
            for n in neighbors:
                if n == best_parent:
                    continue
                if is_path_clear(q_new, n, collision_fn, STEP_SIZE):
                    new_cost = cost_from[q_new] + get_distance(q_new, n)
                    if new_cost < cost_from.get(n, float('inf')):
                        parents_from[n] = q_new
                        cost_from[n] = new_cost

            # -------------------------------------------------------
            # Try to CONNECT to the opposite tree via multi-step extend
            # -------------------------------------------------------
            q_near_other = get_nearest_node(nodes_to, q_new)
            connect_extension = extend_towards(q_near_other, q_new)

            # if multi-step extension from other tree reaches near q_new → connection formed
            if connect_extension:
                q_connect = connect_extension[-1]

                if get_distance(q_connect, q_new) < STEP_SIZE and is_path_clear(q_connect, q_new, collision_fn, STEP_SIZE):

                    # add this new connection node to other tree
                    nodes_to.append(q_connect)
                    parents_to[q_connect] = q_near_other
                    cost_to[q_connect] = cost_to[q_near_other] + get_distance(q_near_other, q_connect)

                    # ---------------------------------------------------
                    # Reconstruct full path: start -> connection -> goal
                    # ---------------------------------------------------
                    path_a = []
                    cur = q_new
                    while cur is not None:
                        path_a.append(cur)
                        cur = parents_from[cur]

                    path_b = []
                    cur = q_connect
                    while cur is not None:
                        path_b.append(cur)
                        cur = parents_to[cur]

                    return path_a[::-1] + path_b

            return None

        # -------------------------------------------------------
        # Bidirectional loop
        # -------------------------------------------------------
        for it in range(MAX_ITERATIONS):

            # Extend from start tree toward random sample, try connecting to goal tree
            res = try_extend(nodes_a, parents_a, cost_a, nodes_b, parents_b, cost_b)
            if res:
                return res

            # Extend from goal tree toward random sample, try connecting to start tree
            res = try_extend(nodes_b, parents_b, cost_b, nodes_a, parents_a, cost_a)
            if res:
                return res

        print("BiRRT* failed to find a path.")
        return None

    # def informed_rrt_star(start, goal, limits, collision_fn):
    #     """Informed RRT* implementation: once a solution is found, sample within the prolate hyperspheroid."""
    #     nodes = [start]
    #     parents = {start: None}
    #     cost = {start: 0.0}
    #     neighbor_radius = 0.4
    #     best_solution_cost = float('inf')
    #     solution_node = None

    #     def sample_in_ellipsoid(start, goal, c_best):
    #         # If no solution yet, sample uniformly
    #         if c_best == float('inf'):
    #             return sample_config(goal, limits, GOAL_BIAS)
    #         # Prolate hyperspheroid sampling in configuration space: approximate by sampling a Gaussian in transformed space
    #         s = np.array(start)
    #         g = np.array(goal)
    #         c_min = np.linalg.norm(g - s)
    #         if c_min == 0:
    #             return tuple(s)
    #         # center
    #         center = (s + g) / 2.0
    #         # rotation is identity in this simplified implementation (we treat axes aligned)
    #         a1 = (g - s) / c_min
    #         # define radii along principal axes
    #         r1 = c_best / 2.0
    #         if c_best**2 - c_min**2 <= 0:
    #             other_radius = 0.0
    #         else:
    #             other_radius = np.sqrt(c_best**2 - c_min**2) / 2.0
    #         # sample in unit n-ball and scale
    #         while True:
    #             # sample in bounding hyper-ellipse box to reduce rejections
    #             sample = np.array([random.uniform(-other_radius, other_radius) for _ in range(len(s))])
    #             # set first component differently to align with major axis
    #             sample[0] = random.uniform(-r1, r1)
    #             q = center + sample
    #             # clip to joint limits
    #             q_clipped = []
    #             for i, (mn, mx) in enumerate(limits):
    #                 q_clipped.append(float(np.clip(q[i], mn, mx)))
    #             q_tuple = tuple(q_clipped)
    #             return q_tuple

    #     for it in range(1, MAX_ITERATIONS+1):
    #         if solution_node is None:
    #             q_rand = sample_config(goal, limits, GOAL_BIAS)
    #         else:
    #             q_rand = sample_in_ellipsoid(start, goal, best_solution_cost)
    #         q_near = get_nearest_node(nodes, q_rand)
    #         q_new = steer(q_near, q_rand, STEP_SIZE)
    #         if collision_fn(q_new) or not is_path_clear(q_near, q_new, collision_fn, STEP_SIZE):
    #             continue
    #         neighbors = [n for n in nodes if get_distance(n, q_new) <= neighbor_radius]
    #         best_parent = q_near
    #         best_cost = cost[q_near] + get_distance(q_near, q_new)
    #         for n in neighbors:
    #             if is_path_clear(n, q_new, collision_fn, STEP_SIZE):
    #                 c = cost[n] + get_distance(n, q_new)
    #                 if c < best_cost:
    #                     best_cost = c
    #                     best_parent = n
    #         nodes.append(q_new)
    #         parents[q_new] = best_parent
    #         cost[q_new] = best_cost
    #         for n in neighbors:
    #             if n == best_parent:
    #                 continue
    #             if is_path_clear(q_new, n, collision_fn, STEP_SIZE):
    #                 new_cost = cost[q_new] + get_distance(q_new, n)
    #                 if new_cost < cost.get(n, float('inf')):
    #                     parents[n] = q_new
    #                     cost[n] = new_cost
    #         # check connect to goal
    #         if get_distance(q_new, goal) <= STEP_SIZE and is_path_clear(q_new, goal, collision_fn, STEP_SIZE):
    #             # potential solution — compute its cost
    #             sol_cost = cost[q_new] + get_distance(q_new, goal)
    #             if sol_cost < best_solution_cost:
    #                 best_solution_cost = sol_cost
    #                 parents[goal] = q_new
    #                 solution_node = goal
    #                 # (we don't immediately return — we allow informed sampling to improve solution until MAX_ITERATIONS)
    #     if solution_node is None:
    #         print("Informed RRT* failed to find a path.")
    #         return None
    #     # reconstruct final path
    #     path = []
    #     cur = goal
    #     while cur is not None:
    #         path.append(cur)
    #         cur = parents[cur]
    #     return path[::-1]

    def informed_rrt_star(start, goal, limits, collision_fn):
        """Informed RRT* with better goal connection."""
        import numpy as np
        
        nodes = [start]
        parents = {start: None}
        cost_dict = {start: 0.0}

        neighbor_radius = 0.4
        best_solution_cost = float('inf')
        solution_found = False
        
        c_min = get_distance(start, goal)
        goal_connection_radius = max(STEP_SIZE * 3.0, 0.5)

        print(f"\n=== Informed RRT* Starting ===")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"c_min (straight-line): {c_min:.4f}")
        print(f"Goal in collision: {collision_fn(goal)}")
        print(f"STEP_SIZE: {STEP_SIZE}")  # ← ADD THIS LINE
        print(f"Goal connection radius: {goal_connection_radius:.4f}")
        print(f"Direct path clear: {is_path_clear(start, goal, collision_fn, STEP_SIZE)}")
        
        # --------------------------------------------------------
        # Informed Sampling
        # --------------------------------------------------------
        def sample_in_ellipsoid(start, goal, c_best):
            """Sample uniformly within prolate hyperellipsoid."""
            if c_best == float('inf'):
                return sample_config(goal, limits, GOAL_BIAS)
            
            s = np.array(start)
            g = np.array(goal)
            c_min_local = np.linalg.norm(g - s)
            
            if c_min_local < 1e-6:
                return sample_config(goal, limits, GOAL_BIAS)
            
            center = (s + g) / 2.0
            a = c_best / 2.0
            c_squared = (c_best / 2.0)**2 - (c_min_local / 2.0)**2
            
            if c_squared <= 0:
                return sample_config(goal, limits, GOAL_BIAS)
            
            b = np.sqrt(c_squared)
            
            d = len(start)
            while True:
                x = np.random.randn(d)
                if np.linalg.norm(x) <= 1.0:
                    break
            
            x_ellipse = np.zeros(d)
            x_ellipse[0] = a * x[0]
            for i in range(1, d):
                x_ellipse[i] = b * x[i]
            
            a1 = (g - s) / c_min_local
            M = np.eye(d)
            M[:, 0] = a1
            
            for i in range(1, d):
                v = np.random.randn(d)
                for j in range(i):
                    v -= np.dot(v, M[:, j]) * M[:, j]
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-10:
                    M[:, i] = v / norm_v
                else:
                    M[:, i] = 0
                    M[i, i] = 1
            
            q_ellipse = center + M @ x_ellipse
            
            q_clipped = []
            for val, (mn, mx) in zip(q_ellipse, limits):
                q_clipped.append(float(np.clip(val, mn, mx)))
            
            q_tuple = tuple(q_clipped)
            
            if get_distance(q_tuple, start) + get_distance(q_tuple, goal) > c_best + 0.1:
                return sample_config(goal, limits, GOAL_BIAS)
            
            return q_tuple

        # --------------------------------------------------------
        # Multi-step extension
        # --------------------------------------------------------
        def extend_towards(q_near, q_rand):
            path_ext = []
            q_curr = q_near
            max_steps = 10  # Limit extension steps

            for _ in range(max_steps):
                q_next = steer(q_curr, q_rand, STEP_SIZE)

                if collision_fn(q_next) or not is_path_clear(q_curr, q_next, collision_fn, STEP_SIZE):
                    break

                path_ext.append(q_next)
                q_curr = q_next

                if get_distance(q_curr, q_rand) < STEP_SIZE:
                    break

            return path_ext

        # --------------------------------------------------------
        # Main loop
        # --------------------------------------------------------
        goal_attempts = 0
        closest_to_goal = float('inf')
        closest_node = None

        for it in range(1, MAX_ITERATIONS + 1):

            if not solution_found:
                q_rand = sample_config(goal, limits, GOAL_BIAS)
            else:
                q_rand = sample_in_ellipsoid(start, goal, best_solution_cost)

            q_near = get_nearest_node(nodes, q_rand)
            extension = extend_towards(q_near, q_rand)
            
            if not extension:
                continue

            q_new = extension[-1]

            # Parent selection
            neighbors = [n for n in nodes if get_distance(n, q_new) <= neighbor_radius]

            best_parent = q_near
            best_cost = cost_dict[q_near] + get_distance(q_near, q_new)

            for n in neighbors:
                if is_path_clear(n, q_new, collision_fn, STEP_SIZE):
                    c = cost_dict[n] + get_distance(n, q_new)
                    if c < best_cost:
                        best_cost = c
                        best_parent = n

            nodes.append(q_new)
            parents[q_new] = best_parent
            cost_dict[q_new] = best_cost

            # Rewire
            for n in neighbors:
                if n == best_parent:
                    continue
                if is_path_clear(q_new, n, collision_fn, STEP_SIZE):
                    new_cost = cost_dict[q_new] + get_distance(q_new, n)
                    if new_cost < cost_dict.get(n, float('inf')):
                        parents[n] = q_new
                        cost_dict[n] = new_cost

            # Track closest approach to goal
            dist_to_goal = get_distance(q_new, goal)
            if dist_to_goal < closest_to_goal:
                closest_to_goal = dist_to_goal
                closest_node = q_new

            # Try connecting to goal - RELAXED threshold
            # goal_threshold = STEP_SIZE * 2.0  # More generous!
            
            if dist_to_goal <= goal_connection_radius: 
                goal_attempts += 1
                if is_path_clear(q_new, goal, collision_fn, STEP_SIZE):
                    sol_cost = cost_dict[q_new] + get_distance(q_new, goal)

                    if sol_cost < best_solution_cost:
                        best_solution_cost = sol_cost
                        solution_found = True
                        parents[goal] = q_new
                        cost_dict[goal] = sol_cost
                        print(f"  ✓ Iteration {it}: Found path! Cost = {best_solution_cost:.4f}")
            
            # Progress updates
            if it % 500 == 0:
                status = f"{best_solution_cost:.4f}" if solution_found else "None"
                print(f"  Iteration {it}/{MAX_ITERATIONS}, Nodes: {len(nodes)}, Best: {status}")
                print(f"    Closest to goal: {closest_to_goal:.4f}, Goal attempts: {goal_attempts}")

        # Return best path
        if not solution_found:
            print(f"✗ Informed RRT* failed to find a path.")
            print(f"   Closest distance to goal: {closest_to_goal:.4f}")
            print(f"   Goal connection attempts: {goal_attempts}")
            print(f"   Total nodes in tree: {len(nodes)}")
            return None

        print(f"✓ Informed RRT* completed. Final cost: {best_solution_cost:.4f}")
        
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parents[cur]
        return path[::-1]

    # --- Main Execution ---
    PLANNER = 'InformedRRT*'  # change this string to select another planner
    print(f"Running {PLANNER}...")
    start_time = time.time()
    # ----------------- PLANNER SELECTION -----------------
    # Choose one of: 'RRT-Connect', 'RRT', 'RRT*', 'BiRRT*', 'InformedRRT*'
    if PLANNER == 'RRT-Connect':
        rrt_path = rrt_connect(start_config, goal_config, joint_limits_list, collision_fn)
    elif PLANNER == 'RRT':
        rrt_path = rrt_basic(start_config, goal_config, joint_limits_list, collision_fn)
    elif PLANNER == 'RRT*':
        rrt_path = rrt_star(start_config, goal_config, joint_limits_list, collision_fn)
    elif PLANNER == 'BiRRT*':
        rrt_path = birrt_star(start_config, goal_config, joint_limits_list, collision_fn)
    elif PLANNER == 'InformedRRT*':
        rrt_path = informed_rrt_star(start_config, goal_config, joint_limits_list, collision_fn)
    else:
        print(f"Unknown planner '{PLANNER}'. Falling back to RRT-Connect.")
        rrt_path = rrt_connect(start_config, goal_config, joint_limits_list, collision_fn)
    # ----------------------------------------------------
    print(f"{PLANNER} finished in {time.time() - start_time:.2f}s")
    
    if rrt_path:
        print(f"{PLANNER} path found with {len(rrt_path)} waypoints.")
        
        # --- Smoothing ---
        print(f"Running smoothing ({SMOOTH_ITERATIONS} iterations)...")
        start_time = time.time()
        smoothed_path = shortcut_smooth(rrt_path, collision_fn, SMOOTH_ITERATIONS)
        print(f"Smoothing finished in {time.time() - start_time:.2f}s")
        print(f"Smoothed path has {len(smoothed_path)} waypoints.")
        
        # --- Visualization ---
        print("Drawing paths...")
        ee_link_id = link_from_name(robots['pr2'], 'l_gripper_tool_frame')
        
        # Get and draw original RRT path (red)
        ee_path_rrt = get_ee_path(robots['pr2'], joint_idx, ee_link_id, rrt_path, interp_step_size=0.01)
        for p in ee_path_rrt:
            draw_sphere_marker(p, 0.005, (1, 0, 0, 0.8)) # Red, now very small
            
        # Get and draw smoothed path (blue)
        ee_path_smooth = get_ee_path(robots['pr2'], joint_idx, ee_link_id, smoothed_path, interp_step_size=0.01)
        for p in ee_path_smooth:
            draw_sphere_marker(p, 0.007, (0, 0, 1, 0.8)) # Blue, slightly larger
        
        # Set the final path to be executed 
        path = smoothed_path
        
    else:
        print("No path found.")

    ######################
    # Execute planned path
    execute_trajectory(robots['pr2'], joint_idx, path, sleep=0.1)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()