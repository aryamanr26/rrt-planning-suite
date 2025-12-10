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

    start_config = tuple(float(x) for x in get_joint_positions(robots['pr2'], joint_idx))
    goal_config = tuple(float(x) for x in (0.5, 0.33, -1.548, 1.557, -1.32, -0.1928))
    path = []
    ### YOUR CODE HERE ###

    step_size = 0.05
    goal_bias = 0.1
    max_iter = 3000

     # Get joint limits as a list of (min, max) tuples
    joint_limits_list = [joint_limits[jn] for jn in joint_names]

    def get_nearest_node(tree, q_sample):
        min_dist = float('inf')
        nearest_node = None

        for node in tree:
            dist = np.linalg.norm(np.array(node) - np.array(q_sample))
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    def steer(q_from, q_to, step_size):
        q_from = np.array(q_from)
        q_to = np.array(q_to)

        dist = np.linalg.norm(q_to - q_from)

        def to_tuple(q):
            return tuple(float(x) for x in q)
        
        if dist < step_size:
            return to_tuple(q_to)

        vec = q_to - q_from
        unit = vec/dist
        q_new = q_from + unit * step_size

        return to_tuple(q_new)

    def SampleRandomNode(goal, limits, bias):
        if random.random() < bias:
            return goal
        sample = []
        ## 6 DOF random config using max and min values of limits
        for (min_val, max_val) in limits:
            sample.append(random.uniform(min_val, max_val))
        return tuple(sample)

    def reconstruct_path(parents_start, parents_goal, connection_node):
        path_start = []
        curr = connection_node

        while curr:
            path_start.append(curr)
            curr = parents_start[curr]

        path_goal = [] 
        curr = parents_goal[connection_node] # Start from connection's parent in B

        while curr:
            path_goal.append(curr)
            curr = parents_goal[curr]

        return path_start[::-1] + path_goal ## path_a (start to connect) & path_b (connect to goal)
    
    def RRTConnect(start, goal, limits, collision_fn):
        T_start = {start}
        T_goal = {goal}
        parents_A = {start: None}
        parents_B = {goal: None}

        for _ in range(max_iter):
            q_rand = SampleRandomNode(goal, limits, goal_bias)

            q_near_start = get_nearest_node(T_start, q_rand)
            q_new_start = steer(q_near_start, q_rand, step_size) ## EXTEND

            if not collision_fn(q_new_start):
                T_start.add(q_new_start)
                parents_A[q_new_start] = q_near_start

                q_near_goal = get_nearest_node(T_goal, q_new_start)
                q_new_goal = steer(q_near_goal, q_new_start, step_size) ## EXTEND

                if not collision_fn(q_new_goal):
                    T_goal.add(q_new_goal)
                    parents_B[q_new_goal] = q_near_goal

                    if np.allclose(q_new_goal, q_new_start, atol=1e-5):
                        print("Connected!")

                        parents_B[q_new_start] = q_near_goal

                        return reconstruct_path(parents_A, parents_B, q_new_start)
    
            T_start, T_goal = T_goal, T_start
            parents_A, parents_B = parents_B, parents_A
            start, goal = goal, start
        
        print("RRT Failed to find a path")
        return None
    
    print("Running RRT-Connect...")
    start_time = time.time()
    rrt_path = RRTConnect(start_config, goal_config, joint_limits_list, collision_fn)
    print(f"RRT-Connect finished in {time.time() - start_time:.2f}s")

    def is_path_clear(q1, q2, collision_fn, step_size=0.05):
        
        dist = np.linalg.norm(np.array(q1) - np.array(q2))
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
            
            if is_path_clear(q_a, q_b, collision_fn, step_size):
                # Create a new path by removing intermediate nodes
                smoothed_path = smoothed_path[:idx1+1] + smoothed_path[idx2:]
                
        return smoothed_path
    
    def get_ee_path(robot, joints, link_id, path, interp_step_size=0.01):
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
            dist = np.linalg.norm(np.array(q1) - np.array(q2))
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
    
    smooth_iterations = 150
    if rrt_path:
        print(f"RRT path found with {len(rrt_path)} waypoints.")
        
        # --- Smoothing ---
        print(f"Running smoothing ({smooth_iterations} iterations)...")
        start_time = time.time()
        smoothed_path = shortcut_smooth(rrt_path, collision_fn, smooth_iterations)
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
