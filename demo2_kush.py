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
    STEP_SIZE = 0.15 # 
    GOAL_BIAS = 0.3 # 
    MAX_ITERATIONS = 5000
    SMOOTH_ITERATIONS = 150 # 

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
    # (existing implementations retained)
    def rrt_basic(start, goal, limits, collision_fn):
        """Simple single-tree RRT returning a path from start to goal if found."""
        nodes = [start]
        parents = {start: None}
        for it in range(MAX_ITERATIONS):
            q_rand = sample_config(goal, limits, GOAL_BIAS)
            q_near = get_nearest_node(nodes, q_rand)
            q_new = steer(q_near, q_rand, STEP_SIZE)
            if collision_fn(q_new):
                continue
            # try to connect to q_near via collision-free interpolation
            if not is_path_clear(q_near, q_new, collision_fn, STEP_SIZE):
                continue
            nodes.append(q_new)
            parents[q_new] = q_near
            # check if we can connect q_new to goal
            if get_distance(q_new, goal) <= STEP_SIZE and is_path_clear(q_new, goal, collision_fn, STEP_SIZE):
                parents[goal] = q_new
                # reconstruct
                path = []
                cur = goal
                while cur is not None:
                    path.append(cur)
                    cur = parents[cur]
                return path[::-1]
        print("RRT basic failed to find a path.")
        return None

    def rrt_star(start, goal, limits, collision_fn):
        """Simple RRT* (single-tree) with rewiring."""
        nodes = [start]
        parents = {start: None}
        cost = {start: 0.0}
        # neighbor radius (static, could be improved)
        neighbor_radius = 0.4

        for it in range(1, MAX_ITERATIONS+1):
            q_rand = sample_config(goal, limits, GOAL_BIAS)
            q_near = get_nearest_node(nodes, q_rand)
            q_new = steer(q_near, q_rand, STEP_SIZE)
            if collision_fn(q_new) or not is_path_clear(q_near, q_new, collision_fn, STEP_SIZE):
                continue
            # find neighbors within radius
            neighbors = [n for n in nodes if get_distance(n, q_new) <= neighbor_radius]
            # choose parent that gives min cost
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
            # rewire neighbors
            for n in neighbors:
                if n == best_parent:
                    continue
                if is_path_clear(q_new, n, collision_fn, STEP_SIZE):
                    new_cost = cost[q_new] + get_distance(q_new, n)
                    if new_cost < cost.get(n, float('inf')):
                        parents[n] = q_new
                        cost[n] = new_cost
            # try connecting to goal if close enough
            if get_distance(q_new, goal) <= STEP_SIZE and is_path_clear(q_new, goal, collision_fn, STEP_SIZE):
                parents[goal] = q_new
                # reconstruct path
                path = []
                cur = goal
                while cur is not None:
                    path.append(cur)
                    cur = parents[cur]
                return path[::-1]
        print("RRT* failed to find a path.")
        return None

    def birrt_star(start, goal, limits, collision_fn):
        """Bidirectional RRT* - simplified: run two RRT* trees and try to connect."""
        # Each tree: nodes list, parents dict, cost dict
        nodes_a = [start]
        parents_a = {start: None}
        cost_a = {start: 0.0}

        nodes_b = [goal]
        parents_b = {goal: None}
        cost_b = {goal: 0.0}

        neighbor_radius = 0.3

        def try_extend(nodes_from, parents_from, cost_from, nodes_to, parents_to):
            q_rand = sample_config(goal, limits, GOAL_BIAS) if nodes_from is nodes_a else sample_config(start, limits, GOAL_BIAS)
            q_near = get_nearest_node(nodes_from, q_rand)
            q_new = steer(q_near, q_rand, STEP_SIZE)
            if collision_fn(q_new) or not is_path_clear(q_near, q_new, collision_fn, STEP_SIZE):
                return None
            # rewire logic like RRT*
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
            for n in neighbors:
                if n == best_parent:
                    continue
                if is_path_clear(q_new, n, collision_fn, STEP_SIZE):
                    new_cost = cost_from[q_new] + get_distance(q_new, n)
                    if new_cost < cost_from.get(n, float('inf')):
                        parents_from[n] = q_new
                        cost_from[n] = new_cost
            # Now try to connect to nearest node in other tree
            q_near_other = get_nearest_node(nodes_to, q_new)
            if get_distance(q_new, q_near_other) <= STEP_SIZE and is_path_clear(q_new, q_near_other, collision_fn, STEP_SIZE):
                # found connection between q_new (in from) and q_near_other (in to)
                # reconstruct path from start->q_new and q_near_other->goal
                # Use parents_from and parents_to
                if nodes_from is nodes_a:
                    # connection node is q_new, corresponding other parent is q_near_other
                    parents_b_copy = parents_b
                    parents_a_copy = parents_a
                    # link for reconstruction: set parents_b[q_new] = q_near_other
                    parents_to[q_new] = q_near_other
                    # build path: start->...->q_new, then q_near_other->...->goal
                    path_a = []
                    cur = q_new
                    while cur is not None:
                        path_a.append(cur)
                        cur = parents_from[cur]
                    path_b = []
                    cur = q_near_other
                    while cur is not None:
                        path_b.append(cur)
                        cur = parents_to[cur]
                    return path_a[::-1] + path_b
                else:
                    # symmetric case
                    parents_to[q_new] = q_near_other
                    path_a = []
                    cur = q_near_other
                    while cur is not None:
                        path_a.append(cur)
                        cur = parents_from[cur]
                    path_b = []
                    cur = q_new
                    while cur is not None:
                        path_b.append(cur)
                        cur = parents_to[cur]
                    return path_a[::-1] + path_b
            return None

        for it in range(MAX_ITERATIONS):
            # alternate extending both trees
            res = try_extend(nodes_a, parents_a, cost_a, nodes_b, parents_b)
            if res:
                return res
            res = try_extend(nodes_b, parents_b, cost_b, nodes_a, parents_a)
            if res:
                return res
        print("BiRRT* failed to find a path.")
        return None

    def informed_rrt_star(start, goal, limits, collision_fn):
        """Informed RRT* implementation: once a solution is found, sample within the prolate hyperspheroid."""
        nodes = [start]
        parents = {start: None}
        cost = {start: 0.0}
        neighbor_radius = 0.4
        best_solution_cost = float('inf')
        solution_node = None

        def sample_in_ellipsoid(start, goal, c_best):
            # If no solution yet, sample uniformly
            if c_best == float('inf'):
                return sample_config(goal, limits, GOAL_BIAS)
            # Prolate hyperspheroid sampling in configuration space: approximate by sampling a Gaussian in transformed space
            s = np.array(start)
            g = np.array(goal)
            c_min = np.linalg.norm(g - s)
            if c_min == 0:
                return tuple(s)
            # center
            center = (s + g) / 2.0
            # rotation is identity in this simplified implementation (we treat axes aligned)
            a1 = (g - s) / c_min
            # define radii along principal axes
            r1 = c_best / 2.0
            if c_best**2 - c_min**2 <= 0:
                other_radius = 0.0
            else:
                other_radius = np.sqrt(c_best**2 - c_min**2) / 2.0
            # sample in unit n-ball and scale
            while True:
                # sample in bounding hyper-ellipse box to reduce rejections
                sample = np.array([random.uniform(-other_radius, other_radius) for _ in range(len(s))])
                # set first component differently to align with major axis
                sample[0] = random.uniform(-r1, r1)
                q = center + sample
                # clip to joint limits
                q_clipped = []
                for i, (mn, mx) in enumerate(limits):
                    q_clipped.append(float(np.clip(q[i], mn, mx)))
                q_tuple = tuple(q_clipped)
                return q_tuple

        for it in range(1, MAX_ITERATIONS+1):
            if solution_node is None:
                q_rand = sample_config(goal, limits, GOAL_BIAS)
            else:
                q_rand = sample_in_ellipsoid(start, goal, best_solution_cost)
            q_near = get_nearest_node(nodes, q_rand)
            q_new = steer(q_near, q_rand, STEP_SIZE)
            if collision_fn(q_new) or not is_path_clear(q_near, q_new, collision_fn, STEP_SIZE):
                continue
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
            for n in neighbors:
                if n == best_parent:
                    continue
                if is_path_clear(q_new, n, collision_fn, STEP_SIZE):
                    new_cost = cost[q_new] + get_distance(q_new, n)
                    if new_cost < cost.get(n, float('inf')):
                        parents[n] = q_new
                        cost[n] = new_cost
            # check connect to goal
            if get_distance(q_new, goal) <= STEP_SIZE and is_path_clear(q_new, goal, collision_fn, STEP_SIZE):
                # potential solution — compute its cost
                sol_cost = cost[q_new] + get_distance(q_new, goal)
                if sol_cost < best_solution_cost:
                    best_solution_cost = sol_cost
                    parents[goal] = q_new
                    solution_node = goal
                    # (we don't immediately return — we allow informed sampling to improve solution until MAX_ITERATIONS)
        if solution_node is None:
            print("Informed RRT* failed to find a path.")
            return None
        # reconstruct final path
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parents[cur]
        return path[::-1]

    # --- Additional planners: PRM, Lazy PRM, PRM*, FMT*, BIT*, A*-grid, STOMP-like optimizer ---

    import heapq
    from math import inf

    def knn_nodes(nodes, q, k):
        """Return k nearest nodes (by get_distance) to q from list nodes."""
        nodes_sorted = sorted(nodes, key=lambda n: get_distance(n, q))
        return nodes_sorted[:k]

    def build_prm(samples, k, limits, collision_fn):
        """Build a PRM graph given list of sample configs (assumed collision-free)."""
        graph = {s: [] for s in samples}
        for i, s in enumerate(samples):
            neighbors = knn_nodes(samples, s, k+1)  # includes itself possibly
            for n in neighbors:
                if n == s:
                    continue
                if is_path_clear(s, n, collision_fn, STEP_SIZE):
                    w = get_distance(s, n)
                    graph[s].append((n, w))
        return graph

    def dijkstra_graph(graph, start, goal):
        """Dijkstra on adjacency dict: graph[node] = [(neighbor, weight), ...]"""
        if start not in graph or goal not in graph:
            return None
        dist = {start: 0.0}
        parent = {start: None}
        pq = [(0.0, start)]
        visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == goal:
                # reconstruct
                path = []
                cur = u
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                return path[::-1]
            for v, w in graph.get(u, []):
                nd = d + w
                if nd < dist.get(v, inf):
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))
        return None

    def prm(start, goal, limits, collision_fn, n_samples=400, k=10):
        """Probabilistic Roadmap (PRM)."""
        samples = [start, goal]
        # sample random collision-free nodes
        tries = 0
        while len(samples) < n_samples and tries < n_samples * 10:
            q = sample_config(goal, limits, GOAL_BIAS)
            if not collision_fn(q):
                samples.append(q)
            tries += 1
        graph = build_prm(samples, k, limits, collision_fn)
        path = dijkstra_graph(graph, start, goal)
        if path is None:
            print("PRM failed to find a path.")
        return path

    def lazy_prm(start, goal, limits, collision_fn, n_samples=300, k=10):
        """Lazy PRM: construct connectivity using proximity, defer edge collision-check until needed."""
        samples = [start, goal]
        tries = 0
        while len(samples) < n_samples and tries < n_samples * 10:
            q = sample_config(goal, limits, GOAL_BIAS)
            if not collision_fn(q):
                samples.append(q)
            tries += 1
        # adjacency by proximity (no edge collision checks yet)
        adj = {s: [] for s in samples}
        for s in samples:
            neighbors = knn_nodes(samples, s, k+1)
            for n in neighbors:
                if n == s:
                    continue
                w = get_distance(s, n)
                adj[s].append((n, w))
        # now run lazy search: Dijkstra but only validate edges when popped
        dist = {start: 0.0}
        parent = {start: None}
        pq = [(0.0, start)]
        visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == goal:
                path = []
                cur = u
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                return path[::-1]
            for v, w in adj.get(u, []):
                # validate this edge now
                if is_path_clear(u, v, collision_fn, STEP_SIZE):
                    nd = d + w
                    if nd < dist.get(v, inf):
                        dist[v] = nd
                        parent[v] = u
                        heapq.heappush(pq, (nd, v))
                # else skip edge (lazy)
        print("Lazy PRM failed to find a path.")
        return None

    def prm_star(start, goal, limits, collision_fn, n_samples=400, k=12):
        """PRM* - PRM with local rewiring (simple variant)."""
        samples = [start, goal]
        tries = 0
        while len(samples) < n_samples and tries < n_samples * 10:
            q = sample_config(goal, limits, GOAL_BIAS)
            if not collision_fn(q):
                samples.append(q)
            tries += 1
        graph = {s: [] for s in samples}
        cost = {}
        # incremental insertion + rewiring
        for s in samples:
            # connect to neighbors
            neighbors = knn_nodes(samples, s, k+1)
            for n in neighbors:
                if n == s:
                    continue
                if is_path_clear(s, n, collision_fn, STEP_SIZE):
                    w = get_distance(s, n)
                    graph[s].append((n, w))
        # run Dijkstra on built graph
        path = dijkstra_graph(graph, start, goal)
        if path is None:
            print("PRM* failed to find a path.")
        return path

    def fmt_star(start, goal, limits, collision_fn, n_samples=400, neighbor_radius=0.6):
        """Simplified FMT*: sample nodes and run a Dijkstra-like expansion constrained by neighbor radius."""
        # sample nodes including start & goal
        samples = [start, goal]
        tries = 0
        while len(samples) < n_samples and tries < n_samples * 10:
            q = sample_config(goal, limits, GOAL_BIAS)
            if not collision_fn(q):
                samples.append(q)
            tries += 1
        # build neighbor lists (without checking collisions yet)
        neighbor_map = {s: [n for n in samples if n != s and get_distance(n, s) <= neighbor_radius] for s in samples}
        # Initialize sets
        V_unvisited = set(samples)
        V_open = set([start])
        parent = {start: None}
        cost = {start: 0.0}
        while V_open:
            # pick lowest-cost node in V_open
            z = min(V_open, key=lambda v: cost.get(v, inf))
            V_open.remove(z)
            V_unvisited.discard(z)
            # consider neighbors of z that are unvisited
            for x in list(V_unvisited):
                if x in neighbor_map[z]:
                    # find y in V_open ∩ neighbor(x) that minimizes cost[y] + dist(y,x)
                    Y = [y for y in V_open if y in neighbor_map[x]]
                    if not Y:
                        continue
                    # try connecting x via best y
                    best_y = None
                    best_cost = inf
                    for y in Y:
                        if is_path_clear(y, x, collision_fn, STEP_SIZE):
                            c = cost.get(y, inf) + get_distance(y, x)
                            if c < best_cost:
                                best_cost = c
                                best_y = y
                    if best_y is not None:
                        cost[x] = best_cost
                        parent[x] = best_y
                        V_open.add(x)
                        V_unvisited.discard(x)
            if goal in parent:
                # reconstruct
                path = []
                cur = goal
                while cur is not None:
                    path.append(cur)
                    cur = parent.get(cur, None)
                return path[::-1]
        print("FMT* failed to find a path.")
        return None

    def bit_star(start, goal, limits, collision_fn, batch_size=200, k=12):
        """Simplified BIT*: batch sampling + PRM* style incremental improvement with informed batches."""
        # Start with small sample set, iteratively add batches and search
        samples = [start, goal]
        graph = {}
        def build_graph(samples):
            g = {s: [] for s in samples}
            for s in samples:
                neighbors = knn_nodes(samples, s, k+1)
                for n in neighbors:
                    if n == s:
                        continue
                    if is_path_clear(s, n, collision_fn, STEP_SIZE):
                        g[s].append((n, get_distance(s, n)))
            return g
        # initial search attempts with increasing batches
        best_path = None
        for batch in range(1 + MAX_ITERATIONS // batch_size):
            # add batch samples
            tries = 0
            while len(samples) < 2 + batch * batch_size and tries < batch_size * 10:
                q = sample_config(goal, limits, GOAL_BIAS)
                if not collision_fn(q):
                    samples.append(q)
                tries += 1
            graph = build_graph(samples)
            p = dijkstra_graph(graph, start, goal)
            if p is not None:
                best_path = p
                # attempt informed refine: limit sampling to ellipsoid around current solution length
                # approximate current best length
                length = sum(get_distance(best_path[i], best_path[i+1]) for i in range(len(best_path)-1))
                # add more samples inside approximate ellipsoid (use informed_rrt_star's sampler idea)
                # but here we simply continue to next batch which will add more samples
            # break early if found
            if best_path:
                break
        if best_path is None:
            print("BIT* failed to find a path.")
        return best_path

    def a_star_grid(start, goal, limits, collision_fn, per_joint=4):
        """A* on discretized joint-space grid (coarse). Returns path of configs if found."""
        # Build discretized grid per joint (uniform)
        grids = []
        for (mn, mx) in limits:
            grids.append(np.linspace(mn, mx, per_joint))
        # helper to convert index tuple to config tuple
        from itertools import product
        index_list = list(product(*[range(per_joint) for _ in limits]))
        idx_to_config = {}
        config_to_idx = {}
        for idx in index_list:
            cfg = tuple(float(grids[i][idx[i]]) for i in range(len(limits)))
            idx_to_config[idx] = cfg
            config_to_idx[cfg] = idx
        # find nearest grid nodes to start & goal
        def nearest_grid(q):
            idx = []
            for i, (mn, mx) in enumerate(limits):
                val = q[i]
                arr = grids[i]
                # pick index with min abs diff
                ind = int(np.argmin(np.abs(arr - val)))
                idx.append(ind)
            return tuple(idx)
        start_idx = nearest_grid(start)
        goal_idx = nearest_grid(goal)
        start_cfg = idx_to_config[start_idx]
        goal_cfg = idx_to_config[goal_idx]
        # Check collision on start/goal grid nodes
        if collision_fn(start_cfg) or collision_fn(goal_cfg):
            print("A*-Grid: start or goal grid node in collision; abort.")
            return None
        # A* over 6D grid neighbors (6-connected)
        import heapq
        open_set = [(0.0, start_idx)]
        g_score = {start_idx: 0.0}
        came_from = {start_idx: None}
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_idx:
                # reconstruct
                path = []
                cur = current
                while cur is not None:
                    path.append(idx_to_config[cur])
                    cur = came_from[cur]
                return path[::-1]
            # neighbors: increment/decrement each joint index by 1
            for i in range(len(limits)):
                for delta in (-1, 1):
                    nei = list(current)
                    nei[i] += delta
                    if nei[i] < 0 or nei[i] >= per_joint:
                        continue
                    nei = tuple(nei)
                    cfg_cur = idx_to_config[current]
                    cfg_nei = idx_to_config[nei]
                    # check edge collision
                    if not is_path_clear(cfg_cur, cfg_nei, collision_fn, STEP_SIZE):
                        continue
                    tentative_g = g_score[current] + get_distance(cfg_cur, cfg_nei)
                    if tentative_g < g_score.get(nei, inf):
                        came_from[nei] = current
                        g_score[nei] = tentative_g
                        f = tentative_g + get_distance(cfg_nei, goal_cfg)
                        heapq.heappush(open_set, (f, nei))
        print("A*-Grid failed to find a path.")
        return None

    def stomp_like(initial_path, collision_fn, iterations=200, noise_scale=0.05):
        """A tiny STOMP-like optimizer: iteratively perturb intermediate waypoints to reduce collisions and length."""
        if initial_path is None:
            return None
        path = [np.array(p) for p in initial_path]
        if len(path) <= 2:
            return initial_path
        for it in range(iterations):
            improved = False
            for i in range(1, len(path)-1):
                p = path[i]
                # apply gaussian perturbation
                delta = np.random.normal(scale=noise_scale, size=p.shape)
                cand = p + delta
                # clip to joint limits
                cand_clipped = np.array([np.clip(cand[j], joint_limits_list[j][0], joint_limits_list[j][1]) for j in range(len(cand))])
                # check local quality: collision-free segments before/after and lower local cost (sum distances)
                prev = path[i-1]
                nxt = path[i+1]
                if is_path_clear(tuple(prev), tuple(cand_clipped), collision_fn, STEP_SIZE) and is_path_clear(tuple(cand_clipped), tuple(nxt), collision_fn, STEP_SIZE):
                    old_cost = get_distance(prev, p) + get_distance(p, nxt)
                    new_cost = get_distance(prev, cand_clipped) + get_distance(cand_clipped, nxt)
                    if new_cost < old_cost:
                        path[i] = cand_clipped
                        improved = True
            if not improved:
                break
        return [tuple(p) for p in path]

    # --- Main Execution ---
    PLANNER = 'PRM'  # change this string to select another planner
    print(f"Running {PLANNER}...")
    start_time = time.time()
    # ----------------- PLANNER SELECTION -----------------
    # Choose one of: 'RRT-Connect', 'RRT', 'RRT*', 'BiRRT*', 'InformedRRT*', 'PRM', 'LazyPRM', 'PRM*', 'FMT*', 'BIT*', 'A*-Grid', 'STOMP'
    # if PLANNER == 'RRT-Connect':
    #     rrt_path = rrt_connect(start_config, goal_config, joint_limits_list, collision_fn)
    # elif PLANNER == 'RRT':
    #     rrt_path = rrt_basic(start_config, goal_config, joint_limits_list, collision_fn)
    # elif PLANNER == 'RRT*':
    #     rrt_path = rrt_star(start_config, goal_config, joint_limits_list, collision_fn)
    # elif PLANNER == 'BiRRT*':
    #     rrt_path = birrt_star(start_config, goal_config, joint_limits_list, collision_fn)
    # elif PLANNER == 'InformedRRT*':
    #     rrt_path = informed_rrt_star(start_config, goal_config, joint_limits_list, collision_fn)
    elif PLANNER == 'PRM':
        rrt_path = prm(start_config, goal_config, joint_limits_list, collision_fn, n_samples=500, k=12)
    elif PLANNER == 'LazyPRM':
        rrt_path = lazy_prm(start_config, goal_config, joint_limits_list, collision_fn, n_samples=400, k=10)
    elif PLANNER == 'PRM*':
        rrt_path = prm_star(start_config, goal_config, joint_limits_list, collision_fn, n_samples=500, k=14)
    elif PLANNER == 'FMT*':
        rrt_path = fmt_star(start_config, goal_config, joint_limits_list, collision_fn, n_samples=400, neighbor_radius=0.6)
    elif PLANNER == 'BIT*':
        rrt_path = bit_star(start_config, goal_config, joint_limits_list, collision_fn, batch_size=200, k=12)
    elif PLANNER == 'A*-Grid':
        rrt_path = a_star_grid(start_config, goal_config, joint_limits_list, collision_fn, per_joint=4)
    elif PLANNER == 'STOMP':
        # STOMP is an optimizer; run basic RRT to get initialization then optimize
        init = rrt_basic(start_config, goal_config, joint_limits_list, collision_fn)
        if init is None:
            rrt_path = None
        else:
            rrt_path = stomp_like(init, collision_fn, iterations=300, noise_scale=0.05)
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
