import numpy as np
from utils import load_env, get_collision_fn_PR2, execute_trajectory
from pybullet_tools.utils import connect, disconnect, wait_if_gui, joint_from_name, get_joint_positions, set_joint_positions, get_joint_info, get_link_pose, link_from_name
import random
import math
random.seed(420)
import networkx as nx
import matplotlib.pyplot as plt
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
    STEP_SIZE = 0.05 # 0.05 PRM
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
    
    def visualize_prm(graph, start=None, goal=None, path=None):
        G = nx.Graph()

        # Build NetworkX graph
        for u, neighbors in graph.items():
            for v, w in neighbors:
                G.add_edge(u, v, weight=w)

        # Node positions (assumes 2D configs)
        pos = {node: (node[0], node[1]) for node in G.nodes}

        plt.figure(figsize=(8, 8))

        # Draw roadmap nodes + edges
        nx.draw(
            G,
            pos,
            node_size=12,
            node_color="lightgray",
            edge_color="gray",
            width=0.5,
            with_labels=False
        )

        # Highlight solution path
        if path is not None and len(path) > 1:
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=path_edges,
                edge_color="blue",
                width=2.5
            )

        # Highlight start
        if start is not None:
            plt.scatter(
                start[0], start[1],
                c="green",
                s=120,
                marker="o",
                edgecolors="black",
                linewidths=1.5,
                label="Start"
            )

        # Highlight goal
        if goal is not None:
            plt.scatter(
                goal[0], goal[1],
                c="red",
                s=120,
                marker="X",
                edgecolors="black",
                linewidths=1.5,
                label="Goal"
            )

        plt.legend()
        plt.axis("equal")
        plt.title("PRM Roadmap with Start and Goal")
        plt.show()

    # --- Additional planners: PRM, Lazy PRM, PRM*, FMT*, BIT*, A*-grid, STOMP-like optimizer ---

    import heapq
    from math import inf

    def build_prm(samples, k, limits, collision_fn):
        """Build PRM graph using node indices instead of raw config tuples."""
        n = len(samples)
        graph = {i: [] for i in range(n)}  # adjacency list

        for i, s in enumerate(samples):
            # find k nearest neighbors by configuration distance
            dists = [(j, get_distance(s, samples[j])) for j in range(n) if j != i]
            dists.sort(key=lambda x: x[1])
            neighbors = dists[:k]

            for j, d in neighbors:
                if is_path_clear(s, samples[j], collision_fn, STEP_SIZE):
                    graph[i].append((j, d))
                    graph[j].append((i, d))  # ensure bidirectional

        return graph


    def dijkstra_graph(graph, start_id, goal_id):
        """Run Dijkstra using node IDs."""
        dist = {start_id: 0.0}
        parent = {start_id: None}
        pq = [(0.0, start_id)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            if u == goal_id:
                # reconstruct
                path_ids = []
                curr = u
                while curr is not None:
                    path_ids.append(curr)
                    curr = parent[curr]
                return path_ids[::-1]

            for v, w in graph.get(u, []):
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))

        return None

    # def prm(start, goal, limits, collision_fn, n_samples=400, k=10):
    #     """Probabilistic Roadmap (PRM)."""
    #     samples = [start, goal]
    #     # sample random collision-free nodes
    #     tries = 0
    #     while len(samples) < n_samples and tries < n_samples * 10:
    #         q = sample_config(goal, limits, GOAL_BIAS)
    #         if not collision_fn(q):
    #             samples.append(q)
    #         tries += 1
    #     graph = build_prm(samples, k, limits, collision_fn)
    #     path = dijkstra_graph(graph, start, goal)
    #     if path is None:
    #         print("PRM failed to find a path.")
    #     return path

    def prm(start, goal, limits, collision_fn, n_samples=400, k=15):
        """Probabilistic Roadmap (PRM)"""

        # 1. Reject invalid start / goal
        if collision_fn(start) or collision_fn(goal):
            print("Start or goal in collision.")
            return None

        # 2. Generate collision-free random samples
        samples = []
        tries = 0
        while len(samples) < n_samples and tries < n_samples * 10:
            q = sample_config(goal, limits, 0.0)
            if not collision_fn(q):
                samples.append(q)
            tries += 1

        # Add start and goal to sample list
        start_id = len(samples)
        samples.append(start)

        goal_id = len(samples)
        samples.append(goal)

        # 3. Build PRM graph
        print("Building PRM graph...")
        graph = build_prm(samples, k, limits, collision_fn)

        # 4. Add start and goal to graph
        print("[Query Phase] Connecting start and goal to PRM graph...")
        graph[start] = []
        graph[goal] = []

        nearest = knn_nodes(samples, start, 10)
        print(f"Distances to start: {[get_distance(start, n) for n in nearest]}")
        # 5. Connect start to roadmap
        for q in knn_nodes(samples, start, 2*k):
            if is_path_clear(start, q, collision_fn, STEP_SIZE):
                w = get_distance(start, q)
                graph[start].append((q, w))
                graph[q].append((start, w))

        # 5. Connect goal to nearest neighbors
        dists_goal = [(i, get_distance(goal, samples[i])) for i in range(len(samples) - 2)]
        dists_goal.sort(key=lambda x: x[1])

        for i, d in dists_goal[:2*k]:
            if is_path_clear(goal, samples[i], collision_fn, STEP_SIZE):
                graph[goal_id].append((i, d))
                graph[i].append((goal_id, d))

        # 7. Graph search
        print(f"Start connections: {len(graph[start])}")
        print(f"Goal connections: {len(graph[goal])}")
        print("Searching for path in PRM graph...")
        path = dijkstra_graph(graph, start, goal)
        visualize_prm(graph, start, goal, path)

        if path_ids is None:
            print("PRM failed to find a path.")
            return None

        # Convert node IDs back to actual configs
        path = [samples[i] for i in path_ids]

        # Optional: visualize roadmap
        visualize_prm(graph, start, goal, path)

        return path


    # def knn_nodes(nodes, q, k):
    #     """Return k nearest nodes (by get_distance) to q from list nodes."""
    #     nodes_sorted = sorted(nodes, key=lambda n: get_distance(n, q))
    #     return nodes_sorted[:k]

    # def build_prm(samples, k, limits, collision_fn):
    #     """Build a PRM graph given list of sample configs (assumed collision-free)."""
    #     graph = {s: [] for s in samples}
    #     for i, s in enumerate(samples):
    #         neighbors = knn_nodes(samples, s, k+1)  # includes itself possibly
    #         for n in neighbors:
    #             if n == s:
    #                 continue
    #             if is_path_clear(s, n, collision_fn, STEP_SIZE):
    #                 w = get_distance(s, n)
    #                 graph[s].append((n, w))
    #                 graph[n].append((s, w))

    #     return graph

    # def dijkstra_graph(graph, start, goal):
    #     """Dijkstra on adjacency dict: graph[node] = [(neighbor, weight), ...]"""
    #     if start not in graph or goal not in graph:
    #         return None
    #     dist = {start: 0.0}
    #     parent = {start: None}
    #     pq = [(0.0, start)]
    #     visited = set()
    #     while pq:
    #         d, u = heapq.heappop(pq)
    #         if u in visited:
    #             continue
    #         visited.add(u)
    #         if u == goal:
    #             # reconstruct
    #             path = []
    #             cur = u
    #             while cur is not None:
    #                 path.append(cur)
    #                 cur = parent[cur]
    #             return path[::-1]
    #         for v, w in graph.get(u, []):
    #             nd = d + w
    #             if nd < dist.get(v, inf):
    #                 dist[v] = nd
    #                 parent[v] = u
    #                 heapq.heappush(pq, (nd, v))
    #     return None

    # # def prm(start, goal, limits, collision_fn, n_samples=400, k=10):
    # #     """Probabilistic Roadmap (PRM)."""
    # #     samples = [start, goal]
    # #     # sample random collision-free nodes
    # #     tries = 0
    # #     while len(samples) < n_samples and tries < n_samples * 10:
    # #         q = sample_config(goal, limits, GOAL_BIAS)
    # #         if not collision_fn(q):
    # #             samples.append(q)
    # #         tries += 1
    # #     graph = build_prm(samples, k, limits, collision_fn)
    # #     path = dijkstra_graph(graph, start, goal)
    # #     if path is None:
    # #         print("PRM failed to find a path.")
    # #     return path

    # def prm(start, goal, limits, collision_fn, n_samples=600, k=15):
    #     """Probabilistic Roadmap (PRM)"""

    #     # 1. Reject invalid start / goal
    #     if collision_fn(start) or collision_fn(goal):
    #         print("Start or goal in collision.")
    #         return None

    #     # 2. Sample collision-free configurations (roadmap only)
    #     samples = []
    #     tries = 0
    #     while len(samples) < n_samples and tries < n_samples * 10:
    #         q = sample_config(goal, limits, GOAL_BIAS)
    #         if not collision_fn(q):
    #             samples.append(q)
    #         tries += 1

    #     # 3. Build roadmap graph (offline)
    #     print("Building PRM graph...")
    #     graph = build_prm(samples, k, limits, collision_fn)

    #     # 4. Add start and goal to graph
    #     print("[Query Phase] Connecting start and goal to PRM graph...")
    #     graph[start] = []
    #     graph[goal] = []

    #     # 5. Connect start to roadmap
    #     for q in knn_nodes(samples, start, 2*k):
    #         if is_path_clear(start, q, collision_fn, STEP_SIZE):
    #             w = get_distance(start, q)
    #             graph[start].append((q, w))
    #             graph[q].append((start, w))

    #     # 6. Connect goal to roadmap
    #     for q in knn_nodes(samples, goal, 2*k):
    #         if is_path_clear(goal, q, collision_fn, STEP_SIZE):
    #             w = get_distance(goal, q)
    #             graph[goal].append((q, w))
    #             graph[q].append((goal, w))

    #     # 7. Graph search
        
    #     print("Searching for path in PRM graph...")
    #     path = dijkstra_graph(graph, start, goal)
    #     visualize_prm(graph, start, goal, path)

    #     if path is None:
    #         print("PRM failed to find a path.")
    #     return path

    def lazy_prm(start, goal, limits, collision_fn, n_samples=300, k=10):
        """Lazy PRM: construct connectivity using proximity, defer edge collision-check until needed."""
        
        print(f"\n=== Lazy PRM Starting ===")
        print(f"Target samples: {n_samples}, k-nearest: {k}")
        
        # 1. Sample collision-free nodes
        samples = [start, goal]
        tries = 0
        while len(samples) < n_samples and tries < n_samples * 10:
            q = sample_config(goal, limits, 0.0)  # No goal bias for PRM
            if not collision_fn(q):
                samples.append(q)
            tries += 1
        
        print(f"✓ Sampled {len(samples)} collision-free nodes (tries: {tries})")
        
        # 2. Build UNDIRECTED graph by proximity (NO collision checks yet)
        graph = {s: [] for s in samples}
        total_edges = 0
        for s in samples:
            neighbors = knn_nodes(samples, s, k+1)
            for n in neighbors:
                if n == s:
                    continue
                w = get_distance(s, n)
                graph[s].append((n, w))  # Add edge s→n
                graph[n].append((s, w))  # Add edge n→s (undirected!)
                total_edges += 1
        
        # Count unique edges (divide by 2 since undirected)
        unique_edges = total_edges // 2
        print(f"✓ Built graph: {len(graph)} nodes, ~{unique_edges} edges (unchecked)")
        print(f"  Start has {len(graph[start])} neighbors")
        print(f"  Goal has {len(graph[goal])} neighbors")
        
        # 3. Lazy search: find path, validate, remove invalid edges, repeat
        checked_edges = set()
        iteration = 0
        
        while True:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            path = dijkstra_graph(graph, start, goal)
            
            if path is None:
                print(f"✗ Lazy PRM failed to find a path after {iteration} iterations.")
                print(f"  Total edges checked: {len(checked_edges)}")
                return None
            
            print(f"  Found candidate path with {len(path)} nodes, {len(path)-1} edges")
            
            # Validate each edge in the path
            all_valid = True
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge = (min(u, v), max(u, v))
                
                if edge not in checked_edges:
                    checked_edges.add(edge)
                    if not is_path_clear(u, v, collision_fn, STEP_SIZE):
                        print(f"  ✗ Edge {i} in collision, removing from graph")
                        # Remove BOTH directions of invalid edge
                        graph[u] = [(n, w) for n, w in graph[u] if n != v]  # Remove u→v
                        graph[v] = [(n, w) for n, w in graph[v] if n != u]  # Remove v→u
                        all_valid = False
                        break
            
            if all_valid:
                print(f"✓ Valid path found!")
                print(f"  Total iterations: {iteration}")
                print(f"  Total edges checked: {len(checked_edges)}")
                print(f"  Path length: {len(path)} nodes")
                return path
            else:
                print(f"  Re-planning without invalid edge...")

    # def prm_star(start, goal, limits, collision_fn, n_samples=400, k=12):
    #     """PRM* - PRM with local rewiring (simple variant)."""
    #     samples = [start, goal]
    #     tries = 0
    #     while len(samples) < n_samples and tries < n_samples * 10:
    #         q = sample_config(goal, limits, GOAL_BIAS)
    #         if not collision_fn(q):
    #             samples.append(q)
    #         tries += 1
    #     graph = {s: [] for s in samples}
    #     cost = {}
    #     # incremental insertion + rewiring
    #     for s in samples:
    #         # connect to neighbors
    #         neighbors = knn_nodes(samples, s, k+1)
    #         for n in neighbors:
    #             if n == s:
    #                 continue
    #             if is_path_clear(s, n, collision_fn, STEP_SIZE):
    #                 w = get_distance(s, n)
    #                 graph[s].append((n, w))
    #     # run Dijkstra on built graph
    #     path = dijkstra_graph(graph, start, goal)
    #     if path is None:
    #         print("PRM* failed to find a path.")
    #     return path
    
    def prm_star(start, goal, limits, collision_fn, n_samples=600, k=15):
        """PRM* with radius-based connections - with adaptive start/goal connection."""
        
        print("\n=== PRM* (Naive) Starting ===")
        
        # 1. Reject invalid start/goal
        if collision_fn(start) or collision_fn(goal):
            print("Start or goal in collision.")
            return None
        
        # 2. Sample collision-free nodes
        samples = []
        tries = 0
        while len(samples) < n_samples and tries < n_samples * 10:
            q = sample_config(goal, limits, 0.0)  # No goal bias
            if not collision_fn(q):
                samples.append(q)
            tries += 1
        
        print(f"✓ Sampled {len(samples)} collision-free nodes")
        
        n = len(samples)
        d = len(start)  # Dimension
        
        # 3. Calculate PRM* connection radius
        gamma = 2.0 * ((1 + 1/d) * (1/math.pi)) ** (1/d)
        r = gamma * (math.log(n) / n) ** (1/d)
        
        print(f"  Dimension: {d}, Connection radius: {r:.4f}")
        
        # 4. Build graph - O(n²) all-pairs check
        print("Building PRM* graph (checking all pairs)...")
        graph = {tuple(s): [] for s in samples}
        edges_checked = 0
        edges_added = 0
        
        for i, s in enumerate(samples):
            s_tuple = tuple(s)
            for j, t in enumerate(samples):
                if i >= j:  # Avoid duplicates and self-loops
                    continue
                
                edges_checked += 1
                t_tuple = tuple(t)
                
                # Check if within radius
                if get_distance(s_tuple, t_tuple) <= r:
                    # Validate edge collision-free
                    if is_path_clear(s_tuple, t_tuple, collision_fn, STEP_SIZE):
                        w = get_distance(s_tuple, t_tuple)
                        graph[s_tuple].append((t_tuple, w))
                        graph[t_tuple].append((s_tuple, w))
                        edges_added += 1
        
        print(f"✓ Graph built: {edges_checked} pairs checked, {edges_added} edges added")
        
        # 5. Add start and goal to graph with k-nearest (not limited by radius)
        print("Connecting start and goal...")
        start_tuple = tuple(start)
        goal_tuple = tuple(goal)
        graph[start_tuple] = []
        graph[goal_tuple] = []
        
        # k = 15  # Connect to 15 nearest neighbors
        
        # Connect start to k-nearest
        start_neighbors = knn_nodes([tuple(s) for s in samples], start_tuple, k)
        for s_tuple in start_neighbors:
            if is_path_clear(start_tuple, s_tuple, collision_fn, STEP_SIZE):
                w = get_distance(start_tuple, s_tuple)
                graph[start_tuple].append((s_tuple, w))
                graph[s_tuple].append((start_tuple, w))
        
        # Connect goal to k-nearest
        goal_neighbors = knn_nodes([tuple(s) for s in samples], goal_tuple, k)
        for s_tuple in goal_neighbors:
            if is_path_clear(goal_tuple, s_tuple, collision_fn, STEP_SIZE):
                w = get_distance(goal_tuple, s_tuple)
                graph[goal_tuple].append((s_tuple, w))
                graph[s_tuple].append((goal_tuple, w))
        
        print(f"  Start connections: {len(graph[start_tuple])}")
        print(f"  Goal connections: {len(graph[goal_tuple])}")
        
        if len(graph[start_tuple]) == 0:
            print("  WARNING: Start could not connect to roadmap!")
        if len(graph[goal_tuple]) == 0:
            print("  WARNING: Goal could not connect to roadmap!")
        
        # 6. Search for path
        print("Searching for path...")
        path = dijkstra_graph(graph, start_tuple, goal_tuple)
        
        if path is None:
            print("✗ PRM* failed to find a path.")
        else:
            print(f"✓ Path found with {len(path)} nodes")
        
        return path
    
    # def fmt_star(start, goal, limits, collision_fn, n_samples=400, neighbor_radius=0.6):
    #     """Simplified FMT*: sample nodes and run a Dijkstra-like expansion constrained by neighbor radius."""
    #     # sample nodes including start & goal
    #     samples = [start, goal]
    #     tries = 0
    #     while len(samples) < n_samples and tries < n_samples * 10:
    #         q = sample_config(goal, limits, GOAL_BIAS)
    #         if not collision_fn(q):
    #             samples.append(q)
    #         tries += 1
    #     # build neighbor lists (without checking collisions yet)
    #     neighbor_map = {s: [n for n in samples if n != s and get_distance(n, s) <= neighbor_radius] for s in samples}
    #     # Initialize sets
    #     V_unvisited = set(samples)
    #     V_open = set([start])
    #     parent = {start: None}
    #     cost = {start: 0.0}
    #     while V_open:
    #         # pick lowest-cost node in V_open
    #         z = min(V_open, key=lambda v: cost.get(v, inf))
    #         V_open.remove(z)
    #         V_unvisited.discard(z)
    #         # consider neighbors of z that are unvisited
    #         for x in list(V_unvisited):
    #             if x in neighbor_map[z]:
    #                 # find y in V_open ∩ neighbor(x) that minimizes cost[y] + dist(y,x)
    #                 Y = [y for y in V_open if y in neighbor_map[x]]
    #                 if not Y:
    #                     continue
    #                 # try connecting x via best y
    #                 best_y = None
    #                 best_cost = inf
    #                 for y in Y:
    #                     if is_path_clear(y, x, collision_fn, STEP_SIZE):
    #                         c = cost.get(y, inf) + get_distance(y, x)
    #                         if c < best_cost:
    #                             best_cost = c
    #                             best_y = y
    #                 if best_y is not None:
    #                     cost[x] = best_cost
    #                     parent[x] = best_y
    #                     V_open.add(x)
    #                     V_unvisited.discard(x)
    #         if goal in parent:
    #             # reconstruct
    #             path = []
    #             cur = goal
    #             while cur is not None:
    #                 path.append(cur)
    #                 cur = parent.get(cur, None)
    #             return path[::-1]
    #     print("FMT* failed to find a path.")
    #     return None

    def fmt_star(start, goal, limits, collision_fn, n_samples=600):
        """FMT*: Fast Marching Tree"""
        
        # 1. Sample all nodes upfront
        samples = [goal]  # Include goal in samples
        while len(samples) < n_samples:
            q = sample_config(goal, limits, 0.0)
            if not collision_fn(q):
                samples.append(q)
        
        n = len(samples)
        d = len(start)
        
        # Calculate connection radius (same as PRM*)
        gamma = 2.0 * ((1 + 1/d) * (1/math.pi)) ** (1/d)
        r = gamma * (math.log(n) / n) ** (1/d)
        
        # 2. Initialize sets
        V_open = {tuple(start)}  # Nodes in tree, not yet expanded
        V_closed = set()          # Nodes already expanded
        V_unvisited = {tuple(s) for s in samples}  # Not yet in tree
        
        # Track costs and parents
        cost = {tuple(start): 0.0}
        parent = {tuple(start): None}
        
        # 3. Grow tree until goal reached
        while V_open:
            # Pick lowest-cost node from open set
            z = min(V_open, key=lambda node: cost[node])
            
            if z == tuple(goal):
                # Reconstruct path
                path = []
                current = z
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return path[::-1]
            
            V_open.remove(z)
            
            # Find neighbors of z within radius r
            X_near = [x for x in V_unvisited if get_distance(z, x) <= r]
            
            # Try to connect neighbors to z
            for x in X_near:
                # Find all nodes in V_open that could reach x
                Y_near = [y for y in V_open if get_distance(y, x) <= r]
                
                # Find best parent for x
                y_min = None
                cost_min = float('inf')
                
                for y in Y_near:
                    if is_path_clear(y, x, collision_fn, STEP_SIZE):
                        c = cost[y] + get_distance(y, x)
                        if c < cost_min:
                            y_min = y
                            cost_min = c
                
                # Add x to tree if connection found
                if y_min is not None:
                    parent[x] = y_min
                    cost[x] = cost_min
                    V_open.add(x)
                    V_unvisited.remove(x)
            
            # Mark z as fully expanded
            V_closed.add(z)
        
        print("FMT* failed to find path")
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
    PLANNER = 'FMT*'  # change this string to select another planner
    print(f"Running {PLANNER}...")
    start_time = time.time()
    # ----------------- PLANNER SELECTION -----------------
    # Choose one of: 'RRT-Connect', 'RRT', 'RRT*', 'BiRRT*', 'InformedRRT*', 'PRM', 'LazyPRM', 'PRM*', 'FMT*', 'BIT*', 'A*-Grid', 'STOMP'
    if PLANNER == 'RRT-Connect':
        rrt_path = rrt_connect(start_config, goal_config, joint_limits_list, collision_fn)
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
        rrt_path = prm_star(start_config, goal_config, joint_limits_list, collision_fn, n_samples=500, k=15)
    elif PLANNER == 'FMT*':
        rrt_path = fmt_star(start_config, goal_config, joint_limits_list, collision_fn, n_samples=400)
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
