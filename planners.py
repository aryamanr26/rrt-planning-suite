import random
# random.seed(420)
import numpy as np
import math
from helper import (
    get_distance, get_nearest_node, sample_config, steer,
    is_path_clear, knn_nodes, dijkstra_graph, visualize_prm
)

# --- RRT Parameters ---
STEP_SIZE = 0.05 #   0.15 for BiRRT*
GOAL_BIAS = 0.1 #   0.3 for BiRRT*
MAX_ITERATIONS = 5000
SMOOTH_ITERATIONS = 200 # 

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

# ---------------------------------------------------
# Utility: Compute path length in configuration space
# ---------------------------------------------------
def compute_path_length(path):
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        total += get_distance(a, b)
    return total


# ---------------------------------------------------
# Utility: Compute EE travel distance
# ---------------------------------------------------
def compute_ee_travel_distance(ee_pts):
    if len(ee_pts) < 2:
        return 0.0
    dist = 0.0
    for p1, p2 in zip(ee_pts[:-1], ee_pts[1:]):
        dist += np.linalg.norm(np.array(p1) - np.array(p2))
    return dist


# ---------------------------------------------------
# Utility: Compute max joint jump
# ---------------------------------------------------
def compute_max_joint_jump(path):
    if len(path) < 2:
        return 0.0
    jumps = [get_distance(a, b) for a, b in zip(path[:-1], path[1:])]
    return max(jumps)


# ---------------------------------------------------
# WRAPPER to track node counts inside planners
# ---------------------------------------------------
def with_node_count(planner_func):
    """
    Wrapper that executes planner, then extracts node count by checking
    global or returned internal data structures.
    Since planners store nodes in a list named 'nodes' OR 'nodes_a'/'nodes_b',
    we detect the node list length dynamically after planner execution.
    """
    before_globals = set(globals().keys())

    result = planner_func()

    # detect newly created node lists
    after_globals = set(globals().keys())
    new_vars = after_globals - before_globals

    node_count = 0

    for var in new_vars:
        val = globals()[var]
        if isinstance(val, list):
            node_count = max(node_count, len(val))

    # Fallback: if planner returned a tuple containing nodes
    if isinstance(result, tuple):
        for part in result:
            if isinstance(part, list):
                node_count = max(node_count, len(part))

    return result, node_count

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

def prm(start, goal, limits, collision_fn, n_samples=400, k=15):
        """Probabilistic Roadmap (PRM)"""

        # 1. Reject invalid start / goal
        if collision_fn(start) or collision_fn(goal):
            print("Start or goal in collision.")
            return None

        # 2. Sample collision-free configurations (roadmap only)
        samples = []
        tries = 0
        while len(samples) < n_samples and tries < n_samples * 10:
            q = sample_config(goal, limits, 0.0)
            if not collision_fn(q):
                samples.append(q)
            tries += 1

        # 3. Build roadmap graph (offline)
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
            print(f"Type of q: {type(q)}, Type in graph keys: {type(list(graph.keys())[0])}")
            print(f"q in graph: {q in graph}")
            print(f"tuple(q) in graph: {tuple(q) in graph if hasattr(q, '__iter__') else 'N/A'}")
            if is_path_clear(start, q, collision_fn, STEP_SIZE):
                w = get_distance(start, q)
                graph[start].append((q, w))
                graph[q].append((start, w))

        # 6. Connect goal to roadmap
        for q in knn_nodes(samples, goal, 2*k):
            if is_path_clear(goal, q, collision_fn, STEP_SIZE):
                w = get_distance(goal, q)
                graph[goal].append((q, w))
                graph[q].append((goal, w))

        # 7. Graph search
        print(f"Start connections: {len(graph[start])}")
        print(f"Goal connections: {len(graph[goal])}")
        print("Searching for path in PRM graph...")
        path = dijkstra_graph(graph, start, goal)
        visualize_prm(graph, start, goal, path)

        if path is None:
            print("PRM failed to find a path.")
        return path

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