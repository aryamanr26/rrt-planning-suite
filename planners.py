import random
import numpy as np
from helper import (
    get_distance, get_nearest_node, sample_config, steer,
    is_path_clear
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
    """Corrected Informed RRT* with multi-step extension + proper goal connection."""
    nodes = [start]
    parents = {start: None}
    cost = {start: 0.0}

    neighbor_radius = 0.4
    best_solution_cost = float('inf')
    solution_found = False

    # --------------------------------------------------------
    # Informed Sampling (Prolate Hyperspheroid approximation)
    # --------------------------------------------------------
    def sample_in_ellipsoid(start, goal, c_best):
        if c_best == float('inf'):
            return sample_config(goal, limits, GOAL_BIAS)

        s = np.array(start)
        g = np.array(goal)
        c_min = np.linalg.norm(g - s)

        if c_min == 0:
            return tuple(s)

        center = (s + g) / 2.0
        r1 = c_best / 2.0

        if c_best**2 - c_min**2 <= 0:
            r_other = 0.0
        else:
            r_other = np.sqrt(c_best**2 - c_min**2) / 2.0

        # Draw until we get a valid sample
        while True:
            sample = np.array([random.uniform(-r_other, r_other) for _ in range(len(s))])
            sample[0] = random.uniform(-r1, r1)

            q = center + sample

            # Clip to joint limits
            q_clipped = []
            for i, (mn, mx) in enumerate(limits):
                q_clipped.append(float(np.clip(q[i], mn, mx)))

            return tuple(q_clipped)

    # --------------------------------------------------------
    # Multi-step extension (same as RRT*)
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Main Informed RRT* loop
    # --------------------------------------------------------
    for it in range(1, MAX_ITERATIONS + 1):

        # If no solution yet → uniform sampling
        # If solution exists → informed sampling
        if not solution_found:
            q_rand = sample_config(goal, limits, GOAL_BIAS)
        else:
            q_rand = sample_in_ellipsoid(start, goal, best_solution_cost)

        # Nearest neighbor
        q_near = get_nearest_node(nodes, q_rand)

        # Multi-step extension
        extension = extend_towards(q_near, q_rand)
        if not extension:
            continue

        # Add last reachable node
        q_new = extension[-1]

        # ---------------- Parent selection (RRT*) ----------------
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

        # ---------------- Rewire neighbors ----------------
        for n in neighbors:
            if n == best_parent:
                continue
            if is_path_clear(q_new, n, collision_fn, STEP_SIZE):
                new_cost = cost[q_new] + get_distance(q_new, n)
                if new_cost < cost.get(n, float('inf')):
                    parents[n] = q_new
                    cost[n] = new_cost

        # ---------------- Try connecting to goal ----------------
        if is_path_clear(q_new, goal, collision_fn, STEP_SIZE):

            sol_cost = cost[q_new] + get_distance(q_new, goal)

            # first solution OR improved solution
            if sol_cost < best_solution_cost:
                best_solution_cost = sol_cost
                solution_found = True
                parents[goal] = q_new
                cost[goal] = sol_cost

                # Continue searching for better paths (Informed RRT*)
                # but do NOT return yet.

    # ---------------- After MAX_ITER — return best found ----------------
    if not solution_found:
        print("Informed RRT* failed to find a path.")
        return None

    # reconstruct best available path
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
