import numpy as np
import random
import heapq
import networkx as nx
import matplotlib.pyplot as plt 
from math import inf
from utils import draw_sphere_marker
from pybullet_tools.utils import (
    set_joint_positions, get_joint_positions, get_link_pose
)

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

def shortcut_smooth(path, collision_fn, iterations, step_size=0.05):
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
        
        if is_path_clear(q_a, q_b, collision_fn, step_size):
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

def knn_nodes(nodes, q, k):
    """Return k nearest nodes (by get_distance) to q from list nodes."""
    nodes_sorted = sorted(nodes, key=lambda n: get_distance(n, q))
    return nodes_sorted[:k]