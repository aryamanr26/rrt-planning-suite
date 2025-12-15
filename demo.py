import time
import numpy as np
from utils import draw_sphere_marker 
from utils import load_env, get_collision_fn_PR2, execute_trajectory
from pybullet_tools.utils import (
    connect, disconnect, wait_if_gui, joint_from_name,
    get_joint_positions, get_joint_info, link_from_name
)

from helper import get_ee_path, shortcut_smooth, get_distance, compute_path_length, compute_ee_travel_distance, compute_max_joint_jump
from planners import (rrt_connect, rrt_basic, rrt_star, 
                      birrt_star, informed_rrt_star, 
                      with_node_count, prm, lazy_prm, prm_star)



# ---------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------
def main():

    connect(use_gui=True)
    robots, obstacles = load_env("pr2table.json")

    joint_names = (
        "l_shoulder_pan_joint", "l_shoulder_lift_joint",
        "l_elbow_flex_joint", "l_upper_arm_roll_joint",
        "l_forearm_roll_joint", "l_wrist_flex_joint"
    )
    joint_idx = [joint_from_name(robots["pr2"], j) for j in joint_names]

    joint_limits = {
        j: (
            get_joint_info(robots["pr2"], idx).jointLowerLimit,
            get_joint_info(robots["pr2"], idx).jointUpperLimit
        )
        for j, idx in zip(joint_names, joint_idx)
    }
    joint_limits_list = [joint_limits[j] for j in joint_names]

    collision_fn = get_collision_fn_PR2(
        robots["pr2"], joint_idx, list(obstacles.values())
    )

    start_config = tuple(get_joint_positions(robots["pr2"], joint_idx))
    goal_config = (0.5, 0.33, -1.548, 1.557, -1.32, -0.1928)

    # --- RRT Parameters ---
    STEP_SIZE = 0.05
    GOAL_BIAS = 0.1
    MAX_ITER = 5000
    SMOOTH_ITER = 200

    # ----------------------------
    # SELECT PLANNER
    # ----------------------------
    PLANNER = "PRM"

    print(f"\n==============================")
    print(f" Running {PLANNER}")
    print(f"==============================")

    # ----------------------------
    # PLAN
    # ----------------------------
    t0 = time.time()

    # Wrap planner in node counter
    if PLANNER == "RRT-Connect":
        rrt_path, node_count = with_node_count(
            lambda: rrt_connect(start_config, goal_config, joint_limits_list, collision_fn)
        )
    elif PLANNER == "RRT":
        rrt_path, node_count = with_node_count(
            lambda: rrt_basic(start_config, goal_config, joint_limits_list, collision_fn)
        )
    elif PLANNER == "RRT*":
        rrt_path, node_count = with_node_count(
            lambda: rrt_star(start_config, goal_config, joint_limits_list, collision_fn)
        )
    elif PLANNER == "BiRRT*":
        rrt_path, node_count = with_node_count(
            lambda: birrt_star(start_config, goal_config, joint_limits_list, collision_fn)
        )
    elif PLANNER == "InformedRRT*":
        rrt_path, node_count = with_node_count(
            lambda: informed_rrt_star(start_config, goal_config, joint_limits_list, collision_fn)
        )
    elif PLANNER == "PRM":
        rrt_path, node_count = with_node_count(
            lambda: prm(start_config, goal_config, joint_limits_list, collision_fn)
        )
    elif PLANNER == "LazyPRM":
        rrt_path, node_count = with_node_count(
            lambda: lazy_prm(start_config, goal_config, joint_limits_list, collision_fn)
        )
    elif PLANNER == "PRM*":
        rrt_path, node_count = with_node_count(
            lambda: prm_star(start_config, goal_config, joint_limits_list, collision_fn)
        )
    else:
        print("Invalid planner.")
        return

    planning_time = time.time() - t0

    # ------------------------------------------------
    # If no solution, print metrics and exit
    # ------------------------------------------------
    if not rrt_path:
        print("\n❌ Planner failed.")
        print(f"Nodes expanded: {node_count}")
        print(f"Planning time: {planning_time:.3f} s")
        disconnect()
        return

    # ------------------------------------------------
    # Compute raw path metrics
    # ------------------------------------------------
    raw_length = compute_path_length(rrt_path)

    print(f"\n--- Raw Path Metrics ---")
    print(f"Nodes expanded           : {node_count}")
    print(f"Planning time            : {planning_time:.3f} s")
    print(f"Raw waypoint count       : {len(rrt_path)}")
    print(f"Raw path length (config) : {raw_length:.4f}")

    # ------------------------------------------------
    # Smooth the path
    # ------------------------------------------------
    path_s = shortcut_smooth(rrt_path, collision_fn, SMOOTH_ITER, STEP_SIZE)
    smooth_length = compute_path_length(path_s)

    print(f"\n--- Smoothed Path Metrics ---")
    print(f"Smoothed waypoint count   : {len(path_s)}")
    print(f"Smoothed path length      : {smooth_length:.4f}")
    print(f"Path improvement          : {(1 - smooth_length / raw_length) * 100:.2f}%")

    # ------------------------------------------------
    # EE Metrics
    # ------------------------------------------------
    link_id = link_from_name(robots["pr2"], "l_gripper_tool_frame")

    # --- ADDED: compute EE path for ORIGINAL RRT ---
    ee_path_rrt = get_ee_path(robots["pr2"], joint_idx, link_id, rrt_path, interp_step_size=0.01)

    ee_path = get_ee_path(robots["pr2"], joint_idx, link_id, path_s, interp_step_size=0.01)
    ee_travel = compute_ee_travel_distance(ee_path)
    max_jump = compute_max_joint_jump(path_s)

    print(f"\n--- End-Effector Metrics ---")
    print(f"EE travel distance (m)    : {ee_travel:.4f}")
    print(f"Max joint jump            : {max_jump:.4f}")

    print("\n==============================")
    print("   EXECUTING TRAJECTORY")
    print("==============================")

    # ------------------------------------------------
    # Visualization: plot BOTH paths
    # ------------------------------------------------

    # ORIGINAL RRT path → RED
    for p in ee_path_rrt:
        draw_sphere_marker(p, 0.005, (1, 0, 0, 0.8)) # Red, now very small

    # SMOOTHED path → BLUE
    for p in ee_path:
        draw_sphere_marker(p, 0.007, (0, 0, 1, 0.8)) # Blue, slightly larger

    execute_trajectory(robots["pr2"], joint_idx, path_s, sleep=0.08)

    wait_if_gui()
    disconnect()


if __name__ == "__main__":
    main()
