import time
import csv
import numpy as np
from utils import draw_sphere_marker 
from utils import load_env, get_collision_fn_PR2, execute_trajectory
from pybullet_tools.utils import (
    connect, disconnect, wait_if_gui, joint_from_name,
    get_joint_positions, get_joint_info, link_from_name
)

from helper import (
    get_ee_path, shortcut_smooth, get_distance,
    compute_path_length, compute_ee_travel_distance, compute_max_joint_jump
)

from planners import (
    rrt_connect, rrt_basic, rrt_star,
    birrt_star, informed_rrt_star, with_node_count
)

# ============================================================
# MAIN SCRIPT WITH MULTI-RUN EVALUATION + CSV EXPORT
# ============================================================
def main():

    # ----------------------------
    # USER SETTINGS
    # ----------------------------
    PLANNER = "RRT-Connect"     # ← select planner here
    N_RUNS  = 20                # ← number of repeated trials

    CSV_FILENAME = f"results_{PLANNER.replace('*','star')}.csv"

    # header for CSV
    csv_header = [
        "run_id",
        "nodes_expanded",
        "planning_time",
        "raw_waypoints",
        "raw_path_length",
        "smooth_waypoints",
        "smooth_path_length",
        "improvement_percent",
        "ee_travel_distance",
        "max_joint_jump"
    ]

    # to store all results
    all_results = []

    # ============================================================
    # REPEAT PLANNING N TIMES
    # ============================================================
    for run in range(1, N_RUNS + 1):

        print("\n==========================================")
        print(f"  RUN {run} / {N_RUNS}  — Planner = {PLANNER}")
        print("==========================================")

        connect(use_gui=False)   # change to True if you want GUI every run
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

        STEP_SIZE = 0.05
        GOAL_BIAS = 0.1
        SMOOTH_ITER = 200

        # ----------------------------
        # PLAN — wrapped with node counter
        # ----------------------------
        t0 = time.time()

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
        else:
            print("Invalid planner.")
            disconnect()
            return

        planning_time = time.time() - t0

        # ----------------------------
        # If failure: store failed row
        # ----------------------------
        if not rrt_path:
            print("❌ Planner failed — storing result as NaN.")
            all_results.append([
                run, node_count, planning_time,
                0, np.nan, 0, np.nan, np.nan, np.nan, np.nan
            ])
            disconnect()
            continue

        # ----------------------------
        # Compute raw metrics
        # ----------------------------
        raw_length = compute_path_length(rrt_path)

        # ----------------------------
        # Smooth the path
        # ----------------------------
        path_s = shortcut_smooth(rrt_path, collision_fn, SMOOTH_ITER, STEP_SIZE)
        smooth_length = compute_path_length(path_s)

        improve = (1 - smooth_length / raw_length) * 100.0

        # ----------------------------
        # EE Metrics
        # ----------------------------
        link_id = link_from_name(robots["pr2"], "l_gripper_tool_frame")

        ee_path = get_ee_path(robots["pr2"], joint_idx, link_id, path_s, interp_step_size=0.01)
        ee_travel = compute_ee_travel_distance(ee_path)
        max_jump = compute_max_joint_jump(path_s)

        # save metrics for this run
        all_results.append([
            run,
            node_count,
            planning_time,
            len(rrt_path),
            raw_length,
            len(path_s),
            smooth_length,
            improve,
            ee_travel,
            max_jump
        ])

        disconnect()

    # ============================================================
    # COMPUTE AVERAGES
    # ============================================================
    arr = np.array(all_results, dtype=float)

    avg_nodes     = np.nanmean(arr[:, 1])
    avg_time      = np.nanmean(arr[:, 2])
    avg_raw_waypts= np.nanmean(arr[:, 3])
    avg_raw_len   = np.nanmean(arr[:, 4])
    avg_smooth_wp = np.nanmean(arr[:, 5])
    avg_smooth_len = np.nanmean(arr[:, 6])
    avg_improve   = np.nanmean(arr[:, 7])
    avg_ee        = np.nanmean(arr[:, 8])
    avg_jump      = np.nanmean(arr[:, 9])

    print("\n==============================================")
    print(f" AVERAGE PERFORMANCE over {N_RUNS} runs")
    print("==============================================")
    print(f"Avg nodes expanded       : {avg_nodes:.1f}")
    print(f"Avg planning time (s)    : {avg_time:.3f}")
    print(f"Avg raw waypoint count   : {avg_raw_waypts:.1f}")
    print(f"Avg raw path length      : {avg_raw_len:.4f}")
    print(f"Avg smoothed waypoints   : {avg_smooth_wp:.1f}")
    print(f"Avg smoothed path length : {avg_smooth_len:.4f}")
    print(f"Avg improvement (%)      : {avg_improve:.2f}")
    print(f"Avg EE travel distance   : {avg_ee:.4f}")
    print(f"Avg max joint jump       : {avg_jump:.4f}")

    # append averages as final row in CSV
    all_results.append([
        "AVG", avg_nodes, avg_time, avg_raw_waypts, avg_raw_len,
        avg_smooth_wp, avg_smooth_len, avg_improve, avg_ee, avg_jump
    ])

    # ============================================================
    # WRITE CSV FILE
    # ============================================================
    with open(CSV_FILENAME, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(all_results)

    print(f"\nCSV saved → {CSV_FILENAME}\n")



if __name__ == "__main__":
    main()
