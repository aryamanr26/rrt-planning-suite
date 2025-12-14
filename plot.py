import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================================
# USER INPUT → CSV FILE NAME
# ================================
CSV_FILE = "results_RRT-Connect.csv"   # <-- change if needed

# ================================
# LOAD CSV
# ================================
df = pd.read_csv(CSV_FILE)

# Remove AVG row from "runs only"
df_runs = df[df["run_id"] != "AVG"].copy()

# Extract numeric values
df_runs["run_id"] = df_runs["run_id"].astype(int)

# Extract average row
df_avg = df[df["run_id"] == "AVG"].iloc[0]

# ========================================================
# Helper function: Plot one metric
# ========================================================
def plot_metric(column, ylabel):
    runs = df_runs["run_id"]
    values = df_runs[column].astype(float)
    avg_value = float(df_avg[column])

    plt.figure(figsize=(10, 5))
    
    # Plot metric values
    plt.plot(runs, values, marker="o", label=f"{ylabel} per run")

    # Plot average line
    plt.axhline(avg_value, color="red", linestyle="--",
                label=f"Average = {avg_value:.4f}")

    # Highlight average point
    plt.scatter([runs.iloc[-1] + 1], [avg_value], 
                color="red", marker="X", s=120,
                label="Average value")

    plt.xlabel("Run ID")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over runs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ========================================================
# GENERATE ALL PLOTS
# ========================================================
plot_metric("nodes_expanded", "Nodes Expanded")
plot_metric("planning_time", "Planning Time (s)")
plot_metric("raw_waypoints", "Raw Waypoint Count")
plot_metric("raw_path_length", "Raw Path Length")
plot_metric("smooth_waypoints", "Smoothed Waypoint Count")
plot_metric("smooth_path_length", "Smoothed Path Length")
plot_metric("improvement_percent", "Improvement (%)")
plot_metric("ee_travel_distance", "EE Travel Distance (m)")
plot_metric("max_joint_jump", "Max Joint Jump")
