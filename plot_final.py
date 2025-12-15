import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# ============================================================
# USER INPUT → Folder containing CSV files
# ============================================================
CSV_FOLDER = "."   # current directory (change if needed)

# Find all CSV files that follow your naming pattern
csv_files = glob.glob(os.path.join(CSV_FOLDER, "results_*.csv"))

if not csv_files:
    print("❌ No results_*.csv files found!")
    exit()

print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(" -", f)

# ============================================================
# Load all CSVs into dictionary {algorithm_name: dataframe}
# ============================================================
algo_data = {}

for file in csv_files:
    algo_name = os.path.basename(file).replace("results_", "").replace(".csv", "")
    df = pd.read_csv(file)

    # Separate runs and average row
    df_runs = df[df["run_id"] != "AVG"].copy()
    df_runs["run_id"] = df_runs["run_id"].astype(int)
    df_avg = df[df["run_id"] == "AVG"].iloc[0]

    algo_data[algo_name] = (df_runs, df_avg)

# ============================================================
# Performance metrics to plot
# ============================================================
metrics = {
    "nodes_expanded": "Nodes Expanded",
    "planning_time": "Planning Time (s)",
    "raw_waypoints": "Raw Waypoint Count",
    "raw_path_length": "Raw Path Length",
    "smooth_waypoints": "Smoothed Waypoint Count",
    "smooth_path_length": "Smoothed Path Length",
    "improvement_percent": "Improvement (%)",
    "ee_travel_distance": "EE Travel Distance (m)",
    "max_joint_jump": "Max Joint Jump"
}

colors = [
    "blue", "red", "green", "orange", "purple", 
    "brown", "cyan", "magenta", "black"
]

# ============================================================
# Ensure "plots" folder exists
# ============================================================
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)
print(f"\n📁 Plots will be saved in: {PLOT_DIR}/\n")

# ============================================================
# Plot function for each metric with multiple algorithms
# ============================================================
def plot_metric_multi(metric_key, ylabel):
    plt.figure(figsize=(10, 5))

    for idx, (algo_name, (df_runs, df_avg)) in enumerate(algo_data.items()):
        color = colors[idx % len(colors)]

        runs = df_runs["run_id"]
        values = df_runs[metric_key].astype(float)
        avg_value = float(df_avg[metric_key])

        # Plot algorithm's run values
        plt.plot(
            runs, values,
            marker="o",
            label=f"{algo_name}",
            color=color
        )

        # # OPTIONAL: average line per algorithm
        # plt.axhline(avg_value, linestyle="--", color=color, alpha=0.5)

    plt.xlabel("Run ID (Iteration)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Comparison Across Algorithms")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # SAVE PLOT
    filename = f"{metric_key}_comparison.png"
    save_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"📌 Saved plot → {save_path}")

    plt.show()

# ============================================================
# GENERATE & SAVE ALL PLOTS
# ============================================================
for key, label in metrics.items():
    plot_metric_multi(key, label)
