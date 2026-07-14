import os
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load uploaded trajectory CSV files
csv_paths = sorted(glob.glob("recorded_robot_paths/*_robot_path.csv"))
# csv_paths = 'recorded_robot_paths'
if not csv_paths:
    raise FileNotFoundError("No *_robot_path.csv files found in /mnt/data.")

frames = []
summaries = []

for path in csv_paths:
    name = os.path.basename(path)
    df = pd.read_csv(path)

    # Standardise column names defensively
    df.columns = [c.strip() for c in df.columns]

    required = ["timestamp_ms", "visibility", "robot_x", "robot_y", "robot_z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

    # Numeric conversion
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop invalid numeric rows
    df = df.dropna(subset=required).copy()
    df["file"] = name
    df["time_s"] = (df["timestamp_ms"] - df["timestamp_ms"].min()) / 1000.0

    # Motion gain column may vary in capitalization
    gain_col = None
    for possible in ["motion_gain", "Motion_gain", "MOTION_GAIN"]:
        if possible in df.columns:
            gain_col = possible
            df[gain_col] = pd.to_numeric(df[gain_col], errors="coerce")
            break

    # Consecutive movement distance
    diffs = df[["robot_x", "robot_y", "robot_z"]].diff()
    df["step_distance_m"] = np.sqrt(
        diffs["robot_x"] ** 2 + diffs["robot_y"] ** 2 + diffs["robot_z"] ** 2
    )

    records = len(df)
    duration = df["time_s"].max() if records else 0
    min_visibility = df["visibility"].min()
    mean_visibility = df["visibility"].mean()
    min_y, max_y = df["robot_y"].min(), df["robot_y"].max()
    min_z, max_z = df["robot_z"].min(), df["robot_z"].max()
    min_x, max_x = df["robot_x"].min(), df["robot_x"].max()

    valid_steps = df["step_distance_m"].dropna()
    above_threshold = (valid_steps >= 0.01).sum()
    below_threshold = (valid_steps < 0.01).sum()
    pct_above = (above_threshold / len(valid_steps) * 100) if len(valid_steps) else 0

    if gain_col:
        gain_values = sorted(df[gain_col].dropna().unique())
        if len(gain_values) == 1:
            gain_text = str(gain_values[0])
        elif len(gain_values) > 1:
            gain_text = f"{min(gain_values)} to {max(gain_values)}"
        else:
            gain_text = "Not recorded"
    else:
        gain_text = "Not recorded"

    summaries.append({
        "file": name,
        "records": records,
        "duration_s": duration,
        "min_visibility": min_visibility,
        "mean_visibility": mean_visibility,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "min_z": min_z,
        "max_z": max_z,
        "steps_above_0_01m": above_threshold,
        "steps_below_0_01m": below_threshold,
        "pct_steps_above_0_01m": pct_above,
        "motion_gain": gain_text
    })

    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
summary_df = pd.DataFrame(summaries)

# Choose a representative trajectory close to 30 seconds if available
summary_df["duration_distance_from_30"] = (summary_df["duration_s"] - 30).abs()
representative_file = summary_df.sort_values("duration_distance_from_30").iloc[0]["file"]
rep_df = all_df[all_df["file"] == representative_file].copy()

# Output directory for plots
out_dir = "/figures/chapter6_draft_plots"
os.makedirs(out_dir, exist_ok=True)

plot_paths = {}

# Plot 1: Records per trajectory file
plt.figure(figsize=(11, 6))
plt.bar(summary_df["file"], summary_df["records"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of recorded samples")
plt.xlabel("Trajectory file")
plt.title("Recorded trajectory samples by file")
plt.tight_layout()
path = os.path.join(out_dir, "figure_6_1_records_per_file.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plot_paths["records_per_file"] = path
plt.show()

# Plot 2: Visibility distribution by file with threshold
visibility_groups = [all_df[all_df["file"] == f]["visibility"].dropna().values for f in summary_df["file"]]
plt.figure(figsize=(11, 6))
plt.boxplot(visibility_groups, labels=summary_df["file"], showfliers=False)
plt.axhline(0.15, linestyle="--", label="Visibility threshold = 0.15")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Visibility score")
plt.xlabel("Trajectory file")
plt.title("Pose landmark visibility distribution by trajectory file")
plt.legend()
plt.tight_layout()
path = os.path.join(out_dir, "figure_6_2_visibility_distribution.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plot_paths["visibility_distribution"] = path
plt.show()

# Plot 3: Representative Y and Z coordinate over time
plt.figure(figsize=(11, 6))
plt.plot(rep_df["time_s"], rep_df["robot_y"], label="Robot Y")
plt.plot(rep_df["time_s"], rep_df["robot_z"], label="Robot Z")
plt.xlabel("Time (s)")
plt.ylabel("Robot coordinate (m)")
plt.title(f"Robot Y and Z coordinates over time: {representative_file}")
plt.legend()
plt.tight_layout()
path = os.path.join(out_dir, "figure_6_3_yz_over_time.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plot_paths["yz_over_time"] = path
plt.show()

# Plot 4: Representative Y-Z workspace path with workspace limits
Y_MIN, Y_MAX = -0.18, 0.18
Z_MIN, Z_MAX = 0.16, 0.34

plt.figure(figsize=(7, 7))
plt.plot(rep_df["robot_y"], rep_df["robot_z"], label="Trajectory path")
plt.scatter(rep_df["robot_y"].iloc[0], rep_df["robot_z"].iloc[0], marker="o", label="Start")
plt.scatter(rep_df["robot_y"].iloc[-1], rep_df["robot_z"].iloc[-1], marker="x", label="End")
plt.plot([Y_MIN, Y_MAX, Y_MAX, Y_MIN, Y_MIN], [Z_MIN, Z_MIN, Z_MAX, Z_MAX, Z_MIN], linestyle="--",
         label="Workspace boundary")
plt.xlabel("Robot Y coordinate (m)")
plt.ylabel("Robot Z coordinate (m)")
plt.title(f"Y-Z workspace trajectory path: {representative_file}")
plt.legend()
plt.axis("equal")
plt.tight_layout()
path = os.path.join(out_dir, "figure_6_4_yz_workspace_path.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plot_paths["yz_workspace_path"] = path
plt.show()

# Plot 5: All recorded points within workspace boundary
plt.figure(figsize=(7, 7))
plt.scatter(all_df["robot_y"], all_df["robot_z"], s=6, alpha=0.35, label="Recorded trajectory points")
plt.plot([Y_MIN, Y_MAX, Y_MAX, Y_MIN, Y_MIN], [Z_MIN, Z_MIN, Z_MAX, Z_MAX, Z_MIN], linestyle="--",
         label="Workspace boundary")
plt.xlabel("Robot Y coordinate (m)")
plt.ylabel("Robot Z coordinate (m)")
plt.title("All recorded trajectory points within Y-Z workspace")
plt.legend()
plt.axis("equal")
plt.tight_layout()
path = os.path.join(out_dir, "figure_6_5_all_points_workspace.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plot_paths["all_points_workspace"] = path
plt.show()

# Plot 6: Consecutive movement distance histogram
valid_distances = all_df["step_distance_m"].dropna()
plt.figure(figsize=(10, 6))
plt.hist(valid_distances, bins=50)
plt.axvline(0.01, linestyle="--", label="Minimum movement threshold = 0.01 m")
plt.xlabel("Consecutive movement distance (m)")
plt.ylabel("Frequency")
plt.title("Distribution of consecutive trajectory movement distances")
plt.legend()
plt.tight_layout()
path = os.path.join(out_dir, "figure_6_6_movement_distance_histogram.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plot_paths["movement_distance_histogram"] = path
plt.show()

# Plot 7: Percentage of movements above threshold by file
plt.figure(figsize=(11, 6))
plt.bar(summary_df["file"], summary_df["pct_steps_above_0_01m"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Movements above 0.01 m threshold (%)")
plt.xlabel("Trajectory file")
plt.title("Estimated command-worthy movement percentage by trajectory file")
plt.tight_layout()
path = os.path.join(out_dir, "figure_6_7_threshold_percentage_by_file.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plot_paths["threshold_percentage_by_file"] = path
plt.show()

# Print compact summary
print("Representative trajectory selected for detailed plots:", representative_file)
print()
print("Overall summary:")
print(f"Total CSV files: {summary_df.shape[0]}")
print(f"Total recorded samples: {len(all_df):,}")
print(f"Total recorded duration (sum of file durations): {summary_df['duration_s'].sum():.2f} seconds")
print(f"Minimum visibility: {all_df['visibility'].min():.3f}")
print(f"Mean visibility: {all_df['visibility'].mean():.3f}")
print(f"X observed range: {all_df['robot_x'].min():.3f} to {all_df['robot_x'].max():.3f} m")
print(f"Y observed range: {all_df['robot_y'].min():.3f} to {all_df['robot_y'].max():.3f} m")
print(f"Z observed range: {all_df['robot_z'].min():.3f} to {all_df['robot_z'].max():.3f} m")
print(f"Records below visibility threshold 0.15: {(all_df['visibility'] < 0.15).sum()}")

print("\nSaved plot files:")
for key, value in plot_paths.items():
    print(f"{key}: {value}")
