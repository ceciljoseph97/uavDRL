import argparse
import logging
import os
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', size=14)
matplotlib.rc('axes', titlesize=16)
matplotlib.rc('axes', labelsize=14)
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rc('legend', fontsize=12)

DARK_BLUE = "#003366"

MODEL_ABBREVIATIONS = {
    "a2c_naturecnn_depth": "A2C-D",
    "a2c_naturecnn_depth_obstacle_weather_leavesfalling": "A2C-DL",
    "a2c_naturecnn_depth_obstacle_weather_rainandfog": "A2C-DRF",
    "a2c_naturecnn_depth_obstacle_weather_snowonly": "A2C-DS",
    "a2c_naturecnn_multi_rgb": "A2C-MRGB",
    "a2c_naturecnn_single_rgb": "A2C-SRGB",
    "a2c_naturecnn_single_rgb_obstacle_weather_leavesfalling": "A2C-SRGBL",
    "a2c_naturecnn_single_rgb_obstacle_weather_rainandfog": "A2C-SRGBRF",
    "a2c_naturecnn_single_rgb_obstacle_weather_snowonly": "A2C-SRGBS",
    "ppo_naturecnn_depth": "PPO-D",
    "ppo_naturecnn_depth_obstacle_weather_leavesfalling": "PPO-DL",
    "ppo_naturecnn_depth_obstacle_weather_rainandfog": "PPO-DRF",
    "ppo_naturecnn_depth_obstacle_weather_snowonly": "PPO-DS",
    "ppo_naturecnn_multi_rgb": "PPO-MRGB",
    "ppo_naturecnn_single_rgb": "PPO-SRGB",
    "ppo_naturecnn_single_rgb_obstacle_weather_leavesfalling": "PPO-SRGBL",
    "ppo_naturecnn_single_rgb_obstacle_weather_rainandfog": "PPO-SRGBRF",
    "ppo_naturecnn_single_rgb_obstacle_weather_snowonly": "PPO-SRGBS",
    "ppo_all_fm": "PPO-DFM under Normal Weather",
    "ppo_all_fm_leavesfalling": "PPO-DFM under LeavesFalling Weather",
    "ppo_all_fm_snowonly": "PPO-DFM under SnowOnly Weather",
    "ppo_all_fm_rainandfog": "PPO-DFM under RainandFog Weather",
    "ppo_all_fm_srgb": "PPO-SRGBFM  under Normal Weather",
    "ppo_all_fm__srgb_lf": "PPO-SRGBFM  under LeavesFalling Weather",
    "ppo_all_fm_srgb_so": "PPO-SRGBFM  under SnowOnly Weather",
    "ppo_all_fm_srgb_rf": "PPO-SRGBFM under RainandFog Weather",
}

def parse_filename_for_abbrev(file_path: str) -> str:
    base_name = os.path.basename(file_path)
    file_core = base_name.replace(".csv", "").replace("_SnowOnly_positions", "")
    return MODEL_ABBREVIATIONS.get(file_core.lower(), file_core)

def load_and_filter_csv(file_path: str, min_travel=18.0):
    df = pd.read_csv(file_path)
    valid_episodes = []
    max_x_shift = 0.0

    for ep in df['Episode'].unique():
        ep_data = df[df['Episode'] == ep]
        x_vals = ep_data['X'].values
        y_vals = ep_data['Y'].values

        if len(x_vals) < 2:
            logging.info(f"File={file_path}: Episode {ep} has insufficient data.")
            continue

        x_shifted = x_vals - x_vals[0]
        y_shifted = y_vals

        if x_shifted.max() >= min_travel:
            if x_shifted.max() > max_x_shift:
                max_x_shift = x_shifted.max()
            valid_episodes.append((x_shifted, y_shifted))

    return valid_episodes, max_x_shift

def main():
    parser = argparse.ArgumentParser(description="Plot UAV trajectory CSVs on subplots.")
    parser.add_argument("--csv_files", nargs='+', required=True,
                        help="List of CSV files. Each must have columns: Episode, X, Y.")
    parser.add_argument("--plot_name", default="subplots.png",
                        help="Output PNG file name.")
    parser.add_argument("--title", default="UAV Trajectories",
                        help="Global title for the figure.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    csv_paths = args.csv_files
    n_files = len(csv_paths)
    n_cols = int(math.ceil(math.sqrt(n_files)))  
    n_rows = int(math.ceil(n_files / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))

    if n_rows * n_cols == 1:
        axes = np.array([[axes]])
    else:
        axes = np.array(axes).reshape(n_rows, n_cols)

    overall_max_x = 0.0
    idx_file = 0

    for r in range(n_rows):
        for c in range(n_cols):
            if idx_file >= n_files:
                axes[r,c].set_visible(False)
                continue

            file_path = csv_paths[idx_file]
            idx_file += 1
            
            label = parse_filename_for_abbrev(file_path)
            episodes_data, local_max_x = load_and_filter_csv(file_path)
            if local_max_x > overall_max_x:
                overall_max_x = local_max_x

            ax = axes[r,c]
            ax.set_title(label, fontsize=14)

            has_plotted = False
            for (x_shifted, y_shifted) in episodes_data:
                if not has_plotted:
                    ax.plot(x_shifted, y_shifted, color=DARK_BLUE, alpha=0.8, label=label)
                    has_plotted = True
                else:
                    ax.plot(x_shifted, y_shifted, color=DARK_BLUE, alpha=0.8)

            ax.set_ylim([-2, 2])
            ax.set_xlim([0, None])
            ax.set_xlabel("Distance traveled (m)")
            ax.set_ylabel("Horizontal Movement (m)")
            ax.grid(True)

            if not episodes_data:
                ax.text(0.3, 0, "No valid episodes", fontsize=10)

    wall_positions = np.arange(3.7, overall_max_x + 3.7, 3.7)
    for r in range(n_rows):
        for c in range(n_cols):
            if not axes[r,c].get_visible():
                continue
            ax = axes[r,c]
            for x_val in wall_positions:
                ax.axvline(x=x_val, color='red', linestyle='--', alpha=0.5, label='_nolegend_')
            ax.axvline(x=-10, color='red', linestyle='--', alpha=0.5, label='Obstacles')

            handles, lbls = ax.get_legend_handles_labels()
            unique_dict = dict(zip(lbls, handles))
            ax.legend(unique_dict.values(), unique_dict.keys(), loc='best')

    plt.suptitle(args.title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(args.plot_name, dpi=150)
    logging.info(f"Plot saved as {args.plot_name}.")

if __name__ == "__main__":
    main()
