import os
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from math import ceil

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.autolayout': False
})

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

def setup_logging(scenario):
    log_dir = os.path.join('inference_outputs', scenario)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{scenario}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized for scenario: '{scenario}'")

def parse_filename_for_abbrev(file_path: str) -> str:
    base_name = os.path.basename(file_path)
    core_name = base_name.replace('.csv', '').replace('_SnowOnly_positions', '').lower()
    return MODEL_ABBREVIATIONS.get(core_name, base_name)

def load_and_filter_csv(file_path: str, min_travel=9.0):
    df = pd.read_csv(file_path)
    valid_episodes = {}
    max_x_shift = 0.0

    for ep in df['Episode'].unique():
        ep_data = df[df['Episode'] == ep]
        x_vals = ep_data['X'].values
        y_vals = ep_data['Y'].values
        z_vals = ep_data['Z'].values

        if len(x_vals) == 0:
            continue

        x_shift = x_vals - x_vals[0]
        if x_shift.max() >= min_travel:
            if x_shift.max() > max_x_shift:
                max_x_shift = x_shift.max()
            valid_episodes[ep] = {
                'X': x_shift,
                'Y': y_vals,
                'Z': z_vals
            }
    return valid_episodes, max_x_shift

def plot_multiple_mean_trajectories_with_walls(filenames, scenario):
    logging.info("Plotting multiple mean trajectories with circular 'Passages' only.")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'gray']
    mean_trajectories = {}
    wall_positions = set()

    for idx, file_path in enumerate(filenames):
        label_abbrev = parse_filename_for_abbrev(file_path)
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read '{file_path}'.")
        except FileNotFoundError:
            logging.error(f"File '{file_path}' not found. Skipping.")
            continue
        except pd.errors.EmptyDataError:
            logging.error(f"File '{file_path}' is empty. Skipping.")
            continue
        except Exception as e:
            logging.error(f"Error reading '{file_path}': {e}. Skipping.")
            continue

        required_cols = {'Episode', 'X', 'Y', 'Z'}
        if not required_cols.issubset(df.columns):
            logging.error(f"File '{file_path}' is missing columns {required_cols}, skipping.")
            continue

        episodes_data, local_max_x = load_and_filter_csv(file_path, min_travel=9.0)
        if not episodes_data:
            logging.warning(f"No valid episodes >=9m in '{file_path}', skipping.")
            continue

        all_x = []
        for ep_data in episodes_data.values():
            all_x.append(ep_data['X'])
        all_x = np.concatenate(all_x)
        x_min, x_max = all_x.min(), all_x.max()
        x_common = np.linspace(x_min, x_max, 500)

        y_list = []
        z_list = []
        for ep_data in episodes_data.values():
            X_ = ep_data['X']
            Y_ = ep_data['Y']
            Z_ = ep_data['Z']
            sorted_idx = np.argsort(X_)
            X_s = X_[sorted_idx]
            Y_s = Y_[sorted_idx]
            Z_s = Z_[sorted_idx]
            unique_X, unique_i = np.unique(X_s, return_index=True)
            if len(unique_X) < 2:
                continue
            Y_u = Y_s[unique_i]
            Z_u = Z_s[unique_i]

            from scipy.interpolate import interp1d
            try:
                fy = interp1d(unique_X, Y_u, bounds_error=False, fill_value=np.nan)
                fz = interp1d(unique_X, Z_u, bounds_error=False, fill_value=np.nan)
                y_list.append(fy(x_common))
                z_list.append(fz(x_common))
            except Exception as e:
                logging.error(f"Interpolation error: {e}")

        if not y_list or not z_list:
            logging.warning(f"After interpolation, no valid data in '{file_path}'. Skipping.")
            continue

        arr_y = np.array(y_list)
        arr_z = np.array(z_list)
        mean_y = np.nanmean(arr_y, axis=0)
        mean_z = np.nanmean(arr_z, axis=0)

        mean_trajectories[idx] = {
            'X': x_common,
            'Y': mean_y,
            'Z': mean_z,
            'color': colors[idx % len(colors)],
            'label': label_abbrev,
            'filename': os.path.basename(file_path)
        }

        walls = np.arange(3.7, x_max + 3.7, 3.7)
        wall_positions.update(walls)

    if not mean_trajectories:
        logging.error("No valid mean trajectories to plot. Exiting.")
        return

    wall_positions = sorted(wall_positions)
    logging.info(f"Walls determined at: {wall_positions}")

    for idx, data in mean_trajectories.items():
        ax.plot(
            data['X'],
            data['Y'],
            data['Z'],
            color=data['color'],
            linewidth=2,
            label=data['label']
        )
        logging.info(f"Plotted mean trajectory from '{data['filename']}' with label '{data['label']}' in color '{data['color']}'.")

    y_min_global, y_max_global = -2, 2
    z_min_global, z_max_global = -2, 2

    ax.set_xlim(left=0)
    ax.set_ylim([y_min_global, y_max_global])
    ax.set_zlim([z_min_global, z_max_global])
    ax.set_xlabel("Distance (X)")
    ax.set_ylabel("Horizontal (Y)")
    ax.set_zlabel("Vertical (Z)")
    ax.set_title(f"Mean Drone Trajectories - {scenario}")
    ax.view_init(elev=10, azim=-135)

    margin = 0.1
    delta_x = 0.5

    for x_val in wall_positions:
        local_y, local_z = [], []
        for traj in mean_trajectories.values():
            X = traj['X']
            Y = traj['Y']
            Z = traj['Z']
            inrange = np.where((X >= x_val - delta_x) & (X <= x_val + delta_x))[0]
            if len(inrange) == 0:
                continue
            local_y.extend(Y[inrange])
            local_z.extend(Z[inrange])

        if not local_y or not local_z:
            logging.warning(f"No data near wall at X={x_val}. Skipping hole creation.")
            continue

        minY = np.nanmin(local_y)
        maxY = np.nanmax(local_y)
        minZ = np.nanmin(local_z)
        maxZ = np.nanmax(local_z)

        minY = max(minY, y_min_global)
        maxY = min(maxY, y_max_global)
        minZ = max(minZ, z_min_global)
        maxZ = min(maxZ, z_max_global)

        if minY >= maxY or minZ >= maxZ:
            continue

        square_minY = max(minY - margin, y_min_global)
        square_maxY = min(maxY + margin, y_max_global)
        square_minZ = max(minZ - margin, z_min_global)
        square_maxZ = min(maxZ + margin, z_max_global)

        centerY = (square_minY + square_maxY) / 2
        centerZ = (square_minZ + square_maxZ) / 2

        widthY = square_maxY - square_minY
        heightZ = square_maxZ - square_minZ
        max_radius = np.sqrt((widthY / 2) ** 2 + (heightZ / 2) ** 2)
        radius = max_radius * 0.9

        y_grid = np.linspace(y_min_global, y_max_global, 300)
        z_grid = np.linspace(z_min_global, z_max_global, 300)
        Yg, Zg = np.meshgrid(y_grid, z_grid)
        Xg = np.full_like(Yg, x_val)
        ax.plot_surface(Xg, Yg, Zg, color='gray', alpha=0.1, linewidth=0, antialiased=False)

        theta = np.linspace(0, 2*np.pi, 200)
        circle_y = centerY + radius * np.cos(theta)
        circle_z = centerZ + radius * np.sin(theta)
        circle_x = np.full_like(circle_y, x_val)

        if x_val == wall_positions[0]:
            ax.plot(
                circle_x,
                circle_y,
                circle_z,
                color='red',
                linewidth=2,
                linestyle='--',
                label="Passages"
            )
        else:
            ax.plot(
                circle_x,
                circle_y,
                circle_z,
                color='red',
                linewidth=2,
                linestyle='--'
            )

    handles, labels = ax.get_legend_handles_labels()
    unique_dict = dict(zip(labels, handles))
    ax.legend(unique_dict.values(), unique_dict.keys())

    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.07)

    out_dir = os.path.join('inference_outputs', scenario)
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"{scenario}_mean_trajectories.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logging.info(f"Mean trajectories saved as '{plot_path}' with circular 'Passages' only, no similarity.")


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean 3D drone trajectories with circular 'Passages' only (no square edges or similarity)."
    )
    parser.add_argument('--scenario', type=str, required=True, help='Scenario name for logging and output.')
    parser.add_argument('filenames', nargs='+', help='CSV filenames with columns [Episode, X, Y, Z].')
    args = parser.parse_args()

    setup_logging(args.scenario)
    logging.info(f"Scenario='{args.scenario}', Filenames={args.filenames}")
    plot_multiple_mean_trajectories_with_walls(args.filenames, args.scenario)


if __name__ == "__main__":
    main()
