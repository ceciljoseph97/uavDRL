import os
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.signal import correlate, correlation_lags
from math import ceil
from itertools import combinations

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

def normalized_cross_correlation_with_lag(x, y):
    x = np.array(x)
    y = np.array(y)
    
    valid = ~np.isnan(x) & ~np.isnan(y)
    x = x[valid]
    y = y[valid]
    
    if len(x) == 0 or len(y) == 0:
        return np.nan, None, None, None

    x = x - np.mean(x)
    y = y - np.mean(y)
    
    corr = correlate(x, y, mode='full')
    lags = correlation_lags(len(x), len(y), mode='full')
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    if norm == 0:
        return np.nan, None, None, None
    corr = corr / norm
    
    max_idx = np.argmax(corr)
    max_corr = corr[max_idx]
    lag_at_max = lags[max_idx]
    
    return max_corr, lag_at_max, corr, lags

def shorten_filename(filename):
    short_name = os.path.splitext(filename)[0]
    parts = short_name.split('_')
    if len(parts) > 5:
        short_name = f"{parts[0]}_{parts[2]}_{parts[3]}_{parts[-2]}__{parts[-1]}"
    return short_name

def plot_multiple_mean_trajectories_with_walls(filenames, scenario):
    logging.info("Starting to plot multiple mean trajectories.")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    mean_trajectories = {}
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'gray']
    wall_positions = set()

    for idx, filename in enumerate(filenames):
        try:
            df = pd.read_csv(filename)
            logging.info(f"Successfully read '{filename}'.")
        except FileNotFoundError:
            logging.error(f"File '{filename}' not found. Skipping.")
            continue
        except pd.errors.EmptyDataError:
            logging.error(f"File '{filename}' is empty. Skipping.")
            continue
        except Exception as e:
            logging.error(f"Error reading '{filename}': {e}. Skipping.")
            continue

        required_columns = {'Episode', 'X', 'Y', 'Z'}
        if not required_columns.issubset(df.columns):
            logging.error(f"File '{filename}' is missing required columns {required_columns}. Skipping.")
            continue

        episodes_data = {}

        for episode in df['Episode'].unique():
            episode_data = df[df['Episode'] == episode]
            x_positions = episode_data['X'].values
            y_positions = episode_data['Y'].values
            z_positions = episode_data['Z'].values

            if len(x_positions) == 0:
                logging.warning(f"Episode {episode} in '{filename}' has no data. Skipping.")
                continue

            x_shifted = x_positions - x_positions[0]

            if x_shifted.max() >= 6:
                episodes_data[episode] = {
                    'X': x_shifted,
                    'Y': y_positions,
                    'Z': z_positions
                }
            else:
                logging.info(f"Episode {episode} in '{filename}' did not travel at least 9 meters. Skipping.")

        if not episodes_data:
            logging.warning(f"No valid episodes in '{filename}'. Skipping this file.")
            continue

        max_x = max([data['X'].max() for data in episodes_data.values()])
        all_x = np.concatenate([data['X'] for data in episodes_data.values()])
        x_min = all_x.min()
        x_max = all_x.max()
        x_common = np.linspace(x_min, x_max, num=500)

        y_interp_list = []
        z_interp_list = []

        for data in episodes_data.values():
            sorted_indices = np.argsort(data['X'])
            X_sorted = data['X'][sorted_indices]
            Y_sorted = data['Y'][sorted_indices]
            Z_sorted = data['Z'][sorted_indices]

            unique_X, unique_indices = np.unique(X_sorted, return_index=True)
            Y_unique = Y_sorted[unique_indices]
            Z_unique = Z_sorted[unique_indices]

            try:
                f_y = interp1d(unique_X, Y_unique, bounds_error=False, fill_value=np.nan)
                f_z = interp1d(unique_X, Z_unique, bounds_error=False, fill_value=np.nan)
            except Exception as e:
                logging.error(f"Interpolation error for file '{filename}', episode data: {e}. Skipping this episode.")
                continue

            y_interp = f_y(x_common)
            z_interp = f_z(x_common)

            y_interp_list.append(y_interp)
            z_interp_list.append(z_interp)

        if not y_interp_list or not z_interp_list:
            logging.warning(f"No valid interpolated data in '{filename}'. Skipping.")
            continue

        y_interp_array = np.array(y_interp_list)
        z_interp_array = np.array(z_interp_list)

        mean_y = np.nanmean(y_interp_array, axis=0)
        mean_z = np.nanmean(z_interp_array, axis=0)

        mean_trajectories[idx] = {
            'X': x_common,
            'Y': mean_y,
            'Z': mean_z,
            'color': colors[idx % len(colors)],
            'filename': os.path.basename(filename)
        }

        wall_pos = np.arange(3.7, x_max + 3.7, 3.7)
        wall_positions.update(wall_pos)

    if not mean_trajectories:
        logging.error("No valid mean trajectories to plot. Exiting.")
        return

    wall_positions = sorted(list(wall_positions))
    logging.info(f"Wall positions determined at: {wall_positions}")

    for idx, traj in mean_trajectories.items():
        ax.plot(
            traj['X'],
            traj['Y'],
            traj['Z'],
            color=traj['color'],
            linewidth=2,
            label=f"{traj['filename']}"
        )
        logging.info(f"Plotted mean trajectory from '{traj['filename']}' with color '{traj['color']}'.")

    y_min_global, y_max_global = -2, 2
    z_min_global, z_max_global = -2, 2

    delta_x = 0.5
    margin = 0.1

    for x_wall in wall_positions:
        local_y = []
        local_z = []

        for traj in mean_trajectories.values():
            X = traj['X']
            Y = traj['Y']
            Z = traj['Z']
            indices = np.where((X >= x_wall - delta_x) & (X <= x_wall + delta_x))[0]
            if len(indices) > 0:
                local_y.extend(Y[indices])
                local_z.extend(Z[indices])

        if not local_y or not local_z:
            logging.warning(f"No data near wall at X={x_wall}. Skipping hole creation.")
            continue

        minY = max(np.nanmin(local_y), y_min_global)
        maxY = min(np.nanmax(local_y), y_max_global)
        minZ = max(np.nanmin(local_z), z_min_global)
        maxZ = min(np.nanmax(local_z), z_max_global)

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

        y = np.linspace(y_min_global, y_max_global, 300)
        z = np.linspace(z_min_global, z_max_global, 300)
        Y_grid, Z_grid = np.meshgrid(y, z)

        square_mask = np.logical_and(
            np.logical_and(Y_grid >= square_minY, Y_grid <= square_maxY),
            np.logical_and(Z_grid >= square_minZ, Z_grid <= square_maxZ)
        )
        D = np.sqrt((Y_grid - centerY) ** 2 + (Z_grid - centerZ) ** 2)
        circular_mask = D <= radius

        hole_mask = np.logical_or(square_mask, circular_mask)

        X_wall = np.full_like(Y_grid, x_wall)
        X_wall = np.ma.array(X_wall, mask=hole_mask)
        Y_wall = np.ma.array(Y_grid, mask=hole_mask)
        Z_wall = np.ma.array(Z_grid, mask=hole_mask)

        ax.plot_surface(
            X_wall,
            Y_wall,
            Z_wall,
            color='gray',
            alpha=0.1,
            linewidth=0,
            antialiased=False
        )
        logging.info(f"Added wall surface at X={x_wall} with Trajectory Variations and Passages.")

        square_edges_y = [square_minY, square_maxY, square_maxY, square_minY, square_minY]
        square_edges_z = [square_minZ, square_minZ, square_maxZ, square_maxZ, square_minZ]
        square_edges_x = [x_wall] * 5
        if x_wall == wall_positions[0]:
            ax.plot(
                square_edges_x,
                square_edges_y,
                square_edges_z,
                color='blue',
                linewidth=2,
                linestyle='--',
                label="LocalMinMaxYZ Variations"
            )
        else:
            ax.plot(
                square_edges_x,
                square_edges_y,
                square_edges_z,
                color='blue',
                linewidth=2,
                linestyle='--'
            )

        theta = np.linspace(0, 2 * np.pi, 200)
        circle_y = centerY + radius * np.cos(theta)
        circle_z = centerZ + radius * np.sin(theta)
        circle_x = np.full_like(circle_y, x_wall)
        if x_wall == wall_positions[0]:
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

    logging.info("Completed plotting walls and holes.")

    print("\nSimilarity Measures between Mean Trajectories:")
    similarity_summary = []

    mean_keys = list(mean_trajectories.keys())
    num_trajs = len(mean_keys)

    trajectory_pairs = list(combinations(mean_keys, 2))
    num_pairs = len(trajectory_pairs)

    if num_pairs == 0:
        logging.warning("Not enough trajectories to compute similarity measures.")
    else:
        pairs_per_figure = 6
        num_figures = ceil(num_pairs / pairs_per_figure)

        for fig_num in range(num_figures):
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            axes = axes.flatten()
            for pair_idx in range(pairs_per_figure):
                global_pair_idx = fig_num * pairs_per_figure + pair_idx
                if global_pair_idx >= num_pairs:
                    axes[pair_idx].axis('off')
                    continue
                i, j = trajectory_pairs[global_pair_idx]
                traj_i = mean_trajectories[i]
                traj_j = mean_trajectories[j]

                x_start = max(traj_i['X'].min(), traj_j['X'].min())
                x_end = min(traj_i['X'].max(), traj_j['X'].max())

                if x_start >= x_end:
                    logging.warning(f"Trajectories '{traj_i['filename']}' and '{traj_j['filename']}' do not overlap in X. Cannot compute similarity.")
                    print(f"\nTrajectories '{traj_i['filename']}' and '{traj_j['filename']}' do not overlap in X. Cannot compute similarity.")
                    axes[pair_idx].axis('off')
                    continue

                x_common_sim = np.linspace(x_start, x_end, num=500)

                f_y_i = interp1d(traj_i['X'], traj_i['Y'], bounds_error=False, fill_value=np.nan)
                f_z_i = interp1d(traj_i['X'], traj_i['Z'], bounds_error=False, fill_value=np.nan)
                f_y_j = interp1d(traj_j['X'], traj_j['Y'], bounds_error=False, fill_value=np.nan)
                f_z_j = interp1d(traj_j['X'], traj_j['Z'], bounds_error=False, fill_value=np.nan)

                y_i_interp = f_y_i(x_common_sim)
                z_i_interp = f_z_i(x_common_sim)
                y_j_interp = f_y_j(x_common_sim)
                z_j_interp = f_z_j(x_common_sim)

                valid_indices = ~(
                    np.isnan(y_i_interp) | np.isnan(y_j_interp) |
                    np.isnan(z_i_interp) | np.isnan(z_j_interp)
                )

                if not np.any(valid_indices):
                    logging.warning(f"No overlapping valid data between '{traj_i['filename']}' and '{traj_j['filename']}'.")
                    print(f"\nNo overlapping valid data between '{traj_i['filename']}' and '{traj_j['filename']}'.")
                    axes[pair_idx].axis('off')
                    continue

                y_i_clean = y_i_interp[valid_indices]
                y_j_clean = y_j_interp[valid_indices]
                z_i_clean = z_i_interp[valid_indices]
                z_j_clean = z_j_interp[valid_indices]

                try:
                    cosine_y = np.dot(y_i_clean, y_j_clean) / (np.linalg.norm(y_i_clean) * np.linalg.norm(y_j_clean) + 1e-10)
                    cosine_z = np.dot(z_i_clean, z_j_clean) / (np.linalg.norm(z_i_clean) * np.linalg.norm(z_j_clean) + 1e-10)
                    cosine_similarity = np.nanmean([cosine_y, cosine_z])
                except Exception as e:
                    cosine_similarity = np.nan
                    logging.error(f"Error computing Cosine similarity between '{traj_i['filename']}' and '{traj_j['filename']}': {e}")

                try:
                    pearson_y, _ = pearsonr(y_i_clean, y_j_clean)
                    pearson_z, _ = pearsonr(z_i_clean, z_j_clean)
                    pearson_mean = np.nanmean([pearson_y, pearson_z])
                except Exception as e:
                    pearson_mean = np.nan
                    logging.error(f"Error computing Pearson correlation between '{traj_i['filename']}' and '{traj_j['filename']}': {e}")

                try:
                    cross_corr_y, lag_y, corr_y, lags_y = normalized_cross_correlation_with_lag(y_i_clean, y_j_clean)
                    cross_corr_z, lag_z, corr_z, lags_z = normalized_cross_correlation_with_lag(z_i_clean, z_j_clean)
                    cross_corr_mean = np.nanmean([cross_corr_y, cross_corr_z])
                except Exception as e:
                    cross_corr_mean = np.nan
                    lag_y = None
                    lag_z = None
                    logging.error(f"Error computing Normalized Cross-Correlation between '{traj_i['filename']}' and '{traj_j['filename']}': {e}")

                short_i = shorten_filename(traj_i['filename'])
                short_j = shorten_filename(traj_j['filename'])

                ax_corr = axes[pair_idx]
                if corr_y is not None and lags_y is not None:
                    ax_corr.plot(lags_y, corr_y, label='Y Cross-Correlation', color='blue')
                if corr_z is not None and lags_z is not None:
                    ax_corr.plot(lags_z, corr_z, label='Z Cross-Correlation', color='red')
                if lag_y is not None:
                    ax_corr.axvline(x=lag_y, color='blue', linestyle='--', label=f'Y Max Lag: {lag_y}')
                if lag_z is not None:
                    ax_corr.axvline(x=lag_z, color='red', linestyle='--', label=f'Z Max Lag: {lag_z}')
                ax_corr.set_title(f"{short_i} vs {short_j}", fontsize=10)
                ax_corr.set_xlabel('Lag', fontsize=8)
                ax_corr.set_ylabel('Normalized Cross-Correlation', fontsize=8)
                ax_corr.legend(fontsize=6)
                ax_corr.grid(True)

                pair_name = f"{traj_i['filename']} vs {traj_j['filename']}"
                similarity_summary.append({
                    'Pair': pair_name,
                    'Cosine Similarity (Mean)': round(cosine_similarity, 4) if not np.isnan(cosine_similarity) else 'N/A',
                    'Pearson Correlation (Mean)': round(pearson_mean, 4) if not np.isnan(pearson_mean) else 'N/A',
                    'Normalized Cross-Correlation (Mean)': round(cross_corr_mean, 4) if not np.isnan(cross_corr_mean) else 'N/A',
                    'Lag at Max NCC (Y)': lag_y if lag_y is not None else 'N/A',
                    'Lag at Max NCC (Z)': lag_z if lag_z is not None else 'N/A'
                })

                print(f"\nBetween '{traj_i['filename']}' and '{traj_j['filename']}':")
                if not np.isnan(cosine_similarity):
                    print(f"  - Cosine Similarity (Mean): {cosine_similarity:.4f}")
                else:
                    print(f"  - Cosine Similarity (Mean): N/A")
                
                if not np.isnan(pearson_mean):
                    print(f"  - Pearson Correlation (Mean): {pearson_mean:.4f}")
                else:
                    print(f"  - Pearson Correlation (Mean): N/A")
                
                if not np.isnan(cross_corr_mean):
                    print(f"  - Normalized Cross-Correlation (Mean): {cross_corr_mean:.4f}")
                else:
                    print(f"  - Normalized Cross-Correlation (Mean): N/A")
                
                if lag_y is not None and lag_z is not None:
                    print(f"  - Lag at Max NCC (Y): {lag_y}, (Z): {lag_z}")
                else:
                    print(f"  - Lag at Max NCC (Y): N/A, (Z): N/A")

                logging.info(f"Similarity between '{traj_i['filename']}' and '{traj_j['filename']}':")
                if not np.isnan(cosine_similarity):
                    logging.info(f"  - Cosine Similarity (Mean): {cosine_similarity:.4f}")
                else:
                    logging.info(f"  - Cosine Similarity (Mean): N/A")
                
                if not np.isnan(pearson_mean):
                    logging.info(f"  - Pearson Correlation (Mean): {pearson_mean:.4f}")
                else:
                    logging.info(f"  - Pearson Correlation (Mean): N/A")
                
                if not np.isnan(cross_corr_mean):
                    logging.info(f"  - Normalized Cross-Correlation (Mean): {cross_corr_mean:.4f}")
                else:
                    logging.info(f"  - Normalized Cross-Correlation (Mean): N/A")
                
                if lag_y is not None and lag_z is not None:
                    logging.info(f"  - Lag at Max NCC (Y): {lag_y}, (Z): {lag_z}")
                else:
                    logging.info(f"  - Lag at Max NCC (Y): N/A, (Z): N/A")

            plt.tight_layout()
            cross_corr_plot_name = os.path.join('inference_outputs', scenario, f"{scenario}_cross_correlation_part{fig_num+1}.png")
            plt.savefig(cross_corr_plot_name)
            plt.close()
            logging.info(f"Cross-correlation plot saved as '{cross_corr_plot_name}'.")

    ax.set_xlabel('Distance Travelled (X)', fontsize=12)
    ax.set_ylabel('Horizontal Variations (Y)', fontsize=12)
    ax.set_zlabel('Vertical Variations (Z)', fontsize=12)
    ax.set_title(f'Mean Drone Trajectories - {scenario}', fontsize=14)

    ax.view_init(elev=10, azim=-135)

    ax.set_ylim([y_min_global, y_max_global])
    ax.set_zlim([z_min_global, z_max_global])

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plot_save_path = os.path.join('inference_outputs', scenario, f"{scenario}_mean_trajectories.png")
    plt.savefig(plot_save_path)
    plt.close()
    logging.info(f"Mean trajectories plotted and saved as '{plot_save_path}'.")

    if similarity_summary:
        summary_df = pd.DataFrame(similarity_summary)
        summary_csv_path = os.path.join('inference_outputs', scenario, f"{scenario}_similarity_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print("\nSummary of Similarity Analyses (Pairwise):")
        print(summary_df.to_string(index=False))
        logging.info(f"Similarity summary saved as '{summary_csv_path}'.")

        evaluate_best_pairs(summary_df)
    else:
        logging.warning("No similarity measures were computed.")

def evaluate_best_pairs(summary_df):
    summary_df['Cosine Similarity (Mean)'] = pd.to_numeric(summary_df['Cosine Similarity (Mean)'], errors='coerce')
    summary_df['Pearson Correlation (Mean)'] = pd.to_numeric(summary_df['Pearson Correlation (Mean)'], errors='coerce')
    summary_df['Normalized Cross-Correlation (Mean)'] = pd.to_numeric(summary_df['Normalized Cross-Correlation (Mean)'], errors='coerce')

    summary_df['Aggregate Similarity Score'] = summary_df[['Cosine Similarity (Mean)', 'Pearson Correlation (Mean)', 'Normalized Cross-Correlation (Mean)']].mean(axis=1)

    sorted_df = summary_df.sort_values(by='Aggregate Similarity Score', ascending=False)

    top_score = sorted_df['Aggregate Similarity Score'].max()
    best_pairs = sorted_df[sorted_df['Aggregate Similarity Score'] == top_score]

    print("\n=== Evaluation of Best Pairs ===")
    for _, row in best_pairs.iterrows():
        print(f"\nBest Pair: {row['Pair']}")
        print(f"  - Cosine Similarity (Mean): {row['Cosine Similarity (Mean)']:.4f}")
        print(f"  - Pearson Correlation (Mean): {row['Pearson Correlation (Mean)']:.4f}")
        print(f"  - Normalized Cross-Correlation (Mean): {row['Normalized Cross-Correlation (Mean)']:.4f}")
        print(f"  - Aggregate Similarity Score: {row['Aggregate Similarity Score']:.4f}")
        print(f"  - Lag at Max NCC (Y): {row['Lag at Max NCC (Y)']}")
        print(f"  - Lag at Max NCC (Z): {row['Lag at Max NCC (Z)']}")
        print("  - Evaluation: This pair exhibits the highest similarity across all metrics and is considered the best pair.")

    top_n = 3
    print(f"\n=== Top {top_n} Pairs Based on Aggregate Similarity ===")
    top_n_df = sorted_df.head(top_n)
    for _, row in top_n_df.iterrows():
        print(f"\nPair: {row['Pair']}")
        print(f"  - Cosine Similarity (Mean): {row['Cosine Similarity (Mean)']:.4f}")
        print(f"  - Pearson Correlation (Mean): {row['Pearson Correlation (Mean)']:.4f}")
        print(f"  - Normalized Cross-Correlation (Mean): {row['Normalized Cross-Correlation (Mean)']:.4f}")
        print(f"  - Aggregate Similarity Score: {row['Aggregate Similarity Score']:.4f}")
        print(f"  - Lag at Max NCC (Y): {row['Lag at Max NCC (Y)']}")
        print(f"  - Lag at Max NCC (Z): {row['Lag at Max NCC (Z)']}")
        print("  - Evaluation: This pair shows strong similarity metrics and is among the top-performing pairs.")

def main():
    parser = argparse.ArgumentParser(description="Plot mean drone trajectories with logging, similarity analysis, and lag evaluation.")
    parser.add_argument('--scenario', type=str, required=True, help='Scenario name for logging and plot titles.')
    parser.add_argument('filenames', nargs='+', help='List of CSV filenames containing drone trajectories.')

    args = parser.parse_args()

    setup_logging(args.scenario)

    logging.info(f"Received command-line arguments: Scenario='{args.scenario}', Filenames={args.filenames}")

    plot_multiple_mean_trajectories_with_walls(args.filenames, args.scenario)

if __name__ == "__main__":
    main()
