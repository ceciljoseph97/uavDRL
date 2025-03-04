import os
import numpy as np
import pandas as pd
import logging
import argparse
import math

def process_position_csv(csv_path, model_name):
    try:
        df = pd.read_csv(csv_path)
        
        required_columns = {'Episode', 'StartPos', 'X', 'Y', 'Z'}
        if not required_columns.issubset(df.columns):
            logging.warning(f"Required columns not found in {csv_path}. Skipping.")
            return None, None
        shift = 0
        if 'single_rgb' in model_name.lower():
            shift = -20
        elif 'multi_rgb' in model_name.lower():
            shift = -40
        flight_distances = []
        holes_reached_list = []

        grouped = df.groupby('Episode')

        for episode, group in grouped:
            x_values = group['X'].dropna().values
            if len(x_values) == 0:
                logging.warning(f"No X values found in Episode {episode} of {csv_path}. Skipping episode.")
                continue
            shifted_x_values = x_values + shift
            max_x = np.max(shifted_x_values)
            flight_distance = max_x
            holes_reached =  round(max_x // 3.7)
            flight_distances.append(flight_distance)
            holes_reached_list.append(holes_reached)

        mean_flight_distance = np.mean(flight_distances) if flight_distances else 0
        mean_holes_reached =  round(np.mean(holes_reached_list)) if holes_reached_list else 0

        return mean_flight_distance, mean_holes_reached

    except Exception as e:
        logging.error(f"Error processing {csv_path}: {e}")
        return None, None

def main(parent_directory, summary_file):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    all_flight_distances = []
    all_holes_reached = []

    summary_data = []

    for item in os.listdir(parent_directory):
        item_path = os.path.join(parent_directory, item)

        if os.path.isdir(item_path):
            csv_files = [f for f in os.listdir(item_path) if f.endswith('_positions.csv')]

            if not csv_files:
                logging.warning(f"No positions CSV found in {item_path}. Skipping.")
                continue

            csv_file = csv_files[0]
            csv_path = os.path.join(item_path, csv_file)

            mean_flight_distance, mean_holes_reached = process_position_csv(csv_path, item)

            if mean_flight_distance is not None and mean_holes_reached is not None:
                all_flight_distances.append(mean_flight_distance)
                all_holes_reached.append(mean_holes_reached)

                summary_data.append({
                    'Model': item,
                    'Flight Distance': mean_flight_distance,
                    'Holes Reached': mean_holes_reached
                })

    overall_mean_flight_distance = np.mean(all_flight_distances) if all_flight_distances else 0
    overall_mean_holes_reached = np.mean(all_holes_reached) if all_holes_reached else 0

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    logging.info(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process position CSVs and generate a summary.")
    parser.add_argument('--parent_dir', type=str, default=os.getcwd(), help='Path to the parent directory')
    parser.add_argument('--output_summary', type=str, default='summary.csv', help='Path to the output summary CSV')

    args = parser.parse_args()

    parent_dir = args.parent_dir
    output_summary = os.path.join(parent_dir, args.output_summary)
    main(parent_dir, output_summary)
