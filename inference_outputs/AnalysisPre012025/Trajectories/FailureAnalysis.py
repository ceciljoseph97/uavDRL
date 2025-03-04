import os
import pandas as pd
import argparse
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

def get_offset_for_model(model_name):
    model_name_lower = model_name.lower()
    if 'multi_rgb' in model_name_lower:
        return -40
    elif 'single_rgb' in model_name_lower:
        return -20
    elif 'depth' in model_name_lower:
        return 0
    else:
        return 0

def compute_failure_percentage(positions_file, model_name, threshold=9.0, total_episodes=100):
    if not os.path.isfile(positions_file):
        logging.warning(f"No positions file found: {positions_file}.")
        return None

    try:
        df_pos = pd.read_csv(positions_file)
        if 'Episode' not in df_pos.columns or 'X' not in df_pos.columns:
            logging.warning(f"Positions file {positions_file} is missing 'Episode' or 'X' column.")
            return None

        last_entries = df_pos.groupby('Episode', as_index=False).tail(1)

        offset = get_offset_for_model(model_name)
        fails = 0
        for _, row in last_entries.iterrows():
            final_x = row['X'] + offset
            if final_x < threshold:
                fails += 1

        fail_percentage = (fails / total_episodes) * 100.0
        return fail_percentage

    except Exception as e:
        logging.error(f"Error reading/parsing {positions_file}: {e}")
        return None

def aggregate_failures(parent_directory, output_file):
    logging.info(f"Starting failure-percentage aggregation in: {parent_directory}")
    known_scenarios = ['gen', 'LeavesFalling', 'RainandFog', 'SnowOnly']

    results = []

    for scenario in known_scenarios:
        scenario_path = os.path.join(parent_directory, scenario)
        if not os.path.isdir(scenario_path):
            logging.warning(f"Scenario folder '{scenario}' not found in {parent_directory}; skipping.")
            continue
        for model_dir in os.listdir(scenario_path):
            model_path = os.path.join(scenario_path, model_dir)
            if not os.path.isdir(model_path):
                continue

            positions_file = os.path.join(model_path, f"{model_dir}_positions.csv")

            fail_percentage = compute_failure_percentage(positions_file, model_name=model_dir)
            if fail_percentage is None:
                logging.warning(f"Skipping model directory '{model_dir}' because no valid positions file was found.")
                continue

            results.append({
                'Model': model_dir,
                'Scenario': scenario,
                'Failure Percentage': fail_percentage
            })

    if not results:
        logging.error("No data aggregated. Exiting without creating CSV.")
        return

    df_results = pd.DataFrame(results)
    df_results['Model'] = df_results['Model'].str.strip()
    df_results['Scenario'] = df_results['Scenario'].str.strip()

    df_results.sort_values(by=['Scenario', 'Model'], inplace=True)

    df_results.to_csv(output_file, index=False)
    logging.info(f"Failure-percentage summary saved to {output_file}")

def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Aggregate the percentage of episodes that did not travel beyond 15m (with offsets)."
    )
    parser.add_argument(
        '--parent_dir',
        type=str,
        default=os.getcwd(),
        help='Path to the parent directory containing scenario folders (gen, LeavesFalling, etc.).'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='failure_summary.csv',
        help='Path to the output aggregated CSV file.'
    )

    args = parser.parse_args()

    if not os.path.isdir(args.parent_dir):
        logging.error(f"Parent directory '{args.parent_dir}' does not exist.")
        return

    aggregate_failures(args.parent_dir, args.output_file)

if __name__ == "__main__":
    main()
