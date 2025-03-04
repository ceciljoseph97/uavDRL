import os
import pandas as pd
import argparse
import logging

def extract_base_model(model_name, scenario):
    if pd.isna(model_name) or pd.isna(scenario):
        return None

    scenario_normalized = scenario.strip().lower()
    model_name_normalized = model_name.strip()

    if model_name_normalized.lower().endswith(scenario_normalized):
        base_model = model_name_normalized[:-len(scenario_normalized)]
        if base_model.endswith('_'):
            base_model = base_model[:-1]
        return base_model.strip()
    else:
        return model_name_normalized

def aggregate_summaries(parent_directory, output_file):
    logging.info(f"Starting aggregation in parent directory: {parent_directory}")
    
    aggregated_data = []
    known_scenarios = ['gen', 'LeavesFalling', 'RainandFog', 'SnowOnly']
    for item in os.listdir(parent_directory):
        item_path = os.path.join(parent_directory, item)
        
        if os.path.isdir(item_path):
            scenario = item.strip()
            if scenario not in known_scenarios:
                logging.warning(f"Directory '{item}' does not match known scenarios {known_scenarios}. Proceeding with extraction.")
            
            summary_csv = os.path.join(item_path, 'summary.csv')
            
            if not os.path.isfile(summary_csv):
                logging.warning(f"No summary.csv found in {item_path}. Skipping.")
                continue
            
            try:
                df = pd.read_csv(summary_csv)
                
                required_columns = {'Model', 'Flight Distance', 'Holes Reached'}
                if not required_columns.issubset(df.columns):
                    logging.warning(f"Required columns missing in {summary_csv}. Skipping.")
                    continue
                
                for index, row in df.iterrows():
                    full_model = str(row['Model']).strip()
                    flight_distance = row['Flight Distance']
                    holes_reached = row['Holes Reached']
                    
                    base_model = extract_base_model(full_model, scenario)
                    
                    if base_model is None:
                        logging.warning(f"Could not extract base model from model '{full_model}' in {summary_csv}. Skipping row.")
                        continue
                    
                    aggregated_data.append({
                        'Model': base_model,
                        'Scenario': scenario,
                        'Flight Distance': flight_distance,
                        'Holes Reached': holes_reached
                    })
                    
            except Exception as e:
                logging.error(f"Error reading {summary_csv}: {e}")
                continue

    if not aggregated_data:
        logging.error("No data aggregated. Exiting without creating summary.")
        return

    aggregated_df = pd.DataFrame(aggregated_data)
    aggregated_df['Model'] = aggregated_df['Model'].str.strip()
    aggregated_df['Scenario'] = aggregated_df['Scenario'].str.strip()
    aggregated_df.sort_values(by=['Model', 'Scenario'], inplace=True)

    aggregated_df.to_csv(output_file, index=False)
    logging.info(f"Aggregated summary saved to {output_file}")

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Aggregate summary.csv files from subdirectories into a total summary.")
    parser.add_argument(
        '--parent_dir',
        type=str,
        default=os.getcwd(),
        help='Path to the parent directory containing subdirectories with summary.csv files.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='total_summary.csv',
        help='Path to the output aggregated CSV file.'
    )
    
    args = parser.parse_args()
    
    parent_directory = args.parent_dir
    output_file = os.path.join(parent_directory, args.output_file)
    
    if not os.path.isdir(parent_directory):
        logging.error(f"Parent directory '{parent_directory}' does not exist.")
        return
    
    aggregate_summaries(parent_directory, output_file)

if __name__ == "__main__":
    main()
