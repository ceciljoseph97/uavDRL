import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


normal_data = {
    'Model': [
        'A2C-D','A2C-DL','A2C-DRF','A2C-DS','A2C-MRGB','A2C-SRGB','A2C-SRGBL','A2C-SRGBRF','A2C-SRGBS',
        'PPO-D','PPO-DL','PPO-DRF','PPO-DS','PPO-MRGB','PPO-SRGB','PPO-SRGBL','PPO-SRGBRF','PPO-SRGBS'
    ],
    'Train_FlightDist': [
        17.841,16.345,18.224,16.863,12.645,13.325,14.885,13.825,12.533,
        13.087,18.634,19.878,13.642,13.260,13.976,14.916,13.893,13.876
    ],
    'Train_Targets': [
        4,4,4,4,3,3,3,3,3,
        3,4,5,3,3,3,3,3,3
    ],
    'Test_FlightDist': [
        17.467,14.112,14.673,14.631,9.454,7.780,10.613,5.45,6.01,
        8.741,16.319,16.994,13.898,9.030,8.216,10.071,13.500,7.712
    ],
    'Test_Targets': [
        4,3,4,3,2,1,2,1,1,
        2,4,4,3,2,2,2,3,1
    ],
    'Collision_Rate': [
        13.0,30.0,25.0,30.0,56.0,73.0,43.0,80.0,86.0,
        69.0,17.0,15.0,34.0,58.0,69.0,59.0,94.0,59.0
    ],
    'Target_Reached': [
        4,3,4,3,2,1,2,1,1,
        2,4,4,3,2,2,2,3,1
    ]
}
df_normal = pd.DataFrame(normal_data)
leaves_data = {
    'Model': [
        'A2C-D','A2C-DL','A2C-DRF','A2C-DS','A2C-MRGB','A2C-SRGB','A2C-SRGBL','A2C-SRGBRF','A2C-SRGBS',
        'PPO-D','PPO-DL','PPO-DRF','PPO-DS','PPO-MRGB','PPO-SRGB','PPO-SRGBL','PPO-SRGBRF','PPO-SRGBS'
    ],
    'Train_FlightDist': [
        11.235,15.432,16.789,14.123,11.575,11.342,13.750,12.27,11.73,
        12.644,17.452,18.750,14.228,8.403,7.184,8.507,5.670,6.547
    ],
    'Train_Targets': [
        3,4,4,3,3,3,3,3,3,
        3,4,4,3,2,1,2,1,1
    ],
    'Test_FlightDist': [
        9.362,12.843,12.582,13.533,8.489,6.714,10.121,4.75,5.65,
        8.809,15.837,16.707,16.500,8.403,7.184,8.507,4.541,6.547
    ],
    'Test_Targets': [
        2,3,3,3,2,1,2,1,1,
        2,4,4,4,2,1,2,1,1
    ],
    'Collision_Rate': [
        66.0,35.0,40.0,36.0,67.0,77.0,51.0,85.0,89.0,
        68.0,22.0,13.0,32.0,66.0,78.0,71.0,93.0,73.0
    ],
    'Target_Reached': [
        2,3,3,3,2,1,2,1,1,
        2,4,4,4,2,1,2,1,1
    ]
}
df_leaves = pd.DataFrame(leaves_data)
rainfog_data = {
    'Model': [
        'A2C-D','A2C-DL','A2C-DRF','A2C-DS','A2C-MRGB','A2C-SRGB','A2C-SRGBL','A2C-SRGBRF','A2C-SRGBS',
        'PPO-D','PPO-DL','PPO-DRF','PPO-DS','PPO-MRGB','PPO-SRGB','PPO-SRGBL','PPO-SRGBRF','PPO-SRGBS'
    ],
    'Train_FlightDist': [
        12.678,15.876,17.529,14.678,12.031,11.901,12.509,9.922,11.532,
        12.520,17.804,17.967,13.881,13.996,11.952,13.512,12.815,12.728
    ],
    'Train_Targets': [
        3,4,4,3,3,3,3,3,3,
        3,4,4,3,3,3,3,3,3
    ],
    'Test_FlightDist': [
        7.218,12.726,14.006,12.782,3.468,2.256,2.496,2.325,2.172,
        8.207,15.473,15.708,16.800,4.966,2.094,3.728,4.937,2.432
    ],
    'Test_Targets': [
        1,3,3,3,0,0,0,0,0,
        2,4,4,4,1,0,0,1,0
    ],
    'Collision_Rate': [
        78.0,40.0,28.0,43.0,100.0,100.0,100.0,100.0,100.0,
        72.0,23.0,22.0,31.0,100.0,100.0,99.0,92.0,100.0
    ],
    'Target_Reached': [
        1,3,3,3,0,0,0,0,0,
        2,4,4,4,1,0,0,1,0
    ]
}
df_rainfog = pd.DataFrame(rainfog_data)
snow_data = {
    'Model': [
        'A2C-D','A2C-DL','A2C-DRF','A2C-DS','A2C-MRGB','A2C-SRGB','A2C-SRGBL','A2C-SRGBRF','A2C-SRGBS',
        'PPO-D','PPO-DL','PPO-DRF','PPO-DS','PPO-MRGB','PPO-SRGB','PPO-SRGBL','PPO-SRGBRF','PPO-SRGBS'
    ],
    'Train_FlightDist': [
        13.890,14.567,15.942,14.596,13.213,12.891,13.980,13.825,12.533,
        13.201,16.992,17.629,12.469,9.428,13.131,14.111,12.151,13.632
    ],
    'Train_Targets': [
        3,3,4,3,3,3,3,3,3,
        3,4,4,3,2,3,3,3,3
    ],
    'Test_FlightDist': [
        7.259,11.872,10.376,12.042,7.236,7.001,10.246,5.45,6.01,
        7.541,15.597,15.462,16.200,9.428,6.668,8.994,4.751,7.451
    ],
    'Test_Targets': [
        1,3,2,3,1,1,2,1,1,
        1,4,4,4,2,1,2,0,1
    ],
    'Collision_Rate': [
        78.0,44.0,53.0,49.0,74.0,78.0,50.0,72.0,86.0,
        72.0,24.0,25.0,43.0,54.0,79.0,69.0,94.0,61.0
    ],
    'Target_Reached': [
        1,3,2,3,1,1,2,1,1,
        1,4,4,4,2,1,2,0,1
    ]
}
df_snow = pd.DataFrame(snow_data)

fm_data = pd.DataFrame({
    'Model': ['PPO-DFM', 'PPO-SRGBFM'],
    'Test_FlightDist': [
        [16.39, 15.73, 15.35, 15.92],
        [11.79, 11.79, 4.68, 11.76]
    ],
    'Collision_Rate': [
        [14.0, 16.0, 20.0, 22.0],
        [39.0, 44.0, 41.0, 46.0]
    ],
    'Target_Reached': [
        [4.0, 4.0, 4.0, 4.0],
        [3.0, 3.0, 1.0, 3.0]
    ]
})

baseline_mapping = {
    'PPO-DFM': {
        'Normal': 'PPO-D',
        'Leaves': 'PPO-DL',
        'RainFog': 'PPO-DRF',
        'Snow': 'PPO-DS'
    },
    'PPO-SRGBFM': {
        'Normal': 'PPO-SRGB',
        'Leaves': 'PPO-SRGBL',
        'RainFog': 'PPO-SRGBRF',
        'Snow': 'PPO-SRGBS'
    }
}

def get_baseline_values(model, condition):
    """Get baseline values for FM model comparison"""
    baseline_model = baseline_mapping[model][condition]
    if condition == 'Normal':
        df = df_normal
    elif condition == 'Leaves':
        df = df_leaves
    elif condition == 'RainFog':
        df = df_rainfog
    else:  # Snow
        df = df_snow
    return df[df['Model'] == baseline_model].iloc[0]

def get_algorithm(model_name):
    return 'A2C' if str(model_name).startswith('A2C') else 'PPO'

def safe_percentage_change(current, baseline, higher_is_better=True):
    if baseline == 0:
        if current == 0:
            return 0
        return float('inf') if higher_is_better else float('-inf')
    
    if higher_is_better:
        return ((current - baseline) / baseline) * 100
    else:
        return (current - baseline)

def compare_performance(df_normal, df_leaves, df_rainfog, df_snow):
    all_results = []
    
    for df_weather, weather in [(df_normal, 'Normal'), (df_leaves, 'Leaves'), 
                               (df_rainfog, 'RainFog'), (df_snow, 'Snow')]:
        
        df = df_weather.copy()
        
        for idx, row in df.iterrows():
            model = row['Model']
            training_condition = get_training_condition(model)
            baseline_df = get_baseline_df(training_condition)
            baseline = baseline_df[baseline_df['Model'] == model].iloc[0]
            flight_change = safe_percentage_change(
                row['Test_FlightDist'], 
                baseline['Test_FlightDist'],
                higher_is_better=True
            )
            
            collision_change = safe_percentage_change(
                row['Collision_Rate'], 
                baseline['Collision_Rate'],
                higher_is_better=False
            )
            if baseline['Target_Reached'] > 0:
                target_change = ((row['Target_Reached'] - baseline['Target_Reached']) / baseline['Target_Reached']) * 100
            else:
                target_change = float('inf') if row['Target_Reached'] > 0 else 0
            
            result = {
                'Model': model,
                'Algorithm': get_algorithm(model),
                'Training_Condition': training_condition,
                'Test_Condition': weather,
                'FlightDist_%Change': flight_change,
                'Collision_%Change': collision_change,
                'Target_%Change': target_change
            }
            all_results.append(result)
    
    return pd.DataFrame(all_results)

def get_training_condition(model_name):
    if model_name.endswith('L'):
        return 'Leaves'
    elif model_name.endswith('RF'):
        return 'RainFog'
    elif model_name.endswith('S'):
        return 'Snow'
    return 'Normal'

def get_baseline_df(condition):
    if condition == 'Leaves':
        return df_leaves
    elif condition == 'RainFog':
        return df_rainfog
    elif condition == 'Snow':
        return df_snow
    return df_normal

fm_results = []
conditions = ['Normal', 'Leaves', 'RainFog', 'Snow']

for idx, row in fm_data.iterrows():
    model = row['Model']
    for i, condition in enumerate(conditions):
        baseline = get_baseline_values(model, condition)
        
        result = {
            'Model': model,
            'Algorithm': 'PPO',
            'Test_Condition': condition,
            'FlightDist_%Change': safe_percentage_change(
                row['Test_FlightDist'][i], 
                baseline['Test_FlightDist'],
                higher_is_better=True
            ),
            'Collision_%Change': safe_percentage_change(
                row['Collision_Rate'][i], 
                baseline['Collision_Rate'],
                higher_is_better=False
            ),
            'Target_%Change': safe_percentage_change(
                row['Target_Reached'][i], 
                baseline['Target_Reached'],
                higher_is_better=True
            )
        }
        fm_results.append(result)

for result in fm_results:
    result['Algorithm'] = 'PPO'

fm_results_df = pd.DataFrame(fm_results)

results_df = compare_performance(df_normal, df_leaves, df_rainfog, df_snow)
combined_results = pd.concat([results_df, fm_results_df], ignore_index=True)

combined_results['Algorithm'] = combined_results.apply(
    lambda x: 'PPO_FM' if 'FM' in str(x['Model']) else x['Algorithm'], 
    axis=1
)

print("\n=== Performance Change by Training and Test Conditions ===")
summary = combined_results.groupby(['Training_Condition', 'Test_Condition']).agg({
    'FlightDist_%Change': 'mean',
    'Collision_%Change': 'mean'
}).round(2)
print(summary)

print("\n=== FM Models Performance Analysis ===")
fm_summary = fm_results_df.groupby(['Model', 'Test_Condition']).agg({
    'FlightDist_%Change': 'mean',
    'Collision_%Change': 'mean'
}).round(2)
print(fm_summary)

algo_summary = combined_results.groupby(['Algorithm', 'Test_Condition']).agg({
    'FlightDist_%Change': 'mean',
    'Collision_%Change': 'mean'
}).round(2)

print("\n=== Algorithm Performance by Conditions ===")
print(algo_summary)

def analyze_variant_performance(data, variant):
    variant_data = data[data['Model'].str.contains(variant)]
    variant_summary = variant_data.groupby(['Algorithm', 'Test_Condition']).agg({
        'FlightDist_%Change': 'mean',
        'Collision_%Change': 'mean'
    }).round(2)
    print(f"\n=== {variant} Variant Performance Analysis ===")
    print(variant_summary)

analyze_variant_performance(combined_results, 'D')
analyze_variant_performance(combined_results, 'SRGB')
analyze_variant_performance(combined_results, 'MRGB')

plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

def create_heatmap(data, metric, title):
    plt.figure(figsize=(12, 10))
    
    if metric == 'Dist':
        column = 'FlightDist_%Change'
    else:  
        column = 'Collision_%Change'
    
    heatmap_data = data.pivot(index='Model', columns='Test_Condition', values=column)
    
    max_val = np.nanmax(heatmap_data.replace([np.inf, -np.inf], np.nan))
    min_val = np.nanmin(heatmap_data.replace([np.inf, -np.inf], np.nan))
    heatmap_data = heatmap_data.replace(np.inf, max_val).replace(-np.inf, min_val)
    
    sns.heatmap(heatmap_data, 
                cmap='RdYlBu_r',
                center=0,
                annot=True,
                fmt='.1f',
                annot_kws={'size': 10},
                cbar_kws={'label': 'Performance Change (%)'})
    
    plt.title(f'{title} Changes Across Weather Conditions', pad=20)
    plt.xlabel('Test Weather', labelpad=10)
    plt.ylabel('Model', labelpad=10)
    plt.tight_layout()
    plt.savefig(f'performance_{metric.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

create_heatmap(combined_results, 'Dist', 'Flight Distance')
create_heatmap(combined_results, 'Coll', 'Collision Rate')

plt.figure(figsize=(12, 6))
metrics = ['Dist', 'Coll']
width = 0.25 
x = np.arange(len(conditions))

for i, metric in enumerate(metrics):
    plt.subplot(1, 2, i+1)
    for j, algo in enumerate(['A2C', 'PPO', 'PPO_FM']):
        values = []
        for cond in conditions:
            val = algo_summary.loc[(algo, cond), f'FlightDist_%Change' if metric=='Dist' 
                               else f'Collision_%Change']
            values.append(val)
        values = np.array(values)
        values[np.isinf(values)] = np.nan
        plt.bar(x + (j-1)*width, values, width, label=algo, alpha=0.8)
    
    plt.title(f'{"Flight Distance" if metric=="Dist" else "Collision Rate"}',
              pad=15, fontsize=14)
    plt.xticks(x, conditions, rotation=45, fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylabel('Performance Change (%)', fontsize=12, labelpad=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

plt.suptitle('Algorithm Performance Comparison Across Weather Conditions', 
             y=1.05, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

def plot_variant_comparison(data, variant):
    plt.figure(figsize=(12, 6))
    metrics = ['Dist', 'Coll']
    width = 0.25
    x = np.arange(len(conditions))
    
    variant_data = data[data['Model'].str.contains(variant)]
    variant_summary = variant_data.groupby(['Algorithm', 'Test_Condition']).agg({
        'FlightDist_%Change': 'mean',
        'Collision_%Change': 'mean'
    }).round(2)
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 2, i+1)
        for j, algo in enumerate(['A2C', 'PPO', 'PPO_FM']):
            if algo in variant_summary.index.get_level_values(0):
                values = []
                for cond in conditions:
                    try:
                        val = variant_summary.loc[(algo, cond), f'FlightDist_%Change' if metric=='Dist' 
                                           else f'Collision_%Change']
                        values.append(val)
                    except:
                        values.append(np.nan)
                values = np.array(values)
                values[np.isinf(values)] = np.nan
                plt.bar(x + (j-1)*width, values, width, label=algo, alpha=0.8)
        
        plt.title(f'{variant} {metric} Performance',
                  pad=15, fontsize=14)
        plt.xticks(x, conditions, rotation=45, fontsize=11)
        plt.yticks(fontsize=11)
        plt.ylabel('Performance Change (%)', fontsize=12, labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
    
    plt.suptitle(f'{variant} Variant Performance Comparison', 
                 y=1.05, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{variant.lower()}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_variant_comparison(combined_results, 'D')
plot_variant_comparison(combined_results, 'SRGB')
plot_variant_comparison(combined_results, 'MRGB')