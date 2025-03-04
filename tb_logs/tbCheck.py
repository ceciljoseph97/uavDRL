import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

#tb_logs_dir = r"C:\Workspace\DRL\tb_logs"
tb_logs_dir = r"D:\Masters\Sem5\uavRL\tb_logs"
tags = [
    ("eval/mean_reward", ["eval/mean_reward"]),
    ("train/policy_gradient_loss", ["train/policy_loss"]),
    ("train/value_loss", ["train/critic_loss"]),
    ("train/entropy_loss", ["train/entropy"])
]

criteria = {
    "eval/mean_reward": "max",
    "train/policy_gradient_loss": "min",
    "train/value_loss": "min",
    "train/entropy_loss": "max"
}
abbreviations = {
    "a2c_naturecnn_depth_obstacle_weather_leavesfalling": "A2C-DL",
    "a2c_naturecnn_depth_obstacle_weather_rainandfog": "A2C-DRF",
    "a2c_naturecnn_depth_obstacle_weather_snowonly": "A2C-DS",
    "a2c_naturecnn_depth": "A2C-D",
    "a2c_naturecnn_multi_rgb_obstacle_weather_off": "A2C-MRGB",
    "a2c_naturecnn_single_rgb_obstacle_weather_leavesfalling": "A2C-SRGBL",
    "a2c_naturecnn_single_rgb_obstacle_weather_rainandfog": "A2C-SRGBRF",
    "a2c_naturecnn_single_rgb_obstacle_weather_snowonly": "A2C-SRGBS",
    "a2c_naturecnn_single_rgb": "A2C-SRGB",
    "ppo_naturecnn_depth_obstacle_weather_clr_snw_rnf_lvf": "PPO-DFM",
    "ppo_naturecnn_depth_obstacle_weather_leavesfalling": "PPO-DL",
    "ppo_naturecnn_depth_obstacle_weather_rainandfog": "PPO-DRF",
    "ppo_naturecnn_depth_obstacle_weather_snowonly": "PPO-DS",
    "ppo_naturecnn_depth": "PPO-D",
    "ppo_naturecnn_multi_rgb": "PPO-MRGB",
    "ppo_naturecnn_single_rgb_obstacle_weather_clr_snw_rnf_lvf": "PPO-SRGBFM",
    "ppo_naturecnn_single_rgb_obstacle_weather_leavesfalling": "PPO-SRGBL",
    "ppo_naturecnn_single_rgb_obstacle_weather_rainandfog": "PPO-SRGBRF",
    "ppo_naturecnn_single_rgb_obstacle_weather_snowonly": "PPO-SRGBS",
    "ppo_naturecnn_single_rgb": "PPO-SRGB"
}

def get_abbreviation(folder_name):
    folder_name = folder_name.strip().lower()
    prefix = "tb_logs_"
    if folder_name.startswith(prefix):
        folder_name = folder_name[len(prefix):]
    
    for key, abbr in sorted(abbreviations.items(), key=lambda x: len(x[0]), reverse=True):
        if folder_name.startswith(key):
            return abbr
    return folder_name

def extract_pytorch_scalars(event_file, tag_groups):
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    
    tag_data = {}
    for primary_tag, aliases in tag_groups:
        for tag in [primary_tag] + aliases:
            if tag in event_acc.Tags()["scalars"]:
                events = event_acc.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                tag_data[primary_tag] = (steps, values)
                break
    return tag_data

def smooth_data(data, window=100):
    if len(data) < window:
        window = len(data)
    return np.convolve(data, np.ones(window) / window, mode='same')

all_data = {primary_tag: {} for primary_tag, _ in tags}

for subdir in os.listdir(tb_logs_dir):
    subdir_path = os.path.join(tb_logs_dir, subdir)
    if os.path.isdir(subdir_path):
        for subsubdir in os.listdir(subdir_path):
            subsubdir_path = os.path.join(subdir_path, subsubdir)
            if os.path.isdir(subsubdir_path):
                for file in os.listdir(subsubdir_path):
                    if file.startswith("events.out.tfevents"):
                        event_file_path = os.path.join(subsubdir_path, file)
                        extracted_data = extract_pytorch_scalars(event_file_path, tags)
                        for primary_tag, values in extracted_data.items():
                            if values:
                                all_data[primary_tag][subsubdir] = values

colors = plt.cm.tab20.colors

for primary_tag, _ in tags:
    plt.figure(figsize=(16, 8))
    found_data = False
    color_idx = 0
    
    print(f"\n=== Summary for tag: {primary_tag} ===")
    
    for algo_name, (steps, vals) in all_data[primary_tag].items():
        if steps and vals:
            smoothed_vals = smooth_data(vals, window=60)
            label = get_abbreviation(algo_name)
            color = colors[color_idx % len(colors)]
            color_idx += 1
            plt.plot(steps, smoothed_vals, label=label, color=color, linewidth=2.5)
            found_data = True
            
            avg_smoothed = np.mean(smoothed_vals)
            max_smoothed = np.max(smoothed_vals)
            print(f"Model: {label}")
            print(f"  Total observed points: {len(vals)}")
            print(f"  Average smoothed value: {avg_smoothed:.2f}")
            print(f"  Max smoothed value: {max_smoothed:.2f}")
    
    plt.xlabel("Steps", fontsize=18)
    plt.ylabel(primary_tag.split("/")[-1], fontsize=18)
    plt.title(f"{primary_tag}", fontsize=20, pad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if found_data:
        plt.legend(fontsize=12, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    else:
        print(f"[No Data] Found for tag: {primary_tag}")
