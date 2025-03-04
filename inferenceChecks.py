import torch
import torch.nn as nn
import os
import gym
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from scripts.network import NatureCNN

ALGORITHMS = {
    "A2C": A2C,
    "PPO": PPO,
}

CNN_ARCHITECTURES = {
    "NatureCNN": NatureCNN,
}

MAX_HOLES = 5

def save_positions_to_csv(all_positions, filename):
    import csv
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Episode', 'StartPos', 'X', 'Y', 'Z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in all_positions:
            episode = data['episode']
            start_pos = data['start_pos']
            positions = data['positions']
            for pos in positions:
                x, y, z = pos
                writer.writerow({'Episode': episode, 'StartPos': start_pos, 'X': x, 'Y': y, 'Z': z})
    logging.info(f"Positions saved to {filename}")

def plot_positions_from_csv(filename, plot_name, suffix):
    df = pd.read_csv(filename)
    plt.figure(figsize=(10, 6))
    
    max_shifted_x = 0
    num_plotted_episodes = 0
    
    for episode in df['Episode'].unique():
        episode_data = df[df['Episode'] == episode]
        x_positions = episode_data['X'].values
        y_positions = episode_data['Y'].values
        
        x_shifted = x_positions - x_positions[0]
        y_shifted = y_positions
        
        if x_shifted.max() >= 18:
            if x_shifted.max() > max_shifted_x:
                max_shifted_x = x_shifted.max()
            
            plt.plot(
                x_shifted, 
                y_shifted, 
                color='blue',
                alpha=0.7
            )
            num_plotted_episodes += 1
        else:
            logging.info(f"Episode {episode} did not travel at least 18 meters, skipping.")
    
    if num_plotted_episodes == 0:
        logging.warning("No episodes traveled at least 18 meters. No plot will be generated.")
        return
    
    plt.ylim([-2, 2])

    wall_positions = np.arange(3.7, max_shifted_x + 3.7, 3.7)
    for x in wall_positions:
        plt.axvline(x=x, color='red', linestyle='--', alpha=0.5)

    plt.xlabel('Distance travelled')
    plt.ylabel('Horizontal Movement')
    plt.title(f'Drone Trajectories - {suffix}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_name)
    logging.info(f"Trajectories plotted and saved as '{plot_name}'.")

def setup_logging(log_file):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])

def main(algorithm, cnn, test_mode, test_type, model_path, suffix):
    output_folder = f"./inference_outputs/{suffix}"
    os.makedirs(output_folder, exist_ok=True)

    log_file = os.path.join(output_folder, 'inference.log')
    setup_logging(log_file)

    with open('scripts/env_config.yml', 'r') as f:
        env_config = yaml.safe_load(f)

    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    if test_mode:
        config["test_mode"] = test_mode
    if test_type:
        config["test_type"] = test_type

    image_shape = (50, 50, 1) if config["test_mode"] == "depth" else (50, 50, 3)

    env = DummyVecEnv([lambda: Monitor(
        gym.make(
            "scripts:test-env-v0",
            ip_address="127.0.0.1",
            image_shape=image_shape,
            env_config=env_config["TrainEnv"],
            input_mode=config["test_mode"],
            test_mode=config["test_type"]
        )
    )])

    env = VecTransposeImage(env)

    policy_kwargs = dict(features_extractor_class=CNN_ARCHITECTURES[cnn])

    model = ALGORITHMS[algorithm].load(
        path=model_path,
        env=env,
        custom_objects={'policy_kwargs': policy_kwargs}
    )

    num_episodes = 100
    all_positions = []
    all_flight_distances = []
    all_holes_reached = []

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        positions = []
        start_pos = None
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            done = dones[0]
            info = infos[0]
            if 'positions' in info:
                positions = info['positions']
                start_pos = info['start_pos']
        
        all_positions.append({'episode': episode, 'start_pos': start_pos, 'positions': positions})
        
        if positions:
            max_x = max(pos[0] for pos in positions)
            flight_distance = max_x
            holes_reached = int(max_x // 3.7)
        else:
            flight_distance = 0
            holes_reached = 0
        
        all_flight_distances.append(flight_distance)
        all_holes_reached.append(holes_reached)
        
        logging.info(f"Episode {episode} completed.")

    csv_filename = os.path.join(output_folder, f'{suffix}_positions.csv')
    save_positions_to_csv(all_positions, filename=csv_filename)

    plot_filename = os.path.join(output_folder, f'{suffix}_trajectories.png')
    plot_positions_from_csv(filename=csv_filename, plot_name=plot_filename, suffix=suffix)

    mean_flight_distance = np.mean(all_flight_distances) if all_flight_distances else 0
    mean_holes_reached = np.mean(all_holes_reached) if all_holes_reached else 0

    logging.info("-----------------------------------")
    logging.info(f"> Total episodes: {num_episodes}")
    logging.info(f"> Flight distance (mean): {mean_flight_distance:.2f}")
    logging.info(f"> Holes reached (mean): {mean_holes_reached:.2f} out of {MAX_HOLES}")
    logging.info("-----------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained RL model.")
    parser.add_argument('--algorithm', type=str, choices=ALGORITHMS.keys(), required=True,
                        help="Algorithm used for training (A2C, PPO)")
    parser.add_argument('--cnn', type=str, choices=CNN_ARCHITECTURES.keys(), required=True,
                        help="CNN architecture used for feature extraction (NatureCNN, EfficientCNN)")
    parser.add_argument('--test_mode', type=str, required=False, default=None,
                        help="Test mode (depth, multi_rgb, single_rgb) to override config file settings")
    parser.add_argument('--test_type', type=str, required=False, default=None,
                        help="Type of test (specific test scenario)")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument('--suffix', type=str, required=False, default='drone_trajectories',
                        help="Suffix for naming the output files and folder (e.g., trajectories)")

    args = parser.parse_args()
    main(args.algorithm, args.cnn, args.test_mode, args.test_type, args.model_path, args.suffix)
