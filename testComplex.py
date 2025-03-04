import gym
import time
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import airsim
from scripts.weather_callback import WeatherChangeCallback

class CustomObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def observation(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] in [1, 3]:
            return np.transpose(obs, (2, 0, 1))
        return obs

def main(model_path, env_select, train_mode, episodes=10, render=False, weather_condition=None):
    if env_select == "obstacle":
        config_file = 'scripts/env_config.yml'
    elif env_select == "forest":
        config_file = 'scripts/checkpoints_envNHForestDed.yaml'
    else:
        config_file = 'scripts/checkpoints_envNHNormalRet.yaml'
        
    with open(config_file, 'r') as f:
        env_config = yaml.safe_load(f)

    if train_mode == "depth":
        image_shape = (84, 84, 1)
    elif train_mode == "multi_rgb":
        image_shape = (84, 84, 3)
    else:
        image_shape = (84, 84, 3)

    env = DummyVecEnv([lambda: CustomObsWrapper(Monitor(
        gym.make(
            "scripts:airsim-env-v2",
            ip_address="127.0.0.1",
            image_shape=image_shape,
            env_config=env_config["TrainEnv"],
            input_mode=train_mode
        )
    ))])

    if weather_condition:
        print(f"Setting weather condition to: {weather_condition}")
        env_instance = env.envs[0]
        drone_client = env_instance.drone
        
        weather_callback = WeatherChangeCallback(
            freq=1000000,
            client=drone_client,
            selected_scenarios=[weather_condition],
            single_weather_condition=True,
            verbose=1
        )
        
        weather_callback._set_weather_once()

    if "ppo" in model_path.lower():
        model = PPO.load(model_path, env=env)
    else:
        model = A2C.load(model_path, env=env)
    
    print(f"Model loaded from {model_path}")
    
    success_count = 0
    collision_count = 0
    timeout_count = 0
    distances = []
    trajectories = []
    
    for episode in range(episodes):
        print(f"\nRunning test episode {episode+1}/{episodes}")
        obs = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            step_count += 1
            
            if render:
                env.render()
                time.sleep(0.01) 
        
        info = info[0]
        
        if 'success' in info and info['success']:
            success_count += 1
            print(f"Episode {episode+1}: SUCCESS! Reward: {episode_reward:.2f}, Steps: {step_count}")
        elif 'collision' in info and info['collision']:
            collision_count += 1
            print(f"Episode {episode+1}: COLLISION! Reward: {episode_reward:.2f}, Steps: {step_count}")
        else:
            timeout_count += 1
            print(f"Episode {episode+1}: TIMEOUT! Reward: {episode_reward:.2f}, Steps: {step_count}")
        
        if 'trajectory' in info:
            trajectories.append(info['trajectory'])
            
            if 'target_position' in info:
                final_pos = info['trajectory'][-1]
                target_pos = info['target_position']
                distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(final_pos, target_pos)))
                distances.append(distance)
                print(f"Final distance to target: {distance:.2f} meters")
    
    print("\n===== TEST RESULTS =====")
    print(f"Total episodes: {episodes}")
    print(f"Success rate: {success_count}/{episodes} ({success_count/episodes*100:.2f}%)")
    print(f"Collision rate: {collision_count}/{episodes} ({collision_count/episodes*100:.2f}%)")
    print(f"Timeout rate: {timeout_count}/{episodes} ({timeout_count/episodes*100:.2f}%)")
    
    if distances:
        print(f"Average final distance to target: {np.mean(distances):.2f} meters")
        print(f"Min distance to target: {np.min(distances):.2f} meters")
        print(f"Max distance to target: {np.max(distances):.2f} meters")
    
    if trajectories:
        visualize_trajectories(trajectories, env_config["TrainEnv"]["sections"])

def visualize_trajectories(trajectories, sections):
    """Visualize drone trajectories in 3D space"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, traj in enumerate(trajectories):
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f'Episode {i+1}', linewidth=2)
        
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='green', s=100, marker='^', label='Start' if i == 0 else "")
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='red', s=100, marker='o', label='End' if i == 0 else "")
    
    for i, section in enumerate(sections):
        if isinstance(section["target"], dict):
            target = [section["target"]["x"], section["target"]["y"], section["target"]["z"]]
        else:
            target = section["target"]
        
        ax.scatter(target[0], target[1], target[2], color='blue', s=200, marker='*', 
                  label=f'Target {i+1}' if i == 0 else f'Target {i+1}')
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Drone Trajectories')
    
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc='upper right')
    
    ax.view_init(elev=30, azim=45)
    
    plt.savefig('drone_trajectories.png', dpi=300, bbox_inches='tight')
    print("Trajectory visualization saved as 'drone_trajectories.png'")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained drone navigation RL agent.")
    parser.add_argument('--model', type=str, required=True,
                        help="Path to the trained model file")
    parser.add_argument('--env', type=str, choices=["obstacle", "forest", "city"], default="forest",
                        help="Test environment selection")
    parser.add_argument('--mode', type=str, choices=["depth", "single_rgb", "multi_rgb"], default="depth",
                        help="Input mode (depth, single_rgb, multi_rgb)")
    parser.add_argument('--episodes', type=int, default=10,
                        help="Number of test episodes to run")
    parser.add_argument('--render', action='store_true',
                        help="Enable rendering during testing")
    parser.add_argument('--weather', type=str, default=None,
                        help="Weather condition to use for testing (e.g., Clear, SnowOnly, RainAndFog)")
    
    args = parser.parse_args()

    main(
        model_path=args.model,
        env_select=args.env,
        train_mode=args.mode,
        episodes=args.episodes,
        render=args.render,
        weather_condition=args.weather
    ) 