import gym
import time
import yaml
import argparse
import torch

from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback

from scripts.network import NatureCNN
from scripts.weather_callback import WeatherChangeCallback

import airsim

ALGORITHMS = {
    "A2C": A2C,
    "PPO": PPO,
    "DQN": DQN,
}

CNN_ARCHITECTURES = {
    "NatureCNN": NatureCNN,
}

def main(algorithm, cnn, total_timesteps, train_mode, env_select, weather_freq,
         use_weather_changes, weather_scenarios, weather_condition):

    config_file = 'scripts/env_config.yml' if env_select == "obstacle" else 'scripts/checkpoints_envNHNormalRet.yaml'
    with open(config_file, 'r') as f:
        env_config = yaml.safe_load(f)

    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    if train_mode:
        config["train_mode"] = train_mode

    if config["train_mode"] == "depth":
        image_shape = (50, 50, 1)
    elif config["train_mode"] == "multi_rgb":
        image_shape = (50, 50, 3)
    else:
        image_shape = (50, 50, 3)

    if env_select == "obstacle":
        env = DummyVecEnv([lambda: Monitor(
            gym.make(
                "scripts:airsim-env-v0",
                ip_address="127.0.0.1",
                image_shape=image_shape,
                env_config=env_config["TrainEnv"],
                input_mode=config["train_mode"]
            )
        )])
    else:
        env = DummyVecEnv([lambda: Monitor(
            gym.make(
                "scripts:airsim-env-v2",
                ip_address="127.0.0.1",
                image_shape=image_shape,
                env_config=env_config["TrainEnv"],
                input_mode=config["train_mode"]
            )
        )])

    env = VecTransposeImage(env)


    env_instance = env.envs[0]
    drone_client = env_instance.drone


    if use_weather_changes:
        if weather_condition:
            print(f"Weather changes enabled. Using single weather condition: {weather_condition}")
            weather_scenarios = [weather_condition]
            single_weather = True
        elif not weather_scenarios:
            print("Weather changes enabled but no scenarios selected. Using default scenarios.")
            weather_scenarios = ["Clear", "SnowOnly", "RainAndFog", "LeavesFalling"]
            single_weather = False
        else:
            print(f"Weather changes enabled. Selected scenarios: {weather_scenarios}")
            single_weather = False


        weather_callback = WeatherChangeCallback(
            freq=weather_freq,
            client=drone_client,
            selected_scenarios=weather_scenarios,
            single_weather_condition=single_weather,
            verbose=1
        )
    else:
        print("Weather changes disabled.")
        weather_callback = None

    policy_kwargs = dict(
        features_extractor_class=CNN_ARCHITECTURES[cnn]
    )

    if use_weather_changes:
        if weather_condition:
            weather_descriptor = f"weather_{weather_condition.lower()}"
        elif weather_scenarios:
            
            abbreviations = {
                "Clear": "clr",
                "LightRain": "lrn",
                "HeavyRain": "hrn",
                "FogOnly": "fog",
                "SnowOnly": "snw",
                "RainAndFog": "rnf",
                "LeavesFalling": "lvf",
                "RainAndLeaves": "rnlv"
            }
            abbr_list = [abbreviations.get(scenario, scenario[:3].lower()) for scenario in weather_scenarios]
            weather_descriptor = "weather_" + "_".join(abbr_list)
        else:
            weather_descriptor = "weather_default"
    else:
        weather_descriptor = "weather_off"


    model_name_prefix = f"{algorithm.lower()}_{cnn.lower()}_{config['train_mode']}_{env_select.lower()}_{weather_descriptor}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ALGORITHMS[algorithm](
        'CnnPolicy',
        env,
        verbose=1,
        seed=1,
        device=device,
        tensorboard_log=f"./tb_logs/tb_logs_{model_name_prefix}/",
        policy_kwargs=policy_kwargs,
    )

    eval_callback = EvalCallback(
        env,
        callback_on_new_best=None,
        n_eval_episodes=20,
        best_model_save_path=f"saved_models/{model_name_prefix}",
        log_path=f"./tb_logs/tb_logs_{model_name_prefix}/",
        eval_freq=1024,
    )

    callbacks = [eval_callback]

    if weather_callback:
        callbacks.append(weather_callback)

    log_name = model_name_prefix + "_" + str(int(time.time()))
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=log_name,
        callback=callbacks,
    )
    model.save(log_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RL agent with the specified algorithm and CNN.")
    parser.add_argument('--algorithm', type=str, choices=ALGORITHMS.keys(), required=True,
                        help="Algorithm to use for training (A2C, PPO, DQN)")
    parser.add_argument('--cnn', type=str, choices=CNN_ARCHITECTURES.keys(), required=True,
                        help="CNN architecture to use for feature extraction (NatureCNN)")
    parser.add_argument('--total_timesteps', type=int, default=35000,
                        help="Total number of timesteps to train the model")
    parser.add_argument('--train_mode', type=str, required=False, default=None,
                        help="Training mode (depth, single_rgb, multi_rgb) to override config file settings")
    parser.add_argument('--env_select', type=str, required=False, default="obstacle",
                        help="Training environment select (obstacle, city) to override config file settings")
    parser.add_argument('--weather_freq', type=int, default=10000,
                        help="Frequency in timesteps to change weather conditions")
    parser.add_argument('--weather', action='store_true',
                        help="Enable dynamic weather changes during training")
    parser.add_argument('--weather_scenarios', type=str, nargs='*', default=[],
                        help="List of weather scenarios to include (e.g., Clear, LightRain, HeavyRain, FogOnly, SnowOnly, RainAndFog, LeavesFalling, RainAndLeaves)")
    parser.add_argument('--weather_condition', type=str, required=False, default=None,
                        help="Single weather condition to apply when weather changes are enabled")
    args = parser.parse_args()

    main(
        algorithm=args.algorithm,
        cnn=args.cnn,
        total_timesteps=args.total_timesteps,
        train_mode=args.train_mode,
        env_select=args.env_select,
        weather_freq=args.weather_freq,
        use_weather_changes=args.weather,
        weather_scenarios=args.weather_scenarios,
        weather_condition=args.weather_condition
    )
