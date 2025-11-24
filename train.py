"""
Training script for the Self-Balancing Robot environment using Stable Baselines3.
"""
import os
import json
import time
import wandb
import argparse
import typing as T
import gymnasium as gym
import numpy as np
from src.env.wrappers.reward import RewardWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC, PPO, TD3, A2C, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

from src.env.self_balancing_robot_env.self_balancing_robot_env import SelfBalancingRobotEnv
from src.env.wrappers.reward import RewardWrapper

def save_configuration(env, xml: str, model: str, filename: str, folder_name: str, iterations: int, processes: int):
    # Save the configuration
    if folder_name is not None:
        path = f"./policies/{folder_name}"
    else:
        path = f"./policies/{filename}"
    
    if not os.path.exists(path):
        # Create the configuration directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
    # Get the unwrapped environment to access its base attributes
    unwrapped_env = env.unwrapped

    # Find the RewardWrapper to access the RewardCalculator
    rw = env
    while hasattr(rw, "env") and not isinstance(rw, RewardWrapper):
        rw = rw.env
    reward_calc = rw.reward_calculator if isinstance(rw, RewardWrapper) else None

    # Save the configuration to a file
    with open(f"{path}/{filename}.json", 'w') as f:
        config = {
            "scene": xml,
            "model": model,
            "iterations": iterations,
            "processes": processes,
            "policy": filename,
            "max_time": unwrapped_env.max_time,
            "max_pitch": unwrapped_env.max_pitch,
            "frame_skip": unwrapped_env.frame_skip,
            "alpha_values": {
                "alpha_pitch_penalty": reward_calc.alpha_pitch_penalty if reward_calc is not None else None,
                "alpha_setpoint_angle_penalty": reward_calc.alpha_setpoint_angle_penalty if reward_calc is not None else None
            }

        }
        json.dump(config, f, indent=4)

def parse_arguments():
    """
    Command line argument parser for training the Self-Balancing Robot environment.
    """
    parser = argparse.ArgumentParser(
        description="Training the self-balancing robot with Deep Reinforcement Learning"
    )

    parser.add_argument("--xml-file", type=str, default="./models/scene.xml",
                       help="Path to the XML file defining the robot and environment (default: ./models/scene.xml)")
    
    parser.add_argument("--model", type=str, default="PPO",
                       help="Model to use for training (default: PPO). Other options: PPO, TD3, A2C, DDPG")
    
    parser.add_argument("--policies-folder", type=str, default="./policies",
                       help="Folder where models are saved/loaded (default: ./policies)")
    
    parser.add_argument("--file-prefix", type=str, default=None,
                       help="Base name for the saved files")
    
    parser.add_argument("--folder-prefix", type=str, default=None,
                       help="Base folder name for the saved files (default: None, uses timestamp)")

    parser.add_argument("--policy", type=str, default=None,
                       help="Name of the model file to load")

    parser.add_argument("--iterations", type=int, default=1_000_000,
                       help="Number of training iterations (default: 1000000)")
    
    parser.add_argument("--processes", type=int, default=50,
                       help="Number of parallel processes for training (default: 50)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use for training (default: cpu). Other options: cpu, cuda, mps")
    
    return parser.parse_args()

def make_env():
    """
    Creates an instance of the SelfBalancingRobotEnv environment wrapped with RewardWrapper.
    """
    def _init():
        environment = SelfBalancingRobotEnv()  # Usa direttamente SelfBalancingRobotEnv
        environment = RewardWrapper(environment)  # Applica il RewardWrapper
        environment = Monitor(environment)
        check_env(environment, warn=True)
        return environment
    return _init

def _parse_model(model_name: str):
    """
    Parses the model name to return the corresponding Stable Baselines3 model class.
    """
    models = {
        "SAC": SAC,
        "PPO": PPO,
        "TD3": TD3,
        "A2C": A2C,
        "DDPG": DDPG
    }
    if model_name in models:
        return models[model_name]
    else:
        return models["SAC"]  # Default to SAC if the model is not recognized

# Log training progress to wandb
class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                r = info["episode"]["r"]
                l = info["episode"]["l"]
                self.episode_rewards.append(r)
                self.episode_lengths.append(l)
                wandb.log({
                    "episode_reward": r,
                    "episode_length": l,
                    "avg_reward_100": float(np.mean(self.episode_rewards[-100:])),
                    "avg_length_100": float(np.mean(self.episode_lengths[-100:])),
                })
        return True

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    XML_FILE = args.xml_file
    MODEL = _parse_model(args.model)
    PROCESSES = args.processes
    ITERATIONS = args.iterations
    MODEL_FILE = args.policy
    FILE_PREFIX = args.file_prefix
    FOLDER_PREFIX = args.folder_prefix
    POLICIES_FOLDER = args.policies_folder
    DEVICE = args.device

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    if FILE_PREFIX is None:
        FILE_PREFIX = f"{MODEL.__name__}_{timestamp}"
    if FOLDER_PREFIX is None:
        FOLDER_PREFIX = FILE_PREFIX
    wandb.init(
        project="self_balancing_robot",
        config={
            "model": args.model,
            "iterations": ITERATIONS,
            "processes": PROCESSES,
            "xml_file": XML_FILE,
            "policies_folder": POLICIES_FOLDER,
            "file_prefix": FILE_PREFIX,
            "folder_prefix": FOLDER_PREFIX
        }
    )
    print("Training configuration:")
    print(f"  - Model: {MODEL.__name__}")
    print(f"  - Processes: {PROCESSES}")
    print(f"  - Iterations: {ITERATIONS}")
    print(f"  - Base file name: {FILE_PREFIX}")
    print(f"  - Policies folder: {POLICIES_FOLDER}")
    print(f"  - Model to load: {MODEL_FILE}")
    print()

    # Wrapper for the algorithm
    vec_env = SubprocVecEnv([make_env() for _ in range(PROCESSES)])
    # Load model if it exists
    try:
        model = MODEL.load(f"{POLICIES_FOLDER}/{MODEL_FILE}/{MODEL_FILE}", env=vec_env)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No pre-trained model found, starting training from scratch.")
        model = MODEL("MlpPolicy", vec_env, device=DEVICE, verbose=1)

    model.learn(total_timesteps=ITERATIONS, progress_bar=True, callback=WandbCallback())

    model.save(f"{POLICIES_FOLDER}/{FOLDER_PREFIX}/{FILE_PREFIX}")
    print(f"Model saved to {POLICIES_FOLDER}/{FOLDER_PREFIX}/{FILE_PREFIX}")
    # Test
    env = make_env()()
    save_configuration(
        env=env, xml=XML_FILE, model=MODEL.__name__, filename=FILE_PREFIX,
        folder_name=FOLDER_PREFIX, iterations=ITERATIONS, processes=PROCESSES
    )

    env.reset()
    obs, _ = env.reset()
    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
            