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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC, PPO, TD3, A2C, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env

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
        
    # Get the unwrapped environment to access its attributes
    unwrapped_env = env.unwrapped
    
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
            "weights": {
                "upright": unwrapped_env.weight_fall_penalty,
            },
            "alpha_values": {
                "yaw_displacement_penalty": unwrapped_env.alpha_yaw_displacement_penalty,
                "pos_displacement_penalty": unwrapped_env.alpha_pos_displacement_penalty,
                "linear_velocity_penalty": unwrapped_env.alpha_linear_velocity_penalty,
                "torque_penalty": unwrapped_env.alpha_torque_penalty
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
    print(f"  - Model: {MODEL}")
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
        model = MODEL.load(f"{POLICIES_FOLDER}/{MODEL_FILE}", env=vec_env)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No pre-trained model found, starting training from scratch.")
        model = MODEL("MlpPolicy", vec_env, verbose=1)

    # Log training progress to wandb
    class WandbCallback:
        def __init__(self):
            self.episode_rewards = []
            self.episode_lengths = []
            self.episode_start_time = time.time()
            self.current_episode_reward = 0

        def __call__(self, locals_, globals_):
            # Aggiorna la reward corrente
            if "reward" in locals_ and locals_["reward"] is not None:
                self.current_episode_reward += locals_["reward"]

            # Quando l'episodio termina, calcola le statistiche
            if "done" in locals_ and locals_["done"]:
                episode_length = time.time() - self.episode_start_time
                self.episode_lengths.append(episode_length)
                self.episode_rewards.append(self.current_episode_reward)

                # Logga reward media e durata media
                avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                avg_length = sum(self.episode_lengths) / len(self.episode_lengths)
                wandb.log({
                    "avg_reward": avg_reward,
                    "avg_episode_length": avg_length,
                    "episode_reward": self.current_episode_reward,
                    "episode_length": episode_length
                })

                # Resetta per il prossimo episodio
                self.episode_start_time = time.time()
                self.current_episode_reward = 0

            return True

    model.learn(total_timesteps=ITERATIONS, progress_bar=True, callback=WandbCallback())

    model.save(f"{POLICIES_FOLDER}/{FOLDER_PREFIX}/{FILE_PREFIX}")

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