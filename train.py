"""
Script di training per il robot auto-bilanciante con Deep Reinforcement Learning.
"""
import os
import json
import time
import argparse
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env

from src.env.self_balancing_robot_env.self_balancing_robot_env import SelfBalancingRobotEnv

def save_configuration(env, policy: str, configuration_path: str = "./configurations/"):
    now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    # Save the configuration
    if not os.path.exists(configuration_path):
        # Create the configuration directory if it doesn't exist
        os.makedirs(os.path.dirname(configuration_path), exist_ok=True)
    # Save the configuration to a file
    with open(f"{configuration_path}/config_{now}.json", 'w') as f:
        config = {
            "max_time": env.max_time,
            "max_pitch": env.max_pitch,
            "frame_skip": env.frame_skip,
            "policy": policy,
            "weights": {
                "upright": env.weight_upright,
                "ang_vel_stability": env.weight_ang_vel_stability,
                "no_linear_movement": env.weight_no_linear_movement,
                "no_yaw_movement": env.weight_no_yaw_movement,
                "control_effort": env.weight_control_effort,
                "action_rate": env.weight_action_rate,
                "fall_penalty": env.weight_fall_penalty
            }
        }
        json.dump(config, f, indent=4)

def parse_arguments():
    """
    Parsing degli argomenti della riga di comando.
    """
    parser = argparse.ArgumentParser(
        description="Training del robot auto-bilanciante con Deep Reinforcement Learning"
    )
    
    parser.add_argument("--policies-folder", type=str, default="./policies",
                       help="Folder where models are saved/loaded (default: ./policies)")
    
    parser.add_argument("--file-base-name", type=str, default="new_reward_",
                       help="Base name for the saved files (default: new_reward_)")

    parser.add_argument("--policy", type=str, default="new_reward_2025-07-14_09-07-58",
                       help="Name of the model file to load")

    parser.add_argument("--iterations", type=int, default=1_000_000,
                       help="Number of training iterations (default: 1000000)")
    
    return parser.parse_args()

def make_env():
    """
    Creates an instance of the SelfBalancingRobotEnv environment.
    """
    def _init():
        environment = gym.make("SelfBalancingRobot-v0")
        environment = Monitor(environment)
        check_env(environment, warn=True)
        return environment
    return _init

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    MODEL = SAC
    POLICIES_FOLDER = args.policies_folder
    FILE_BASE_NAME = args.file_base_name
    MODEL_FILE = args.policy
    ITERATIONS = args.iterations
    
    print("Training configuration:")
    print(f"  - Policies folder: {POLICIES_FOLDER}")
    print(f"  - Base file name: {FILE_BASE_NAME}")
    print(f"  - Model to load: {MODEL_FILE}")
    print(f"  - Iterations: {ITERATIONS}")
    print()

    # Wrapper for the algorithm
    vec_env = SubprocVecEnv([make_env() for _ in range(100)])
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    # Load model if it exists
    try:
        model = MODEL.load(f"{POLICIES_FOLDER}/{MODEL_FILE}", env=vec_env)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No pre-trained model found, starting training from scratch.")
        model = MODEL("MlpPolicy", vec_env, verbose=1)

    model.learn(total_timesteps=ITERATIONS, progress_bar=True)
    model.save(f"{POLICIES_FOLDER}/{FILE_BASE_NAME}{timestamp}")

    # Test
    env = gym.make("SelfBalancingRobot-v0")
    save_configuration(env=env, policy=MODEL_FILE)

    env.reset()
    obs, _ = env.reset()
    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
