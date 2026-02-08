"""
Training script for the Self-Balancing Robot environment using Stable Baselines3.
"""
import os
import csv
import json
import time
import wandb
import shutil
import tarfile
import numpy as np
import typing as T
import gymnasium as gym

from src.utils.files import backup, compress_and_remove
from src.utils.parser import parse_train_arguments, parse_model

from src.env.robot import SelfBalancingRobotEnv
from src.env.wrappers.reward import RewardWrapper
from src.env.wrappers.observations import ObservationWrapper

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC, PPO, TD3, A2C, DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback


def save_configuration(env, xml: str, model: str, folder_name: str, iterations: int, processes: int):
    # Save the configuration
    if folder_name is not None:
        path = f"./policies/{folder_name}"
    else:
        path = f"./policies/{folder_name}"

    if not os.path.exists(path):
        # Create the configuration directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Get the unwrapped environment to access its base attributes
    unwrapped_env = env.unwrapped

    # Find the RewardWrapper to access the RewardCalculator
    rw = env
    while hasattr(rw, "env") and not isinstance(rw, RewardWrapper):
        rw = rw.env
    reward_calc = rw.reward_calculator if isinstance(
        rw, RewardWrapper) else None

    # Save the configuration to a file
    with open(f"{path}/config.json", 'w') as f:
        config = {
            "scene": xml,
            "model": model,
            "iterations": iterations,
            "processes": processes,
            "policy": "policy",
            "max_time": unwrapped_env.max_time,
            "max_pitch": unwrapped_env.max_pitch,
            "frame_skip": unwrapped_env.frame_skip,
            "alpha_values": {
                "alpha_pitch_penalty": reward_calc.alpha_pitch_penalty if reward_calc is not None else None,
                "alpha_yaw_speed_penalty": reward_calc.alpha_yaw_speed_penalty if reward_calc is not None else None,
                "alpha_ctrl_variation_penalty": reward_calc.alpha_ctrl_variation_penalty if reward_calc is not None else None,
                "alpha_x_vel_penalty": reward_calc.alpha_x_vel_penalty if reward_calc is not None else None,
                "alpha_equilibrium": reward_calc.alpha_equilibrium if reward_calc is not None else None,
                "alpha_linear_equilibrium": reward_calc.alpha_linear_equilibrium if reward_calc is not None else None
            }

        }
        json.dump(config, f, indent=4)


def make_env():
    """
    Creates an instance of the SelfBalancingRobotEnv environment wrapped with RewardWrapper.
    """
    def _init():
        environment = SelfBalancingRobotEnv()
        environment = ObservationWrapper(environment)
        environment = RewardWrapper(environment)
        environment = Monitor(environment)
        check_env(environment, warn=True)
        return environment
    return _init

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


def _run(TEST_STEPS, env, writer: csv.writer = None):
    env.reset()
    obs, _ = env.reset()
    for _ in range(TEST_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        try:
            env.render()
        except Exception as e:
            break
        except KeyboardInterrupt:
            compress_and_remove(folder_to_compress, POLICY_SCP)
            break
        if terminated or truncated:
            obs, _ = env.reset()
        if writer is not None:
            writer.writerow(obs.tolist())


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_train_arguments()
    XML_FILE = args.xml_file
    MODEL = parse_model(args.model)
    PROCESSES = args.processes
    ITERATIONS = args.iterations
    MODEL_FILE = args.policy
    FILE_PREFIX = args.file_prefix
    FOLDER_PREFIX = args.folder_prefix
    POLICIES_FOLDER = args.policies_folder
    DEVICE = args.device
    N_STEPS = args.n_steps
    BUFFER_SIZE = args.buffer_size
    REGISTER_DATASET = args.register_dataset
    TEST_STEPS = args.test_steps
    HEADLESS = args.headless
    POLICY_SCP = args.policy_scp

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
            "n_steps": N_STEPS,
            "buffer_size": BUFFER_SIZE,
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
    print(f"  - N Steps (PPO/A2C): {N_STEPS}")
    print(f"  - Buffer Size (SAC/TD3/DDPG): {BUFFER_SIZE}")
    print(f"  - Total batch size: {PROCESSES * N_STEPS}")
    print(f"  - Base file name: {FILE_PREFIX}")
    print(f"  - Policies folder: {POLICIES_FOLDER}")
    print(f"  - Model to load: {MODEL_FILE}")
    print(f"  - Headless mode: {HEADLESS}")
    print(f"  - SCP: {POLICY_SCP}")
    print()

    # Wrapper for the algorithm
    vec_env = SubprocVecEnv([make_env() for _ in range(PROCESSES)])
    # Load model if it exists
    try:
        model = MODEL.load(
            f"{POLICIES_FOLDER}/{MODEL_FILE}/{MODEL_FILE}", env=vec_env)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No pre-trained model found, starting training from scratch.")

    # Create model with appropriate parameters based on algorithm type
    if MODEL in [PPO, A2C]:
        model = MODEL("MlpPolicy", vec_env, n_steps=N_STEPS,
                      device=DEVICE, verbose=1)
    elif MODEL in [SAC, TD3, DDPG]:
        model = MODEL("MlpPolicy", vec_env,
                      buffer_size=BUFFER_SIZE, device=DEVICE, verbose=1)
    else:
        model = MODEL("MlpPolicy", vec_env, device=DEVICE, verbose=1)

    model.learn(total_timesteps=ITERATIONS,
                progress_bar=True, callback=WandbCallback())

    model.save(f"{POLICIES_FOLDER}/{FOLDER_PREFIX}/policy")
    print(f"Model saved to {POLICIES_FOLDER}/{FOLDER_PREFIX}/policy.zip")

    # Test
    env = make_env()()
    save_configuration(
        env=env, xml=XML_FILE, model=MODEL.__name__,
        folder_name=FOLDER_PREFIX, iterations=ITERATIONS, processes=PROCESSES
    )

    folder_to_compress = backup(POLICIES_FOLDER, FOLDER_PREFIX, XML_FILE)

    # Start testing
    if REGISTER_DATASET and not HEADLESS:
        with open(f"{POLICIES_FOLDER}/{FOLDER_PREFIX}/dataset.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Pitch", "Y angular Velocity Y", "Accel X", "Accel Z", "Left Wheel Velocity",
                            "Right Wheel Velocity", "Yaw angular Velocity Z", "Left Motor Command", "Right Motor Command"])
            _run(TEST_STEPS, env, writer)
    elif not REGISTER_DATASET and not HEADLESS:
        _run(TEST_STEPS, env)

    compress_and_remove(folder_to_compress, POLICY_SCP)
