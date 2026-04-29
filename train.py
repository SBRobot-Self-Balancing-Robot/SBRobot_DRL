"""
Training script for the Self-Balancing Robot environment using Stable Baselines3.
Adapted to include iterative logging, plotting, and video generation.
"""
import os
import json
import time
import wandb
import numpy as np
import typing as T
import pandas as pd
import matplotlib.pyplot as plt
import imageio

from src.utils.files import backup, compress_and_remove
from src.utils.parser import parse_train_arguments, parse_model

from src.env.robot import SelfBalancingRobotEnv
from src.env.wrappers.reward.reward import RewardWrapper
from src.env.wrappers.observations import ObservationWrapper

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC, PPO, TD3, A2C, DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# --- Utility Functions for Plotting (Replica of your reference utils) ---
def plot_data_line(dfs, xaxis, value, condition, smooth, title, output):
    """Simple matplotlib implementation to replace external plotter."""
    plt.figure(figsize=(10, 6))
    for df in dfs:
        if value in df.columns:
            # Simple moving average for smoothing
            data = df[value].rolling(window=smooth, min_periods=1).mean()
            plt.plot(df[xaxis], data, label=condition)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(value)
    plt.grid(True)
    plt.savefig(output)
    plt.close()

def plot_reward_components(df, output):
    """Simple stacked plot for components."""
    # Assuming columns other than 'Training Steps', 'Condition', 'Reward', 'Std' are components
    excluded = ['Training Steps', 'Condition', 'Reward', 'Std']
    components = [c for c in df.columns if c not in excluded]
    
    if not components:
        return

    plt.figure(figsize=(10, 6))
    plt.stackplot(df['Training Steps'], [df[c] for c in components], labels=components)
    plt.legend(loc='upper left')
    plt.title("Reward Components")
    plt.xlabel("Steps")
    plt.savefig(output.replace('.html', '.png')) # Saving as png for simplicity
    plt.close()

# --- Custom Callback for Data Collection ---
class RewardCollectorCallback(BaseCallback):
    """
    Callback meant to replicate the data structure of the reference code.
    Collects rewards and potential components from 'infos'.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.data = {
            'rewards': [],
            'components': [], # List of dicts
            'std': []
        }
        self.column_order = []

    def _on_step(self) -> bool:
        # Get the rewards for the current step (vectorized env support)
        rewards = self.locals['rewards']
        infos = self.locals['infos']
        
        # We take the mean reward across processes for logging purposes
        avg_reward = np.mean(rewards)
        self.data['rewards'].append(avg_reward)
        self.data['std'].append(np.std(rewards))

        # Check for reward components in infos
        # Assuming the environment puts a 'reward_components' dict in info
        comp_data = {}
        if len(infos) > 0 and 'reward_components' in infos[0]:
            # Aggregate components across envs if needed, here taking first env for simplicity
            comp_data = infos[0]['reward_components']
        
        self.data['components'].append(comp_data)
        
        # Update column order if new keys appear
        for k in comp_data.keys():
            if k not in self.column_order:
                self.column_order.append(k)

        return True

# --- WandB Callback (Preserved) ---
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

def save_configuration(env, xml: str, model: str, folder_name: str, iterations: int, processes: int):
    path = f"./policies/{folder_name}"
    os.makedirs(path, exist_ok=True)

    unwrapped_env = env.unwrapped

    # Walk the wrapper chain to find RewardWrapper
    rw = env
    while hasattr(rw, "env") and not isinstance(rw, RewardWrapper):
        rw = rw.env
    rc = rw.reward_calculator if isinstance(rw, RewardWrapper) else None

    with open(f"{path}/config.json", 'w') as f:
        config = {
            "scene": xml,
            "model": model,
            "iterations": iterations,
            "processes": processes,
            "policy": "policy",
            "max_time": getattr(unwrapped_env, "max_time", None),
            "reward_weights": {
                "w_balance":  rc.w_balance  if rc else None,
                "w_velocity": rc.w_velocity if rc else None,
                "w_heading":  rc.w_heading  if rc else None,
                "w_smooth":   rc.w_smooth   if rc else None,
            },
            "reward_sigmas": {
                "sigma_pitch": rc.sigma_pitch if rc else None,
                "sigma_vel":   rc.sigma_vel   if rc else None,
            },
        }
        json.dump(config, f, indent=4)

def make_env(render_mode=None, initial_curriculum_phase: int = 0):
    """
    Creates an instance of the SelfBalancingRobotEnv environment wrapped with RewardWrapper.
    Pass initial_curriculum_phase>0 to resume training from a mid-curriculum checkpoint
    without repeating the early phases.
    """
    def _init():
        environment = SelfBalancingRobotEnv(
            initial_curriculum_phase=initial_curriculum_phase,
            render_mode=render_mode,
            randomization_scale=1.0,
        )
        environment = ObservationWrapper(environment)
        environment = RewardWrapper(environment)
        environment = Monitor(environment)
        check_env(environment, warn=True)
        return environment
    return _init

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
    WANDB = args.wandb
    CURRICULUM_PHASE = args.curriculum_phase

    print("\n--- TRAINING CONFIGURATION ---")
    print(f"* XML File: {XML_FILE}")
    print(f"* Model: {MODEL.__name__}")
    print(f"* Processes: {PROCESSES}")
    print(f"* Iterations: {ITERATIONS}")
    print(f"* Model File: {MODEL_FILE}")
    print(f"* File Prefix: {FILE_PREFIX}")
    print(f"* Folder Prefix: {FOLDER_PREFIX}")
    print(f"* Policies Folder: {POLICIES_FOLDER}")
    print(f"* Device: {DEVICE}")
    print(f"* N Steps: {N_STEPS}")
    print(f"* Buffer Size: {BUFFER_SIZE}")
    print(f"* Register Dataset: {REGISTER_DATASET}")
    print(f"* Test Steps: {TEST_STEPS}")
    print(f"* Headless: {HEADLESS}")
    print(f"* Policy SCP: {POLICY_SCP}")
    print(f"* WandB: {WANDB}")
    print(f"* Curriculum Phase: {CURRICULUM_PHASE}")
    print(f"* Entropy Coef: {args.ent_coef}")
    print("-----------------------------\n")
    
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    if FILE_PREFIX is None:
        FILE_PREFIX = f"{MODEL.__name__}_{timestamp}"
    if FOLDER_PREFIX is None:
        FOLDER_PREFIX = FILE_PREFIX
    
    # Setup Paths
    base_save_path = os.path.join(POLICIES_FOLDER, FOLDER_PREFIX)
    os.makedirs(base_save_path, exist_ok=True)
    video_path_base = os.path.join(base_save_path, "videos")
    plot_path_base = os.path.join(base_save_path, "plots")
    os.makedirs(video_path_base, exist_ok=True)
    os.makedirs(plot_path_base, exist_ok=True)

    if WANDB:
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
    
    # Wrapper for the algorithm
    vec_env = SubprocVecEnv([make_env(initial_curriculum_phase=CURRICULUM_PHASE) for _ in range(PROCESSES)])

    # Load model if it exists — the model is always saved as "policy.zip" inside the folder
    model_load_path = f"{POLICIES_FOLDER}/{MODEL_FILE}/policy"
    start_fresh = True
    
    # Create model structure first
    model_kwargs = {"device": DEVICE, "verbose": 1, "learning_rate": args.learning_rate}
    if MODEL in [PPO, A2C]:
        model_kwargs["n_steps"] = N_STEPS
        model_kwargs["ent_coef"] = args.ent_coef
        model_kwargs["gamma"] = args.gamma
        policy_type = "MlpPolicy"
    elif MODEL in [SAC, TD3, DDPG]:
        model_kwargs["buffer_size"] = BUFFER_SIZE
        policy_type = "MlpPolicy"
    else:
        policy_type = "MlpPolicy"

    try:
        if os.path.exists(model_load_path + ".zip"):
            model = MODEL.load(model_load_path, env=vec_env, **model_kwargs)
            print(f"Model loaded successfully from {model_load_path}")
            start_fresh = False
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("No pre-trained model found, starting training from scratch.")
        model = MODEL(policy_type, vec_env, **model_kwargs)

    # --- Setup Callbacks ---
    reward_callback = RewardCollectorCallback()
    callbacks = [reward_callback]
    if WANDB:
        callbacks.append(WandbCallback())
    
    combined_callback = CallbackList(callbacks)

    # --- Training Loop Configuration ---
    # Split the overall training into multiple cycles, each of a given time duration
    CYCLE_DURATION = 180 # seconds for the iteration
    NUM_CYCLES = 10
    TOTAL_TIMESTEPS_PER_CYCLE = ITERATIONS // NUM_CYCLES
    
    print(f"Starting training loop: {NUM_CYCLES} cycles of {TOTAL_TIMESTEPS_PER_CYCLE} steps each.")

    for i in range(NUM_CYCLES):
        current_iter_str = f"{i+1}/{NUM_CYCLES}"
        print(f"\n--- Training Iteration {current_iter_str} ---")

        # 1. Train
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS_PER_CYCLE,
            callback=combined_callback,
            reset_num_timesteps=False, # Important for continuity
            progress_bar=True
        )

        # 2. Save Model
        current_model_path = os.path.join(base_save_path, "policy")
        model.save(current_model_path)
        print(f"Model saved to {current_model_path}")

        # 3. Process and Save Reward Data
        print("Processing and saving reward data...")
        if not reward_callback.data['rewards']:
            print("Warning: No reward data collected.")
        else:
            steps_collected = len(reward_callback.data['rewards'])
            # Create Dataframe
            df_rewards = pd.DataFrame({
                'Training Steps': range(steps_collected),
                'Reward': reward_callback.data['rewards'],
                'Std': reward_callback.data['std'],
                'Condition': 'Training'
            })
            
            # Add components if they exist
            if reward_callback.data['components'] and any(reward_callback.data['components']):
                 df_components = pd.DataFrame(reward_callback.data['components'])
                 full_data = pd.concat([df_rewards, df_components], axis=1)
            else:
                full_data = df_rewards

            # Save CSV
            csv_path = os.path.join(base_save_path, 'rewards.csv')
            full_data.to_csv(csv_path, index=False)

        # 4. Evaluate and Save Video
        print(f"Evaluating and saving video for iteration {i+1}...")
        eval_video_filename = os.path.join(video_path_base, f'run_iter_{i+1}.mp4')
        
        # Create a specific environment for evaluation with render mode enabled
        # Note: We create a single env, not vectorized, for video recording
        eval_env = make_env(render_mode="rgb_array")()
        
        try:
            obs, _ = eval_env.reset()
            done = False
            truncated = False
            frames = []

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)

                frame = eval_env.render()
                if frame is not None:
                    frames.append(frame)

                done = terminated

                if len(frames) > 2000:
                    break

            if frames:
                imageio.mimsave(eval_video_filename, frames, fps=30)
                print(f"Video saved to {eval_video_filename}")
            else:
                print("No frames captured, video not saved.")

        except Exception as e:
            print(f"Error during video evaluation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if 'eval_env' in locals():
                eval_env.close()
            time.sleep(1)
            del eval_env

    print("\n--- Training Loop Complete ---")

    # Final Save Configuration
    temp_env = make_env()()
    save_configuration(
        env=temp_env, xml=XML_FILE, model=MODEL.__name__,
        folder_name=FOLDER_PREFIX, iterations=ITERATIONS, processes=PROCESSES
    )
    temp_env.close()

    # Create backup/zip
    folder_to_compress = backup(POLICIES_FOLDER, FOLDER_PREFIX, XML_FILE)
    compress_and_remove(folder_to_compress, POLICY_SCP)
    
    vec_env.close()