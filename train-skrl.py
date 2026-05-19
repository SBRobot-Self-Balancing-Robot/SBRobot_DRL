"""
Training script for the Self-Balancing Robot environment using skrl 2.0.
Adapted from train.py (Stable Baselines3) to use skrl agents.
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import imageio
import wandb

# skrl 2.0
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.agents.torch.sac import SAC, SAC_CFG
from skrl.agents.torch.td3 import TD3, TD3_CFG
from skrl.agents.torch.ddpg import DDPG, DDPG_CFG
from skrl.utils import set_seed

# SB3 Monitor (gymnasium-compatible wrapper)
from stable_baselines3.common.monitor import Monitor

# Gymnasium vectorized env (replaces SB3's SubprocVecEnv)
from gymnasium.vector import AsyncVectorEnv

# Project
from src.utils.files import backup, compress_and_remove
from src.utils.parser import parse_train_arguments
from src.env.robot import SelfBalancingRobotEnv
from src.env.wrappers.reward.reward import RewardWrapper
from src.env.wrappers.observations import ObservationWrapper

try:
    from tqdm import tqdm as _tqdm
    def tqdm(x, **kwargs):
        return _tqdm(x, **kwargs)
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ─── Neural Network Models ────────────────────────────────────────────────────

class GaussianPolicy(GaussianMixin, Model):
    """Stochastic Gaussian policy for PPO / SAC."""
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False):
        Model.__init__(self,
                       observation_space=observation_space,
                       action_space=action_space,
                       device=device)
        GaussianMixin.__init__(self,
                               clip_actions=clip_actions,
                               clip_log_std=True,
                               min_log_std=-20.0,
                               max_log_std=2.0,
                               reduction="sum")
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, self.num_actions),
        )
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role=""):
        return self.net(inputs["observations"]), {"log_std": self.log_std}


class DeterministicPolicy(DeterministicMixin, Model):
    """Deterministic policy for TD3 / DDPG."""
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False):
        Model.__init__(self,
                       observation_space=observation_space,
                       action_space=action_space,
                       device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, self.num_actions),
            nn.Tanh(),
        )

    def compute(self, inputs, role=""):
        return self.net(inputs["observations"]), {}


class ValueFunction(DeterministicMixin, Model):
    """State-value network V(s) for PPO."""
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False):
        Model.__init__(self,
                       observation_space=observation_space,
                       action_space=action_space,
                       device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role=""):
        return self.net(inputs["observations"]), {}


class QFunction(DeterministicMixin, Model):
    """Q-value network Q(s,a) for SAC / TD3 / DDPG."""
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False):
        Model.__init__(self,
                       observation_space=observation_space,
                       action_space=action_space,
                       device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role=""):
        x = torch.cat([inputs["observations"], inputs["taken_actions"]], dim=-1)
        return self.net(x), {}


# ─── Environment Factory ──────────────────────────────────────────────────────

def make_env():
    def _init():
        env = SelfBalancingRobotEnv()
        env = ObservationWrapper(env)
        env = RewardWrapper(env)  # type: ignore[arg-type]
        env = Monitor(env)
        return env
    return _init


# ─── Agent Builder ────────────────────────────────────────────────────────────

def build_agent(algorithm, env, device, base_save_path,
                n_steps, buffer_size, iterations):
    obs_space = env.observation_space
    act_space = env.action_space
    num_envs  = env.num_envs

    exp_cfg = {
        "directory": base_save_path,
        "experiment_name": "",
        "write_interval": 1000,
        "checkpoint_interval": max(1, iterations // 10),
    }

    if algorithm == "PPO":
        memory = RandomMemory(memory_size=n_steps, num_envs=num_envs, device=device)
        policy = GaussianPolicy(obs_space, act_space, device, clip_actions=True)
        value  = ValueFunction(obs_space, act_space, device)

        cfg = PPO_CFG()
        cfg.rollouts          = n_steps
        cfg.learning_epochs   = 10
        cfg.mini_batches      = 4
        cfg.discount_factor   = 0.99
        cfg.gae_lambda        = 0.95
        cfg.learning_rate     = 3e-4
        cfg.grad_norm_clip    = 0.5
        cfg.ratio_clip        = 0.2
        cfg.value_loss_scale  = 0.5
        cfg.entropy_loss_scale = 0.01
        cfg.experiment        = exp_cfg

        return PPO(models={"policy": policy, "value": value},
                   memory=memory,
                   observation_space=obs_space,
                   action_space=act_space,
                   device=device,
                   cfg=cfg)

    elif algorithm == "SAC":
        memory          = RandomMemory(memory_size=buffer_size, num_envs=num_envs, device=device)
        policy          = GaussianPolicy(obs_space, act_space, device, clip_actions=True)
        critic_1        = QFunction(obs_space, act_space, device)
        critic_2        = QFunction(obs_space, act_space, device)
        target_critic_1 = QFunction(obs_space, act_space, device)
        target_critic_2 = QFunction(obs_space, act_space, device)

        cfg = SAC_CFG()
        cfg.gradient_steps  = 1
        cfg.batch_size      = 256
        cfg.discount_factor = 0.99
        cfg.polyak          = 0.005
        cfg.learning_rate   = 3e-4
        cfg.random_timesteps = 1000
        cfg.learning_starts  = 1000
        cfg.experiment       = exp_cfg

        return SAC(
            models={"policy": policy,
                    "critic_1": critic_1, "critic_2": critic_2,
                    "target_critic_1": target_critic_1, "target_critic_2": target_critic_2},
            memory=memory,
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            cfg=cfg,
        )

    elif algorithm == "TD3":
        memory          = RandomMemory(memory_size=buffer_size, num_envs=num_envs, device=device)
        policy          = DeterministicPolicy(obs_space, act_space, device, clip_actions=True)
        target_policy   = DeterministicPolicy(obs_space, act_space, device, clip_actions=True)
        critic_1        = QFunction(obs_space, act_space, device)
        critic_2        = QFunction(obs_space, act_space, device)
        target_critic_1 = QFunction(obs_space, act_space, device)
        target_critic_2 = QFunction(obs_space, act_space, device)

        cfg = TD3_CFG()
        cfg.gradient_steps  = 1
        cfg.batch_size      = 256
        cfg.discount_factor = 0.99
        cfg.polyak          = 0.005
        cfg.learning_rate   = 3e-4
        cfg.random_timesteps = 1000
        cfg.learning_starts  = 1000
        cfg.experiment       = exp_cfg

        return TD3(
            models={"policy": policy, "target_policy": target_policy,
                    "critic_1": critic_1, "critic_2": critic_2,
                    "target_critic_1": target_critic_1, "target_critic_2": target_critic_2},
            memory=memory,
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            cfg=cfg,
        )

    elif algorithm in ("DDPG", "A2C"):
        if algorithm == "A2C":
            print("Warning: A2C not natively supported; falling back to DDPG.")
        memory        = RandomMemory(memory_size=buffer_size, num_envs=num_envs, device=device)
        policy        = DeterministicPolicy(obs_space, act_space, device, clip_actions=True)
        target_policy = DeterministicPolicy(obs_space, act_space, device, clip_actions=True)
        critic        = QFunction(obs_space, act_space, device)
        target_critic = QFunction(obs_space, act_space, device)

        cfg = DDPG_CFG()
        cfg.gradient_steps  = 1
        cfg.batch_size      = 256
        cfg.discount_factor = 0.99
        cfg.polyak          = 0.005
        cfg.learning_rate   = 3e-4
        cfg.random_timesteps = 1000
        cfg.learning_starts  = 1000
        cfg.experiment       = exp_cfg

        return DDPG(
            models={"policy": policy, "target_policy": target_policy,
                    "critic_1": critic, "target_critic_1": target_critic},
            memory=memory,
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            cfg=cfg,
        )

    raise ValueError(f"Unsupported algorithm: {algorithm}")


# ─── Video Generation ─────────────────────────────────────────────────────────

def generate_video(agent, video_path, device):
    eval_env = make_env()()
    try:
        obs, _ = eval_env.reset()
        frames = []
        done = truncated = False
        agent.enable_models_training_mode()  # disables dropout/batchnorm noise
        while not (done or truncated) and len(frames) < 2000:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _ = agent.act(obs_t, None, timestep=0, timesteps=1)
            obs, _, done, truncated, _ = eval_env.step(action.cpu().numpy().flatten())
            frame = eval_env.render()
            if frame is not None:
                frames.append(frame)
        if frames:
            imageio.mimsave(video_path, frames, fps=30)
            print(f"Video saved to {video_path}")
        else:
            print("No frames captured, video not saved.")
    except Exception as e:
        print(f"Video generation error: {e}")
        import traceback; traceback.print_exc()
    finally:
        agent.enable_training_mode()
        eval_env.close()


# ─── Save Configuration ───────────────────────────────────────────────────────

def save_configuration(xml, algorithm, folder_name, iterations, processes, policies_folder):
    path = os.path.join(policies_folder, folder_name)
    os.makedirs(path, exist_ok=True)

    temp_env = make_env()()
    unwrapped = temp_env.unwrapped
    rw = temp_env
    while hasattr(rw, "env") and not isinstance(rw, RewardWrapper):
        rw = rw.env
    rc = rw.reward_calculator if isinstance(rw, RewardWrapper) else None

    config = {
        "scene": xml,
        "model": algorithm,
        "trainer": "skrl",
        "iterations": iterations,
        "processes": processes,
        "policy": "policy",
        "max_time": getattr(unwrapped, "max_time", None),
        "alpha_values": {
            "heading": rc.heading_weight if rc else None,
            "velocity": rc.velocity_weight if rc else None,
            "control_variety": rc.control_variety_weight if rc else None,
        },
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    temp_env.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_train_arguments()
    XML_FILE        = args.xml_file
    ALGORITHM       = args.model.upper()
    PROCESSES       = args.processes
    ITERATIONS      = args.iterations
    MODEL_FILE      = args.policy
    FILE_PREFIX     = args.file_prefix
    FOLDER_PREFIX   = args.folder_prefix
    POLICIES_FOLDER = args.policies_folder
    DEVICE          = args.device
    N_STEPS         = args.n_steps
    BUFFER_SIZE     = args.buffer_size
    HEADLESS        = args.headless
    POLICY_SCP      = args.policy_scp
    USE_WANDB       = args.wandb

    print("\n--- TRAINING CONFIGURATION (skrl 2.0) ---")
    print(f"* XML File:     {XML_FILE}")
    print(f"* Algorithm:    {ALGORITHM}")
    print(f"* Processes:    {PROCESSES}")
    print(f"* Iterations:   {ITERATIONS}")
    print(f"* Model File:   {MODEL_FILE}")
    print(f"* Device:       {DEVICE}")
    print(f"* N Steps:      {N_STEPS}")
    print(f"* Buffer Size:  {BUFFER_SIZE}")
    print(f"* Headless:     {HEADLESS}")
    print(f"* WandB:        {USE_WANDB}")
    print("-----------------------------------------\n")

    set_seed(42)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    if FILE_PREFIX is None:
        FILE_PREFIX = f"skrl_{ALGORITHM}_{timestamp}"
    if FOLDER_PREFIX is None:
        FOLDER_PREFIX = FILE_PREFIX

    base_save_path  = os.path.join(POLICIES_FOLDER, FOLDER_PREFIX)
    video_path_base = os.path.join(base_save_path, "videos")
    os.makedirs(video_path_base, exist_ok=True)

    if USE_WANDB:
        wandb.init(
            project="self_balancing_robot_skrl",
            config={
                "algorithm": ALGORITHM,
                "iterations": ITERATIONS,
                "processes": PROCESSES,
                "n_steps": N_STEPS,
                "buffer_size": BUFFER_SIZE,
                "xml_file": XML_FILE,
            }
        )

    # ── Vectorized environment ─────────────────────────────────────────────────
    vec_env = AsyncVectorEnv([make_env() for _ in range(PROCESSES)])
    env = wrap_env(vec_env)

    # ── Build agent ────────────────────────────────────────────────────────────
    agent = build_agent(ALGORITHM, env, DEVICE, base_save_path,
                        N_STEPS, BUFFER_SIZE, ITERATIONS)

    if MODEL_FILE:
        ckpt_path = os.path.join(POLICIES_FOLDER, MODEL_FILE, "policy.pt")
        if os.path.exists(ckpt_path):
            agent.load(ckpt_path)
            print(f"Loaded checkpoint from {ckpt_path}")
        else:
            print(f"No checkpoint at {ckpt_path}, starting from scratch.")

    # ── Manual training loop (cycle-based for periodic saves / videos) ─────────
    # The manual loop is used instead of SequentialTrainer so that we can save
    # checkpoints and generate evaluation videos at the end of each cycle
    # without re-initialising the agent (which trainer.train() would do).
    NUM_CYCLES  = 10
    CYCLE_STEPS = ITERATIONS // NUM_CYCLES

    agent.init()
    agent.enable_training_mode()
    observations, _ = env.reset()
    states = observations  # no privileged state in this environment

    episode_rewards: list = []
    episode_lengths: list = []

    for cycle in range(NUM_CYCLES):
        start_t = cycle * CYCLE_STEPS
        end_t   = (cycle + 1) * CYCLE_STEPS
        print(f"\n--- Cycle {cycle + 1}/{NUM_CYCLES}  [{start_t} → {end_t} steps] ---")

        for t in tqdm(range(start_t, end_t), desc=f"Cycle {cycle + 1}"):
            agent.pre_interaction(timestep=t, timesteps=ITERATIONS)

            with torch.no_grad():
                obs_device  = observations.to(DEVICE) if isinstance(observations, torch.Tensor) else torch.tensor(observations, dtype=torch.float32, device=DEVICE)
                states_device = obs_device
                actions, _ = agent.act(obs_device, states_device,
                                       timestep=t, timesteps=ITERATIONS)

            next_observations, rewards, terminated, truncated, infos = env.step(actions)
            next_states = next_observations

            agent.record_transition(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=t,
                timesteps=ITERATIONS,
            )
            agent.post_interaction(timestep=t, timesteps=ITERATIONS)

            # WandB episode logging.
            # gymnasium vector env returns infos as dict-of-arrays;
            # "_episode" is a boolean mask for envs that just finished.
            if USE_WANDB and isinstance(infos, dict) and "episode" in infos:
                mask = infos.get("_episode", np.ones(PROCESSES, dtype=bool))
                for idx, done_ep in enumerate(mask):
                    if done_ep:
                        r = float(infos["episode"]["r"][idx])
                        l = float(infos["episode"]["l"][idx])
                        episode_rewards.append(r)
                        episode_lengths.append(l)
                        wandb.log({
                            "episode_reward": r,
                            "episode_length": l,
                            "avg_reward_100": float(np.mean(episode_rewards[-100:])),
                            "avg_length_100": float(np.mean(episode_lengths[-100:])),
                        })

            observations = next_observations
            states = next_states

        # Checkpoint at end of cycle
        save_path = os.path.join(base_save_path, "policy.pt")
        agent.save(save_path)
        print(f"Model saved → {save_path}")

        # Evaluation video
        if not HEADLESS:
            video_path = os.path.join(video_path_base, f"run_cycle_{cycle + 1}.mp4")
            generate_video(agent, video_path, DEVICE)

    print("\n--- Training Complete ---")

    # ── Post-training ──────────────────────────────────────────────────────────
    save_configuration(XML_FILE, ALGORITHM, FOLDER_PREFIX,
                       ITERATIONS, PROCESSES, POLICIES_FOLDER)

    folder_to_compress = backup(POLICIES_FOLDER, FOLDER_PREFIX, XML_FILE)
    compress_and_remove(folder_to_compress, POLICY_SCP)

    env.close()
    if USE_WANDB:
        wandb.finish()
