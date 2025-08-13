import typing as T
import gymnasium as gym
import numpy as np
from src.env.self_balancing_robot_env.self_balancing_robot_env import SelfBalancingRobotEnv

class RewardWrapper(gym.Wrapper):
    """
    Wrapper for the SelfBalancingRobotEnv to modify the reward structure.
    """
    def __init__(self, env: SelfBalancingRobotEnv):
        super().__init__(env)
        self.reward_calculator = RewardCalculator()

    def step(self, action):
        """
        Executes one step in the environment with the given action.
        
        Args:
            action: The action to take in the environment.
        Returns:
            obs: The observation after taking the action.
            reward: The modified reward after taking the action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode was truncated.
            info: Additional information from the environment.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        torque_penalty = 0.01 * np.sum(np.square(action))
        reward -= torque_penalty

        if not terminated and not truncated:
            reward += 1.0
        
        reward += self.reward_calculator.compute_reward(self.env)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment.
        
        Args:
            **kwargs: Optional arguments for the reset.
        Returns:
            Initial observation and additional info.
        """
        # reset dello stato interno della reward (anchor, contatori, ecc.)
        self.reward_calculator.reset()
        return self.env.reset(**kwargs)

class RewardCalculator:
    """
    Class to compute the reward for the SelfBalancingRobotEnv.
    """
    def __init__(self, 
                 alpha_yaw_displacement_penalty=0.3, 
                 alpha_pos_displacement_penalty=0.01,
                 alpha_linear_velocity_penalty=0.001, 
                 alpha_torque_penalty=0.025, 
                 weight_fall_penalty=100.0,
                 
                 alpha_yaw_magnitude_penalty=0.3,
                 anchor_lock_steps=20,
                 yaw_settle_thresh=0.08,
                 lin_settle_thresh=0.05,
                 yaw_disp_settle_thresh=0.01,
                 torque_persist_penalty_scale=0.02
                 ):
        self.alpha_yaw_displacement_penalty = alpha_yaw_displacement_penalty
        self.alpha_pos_displacement_penalty = alpha_pos_displacement_penalty
        self.alpha_linear_velocity_penalty = alpha_linear_velocity_penalty
        self.alpha_torque_penalty = alpha_torque_penalty
        self.weight_fall_penalty = weight_fall_penalty

        self.alpha_yaw_magnitude_penalty = alpha_yaw_magnitude_penalty
        self.anchor_lock_steps = anchor_lock_steps
        self.yaw_settle_thresh = yaw_settle_thresh
        self.lin_settle_thresh = lin_settle_thresh
        self.yaw_disp_settle_thresh = yaw_disp_settle_thresh
        self.torque_persist_penalty_scale = torque_persist_penalty_scale

        # internal state for the reward calculator
        self.negative_sign = 0
        self.positive_sign = 0
        self.anchor_position = None
        self.stable_counter = 0

    def compute_reward(self, env) -> float:
        """
        Compute the reward for the current step, focused on self-balancing and staying still.

        Args:
            env: environment instance of SelfBalancingRobotEnv.

        Returns:
            float: computed reward value.
        """
        yaw_displacement = abs(env.yaw - env.last_yaw)
        yaw_displacement_penalty = self._kernel(yaw_displacement, alpha=self.alpha_yaw_displacement_penalty)
        if getattr(env, "count_yaw", None) is None:
            env.count_yaw = 0
        if env.count_yaw == 10:
            env.last_yaw = env.yaw
            env.count_yaw = 0
        env.count_yaw += 1

        yaw_magnitude_penalty = self._kernel(abs(env.yaw), alpha=self.alpha_yaw_magnitude_penalty)

        linear_norm = np.linalg.norm([env.linear_vel[0], env.linear_vel[1]])
        linear_penalty = self._kernel(float(linear_norm), alpha=self.alpha_linear_velocity_penalty)

        if self.anchor_position is None:
            if (abs(env.yaw) < self.yaw_settle_thresh and
                linear_norm < self.lin_settle_thresh and
                yaw_displacement < self.yaw_disp_settle_thresh):
                self.stable_counter += 1
                if self.stable_counter >= self.anchor_lock_steps:
                    self.anchor_position = env.data.qpos[:2].copy()
            else:
                self.stable_counter = 0
            pos_penalty = 1.0
        else:
            pos_error = np.linalg.norm(env.data.qpos[:2] - self.anchor_position)
            pos_penalty = self._kernel(float(pos_error), alpha=self.alpha_pos_displacement_penalty)

        if env.torque_l * env.torque_r > 0:
            if np.sign(env.torque_l) > 0:
                self.positive_sign += 1.0
                self.negative_sign = 0
            else:
                self.negative_sign += 1.0
                self.positive_sign = 0
        else:
            self.positive_sign = 0
            self.negative_sign = 0
        persistence = min(self.negative_sign + self.positive_sign, 50)
        persistence_penalty = self.torque_persist_penalty_scale * persistence

        reward = (yaw_displacement_penalty * yaw_magnitude_penalty) * linear_penalty * pos_penalty
        reward -= persistence_penalty

        if env._is_truncated():
            reward -= (self.weight_fall_penalty + 10 * yaw_displacement)
        elif env._is_terminated():
            terminal_bonus = 500.0 * pos_penalty * linear_penalty * yaw_magnitude_penalty
            reward += terminal_bonus - 3 * (self.positive_sign + self.negative_sign)

        return reward

    def reset(self):
        """
        Reset of the internal state of the reward calculator.
        """
        self.negative_sign = 0
        self.positive_sign = 0
        self.anchor_position = None
        self.stable_counter = 0

    def _kernel(self, x: float, alpha: float) -> float:
        """
        Gaussian kernel function for reward computation.
        Args:
            x (float): The input value.
            alpha (float): The bandwidth parameter for the Gaussian kernel. 
        Returns:
            float: The value of the Gaussian kernel at x.
        """
        return np.exp(-(x**2)/alpha)
