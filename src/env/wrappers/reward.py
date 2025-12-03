import numpy as np
import typing as T
import gymnasium as gym
from scipy.spatial.transform import Rotation as R
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

        reward += self.reward_calculator.compute_reward(self.env)

        if terminated and not truncated:
            reward += 10 
        elif truncated:
            reward -= 100 * (1 - (self.env.env.data.time / self.env.env.max_time))

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment.
        
        Args:
            **kwargs: Optional arguments for the reset.
        Returns:
            Initial observation and additional info.
        """
        return self.env.reset(**kwargs)

class RewardCalculator:
    """
    Class to compute the reward for the SelfBalancingRobotEnv.
    """
    def __init__(self, alpha_pitch_penalty = 0.005, 
                 alpha_yaw_speed_penalty = 0.0001, 
                 alpha_x_vel_penalty = 0.00001,
                 alpha_equilibrium = 0.03,
                 alpha_ctrl_variation_penalty = 0.15,
                 alpha_ctrl_intensity_penalty = 0.01,
                 alpha_yaw_equilibrium = 1.0,
                 alpha_linear_equilibrium = 1.5):
        self.alpha_pitch_penalty = alpha_pitch_penalty
        self.alpha_yaw_speed_penalty = alpha_yaw_speed_penalty
        self.alpha_ctrl_variation_penalty = alpha_ctrl_variation_penalty
        self.alpha_x_vel_penalty = alpha_x_vel_penalty
        self.alpha_equilibrium = alpha_equilibrium
        self.alpha_ctrl_intensity_penalty = alpha_ctrl_intensity_penalty
        self.alpha_yaw_equilibrium = alpha_yaw_equilibrium
        self.alpha_linear_equilibrium = alpha_linear_equilibrium

        self.past_ctrl = np.array([0.0, 0.0])

    def compute_reward(self, env) -> float:
        """
        Compute the reward for the current step using the current env API.
        
        Reward = 1.0 (alive bonus) - |x_vel| - 0.1 * |w_yaw|
        """
        # Angular velocities
        w_roll, w_pitch, w_yaw = env.real_angular_velocity  # in radians/sec

        # Linear velocities
        x_vel = env.x_vel  # in m/s

        # Position
        x_pos = env.env.data.qpos[0]

        # Improved reward: 
        # - Alive bonus: 1.0
        # - Velocity penalty: |x_vel| (stay still)
        # - Yaw rate penalty: 0.1 * |w_yaw| (stay straight)
        # - Position penalty: 0.1 * |x_pos| (prevent drift)
        # - Pitch rate penalty: alpha * |w_pitch| (dampen oscillations)
        # - Control variation penalty: alpha * |ctrl - past_ctrl|^2 * kernel(pitch) (smooth movement in equilibrium)
        # - Yaw equilibrium penalty: alpha * |w_yaw| * kernel(pitch) (prevent rotation in equilibrium)
        # - Linear equilibrium penalty: alpha * |x_vel| * kernel(pitch) (prevent linear drift in equilibrium)
                
        current_ctrl = env.env.data.ctrl.copy()
        ctrl_variation = np.sum(np.square(current_ctrl - self.past_ctrl))
        
        # Scale penalty by how close we are to equilibrium (pitch = 0)
        equilibrium_factor = self._kernel(env.pitch, self.alpha_equilibrium)

        # To avoid the robot to cheat, causing a bang-bang control, we add a penalty for the control variation
        # this penalty is never null, even if the equilibrium_factor is null
        base_penalty = 0.01 * ctrl_variation
        stability_penalty = (self.alpha_ctrl_variation_penalty - 0.01) * ctrl_variation * equilibrium_factor
        total_ctrl_penalty = base_penalty + stability_penalty

        reward = 1.0 - abs(x_vel) - 0.1 * abs(w_yaw) - 0.75 * abs(x_pos) * equilibrium_factor - \
        self.alpha_pitch_penalty * abs(w_pitch) - \
        total_ctrl_penalty - \
        self.alpha_yaw_equilibrium * abs(w_yaw) * equilibrium_factor - \
        self.alpha_linear_equilibrium * abs(x_vel) * equilibrium_factor - \
        self.alpha_ctrl_intensity_penalty * abs(np.linalg.norm(current_ctrl)) * equilibrium_factor

        self.past_ctrl = current_ctrl

        return float(reward)

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