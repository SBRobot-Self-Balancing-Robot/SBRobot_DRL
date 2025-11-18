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

        if not terminated and not truncated:
            reward += 0.1
        else:
            reward -= 1.0
        
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
        return self.env.reset(**kwargs)

class RewardCalculator:
    """
    Class to compute the reward for the SelfBalancingRobotEnv.
    """
    def __init__(self, alpha_pitch_penalty = 0.003, alpha_setpoint_angle_penalty = 0.001):
        self.alpha_pitch_penalty = alpha_pitch_penalty
        self.alpha_setpoint_angle_penalty = alpha_setpoint_angle_penalty

    def compute_reward(self, env) -> float:
        """
        Compute the reward for the current step using the current env API.

        Usa:
          - env.pitch
          - env.angular_velocity (array)
          - env.right_wheel_velocity, env.left_wheel_velocity
          - env.data.qpos[:2] per la posizione (x,y)
        """
        # Extract necessary state variables from the environment
        # Pitch angle
        quaternion_angles = env.data.qpos[3:7]  # quaternion [w, x, y, z]
        r = env.R.from_quat([quaternion_angles[1], quaternion_angles[2], quaternion_angles[3], quaternion_angles[0]]) # Rearrange to [x, y, z, w]
        roll, pitch, yaw = r.as_euler('xyz', degrees=False) # in radians

        # Wheel velocities
        left_wheel_vel  = env.data.qvel[7]
        right_wheel_vel = env.data.qvel[8]

        # Setpoint speed error
        left_setpoint_speed_error  = left_wheel_vel  - env.speed_setpoints[0]
        right_setpoint_speed_error = right_wheel_vel - env.speed_setpoints[1]


        # Reward composition
        reward = 0.9 * (
            0.3 * self._kernel(pitch, self.alpha_pitch_penalty) +
            0.35 * self._kernel(left_setpoint_speed_error, self.alpha_setpoint_angle_penalty) +
            0.35 * self._kernel(right_setpoint_speed_error, self.alpha_setpoint_angle_penalty)
        )

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