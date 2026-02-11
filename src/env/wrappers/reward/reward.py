import numpy as np
import typing as T
import gymnasium as gym
from src.utils.math import signed_sin
from scipy.spatial.transform import Rotation as R
from src.env.robot import SelfBalancingRobotEnv

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

        reward += self.reward_calculator.compute_reward(self.env) # type: ignore

        if terminated and not truncated:
            reward += 50
        elif truncated:
            reward -= 200 * (1 - (self.env.env.data.time / self.env.env.max_time))

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
    HEADING LOCK VERSION: Keeps the robot facing a specific direction (Yaw=0).
    """
    def __init__(self, 
                    heading_weight: float = 1.0, 
                    velocity_weight: float = 0.7, 
                    control_variety_weight: float = 1.0):
        self.heading_weight = heading_weight
        self.velocity_weight = velocity_weight
        self.control_variety_weight = control_variety_weight
    
    def _heading_error(self, env: SelfBalancingRobotEnv) -> float:
        # take the quaternion from the environment and convert it to a rotation matrix
        quat = env.env.data.qpos[3:7]  # Assuming the quaternion is in the order [w, x, y, z]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) # Rearrange to [x, y, z, w]
        rot_matrix = r.as_matrix()
        # The forward direction in the robot's local frame is typically along the x-axis
        forward_vector = rot_matrix[:2, 0]  # Get the first column of the rotation
        # Normalize the forward vector
        norm = np.linalg.norm(forward_vector)
        if norm > 1e-6:
            forward_vector /= norm
        else:
            forward_vector = np.array([0.0, 0.0])
        
        return env.env.pose_control.heading_error(forward_vector)
    
    def _velocity_error(self, env: SelfBalancingRobotEnv) -> float:
        # Compute the velocity error based on the robot's current speed
        current_speed = (env.env.data.qvel[0] + env.env.data.qvel[1]) / 2  # Average of left and right wheel velocities
        return env.env.pose_control.velocity_error(current_speed)
    
    def _pitch(self, env: SelfBalancingRobotEnv) -> float:
        quat = env.env.data.qpos[3:7]  # Assuming the quaternion is in the order [w, x, y, z]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) # Rearrange to [x, y, z, w]
        return r.as_euler('xyz', degrees=False)[1]  # Return the pitch angle in radians
    
    def compute_reward(self, env: SelfBalancingRobotEnv) -> float:
        """
        Compute the reward based on the current state of the environment.
        
        Args:
            env: The environment instance to compute the reward for.
        Returns:
            float: The computed reward.
        """
        heading_error = self._heading_error(env)
        velocity_error = self._velocity_error(env)
        ctrl_variation = env.ctrl_variation
        ctrl = env.ctrl
        pitch = self._pitch(env)
        
        # if the heading error is less than a given threshold, update the heading angle
        if heading_error < 0.1:
            env.env.pose_control.update_heading()
        
        reward = self.heading_weight * abs(heading_error) * env.env.data.time + \
                 self.velocity_weight * (velocity_error) + \
                 self.control_variety_weight * np.linalg.norm(ctrl_variation) + \
                 self._kernel(abs(pitch), alpha=0.01) * self._kernel(np.linalg.norm(ctrl), alpha=0.5)                
        
        
        # reward = -(self._kernel(heading_error, alpha=0.1)+ 
        #            self._kernel(velocity_error, alpha=0.05)+ 
        #            self._kernel(float(np.linalg.norm(ctrl_variation)), alpha=0.25))
        return -reward # type: ignore

    def _kernel(self, x: float, alpha: float) -> float:
        return np.exp(-(x**2)/alpha)