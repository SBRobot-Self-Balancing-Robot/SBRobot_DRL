
import typing as T
import numpy as np
import gymnasium as gym
import time
import mujoco
from mujoco import MjModel, MjData
import os
from scipy.spatial.transform import Rotation as R
from mujoco.viewer import launch_passive


class SelfBalancingRobotEnv(gym.Env):
    
    def __init__(self, environment_path: str = "./models/scene.xml", max_time: float = 10.0, max_pitch: float = 1.0472, frame_skip: int = 10):
        """
        Initialize the SelfBalancingRobot environment.
        
        Args:
            environment_path (str): Path to the MuJoCo model XML file.
        """
        # Initialize the environment
        super().__init__()
        self.viewer = None
        full_path = os.path.abspath(environment_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")
        self.model = MjModel.from_xml_path(full_path)
        self.data = MjData(self.model)
        self.max_time = max_time  # Maximum time for the episode
        self.frame_skip = frame_skip  # Number of frames to skip in each step   

        # Action and observation spaces
        # Observation space: inclination angle, angular velocity, linear position and velocity (could be added: other axis for position and last action)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64)

        # Action space: torque applied to the wheels
        self.action_limit = 10.0
        self.action_space = gym.spaces.Box(low=np.array([-self.action_limit, -self.action_limit]), high=np.array([self.action_limit, self.action_limit]), dtype=np.float64)

        self.max_pitch = max_pitch  # Maximum pitch angle before truncation
        
        # If last action is needed, uncomment the following line
        # self.last_action = [0.0, 0.0]
        

    def step(self, action: T.Tuple[float, float]) -> T.Tuple[np.ndarray, float, bool, dict]:
        """
        Perform a step in the environment.
        
        Args:
            action (np.ndarray or list): The action to be taken, which is a torque applied
            to the wheels of the robot.

        Returns:
            T.Tuple[np.ndarray, float, bool, dict]: A tuple containing:
                - obs (np.ndarray): The observation of the environment (pitch angle, pitch velocity, x position, and x velocity).
                - reward (float): The reward received after taking the action   
                - terminated (bool): Whether the episode has terminated.
                - truncated (bool): Whether the episode has been truncated.
                - info (dict): Additional information about the environment.
        """
        self.data.ctrl[:] = np.clip(action, -self.action_limit, self.action_limit)  # Apply action (torque to the wheels)
        for skip in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)  # Step the simulation
        obs = self._get_obs()
        reward = self._compute_reward()  # Compute the reward based on the pitch angle
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        # info = self._get_info()
        if truncated:
            reward = -100 # Penalize truncation

        return obs, reward, terminated, truncated, {}
        
    def reset(self, seed: T.Optional[int] = None, options: T.Optional[dict] = None) -> np.ndarray:
        """
        Reset the environment to an initial state.
        
        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options for resetting the environment.
            
        Returns:
            T.Dict[str, T.Any]: A dictionary containing the initial observation of the environment (pitch angle, pitch velocity, x position, and x velocity).
        """
        # Seed the random number generator
        if seed is not None:
            np.random.seed(seed)
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)  # Reset the simulation data
        self._initialize_random_state()
        # info = self._get_info()
        obs = self._get_obs()
        return obs, {}
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): The mode in which to render the environment. Default is 'human'.
        """
        if self.viewer is None:
            self.viewer = launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()
            time.sleep(self.model.opt.timestep * self.frame_skip)  # Sleep for the duration of the frame skip
        else:
            raise RuntimeError("Viewer is not running. Please reset the environment or start the viewer.")

    def _compute_reward(self) -> float:
        """
        Compute the reward for the current step.
        
        Args:
            action (T.Tuple[float, float]): The action taken in the environment.
        
        Returns:
            float: The computed reward.
        """
        sensor_data = self.data.sensordata
        torques = self.data.ctrl  # Get the torques applied to the wheels
        
        obs = self._get_obs()
        
        vel = self.data.qvel[0:2]
        pos = self.data.qpos[0:2]
        # Reward is based on the pitch angle: closer to 0 is better
        # The reward is negative to encourage the robot to balance
        # pitch_reward = float((-abs(pitch)**4)*(max_reward/self.max_pitch**4) + max_reward)
        torque_norm = np.linalg.norm(torques)
        
        pitch_component = self._pitch_reward_component(alpha=0.3)  # Pitch reward component
        vel_component = self._velocity_reward_component(alpha=0.1)
        gyro_component = self._gyro_reward_component(alpha=0.05)  # Gyroscope reward component
        wheel_sign, wheel_vel, wheel_delta = self._wheels_reward_component(alpha=0.1)


        reward =  pitch_component + 2 * gyro_component - 0.01 * torque_norm
        return float(reward)
    
    def _pitch_reward_component(self, alpha: float) -> float:
        """"
        Compute the reward based on the pitch angle.
        """
        pitch = self._get_obs()[0]  # Pitch angle from the observation
        return self._kernel(float(pitch), alpha)  # Reward based on the pitch angle using a Gaussian kernel
    
    def _velocity_reward_component(self, alpha: float) -> float:
        """ 
        Compute the reward based on the robot's velocity.
        
        Returns:
            float: The computed reward based on the robot's velocity.
        """
        vel = self.data.qvel[1:3]
        vel_norm = np.linalg.norm(vel)
        velocity_reward = self._kernel(float(vel_norm), alpha)

        return velocity_reward
    
    def _wheels_reward_component(self, alpha: float) -> T.Tuple[float, float, float]:
        """
        Compute the reward based on the wheel torques.
        
        Returns:
            float: The computed reward based on the wheel torques.
        """
        
        wheel_vel = self.data.qvel[6:8]
        # take the sign of the product of the wheel velocities to determine the direction of rotation
        sign_component = np.sign(wheel_vel[0] * wheel_vel[1])  # Reward based on the direction of the wheel velocities
        wheel_delta = abs(wheel_vel[0] - wheel_vel[1])  # Difference between the wheel velocities
        intensity_component = np.linalg.norm(wheel_vel)
        intensity_component = self._kernel(float(intensity_component), alpha)  # Reward based on the intensity of the wheel velocities
        # Reward is based on the absolute value of the wheel torques
        return float(sign_component), float(intensity_component), float(wheel_delta)

    def _gyro_reward_component(self, alpha: float) -> float:
        """
        Compute the reward based on the gyroscope data.
        
        Returns:
            float: The computed reward based on the gyroscope data.
        """
        gyro = self.data.sensordata[4:6] 
        gyro_norm = np.linalg.norm(gyro)
        gyro_reward = self._kernel(float(gyro_norm), float(alpha))  # Reward based on the gyroscope data
        return gyro_reward
    
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
        
    def _get_obs(self) -> np.ndarray:
        """
        Get the current observation of the environment.
        
        Returns:
            np.ndarray: The observation vector containing the pitch angle, pitch velocity, x position, and x velocity.
        """
        quat_xyzw = self.data.qpos[3:7][[1, 2, 3, 0]]  # convert [w, x, y, z] → [x, y, z, w]
        euler = R.from_quat(quat_xyzw).as_euler('xyz')

        pitch = euler[1]
        pitch_vel = self.data.qvel[4] # for the moment it is useless

        gyro = self.data.sensordata[3:6]  # Gyroscope data
        
        left_wheel_torque = self.data.ctrl[0]
        right_wheel_torque = self.data.ctrl[1]
        
        vel = self.data.qvel[0:2]  # x, y velocity

        return np.array([pitch, pitch_vel, gyro[2], left_wheel_torque, right_wheel_torque, vel[0], vel[1]], dtype=np.float64)

    def _get_info(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _is_terminated(self) -> bool:
        """
        Check if the episode is terminated.
        
        Returns:
            bool: True if the episode is terminated, False otherwise.
        """
        # Terminated when the robot falls or reaches the final position (equilibrium)
        return self._is_truncated() or self.data.time >= self.max_time    
        #return self.data.time >= self.max_time
    
    def _is_truncated(self) -> bool:
        # Ottieni orientamento attuale in angoli Euler
        quat_xyzw = self.data.qpos[3:7][[1, 2, 3, 0]]
        euler = R.from_quat(quat_xyzw).as_euler('xyz')

        pitch = euler[1]

        # Truncate if the pitch angle is too high or if the robot is too low
        return bool(abs(pitch) > self.max_pitch)

    def _initialize_random_state(self):
        # Reset position and velocity
        self.data.qpos[:3] = [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.25]  # Initial position (x, y, z)
        self.data.qvel[:] = 0.0  # Initial speed

        # Euler angles: Roll=0, Pitch=random, Yaw=random
        euler = [
            0.0,
            np.random.uniform(-0.6, 0.6),
            np.random.uniform(-np.pi, np.pi)
        ]

        # Euler → Quaternion [x, y, z, w]
        quat_xyzw = R.from_euler('xyz', euler).as_quat()
        self.data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]