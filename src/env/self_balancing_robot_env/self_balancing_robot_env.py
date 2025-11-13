"""
Environment for a self-balancing robot using MuJoCo.
"""
import os
import time
import mujoco
import numpy as np
import typing as T
import gymnasium as gym
from mujoco import MjModel, MjData
from mujoco.viewer import launch_passive
from scipy.spatial.transform import Rotation as R


class SelfBalancingRobotEnv(gym.Env):
    
    def __init__(self, environment_path: str = "./models/scene.xml", max_time: float = 10.0, max_pitch: float = 0.8, frame_skip: int = 10):
        """
        Initialize the SelfBalancingRobot environment.
        
        Args:
            environment_path (str): Path to the MuJoCo model XML file.
            max_time (float): Maximum time for the episode.
            max_pitch (float): Maximum pitch angle before truncation.
            frame_skip (int): Number of frames to skip in each step.
        """
        # Initialize the environment
        super().__init__()
        self.viewer = None
        full_path = os.path.abspath(environment_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")
        self.model = MjModel.from_xml_path(full_path)
        self.data = MjData(self.model)
        self.max_time = max_time # Maximum time for the episode
        self.frame_skip = frame_skip # Number of frames to skip in each step   

        # Observation space: pitch, roll, yaw, angle_acceleration_x, angle_acceleration_y, linear_velocity_x, linear_velocity_y, pos_x, pos_y
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float64)
        # Action space: torque applied to the wheels
        self.action_limit = 0.135  # Maximum torque that can be applied to the wheels [N·m]
        self.action_space = gym.spaces.Box(low=np.array([-self.action_limit, -self.action_limit]), high=np.array([self.action_limit, self.action_limit]), dtype=np.float64)
        
        # Initialize the environment attributes
        self.weight_fall_penalty = 100.0 # Penalty for falling
        self.max_pitch = max_pitch # Maximum pitch angle before truncation
        
        # Initialize observation values
        self.roll, self.pitch, self.yaw = 0.0, 0.0, 0.0 # Orientation angles of the robot [roll, pitch, yaw]
        self.angle_acceleration = np.zeros(3) # Angular acceleration of the robot [gyro_x, gyro_y, gyro_z]
        self.linear_velocity = np.zeros(3) # Linear velocity of the robot [linear_velocity_x, linear_velocity_y, linear_velocity_z]
        self.wheels_velocity = np.zeros(2) # Angular velocity of the wheels [wheel_left_velocity, wheel_right_velocity]


    def step(self, action: T.Tuple[float, float]) -> T.Tuple[np.ndarray, float, bool, bool, dict]:
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
        
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)  # Step the simulation
        
        obs = self._get_obs()
        
        terminated = self._is_terminated()
        
        truncated = self._is_truncated()
        
        return obs, 0.0, terminated, truncated, {}

    def reset(self, seed: T.Optional[int] = None, options: T.Optional[dict] = None) -> T.Tuple[np.ndarray, dict]:
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
        self._last_action = np.zeros(2) # Resetta anche l'ultima azione
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

    def _get_body_angular_acceleration(self) -> np.ndarray:
        """
        Get the angular velocities of the robot's body.
        
        Returns:
            np.ndarray: The angular velocities of the robot's body in the order [gyro_x, gyro_y, gyro_z].
        """
        return self.data.qvel[3:6]

    def _get_robot_linear_velocity(self) -> np.ndarray:
        """
        Get the linear velocity of the robot.
        
        Returns:
            np.ndarray: The linear velocity of the robot in the order [linear_velocity_x, linear_velocity_y, linear_velocity_z].
        """
        return self.data.qvel[0:3]
    
    def _get_body_orientation_angles(self) -> T.Tuple[float, float, float]:
        """
        Get the orientation angles of the robot's body in Euler angles (roll, pitch, yaw).
        
        Returns:
            T.Tuple[float, float, float]: The roll, pitch, and yaw angles of the robot's body.
        """
        # Usare questi parametri per stimare l'orientamento
        self.angle_acceleration = self._get_body_angular_acceleration() # [gyro_x, gyro_y, gyro_z]
        self.linear_velocity = self._get_robot_linear_velocity() # [vel_x, vel_y, vel_z]

        # DA IMPLEMENTARE CORRETTAMENTE
        return 0, 0, 0
    
    def _get_wheels_angular_velocity(self) -> T.Tuple[float, float]:
        """
        Get the angular velocities of the robot's wheels.
        
        Returns:
            T.Tuple[float, float]: The angular velocities of the left and right wheels.
        """

        # DA IMPLEMENTARE CORRETTAMENTE
        return 0, 0

    def _get_obs(self) -> np.ndarray:
        """
        Get the current observation of the environment.
        
        Returns:
            np.ndarray: The observation vector.
            
            Ho mantenuto l'output di 7 elementi, che ora include:
            [pitch, roll, yaw, angle_acceleration_x, angle_acceleration_y, linear_velocity_x, linear_velocity_y]
        """
        self.roll, self.pitch, self.yaw = self._get_body_orientation_angles()
        self.right_wheel_velocity, self.left_wheel_velocity = self._get_wheels_angular_velocity()
        
        return np.array([
            self.pitch,                    
            self.yaw,
            self.right_wheel_velocity,
            self.left_wheel_velocity         
        ], dtype=np.float64)
    
    def _get_info(self):
        return {}

    def _is_terminated(self) -> bool:
        """
        Check if the episode is terminated.
        
        Returns:
            bool: True if the episode is terminated, False otherwise.
        """
        # Terminated when the robot falls or the maximum time is reached
        return self._is_truncated() or self.data.time >= self.max_time
    
    def _is_truncated(self) -> bool:
        """
        Truncate the episode if the robot's pitch or roll exceeds the maximum allowed values
        or if the robot exhibits bad motion (high linear velocity in both x and y directions).
        
        Returns:
            bool: True if the episode is truncated, False otherwise.
        """
        return bool(
            abs(self.pitch) > self.max_pitch or 
            abs(self.roll) > 0.06
            )


    def _initialize_random_state(self):
        # Reset position and velocity
        self.data.qpos[:3] = [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.25]  # Initial position (x, y, z)
        self.data.qvel[:] = 0.0  # Initial speed

        # Euler angles: Roll=0, Pitch=random, Yaw=random
        euler = [
            0.0, # Roll
            np.random.uniform(-0.6, 0.6), # Pitch
            np.random.uniform(-np.pi, np.pi) # Yaw
        ]

        # Euler → Quaternion [x, y, z, w]
        quat_xyzw = R.from_euler('xyz', euler).as_quat()
        self.data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]