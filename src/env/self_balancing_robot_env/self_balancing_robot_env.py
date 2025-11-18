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
    
    def __init__(self, environment_path: str = "./models/scene.xml", max_time: float = 10.0, max_pitch: float = 0.8, frame_skip: int = 5):
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
        self.time_step = self.model.opt.timestep * self.frame_skip # Effective time step of the environment

        # Observation space: pitch, wheel velocities
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)
        # Encoder resolution:
        self.encoder_resolution = (2 * np.pi)/8192 # Minimum angular change detectable by the wheel encoders [radians]

        # Initialize the environment attributes
        self.weight_fall_penalty = 100.0 # Penalty for falling
        self.max_pitch = max_pitch # Maximum pitch angle before truncation
        
        # Initialize observation values
        self.roll, self.pitch, self.yaw = 0.0, 0.0, 0.0 # Orientation angles of the robot [roll, pitch, yaw]
        self.linear_acceleration = np.zeros(3) # Linear acceleration of the robot [gyro_x, gyro_y, gyro_z]
        self.angular_velocity = np.zeros(3) # Angular velocity of the robot [angular_velocity_x, angular_velocity_y, angular_velocity_z]
        self.wheels_position = np.zeros(2) # Angular position of the wheels [wheel_left_position, wheel_right_position]

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
        self.data.ctrl[:] = action
        
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


    def _get_body_linear_acceleration(self) -> np.ndarray:
        """
        Get the linear accelerations of the robot's body.
        
        Returns:
            np.ndarray: The linear accelerations of the robot's body in the order [acceleration_x, acceleration_y, acceleration_z].
        """
        # Index of the gyroscope sensor
        accel_id = self.model.sensor_name2id("accelerometer")

        # Address of the gyroscope sensor data
        accel_adr = self.model.sensor_adr[accel_id]

        return self.data.sensordata[accel_adr : accel_adr + 3]


    def _dirty_gyro(self, gyro_data: np.ndarray) -> np.ndarray:
        """
        Simulate noise in the gyroscope data.
        
        Args:
            gyro_data (np.ndarray): The raw gyroscope data.
        """
        FS = 131.0              # LSB/(°/s)
        DEG2RAD = np.pi/180
        RAD2DEG = 180/np.pi
        FSR = 250.0             # Full scale range in °/s

        # Full scale conversion
        gyro_data = np.clip(gyro_data * RAD2DEG, -FSR, FSR)

        # Turn rad/s to raw data
        gyro_raw = gyro_data * FS

        # Add noise
        # Sensitivity Scale Factor Tolerance ±3%
        gyro_raw *= (1 + np.random.uniform(-0.03, 0.03, size=3))

        # Non-linearity ±2%
        gyro_raw += 0.002 * (gyro_raw ** 2)

        # Cross-axis sensitivity ±2%
        cross = np.eye(3) + np.random.uniform(-0.02, 0.02, size=(3,3))
        gyro_raw = cross @ gyro_raw

        # Turn raw data back to rad/s
        gyro_data_noisy = gyro_raw / FS * DEG2RAD

        return gyro_data_noisy

    def _get_robot_angular_velocity(self) -> np.ndarray:
        """
        Get the angular velocity of the robot.
        
        Returns:
            np.ndarray: The angular velocity of the robot in the order [angular_velocity_x, angular_velocity_y, angular_velocity_z].
        """
        # Index of the gyroscope sensor
        gyro_id = self.model.sensor_name2id("gyroscope")

        # Address of the gyroscope sensor data
        gyro_adr = self.model.sensor_adr[gyro_id]

        # Get the gyroscope data
        gyro_data = self._dirty_gyro(self.data.sensordata[gyro_adr : gyro_adr + 3])

        return gyro_data
    
    def _get_body_orientation_angles(self) -> T.Tuple[float, float, float]:
        """
        Get the orientation angles of the robot's body in Euler angles (roll, pitch, yaw).
        
        Returns:
            T.Tuple[float, float, float]: The roll, pitch, and yaw angles of the robot's body.
        """
        self.linear_acceleration = self._get_body_linear_acceleration() # [accel_x, accel_y, accel_z]
        self.angular_velocity = self._get_robot_angular_velocity() # [gyro_x, gyro_y, gyro_z]

        # Complementary filter to estimate pitch angle
        self.pitch = 0.996 * (self.pitch + self.angular_velocity[1] * self.time_step) - 0.004 * np.arctan2(self.linear_acceleration[0], self.linear_acceleration[2]) * 180.0 / np.pi
        self.roll = 0.996 * (self.roll + self.angular_velocity[0] * self.time_step) - 0.004 * np.arctan2(self.linear_acceleration[1], self.linear_acceleration[2]) * 180.0 / np.pi
        self.yaw += self.angular_velocity[2] * self.time_step
        
        return self.pitch, self.roll, self.yaw
    
    def _get_wheels_angular_velocity(self) -> T.Tuple[float, float]:
        """
        Get the angular velocities of the robot's wheels.
        
        Returns:
            T.Tuple[float, float]: The angular velocities of the left and right wheels.
        """
        # Index of the wheel position sensors
        left_pos_id = self.model.sensor_name2id("left_wheel_pos")
        right_pos_id = self.model.sensor_name2id("right_wheel_pos")

        # Address of the wheel position sensor data
        left_pos_adr = self.model.sensor_adr[left_pos_id]
        right_pos_adr = self.model.sensor_adr[right_pos_id]

        # Get the wheel positions
        left_pos = self.data.sensordata[left_pos_adr]
        right_pos = self.data.sensordata[right_pos_adr]

        # Quantization according to encoder resolution
        left_pos = np.floor(left_pos / self.encoder_resolution) * self.encoder_resolution
        right_pos = np.floor(right_pos / self.encoder_resolution) * self.encoder_resolution
        
        # Compute wheel angular velocities
        left_speed = (left_pos - self.wheels_position[0]) / self.time_step
        right_speed = (right_pos - self.wheels_position[1]) / self.time_step
        self.wheels_position[0] = left_pos
        self.wheels_position[1] = right_pos 
        return left_speed, right_speed

    def _get_obs(self) -> np.ndarray:
        """
        Get the current observation of the environment.
        
        Returns:
            np.ndarray: The observation vector.
            
            Ho mantenuto l'output di 7 elementi, che ora include:
            [pitch, roll, yaw, linear_acceleration_x, linear_acceleration_y, angular_velocity_x, angular_velocity_y]
        """
        self.pitch, self.roll, self.yaw = self._get_body_orientation_angles()
        self.right_wheel_velocity, self.left_wheel_velocity = self._get_wheels_angular_velocity()
        
        return np.array([
            self.pitch,                    
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
        Truncate the episode if the robot's pitch exceeds the maximum allowed value.
        
        Returns:
            bool: True if the episode is truncated, False otherwise.
        """
        return bool(
            abs(self.pitch) > self.max_pitch
            )


    def _initialize_random_state(self):
        # Reset position and velocity
        self.data.qpos[:3] = [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.25]  # Initial position (x, y, z)
        self.data.qvel[:] = 0.0  # Initial speed

        # Euler angles: Roll=0, Pitch=random, Yaw=random
        self.pitch = np.random.uniform(-0.6, 0.6)
        self.yaw = np.random.uniform(-np.pi, np.pi)
        self.roll = 0.0
        euler = [
            self.roll, # Roll
            self.pitch, # Pitch
            self.yaw  # Yaw
        ]

        # Euler → Quaternion [x, y, z, w]
        quat_xyzw = R.from_euler('xyz', euler).as_quat()
        self.data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]