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

FS_ACCEL = 16384.0            # LSB/g
FSR_ACCEL = 2                 # Full scale range in g
FS_GYRO = 131.0               # LSB/(°/s)
FSR_GYRO = 250.0              # Full scale range in °/s

g = 9.81                      # Gravitational acceleration in m/s^2
DEG2RAD = (np.pi)/180         # Degrees to radians conversion factor
RAD2DEG = 180/(np.pi)         # Radians to degrees conversion factor

class SelfBalancingRobotEnv(gym.Env):
    def __init__(self, environment_path: str = "./models/scene.xml", max_time: float = 10.0, max_pitch: float = 0.5, frame_skip: int = 10):
        """
        Initialize the SelfBalancingRobot environment.
        
        Args:
            environment_path (str): Path to the MuJoCo model XML file.
            max_time (float): Maximum time for the episode.
            max_pitch (float): Maximum pitch angle before truncation.
            frame_skip (int): Number of frames to skip in each step.
        """
        super().__init__()
        self.viewer = None
        full_path = os.path.abspath(environment_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")
        
        self.model = MjModel.from_xml_path(full_path)
        self.data = MjData(self.model)
        self.max_time = max_time        # Maximum time for the episode
        self.frame_skip = frame_skip    # Number of frames to skip in each step  
        self.time_step = self.model.opt.timestep * self.frame_skip # Effective time step of the environment

        # Observation space: pitch, wheel velocities
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        
        # Action space
        ctrl_ranges = self.model.actuator_ctrlrange
        
        self.low  = ctrl_ranges[:, 0]
        self.high = ctrl_ranges[:, 1]

        self.action_space = gym.spaces.Box(
            low=self.low,
            high=self.high,
            dtype=np.float32
        )
        
        # Sensor parameters:
        self.accel_calib_scale = 0.0 # Accelerometer calibration scale factor
        self.encoder_resolution = (2 * np.pi)/8192 # Minimum angular change detectable by the wheel encoders [radians]

        # Initialize the environment attributes
        self.max_pitch = max_pitch # Maximum pitch angle before truncation
        self.setpoint = [0.0, 0.0] # [velocity_setpoint, angular_velocity_setpoint (steering)]
        
        # Initialize observation values
        self.roll, self.pitch, self.yaw = 0.0, 0.0, 0.0 # Orientation angles of the robot [roll, pitch, yaw]
        self.linear_acceleration = np.zeros(3) # Linear acceleration of the robot [gyro_x, gyro_y, gyro_z]
        self.angular_velocity = np.zeros(3) # Angular velocity of the robot [angular_velocity_x, angular_velocity_y, angular_velocity_z]
        self.wheels_position = np.zeros(2) # Angular position of the wheels [wheel_left_position, wheel_right_position]
        self.wheels_real_velocity = np.zeros(2) # Ideal angular velocity of the wheels [wheel_left_velocity, wheel_right_velocity]z

        self.past_pitch = 0.0
        self.past_wz = 0.0
        self.past_wx = 0.0
        self.past_ctrl = np.array([0.0, 0.0])

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
        action = np.clip(action, self.low, self.high)
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
        accel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer")

        # Address of the gyroscope sensor data
        accel_adr = self.model.sensor_adr[accel_id]

        # Get the accelerometer data
        accel_data = self._dirty_accel(self.data.sensordata[accel_adr : accel_adr + 3])

        return self.data.sensordata[accel_adr : accel_adr + 3]

    def _get_robot_angular_velocity(self) -> np.ndarray:
        """
        Get the angular velocity of the robot.
        
        Returns:
            np.ndarray: The angular velocity of the robot in the order [angular_velocity_x, angular_velocity_y, angular_velocity_z].
        """
        # Index of the gyroscope sensor
        gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "gyroscope")

        # Address of the gyroscope sensor data
        gyro_adr = self.model.sensor_adr[gyro_id]

        # Get the gyroscope data
        gyro_data = self._dirty_gyro(self.data.sensordata[gyro_adr : gyro_adr + 3])

        return self.data.sensordata[gyro_adr : gyro_adr + 3]
    
    def _get_body_orientation_angles(self) -> T.Tuple[float, float, float]:
        """
        Get the orientation angles of the robot's body in Euler angles (roll, pitch, yaw).
        
        Returns:
            T.Tuple[float, float, float]: The roll, pitch, and yaw angles of the robot's body.
        """
        # Complementary filter to estimate pitch angle
        pitch = 0.996 * (self.pitch + self.angular_velocity[1] * self.time_step) - 0.004 * np.arctan2(self.linear_acceleration[0], self.linear_acceleration[2])
        roll = 0.996 * (self.roll + self.angular_velocity[0] * self.time_step) - 0.004 * np.arctan2(self.linear_acceleration[1], self.linear_acceleration[2])
        yaw =  self.yaw + self.angular_velocity[2] * self.time_step 
        
        return pitch, roll, yaw
    
    def _get_wheels_angular_velocity(self) -> T.Tuple[float, float]:
        """
        Get the angular velocities of the robot's wheels.
        
        Returns:
            T.Tuple[float, float]: The angular velocities of the left and right wheels.
        """
        # Index of the wheel position sensors
        left_pos_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "left_wheel_pos")
        right_pos_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "right_wheel_pos")

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

    def _get_wheels_real_angular_velocity(self) -> T.Tuple[float, float]:
        """
        Get the real angular velocities of the robot's wheels without quantization.
        Returns:
            T.Tuple[float, float]: The angular velocities of the left and right wheels.
        """
        # Index of the wheel position sensors
        left_vel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "left_wheel_vel")
        right_vel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "right_wheel_vel")

        # Address of the wheel position sensor data
        left_vel_adr = self.model.sensor_adr[left_vel_id]
        right_vel_adr = self.model.sensor_adr[right_vel_id]

        # Get the wheel positions
        left_vel = self.data.sensordata[left_vel_adr]
        right_vel = self.data.sensordata[right_vel_adr]
        
        return left_vel, right_vel

    def _get_obs(self) -> np.ndarray:
        """
        Get the current observation of the environment.
       
        Returns:
            np.ndarray: The observation vector containing normalized sensor data and setpoints.
            - pitch
            - angular_velocity_y (pitch rate)
            - linear_acceleration_x
            - linear_acceleration_z
            - left_wheel_velocity
            - right_wheel_velocity
            - angular_velocity_z (yaw rate)
            - setpoint linear velocity
            - setpoint angular velocity (steering) 
       """
        # --- 1. Get sensors data ---
        self.wheels_real_velocity = self._get_wheels_real_angular_velocity()
        self.linear_acceleration = self._get_body_linear_acceleration()
        self.angular_velocity = self._get_robot_angular_velocity()
        #self.pitch, self.roll, self.yaw = self._get_body_orientation_angles() DA SISTEMARE
        quat = self.data.qpos[3:7]  # quaternion [w, x, y, z]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) # Rearrange to [x, y, z, w]
        self.roll, self.pitch, self.yaw = r.as_euler('xyz', degrees=False) # in radians
        
        # --- 2. Normalize sensor data ---
        # Constants (make sure they are defined in the class or globally)
        # Get the radius of the wheels from the model
        try:
            wheel_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_wheel_geom")
        except AttributeError:
        # Fallback per vecchie versioni (mujoco-py)
            wheel_geom_id = self.model.geom_name2id("right_wheel_geom")
        WHEEL_RADIUS = self.model.geom_size[wheel_geom_id][0]  # Assuming both wheels have the same max speed
        
        # Get the max wheel speed from the model
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
        MAX_WHEEL_SPEED = self.model.actuator_ctrlrange[actuator_id][1]
        # Pitch: Normalized over 90 degrees (pi/2)
        norm_pitch = self.pitch / (np.pi/2)
        
        # Angular Velocities (Gyro): Normalized over Full Scale Range (FSR)
        norm_w_x = self.angular_velocity[0] / (FSR_GYRO * DEG2RAD) # Roll rate
        norm_w_y = self.angular_velocity[1] / (FSR_GYRO * DEG2RAD) # Pitch rate
        norm_w_z = self.angular_velocity[2] / (FSR_GYRO * DEG2RAD) # Yaw rate (real)

        # Acceleration: Normalized over FSR
        norm_a_x = self.linear_acceleration[0] / (FSR_ACCEL * g)
        norm_a_z = self.linear_acceleration[2] / (FSR_ACCEL * g)

        # Wheel Speeds: Normalized over maximum speed
        norm_wheel_vel_left = self.wheels_real_velocity[0] / MAX_WHEEL_SPEED
        norm_wheel_vel_right = self.wheels_real_velocity[1] / MAX_WHEEL_SPEED

        # --- 3. Normalize COMMANDS (TARGET) ---
        # Using self.setpoint = [vel, dir]
        
        # A. Target Linear Velocity (self.setpoint[0]) in m/s
        # Convert it to "rad/s at the wheels" to make it comparable with norm_wheel_vel_left/right
        target_wheel_omega = self.setpoint[0] / WHEEL_RADIUS
        norm_target_lin = target_wheel_omega / MAX_WHEEL_SPEED

        # B. Target Steering (self.setpoint[1]) in rad/s
        # Normalize it with the same factor as the gyroscope (Z axis), to compare it with norm_w_z
        norm_target_ang = self.setpoint[1] / (FSR_GYRO * DEG2RAD)

        # Normalize past values
        norm_past_pitch = self.past_pitch / (np.pi/2)
        norm_past_wz = self.past_wz / (FSR_GYRO * DEG2RAD)
        norm_past_wx = self.past_wx / (FSR_GYRO * DEG2RAD)

        # --- 4. Construct Observation Vector ---
        obsv  = np.array([  
            norm_pitch,             # 1. Balance State
            norm_past_pitch,        # 2. Past Balance State
            norm_w_y,               # 3. Pitch Dynamics
            norm_a_x,               # 4. Accel X
            norm_a_z,               # 5. Accel Z
            norm_w_z,               # 6. Yaw Dynamics
            norm_past_wz,           # 7. Past Yaw Dynamics
            #norm_w_x,               # 8. Roll Dynamics
            #norm_past_wx,           # 9. Past Roll Dynamics
            self.data.ctrl[0],      # 10. Left Motor Command
            self.data.ctrl[1],      # 11. Right Motor Command
            self.past_ctrl[0],     # 12. Past Left Motor Command
            self.past_ctrl[1],     # 13. Past Right Motor Command
        ], dtype=np.float32)

        self.past_pitch = self.pitch
        self.past_wz = self.angular_velocity[2]
        self.past_wx = self.angular_velocity[0]
        self.past_ctrl = self.data.ctrl.copy()

        return obsv

    # Environment termination and truncation conditions and initialization
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
        quat = self.data.qpos[3:7]  # quaternion [w, x, y, z]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) # Rearrange to [x, y, z, w]
        _, pitch, _ = r.as_euler('xyz', degrees=False) # in radians
        return bool(
            abs(self.pitch) > self.max_pitch
            )

    def _initialize_random_state(self):
        # Reset ctrl inputs
        self.data.ctrl[:] = 0.0

        # Initialize accelerometer initial calibration scale
        self.accel_calib_scale = 1.0 + np.random.uniform(-0.03, 0.03, size=3)

        # Reset position and velocity
        self.data.qpos[:3] = [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.255]  # Initial position (x, y, z)
        self.data.qvel[:] = 0.0  # Initial speed

        # Euler angles: Roll=0, Pitch=random, Yaw=random
        euler = [
            0.0, # Roll
            0.0, # Pitch
            np.random.uniform(-np.pi, np.pi) # Yaw
        ]

        # Create a function to give a random direction to follow (evaluate also the velocity)
        
        # Euler → Quaternion [x, y, z, w]
        quat_xyzw = R.from_euler('xyz', euler).as_quat()
        self.data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

        self.past_pitch = 0.0
        self.past_wz = 0.0
        self.past_wx = 0.0
        self.past_ctrl = np.array([0.0, 0.0])

   
    # Function not used  
    def _dirty_gyro(self, gyro_data: np.ndarray) -> np.ndarray:
            """
            Simulate noise in the gyroscope data.
            
            Args:
                gyro_data (np.ndarray): The raw gyroscope data.
            """
            # Full scale conversion
            gyro_data = np.clip(gyro_data * RAD2DEG, -FSR_GYRO, FSR_GYRO)

            # Turn °/s to raw data
            gyro_raw = gyro_data * FS_GYRO

            # Add noise
            # Sensitivity Scale Factor Tolerance ±3%
            gyro_raw *= (1 + np.random.uniform(-0.03, 0.03, size=3))

            # Non-linearity ±0.2%
            gyro_raw += 0.002 * (gyro_raw ** 2)

            # Cross-axis sensitivity ±2%
            cross = np.eye(3) + np.random.uniform(-0.02, 0.02, size=(3,3))
            gyro_raw = cross @ gyro_raw

            # Turn raw data back to rad/s
            gyro_data_noisy = gyro_raw / FS_GYRO * DEG2RAD 

            return gyro_data_noisy
    
    def _dirty_accel(self, accel_data: np.ndarray) -> np.ndarray:
        """
        Simulate noise in the accelerometer data.
        
        Args:
            accel_data (np.ndarray): The raw accelerometer data.
        """
        
        # Full scale conversion
        accel_data = np.clip(accel_data / g, -FSR_ACCEL, FSR_ACCEL)

        # Turn g to raw data
        accel_raw = accel_data * FS_ACCEL

        # Add noise
        # Initial Calibration Tolerance ±3%
        accel_raw *= self.accel_calib_scale

        # Non-linearity ±0.5%
        accel_raw += 0.005 * (accel_raw ** 2)

        # Cross-axis sensitivity ±2%
        cross = np.eye(3) + np.random.uniform(-0.02, 0.02, size=(3,3))
        accel_raw = cross @ accel_raw

        # Turn raw data back to g
        accel_data_noisy = accel_raw / FS_ACCEL * g

        return accel_data_noisy
      
     
     
        