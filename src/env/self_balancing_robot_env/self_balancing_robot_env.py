"""
Environment for a self-balancing robot using MuJoCo.
"""
import os
import time
import mujoco
import numpy as np
import typing as T
import gymnasium as gym
from ahrs.filters import Madgwick
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
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
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
        
        # Madgwick filter for orientation estimation
        self.madgwick = Madgwick(frequency=1.0/self.time_step, beta=0.033)
        
        # Initialize observation values
        self.roll, self.pitch, self.yaw = 0.0, 0.0, 0.0 # Orientation angles of the robot [roll, pitch, yaw]
        self.linear_acceleration = np.zeros(3) # Linear acceleration of the robot [gyro_x, gyro_y, gyro_z]
        self.angular_velocity = np.zeros(3) # Angular velocity of the robot [angular_velocity_x, angular_velocity_y, angular_velocity_z]
        self.wheels_position = np.zeros(2) # Angular position of the wheels [wheel_left_position, wheel_right_position]
        self.wheels_velocity = np.zeros(2) # Quantized angular velocity of the wheels [wheel_left_velocity, wheel_right_velocity]
        self.x_vel = 0.0 # Linear velocity in the x direction

        # Ideal data for reward computation
        self.wheels_real_velocity = np.zeros(2) # Ideal angular velocity of the wheels [wheel_left_velocity, wheel_right_velocity]
        self.real_linear_acceleration = np.zeros(3) # Ideal linear acceleration of the robot [accel_x, accel_y, accel_z]
        self.real_angular_velocity = np.zeros(3) # Ideal angular velocity of the robot [gyro_x, gyro_y, gyro_z]     

        # Offset angle at the beginning of the simulation
        self.offset_angle = self._get_offset()

        # Initialize past values for observation
        self.past_pitch = 0.0
        self.past_wz = 0.0
        self.past_ctrl = np.array([0.0, 0.0])
        self.past_x_vel = 0.0
        self.past_wheels_velocity = np.zeros(2)

        # Save initial masses for randomization
        self.initial_masses = self.model.body_mass.copy()

        # Save original body positions for randomization
        self.original_body_ipos = self.model.body_ipos.copy()

        # Save original IMU position for randomization
        self.imu_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "IMU")
        self.original_imu_pos = self.model.body_pos[self.imu_id].copy()
        self.original_imu_quat = self.model.body_quat[self.imu_id].copy()


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

    def _get_offset(self) -> float:
        """
        Get the initial offset angles of the robot's body in Euler angles (roll, pitch, yaw) 
        wrt the ideal 0 position of the robot.
        
        Returns:
            T.Tuple[float, float, float]: The roll, pitch, and yaw angles of the robot's body.
        """
        mujoco.mj_kinematics(self.model, self.data)
        chassis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Chassis")
        wheel_L_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "WheelL")
        com_total = self.data.subtree_com[chassis_id]
        pivot_pos = self.data.xpos[wheel_L_id] 
        
        dx = com_total[0] - pivot_pos[0]
        dz = com_total[2] - pivot_pos[2]
        
        angle_rad = -np.arctan2(dx, dz)
        
        return angle_rad

    # Sensor reading and observation construction
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
        # pitch = 0.996 * (self.pitch + self.angular_velocity[1] * self.time_step) - 0.004 * np.arctan2(self.linear_acceleration[0], self.linear_acceleration[2])
        # roll = 0.996 * (self.roll + self.angular_velocity[0] * self.time_step) - 0.004 * np.arctan2(self.linear_acceleration[1], self.linear_acceleration[2])
        # yaw =  self.yaw + self.angular_velocity[2] * self.time_step 
        
        # return pitch, roll, yaw
        # 1. Update the Quaternion State
        # The filter calculates the NEW quaternion based on the OLD quaternion + Sensor Data
        self.Q = self.madgwick.updateIMU(
            self.Q, 
            gyr=self.angular_velocity,     # Gyro [x, y, z] in rad/s
            acc=self.linear_acceleration   # Accel [x, y, z] in m/s^2 or g
        )

        # 2. Convert Quaternion to Euler
        # AHRS uses [w, x, y, z], Scipy uses [x, y, z, w] -> We must reorder
        r = R.from_quat([self.Q[1], self.Q[2], self.Q[3], self.Q[0]])
        
        # Get angles (standard aerospace sequence is usually 'xyz' -> roll, pitch, yaw)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        
        # 3. Update legacy class attributes (Optional, only if other parts of your code read these directly)
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

        # 4. Return in the SPECIFIC order your previous function defined: (Pitch, Roll, Yaw)
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

    # Ideal data for reward computation
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

    def _get_linear_velocity(self) -> T.Tuple[float, float]:
        """
        Get the linear velocities of the robot in the x and y directions.
        
        Returns:
            T.Tuple[float, float]: The linear velocities in the x and y directions.
        """
        x_vel, y_vel = self.data.qvel[0:2]
        
        return x_vel, y_vel

    def _get_real_body_linear_acceleration(self) -> np.ndarray:
        """
        Get the ideal linear accelerations of the robot's body without noise.
        
        Returns:
            np.ndarray: The linear accelerations of the robot's body in the order [acceleration_x, acceleration_y, acceleration_z].
        """
        # Index of the ideal accelerometer sensor
        accel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ideal_accelerometer")

        # Address of the ideal accelerometer sensor data
        accel_adr = self.model.sensor_adr[accel_id]

        # Get the ideal accelerometer data
        accel_data = self.data.sensordata[accel_adr : accel_adr + 3]

        return accel_data

    def _get_real_robot_angular_velocity(self) -> np.ndarray:
        """
        Get the ideal angular velocity of the robot without noise.
        
        Returns:
            np.ndarray: The angular velocity of the robot in the order [angular_velocity_x, angular_velocity_y, angular_velocity_z].
        """
        # Index of the ideal gyroscope sensor
        gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ideal_gyroscope")

        # Address of the ideal gyroscope sensor data
        gyro_adr = self.model.sensor_adr[gyro_id]

        # Get the ideal gyroscope data
        gyro_data = self.data.sensordata[gyro_adr : gyro_adr + 3]

        return gyro_data

    # Observation construction
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
        self.wheels_velocity = self._get_wheels_angular_velocity()
        self.linear_acceleration = self._get_body_linear_acceleration()
        self.angular_velocity = self._get_robot_angular_velocity()
        self.pitch, self.roll, self.yaw = self._get_body_orientation_angles()
        self.x_vel, self.y_vel = self._get_linear_velocity()
        # quat = self.data.qpos[3:7]  # quaternion [w, x, y, z]
        # r = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) # Rearrange to [x, y, z, w]
        # self.roll, self.pitch, self.yaw = r.as_euler('xyz', degrees=False) # in radians
        
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
        compensated_pitch = self.pitch - self.offset_angle
        norm_pitch = self.pitch / (np.pi/2)
        
        # Ideal data for reward computation 
        self.wheels_real_velocity = self._get_wheels_real_angular_velocity()
        self.real_linear_acceleration = self._get_real_body_linear_acceleration()
        self.real_angular_velocity = self._get_real_robot_angular_velocity()

        # Angular Velocities (Gyro): Normalized over Full Scale Range (FSR)
        norm_w_x = self.angular_velocity[0] / (FSR_GYRO * DEG2RAD) # Roll rate
        norm_w_y = self.angular_velocity[1] / (FSR_GYRO * DEG2RAD) # Pitch rate
        norm_w_z = self.angular_velocity[2] / (FSR_GYRO * DEG2RAD) # Yaw rate (real)

        # Acceleration: Normalized over FSR
        norm_a_x = self.linear_acceleration[0] / (FSR_ACCEL * g)
        norm_a_z = self.linear_acceleration[2] / (FSR_ACCEL * g)

        # Wheel Speeds: Normalized over maximum speed
        norm_wheel_left = self.wheels_velocity[0] / MAX_WHEEL_SPEED
        norm_wheel_right = self.wheels_velocity[1] / MAX_WHEEL_SPEED

        # Linear Velocities: Normalized over maximum speed
        norm_x_vel = self.x_vel / 0.6
        norm_y_vel = self.y_vel / 0.6

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
        norm_past_x_vel = self.past_x_vel / 0.6
        norm_past_wheel_left = self.past_wheels_velocity[0] / MAX_WHEEL_SPEED
        norm_past_wheel_right = self.past_wheels_velocity[1] / MAX_WHEEL_SPEED

        # --- 4. Construct Observation Vector ---
        obsv  = np.array([
            # Pitch and related dynamics
            norm_pitch,             # 1. Balance State
            norm_past_pitch,        # 2. Past Balance State
            norm_w_y,               # 3. Pitch Dynamics
            norm_a_x,               # 4. Accel X
            norm_a_z,               # 5. Accel Z

            # Yaw dynamics
            norm_w_z,               # 6. Yaw Dynamics
            norm_past_wz,           # 7. Past Yaw Dynamics
            
            # Wheels dynamics
            self.data.ctrl[0],      # 8. Left Motor Command
            self.data.ctrl[1],      # 9. Right Motor Command
            self.past_ctrl[0],      # 10. Past Left Motor Command
            self.past_ctrl[1],      # 11. Past Right Motor Command

            # Wheels velocities
            norm_wheel_left,        # 12. Left Wheel Velocity
            norm_wheel_right,       # 13. Right Wheel Velocity
            norm_past_wheel_left,   # 14. Past Left Wheel Velocity
            norm_past_wheel_right,  # 15. Past Right Wheel Velocity

        ], dtype=np.float32)

        self.past_pitch = self.pitch.copy()
        self.past_wz = self.angular_velocity[2].copy()
        self.past_x_vel = self.x_vel.copy()
        self.past_ctrl = self.data.ctrl.copy()
        self.past_wheels_velocity = (self.wheels_velocity[0], self.wheels_velocity[1])

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
            abs(pitch) > self.max_pitch
            )


    def _space_positioning(self):
        """
        Randomly position the robot within defined ranges for x, y, pitch, and yaw.
        """
        # Random position
        self.data.qpos[:3] = [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.255]  # Initial position (x, y, z)

        # Reandom orientation
        euler = [
            0.0, # Roll
            np.random.uniform(-0.05, 0.05), # Pitch
            np.random.uniform(-np.pi, np.pi) # Yaw
        ]
        # Euler → Quaternion [x, y, z, w]
        quat_xyzw = R.from_euler('xyz', euler).as_quat()

        self.data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

        self.Q = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    def _reset_params(self):
        """
        Reset environment parameters to default values.
        """
        # Reset ctrl inputs
        self.data.ctrl[:] = 0.0

        # Reset speeds
        self.data.qvel[:] = 0.0

        # Reset past values
        self.past_pitch = 0.0
        self.past_wz = 0.0
        self.past_ctrl = np.array([0.0, 0.0])
        self.past_wheels_velocity = np.array([0.0, 0.0])
    
    def _randomize_masses(self):
        """
        Randomize the masses of the robot's components within ±10% of their original values.
        """
        for i in range(self.model.nbody):
            random_factor = np.random.uniform(0.9, 1.1)
            self.model.body_mass[i] = self.initial_masses[i] * random_factor

    def _randomize_com(self):
        """
        Randomize the center of mass of each body by adding a small positional offset (±2 mm).
        """
        for i in range(self.model.nbody):
            offset = np.random.uniform(-0.002, 0.002, size=3)  # ±2 mm
            self.model.body_ipos[i] = self.original_body_ipos[i] + offset

    def _randomize_imu_pose(self):
        """
        Randomize the IMU pose by adding small positional and rotational offsets.
        """
        # Random position offset ±5 mm
        pos_offset = np.random.uniform(-0.005, 0.005, size=3)
        self.model.body_pos[self.imu_id] = self.original_imu_pos + pos_offset

        # Random rotation offset ±0.01 rad (about 0.57°) per axis
        euler_offset = np.random.uniform(-0.01, 0.01, size=3)
        quat_offset = R.from_euler("xyz", euler_offset).as_quat() 
        # Convert to MuJoCo format [w, x, y, z]
        quat_offset_mj = np.array([quat_offset[3], quat_offset[0], quat_offset[1], quat_offset[2]])

        # Multiply the original IMU quaternion by the offset
        new_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(new_quat, self.original_imu_quat, quat_offset_mj)
        self.model.body_quat[self.imu_id] = new_quat

    def _randomize_actuator_gains(self):
        """
        Randomize the actuator gains within ±20% of their original values.
        """
        left_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
        right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_motor")


        # varia kv del ±20%
        self.model.actuator_gainprm[left_id][0] = np.random.uniform(16, 24)
        self.model.actuator_gainprm[right_id][0] = np.random.uniform(16, 24)

    def _randomize_wheel_friction(self):
        """
        Randomize the friction parameters of the wheels within ±10% of given values.
        """
        # random ±10% per ogni componente dell'attrito
        sliding   = np.random.uniform(0.8, 1.0)   # 0.9 ±10%
        torsional = np.random.uniform(0.045, 0.055) # 0.05 ±10%
        rolling   = np.random.uniform(0.0018, 0.0022) # 0.002 ±10%

        for wheel in ["WheelL_collision", "WheelR_collision"]:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, wheel)
            self.model.geom_friction[geom_id] = [sliding, torsional, rolling]

    def _initialize_random_state(self):
        # Position the robot randomly within a small range
        self._space_positioning()

        # Reset other parameters
        self._reset_params()

        # Randomize masses
        self._randomize_masses()

        # Randomize center of mass
        self._randomize_com()

        # Randomize IMU pose
        self._randomize_imu_pose()

        # Randomize actuator gains
        self._randomize_actuator_gains()

        # Randomize wheels friction
        self._randomize_wheel_friction()

        # Initialize accelerometer initial calibration scale
        self.accel_calib_scale = 1.0 + np.random.uniform(-0.03, 0.03, size=3)


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
      
     
     
        