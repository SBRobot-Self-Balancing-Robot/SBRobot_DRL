import mujoco
import numpy as np
import typing as T
import gymnasium as gym
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R
from src.env.self_balancing_robot_env.self_balancing_robot_env import SelfBalancingRobotEnv

FS_ACCEL = 16384.0            # LSB/g
FSR_ACCEL = 2                 # Full scale range in g
FS_GYRO = 131.0               # LSB/(°/s)
FSR_GYRO = 250.0              # Full scale range in °/s

g = 9.81                      # Gravitational acceleration in m/s^2
DEG2RAD = (np.pi)/180         # Degrees to radians conversion factor
RAD2DEG = 180/(np.pi)         # Radians to degrees conversion factor

class ObservationWrapper(gym.Wrapper):
    """
    Wrapper for the SelfBalancingRobotEnv to modify the observation structure.
    """
    def __init__(self, env: SelfBalancingRobotEnv):
        super().__init__(env)

        # Sensor parameters:
        self.encoder_resolution = (2 * np.pi)/8192 # Minimum angular change detectable by the wheel encoders [radians]
        self.accel_calib_scale = 0.0 # Accelerometer calibration scale factor

        # Madgwick filter for orientation estimation
        self.madgwick = Madgwick(frequency=1.0/self.env.time_step, beta=0.033)

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

        # Initialize past values for observation
        self.past_pitch = 0.0
        self.past_wz = 0.0
        self.past_ctrl = np.array([0.0, 0.0])
        self.past_x_vel = 0.0
        self.past_wheels_velocity = np.zeros(2)


    def step(self, action):
        """
        Executes one step in the environment with the given action.
        
        Args:
            action: The action to take in the environment.
        Returns:
            obs: The modified observation after taking the action.
            reward: The reward from the environment.
            terminated: Whether the episode has ended.
            truncated: Whether the episode was truncated.
            info: Additional information from the environment.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = self._get_obs()

        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """
        Reset the environment.
        
        Args:
            **kwargs: Optional arguments for the reset.
        Returns:
            Initial modified observation and additional info.
        """
        obs, info = self.env.reset(**kwargs)

        obs = self._get_obs()

        return obs, info
    
    # Sensor reading and observation construction
    def _get_body_linear_acceleration(self) -> np.ndarray:
        """
        Get the linear accelerations of the robot's body.
        
        Returns:
            np.ndarray: The linear accelerations of the robot's body in the order [acceleration_x, acceleration_y, acceleration_z].
        """
        # Index of the gyroscope sensor
        accel_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer")

        # Address of the gyroscope sensor data
        accel_adr = self.env.model.sensor_adr[accel_id]

        # Get the accelerometer data
        accel_data = self._dirty_accel(self.env.data.sensordata[accel_adr : accel_adr + 3])

        return self.env.data.sensordata[accel_adr : accel_adr + 3]

    def _get_robot_angular_velocity(self) -> np.ndarray:
        """
        Get the angular velocity of the robot.
        
        Returns:
            np.ndarray: The angular velocity of the robot in the order [angular_velocity_x, angular_velocity_y, angular_velocity_z].
        """
        # Index of the gyroscope sensor
        gyro_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SENSOR, "gyroscope")

        # Address of the gyroscope sensor data
        gyro_adr = self.env.model.sensor_adr[gyro_id]

        # Get the gyroscope data
        gyro_data = self._dirty_gyro(self.env.data.sensordata[gyro_adr : gyro_adr + 3])

        return self.env.data.sensordata[gyro_adr : gyro_adr + 3]
    
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
        self.env.Q = self.madgwick.updateIMU(
            self.env.Q, 
            gyr=self.angular_velocity,     # Gyro [x, y, z] in rad/s
            acc=self.linear_acceleration   # Accel [x, y, z] in m/s^2 or g
        )

        # 2. Convert Quaternion to Euler
        # AHRS uses [w, x, y, z], Scipy uses [x, y, z, w] -> We must reorder
        r = R.from_quat([self.env.Q[1], self.env.Q[2], self.env.Q[3], self.env.Q[0]])
        
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
        left_pos_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SENSOR, "left_wheel_pos")
        right_pos_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SENSOR, "right_wheel_pos")

        # Address of the wheel position sensor data
        left_pos_adr = self.env.model.sensor_adr[left_pos_id]
        right_pos_adr = self.env.model.sensor_adr[right_pos_id]

        # Get the wheel positions
        left_pos = self.env.data.sensordata[left_pos_adr]
        right_pos = self.env.data.sensordata[right_pos_adr]

        # Quantization according to encoder resolution
        left_pos = np.floor(left_pos / self.encoder_resolution) * self.encoder_resolution
        right_pos = np.floor(right_pos / self.encoder_resolution) * self.encoder_resolution
        
        # Compute wheel angular velocities
        left_speed = (left_pos - self.wheels_position[0]) / self.env.time_step
        right_speed = (right_pos - self.wheels_position[1]) / self.env.time_step
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
        left_vel_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SENSOR, "left_wheel_vel")
        right_vel_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SENSOR, "right_wheel_vel")

        # Address of the wheel position sensor data
        left_vel_adr = self.env.model.sensor_adr[left_vel_id]
        right_vel_adr = self.env.model.sensor_adr[right_vel_id]

        # Get the wheel positions
        left_vel = self.env.data.sensordata[left_vel_adr]
        right_vel = self.env.data.sensordata[right_vel_adr]
        
        return left_vel, right_vel

    def _get_linear_velocity(self) -> T.Tuple[float, float]:
        """
        Get the linear velocities of the robot in the x and y directions.
        
        Returns:
            T.Tuple[float, float]: The linear velocities in the x and y directions.
        """
        x_vel, y_vel = self.env.data.qvel[0:2]
        
        return x_vel, y_vel

    def _get_real_body_linear_acceleration(self) -> np.ndarray:
        """
        Get the ideal linear accelerations of the robot's body without noise.
        
        Returns:
            np.ndarray: The linear accelerations of the robot's body in the order [acceleration_x, acceleration_y, acceleration_z].
        """
        # Index of the ideal accelerometer sensor
        accel_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SENSOR, "ideal_accelerometer")

        # Address of the ideal accelerometer sensor data
        accel_adr = self.env.model.sensor_adr[accel_id]

        # Get the ideal accelerometer data
        accel_data = self.env.data.sensordata[accel_adr : accel_adr + 3]

        return accel_data

    def _get_real_robot_angular_velocity(self) -> np.ndarray:
        """
        Get the ideal angular velocity of the robot without noise.
        
        Returns:
            np.ndarray: The angular velocity of the robot in the order [angular_velocity_x, angular_velocity_y, angular_velocity_z].
        """
        # Index of the ideal gyroscope sensor
        gyro_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SENSOR, "ideal_gyroscope")

        # Address of the ideal gyroscope sensor data
        gyro_adr = self.env.model.sensor_adr[gyro_id]

        # Get the ideal gyroscope data
        gyro_data = self.env.data.sensordata[gyro_adr : gyro_adr + 3]

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
            wheel_geom_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "right_wheel_geom")
        except AttributeError:
        # Fallback per vecchie versioni (mujoco-py)
            wheel_geom_id = self.env.model.geom_name2id("right_wheel_geom")
        WHEEL_RADIUS = self.env.model.geom_size[wheel_geom_id][0]  # Assuming both wheels have the same max speed
        
        # Get the max wheel speed from the model
        actuator_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
        MAX_WHEEL_SPEED = self.env.model.actuator_ctrlrange[actuator_id][1]
        # Pitch: Normalized over 90 degrees (pi/2)
        compensated_pitch = self.pitch - self.env.offset_angle
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
        target_wheel_omega = self.env.setpoint[0] / WHEEL_RADIUS
        norm_target_lin = target_wheel_omega / MAX_WHEEL_SPEED

        # B. Target Steering (self.setpoint[1]) in rad/s
        # Normalize it with the same factor as the gyroscope (Z axis), to compare it with norm_w_z
        norm_target_ang = self.env.setpoint[1] / (FSR_GYRO * DEG2RAD)

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
            self.env.data.ctrl[0],      # 8. Left Motor Command
            self.env.data.ctrl[1],      # 9. Right Motor Command
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
        self.past_ctrl = self.env.data.ctrl.copy()
        self.past_wheels_velocity = (self.wheels_velocity[0], self.wheels_velocity[1])

        return obsv
    
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
 