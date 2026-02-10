import mujoco
import numpy as np
import typing as T
import gymnasium as gym
from src.utils.math import signed_sin
from ahrs.filters import Madgwick
from src.env.robot import SelfBalancingRobotEnv
from scipy.spatial.transform import Rotation as R

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

        # Sensor parameters
        self.encoder_resolution = (2 * np.pi)/8192 # Minimum angular change detectable by the wheel encoders [radians]

        # Madgwick filter for orientation estimation
        self.madgwick = Madgwick(frequency=1.0/self.env.time_step, beta=0.033)

        # Initialize observation values
        self.direction_vector = np.array([0.0, 0.0]) # Normal vector pointing upwards in the robot's frame
        self.roll, self.pitch, self.yaw = 0.0, 0.0, 0.0 # Orientation angles of the robot [roll, pitch, yaw]
        self.linear_acceleration = np.zeros(3) # Linear acceleration of the robot [gyro_x, gyro_y, gyro_z]
        self.angular_velocity = np.zeros(3) # Angular velocity of the robot [angular_velocity_x, angular_velocity_y, angular_velocity_z]
        self.wheels_position = np.zeros(2) # Angular position of the wheels [wheel_left_position, wheel_right_position]
        self.wheels_velocity = np.zeros(2) # Quantized angular velocity of the wheels [wheel_left_velocity, wheel_right_velocity]
        self.ctrl = np.zeros(2) # Control commands sent to the motors [left_motor_command, right_motor_command]
        
        # Initialize past values for observation
        self.past_ctrl = np.array([0.0, 0.0])

        # Control variation
        self.ctrl_variation = np.array([0.0, 0.0])

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

        self._reset_params()

        return obs, info
    
    def _reset_params(self):
        """
        Reset environment parameters to default values.
        """
        # Reset past control commands
        self.past_ctrl = np.array([0.0, 0.0])

        # Reset control variation
        self.ctrl_variation = np.array([0.0, 0.0])
        

    # Sensor reading and observation construction
    def _get_wheels_data(self) -> T.Tuple[float, float, float, float]:
        """
        Get the angular velocities of the robot's wheels.
        
        Returns:
            T.Tuple[float, float, float, float]: The positions and angular velocities of the left and right wheels.

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
        return left_pos, right_pos, left_speed, right_speed

    def _get_body_linear_acceleration(self) -> np.ndarray:
        """
        Get the linear accelerations of the robot's body.
        
        Returns:
            np.ndarray: The linear accelerations of the robot's body in the order [acceleration_x, acceleration_y, acceleration_z].
        """
        # Index of the gyroscope sensor
        accel_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer") # with noise

        # Address of the gyroscope sensor data
        accel_adr = self.env.model.sensor_adr[accel_id]

        return self.env.data.sensordata[accel_adr : accel_adr + 3]

    def _get_robot_angular_velocity(self) -> np.ndarray:
        """
        Get the angular velocity of the robot.
        
        Returns:
            np.ndarray: The angular velocity of the robot in the order [angular_velocity_x, angular_velocity_y, angular_velocity_z].
        """
        # Index of the gyroscope sensor
        gyro_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SENSOR, "gyroscope") # with noise

        # Address of the gyroscope sensor data
        gyro_adr = self.env.model.sensor_adr[gyro_id]

        return self.env.data.sensordata[gyro_adr : gyro_adr + 3]

    def _get_body_orientation_angles(self, angular_velocity, linear_acceleration) -> T.Tuple[np.ndarray, float, float, float]:
        """
        Get the orientation angles of the robot's body in Euler angles (roll, pitch, yaw).
        
        Returns:
            T.Tuple[np.ndarray, float, float, float]: The direction vector and the roll, pitch, and yaw angles of the robot's body.
        """
        # 1. Update the Quaternion State
        # The filter calculates the NEW quaternion based on the OLD quaternion + Sensor Data
        self.env.Q = self.madgwick.updateIMU(
            self.env.Q, 
            gyr=angular_velocity,     # Gyro [x, y, z] in rad/s
            acc=linear_acceleration   # Accel [x, y, z] in m/s^2 or g
        )

        # 2. Convert Quaternion to Euler
        # AHRS uses [w, x, y, z], Scipy uses [x, y, z, w] -> We must reorder
        r = R.from_quat([self.env.Q[1], self.env.Q[2], self.env.Q[3], self.env.Q[0]])
        
        # Get the rotation matrix
        rot_matrix = r.as_matrix()
        xy_vector = rot_matrix[:2,0]
        norm = np.linalg.norm(xy_vector)
        if norm > 1e-6:
            unit_vector = xy_vector / norm
        else:
            unit_vector= np.zeros(2)

        # Get angles (standard aerospace sequence is usually 'xyz' -> roll, pitch, yaw)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        
        # 3. Return in the SPECIFIC order your previous function defined: (Pitch, Roll, Yaw)
        return unit_vector, pitch, roll, yaw

    # Observation construction

    def _normalize_value(self, value, range) -> float:
        """Normalize a value to the range [-1, 1]."""
        return (value / range) if range != 0 else 0
    
    def _value_variation(self, current_value, past_value) -> float:
        """Compute the variation of a value over time."""
        return (current_value - past_value)

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
        # Data from encoder: left_pos, right_pos, left_speed, right_speed
        self.wheels_position[0], self.wheels_position[1], self.wheels_velocity[0], self.wheels_velocity[1] = self._get_wheels_data()
        
        # Data from IMU: linear_acceleration [ax, ay, az], angular_velocity [wx, wy, wz], orientation angles (roll, pitch, yaw), direction_vector
        self.linear_acceleration = self._get_body_linear_acceleration()
        self.angular_velocity = self._get_robot_angular_velocity()
        self.direction_vector, self.pitch, self.roll, self.yaw = self._get_body_orientation_angles(self.angular_velocity, self.linear_acceleration)
        
        # Control commands sent to the motors
        self.ctrl = self.env.data.ctrl.copy() # Control commands sent to the motors [left_motor_command, right_motor_command]

        # --- 2. Normalize sensor data ---
        # Get the max wheel speed from the model
        actuator_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
        MAX_WHEEL_SPEED = self.env.model.actuator_ctrlrange[actuator_id][1]
        # Pitch: Normalized over 90 degrees (pi/2)
        norm_pitch = self._normalize_value(self.pitch, np.pi/2)

        # Angular Velocities (Gyro): Normalized over Full Scale Range (FSR)
        norm_w_y = self._normalize_value(self.angular_velocity[1], FSR_GYRO * DEG2RAD) # Pitch rate (real)
        norm_w_z = self._normalize_value(self.angular_velocity[2], FSR_GYRO * DEG2RAD) # Yaw rate (real)

        # Wheel Speeds: Normalized over maximum speed
        norm_wheel_left_vel = self._normalize_value(self.wheels_velocity[0], MAX_WHEEL_SPEED)
        norm_wheel_right_vel = self._normalize_value(self.wheels_velocity[1], MAX_WHEEL_SPEED)

        # Normalize control commands
        norm_ctrl_left = self._normalize_value(self.ctrl[0], MAX_WHEEL_SPEED)
        norm_ctrl_right = self._normalize_value(self.ctrl[1], MAX_WHEEL_SPEED)

        # Normalize past values
        norm_past_ctrl_left = self._normalize_value(self.past_ctrl[0], MAX_WHEEL_SPEED)
        norm_past_ctrl_right = self._normalize_value(self.past_ctrl[1], MAX_WHEEL_SPEED)

        # --- 3. values variation ---
        # Compute the variation of control commands over time (derivative)
        self.ctrl_variation[0] = self._value_variation(norm_ctrl_left, norm_past_ctrl_left)
        self.ctrl_variation[1] = self._value_variation(norm_ctrl_right, norm_past_ctrl_right)
        
        # --- 4. Setpoints and Errors ---
        heading_error = self.env.pose_control.heading_error(self.direction_vector) # type: ignore
        velocity_error = self.env.pose_control.velocity_error((norm_wheel_left_vel + norm_wheel_right_vel)/2) # type: ignore
        
        
        # --- 5. Construct Observation Vector ---
        obsv = np.array([
            # Pitch and related dynamics
            norm_pitch,                 # 1. Balance State
            norm_w_y,                   # 2. Pitch Dynamics

            # Yaw dynamics
            norm_w_z,                   # 3. Yaw Dynamics
            
            # Wheels dynamics
            norm_ctrl_left,             # 4. Left Motor Command
            norm_ctrl_right,            # 5. Right Motor Command
            self.ctrl_variation[0],     # 6. Left Motor Command Variation
            self.ctrl_variation[1],     # 7. Right Motor Command Variation

            # Wheels velocities
            norm_wheel_left_vel,        # 8. Left Wheel Velocity
            norm_wheel_right_vel,       # 9. Right Wheel Velocity
            (norm_wheel_left_vel + norm_wheel_right_vel)/2, # 10. Linear Velocity (Speed)
            
            heading_error,              # 11. Heading Error (Direction)
            velocity_error              # 12. Velocity Error (Speed)

        ], dtype=np.float32)

        self.past_ctrl = self.ctrl.copy() # Update past control for the next step

        return obsv