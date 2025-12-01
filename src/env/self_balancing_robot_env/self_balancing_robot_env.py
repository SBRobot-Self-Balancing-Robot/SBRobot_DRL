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
        
        # Initialize the environment attributes
        self.max_pitch = max_pitch # Maximum pitch angle before truncation
        self.setpoint = [0.0, 0.0] # [velocity_setpoint, angular_velocity_setpoint (steering)]        

        # Offset angle at the beginning of the simulation
        self.offset_angle = self._get_offset()

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
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        return [], 0.0, terminated, truncated, {}

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
        return [], {}
    
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

    # Environment termination and truncation conditions
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

    # Initialize and randomization methods
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

    # Initialize random state
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