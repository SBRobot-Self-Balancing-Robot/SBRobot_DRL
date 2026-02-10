"""
Environment for a self-balancing robot using MuJoCo.
"""
import os
import time
import cv2
import mujoco
import numpy as np
import typing as T
import gymnasium as gym
from mujoco import MjModel, MjData
from mujoco.viewer import launch_passive
from src.env.control.pose_control import PoseControl
from scipy.spatial.transform import Rotation as R

class SelfBalancingRobotEnv(gym.Env):
    def __init__(self, 
                 environment_path: str = "./models/scene.xml", 
                 max_time: float = 10.0, 
                 max_pitch: float = 0.5, 
                 frame_skip: int = 10, 
                 render_mode: T.Optional[str] = None, 
                 width: int = 720,
                 height: int = 480,
                 render_fps: int = 30,
                 save_video: bool = False,
                 video_path: str = "videos/simulation.mp4",):
        """
        Initialize the SelfBalancingRobot environment.
        
        Args:
            environment_path (str): Path to the MuJoCo model XML file.
            max_time (float): Maximum time for the episode.
            max_pitch (float): Maximum pitch angle before truncation.
            frame_skip (int): Number of frames to skip in each step.
            render_mode (str, optional): The mode for rendering. Defaults to None.
            width (int, optional): Width of the render window. Defaults to 720.
            height (int, optional): Height of the render window. Defaults to 480.
            render_fps (int, optional): Frames per second for rendering. Defaults to 30.
            save_video (bool, optional): Whether to save the video. Defaults to False.
            video_path (str, optional): Path to save the video. Defaults to "videos/simulation.mp4".
        """
        super().__init__()
        full_path = os.path.abspath(environment_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")
        # Load MuJoCo model and data
        try:
            self.model = MjModel.from_xml_path(full_path)
            self.data = MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model from {full_path}: {e}") from e
        
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
        self.imu_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, " ")
        self.original_imu_pos = self.model.body_pos[self.imu_id].copy()
        self.original_imu_quat = self.model.body_quat[self.imu_id].copy()

        # Initialize pose control
        self.pose_control = PoseControl()
        
        # Rendering setup
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.render_fps = render_fps
        
        self.renderer: T.Optional[mujoco.Renderer] = None
        self.viewer: T.Optional[mujoco.viewer.Handle] = None
        
        self._render_callbacks: T.List[T.Callable[[], None]] = []
        
        self._sim_start_time: T.Optional[float] = None
        self._frame_count: int = 0

        # Camera and Scene options
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 1.0  # Distance from the robot
        self.camera.elevation = -30  # Camera elevation angle
        self.camera.azimuth = 120  # Camera azimuth angle

        # Set up scene options.
        self.scene_option = mujoco.MjvOption()
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        self.scene_option.frame = mujoco.mjtFrame.mjFRAME_SITE
        self.scene_option.geomgroup[:] = 1
        
        # Video recording
        self.save_video = save_video
        self.video_path = video_path
        self.video_writer: T.Optional[cv2.VideoWriter] = None

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
        
        if self.render_mode == "human":
            self._sim_start_time = time.time()
            if self.viewer is None:
                try:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

                    # --- Apply visualization options to the viewer ---
                    # Copy flags from self.scene_option to viewer.opt
                    self.viewer.opt.flags[:] = self.scene_option.flags[:]
                    # Copy frame setting
                    self.viewer.opt.frame = self.scene_option.frame
                    # Copy geom group settings
                    self.viewer.opt.geomgroup[:] = self.scene_option.geomgroup[:]
                    # --- End of applying visualization options ---

                    # Apply camera settings from self.camera to the viewer
                    self.viewer.cam.distance = self.camera.distance
                    self.viewer.cam.elevation = self.camera.elevation
                    self.viewer.cam.azimuth = self.camera.azimuth
                    # Set initial lookat based on current qpos (might be randomized)
                    robot_pos = self.data.qpos[:3]
                    self.viewer.cam.lookat[:] = robot_pos

                except Exception as e:
                    print(f"Warning: Could not launch MuJoCo viewer: {e}")
                    self.viewer = None
            elif self.viewer.is_running():
                self.viewer.sync()
        
        # Initialize video writer if saving video
        if self.save_video and self.video_writer is None:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.render_fps, (self.width, self.height))

        observation = self._get_obs()
        info = self._get_info() # Get initial info (usually empty)

        return observation, info
    
    def _get_obs(self) -> np.ndarray:
        """Default observation: returns all available sensor data."""
        return self.data.sensordata.copy()

    def _get_info(self) -> T.Dict[str, T.Any]:
        """Returns basic information about the environment state."""
        # Base environment provides minimal info. Wrappers add more.
        return {"time": self.data.time}
        
    def add_render_callback(self, callback: T.Callable[[], None]):
        """Registers a function to be called during the render cycle."""
        if callable(callback):
            self._render_callbacks.append(callback)
        else:
            print("Warning: Tried to add a non-callable render callback.")
    
    def render(self):
        """Renders the environment based on the render_mode."""
        if self.render_mode is None:
            return None

        # Throttle rendering based on render_fps
        sim_time = self.data.time
        expected_frames = int(sim_time * self.render_fps)
        if self._frame_count >= expected_frames:
            return None # Skip frame
        self._frame_count += 1

        # Initialize renderer if needed
        if self.renderer is None:
            try:
                self.renderer = mujoco.Renderer(self.model, width=self.width, height=self.height)
            except Exception as e:
                 print(f"Warning: Failed to initialize MuJoCo renderer: {e}")
                 self.render_mode = None # Disable rendering
                 return None

        try:
            self.renderer.update_scene(self.data, scene_option=self.scene_option, camera=self.camera)

            if self.viewer is not None:
                # Reset the user geom buffer for the viewer
                self.viewer.user_scn.ngeom = 0

            # Call registered render callbacks
            for callback in self._render_callbacks:
                    try:
                        # Pass self in case the callback needs access to env methods directly
                        # Or adjust callback signature if it only needs renderer/scene
                        callback()
                    except Exception as e:
                        print(f"Warning: Error during render callback: {e}")

        except mujoco.FatalError as e:
             print(f"Warning: MuJoCo error during scene update: {e}")
             return None # Skip rendering this frame

        # Get pixel data
        try:
            pixels = self.renderer.render()
        except mujoco.FatalError as e:
             print(f"Warning: MuJoCo error during rendering: {e}")
             return None # Skip rendering this frame

        # Save to video if enabled
        if self.save_video and self.video_writer is not None:
            pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            self.video_writer.write(pixels_bgr)

        # Handle different render modes
        if self.render_mode == "rgb_array":
            return pixels

        elif self.render_mode == "human":
            # Check if viewer exists and is running before syncing
            if self.viewer is not None and self.viewer.is_running():
                # Wait until wall-clock time catches up to simulation time.
                if self._sim_start_time is None:
                    self._sim_start_time = time.time()

                desired_wall_time = self._sim_start_time + self.data.time
                current_wall_time = time.time()
                wait_time = desired_wall_time - current_wall_time

                if wait_time > 0:
                    time.sleep(wait_time)

                # Sync the viewer with the simulation.
                try: # Add a try-except around sync as well, just in case
                    self.viewer.sync()
                except Exception as e:
                    print(f"Error during viewer sync: {e}")
                    # Optionally close the viewer or handle the error
                    if self.viewer:
                        self.viewer.close()
                    self.viewer = None # Mark viewer as unusable

                # Check again if the viewer was closed by the user during sync/wait
                if self.viewer is not None and not self.viewer.is_running():
                    print("Viewer stopped by the user")
                    self.viewer.close()
                    self.viewer = None
            elif self.viewer is not None and not self.viewer.is_running():
                 # Handle case where viewer was previously initialized but is now closed
                 print("Viewer is closed.")
                 self.viewer.close() # Ensure cleanup
                 self.viewer = None
            # else: viewer is None (failed to initialize or already cleaned up)

            return None # Always return None for human mode

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
            np.random.uniform(-0.1, 0.1), # Pitch
            np.random.uniform(-np.pi, np.pi) # Yaw
        ]
        # Euler → Quaternion [x, y, z, w]
        quat_xyzw = R.from_euler('xyz', euler).as_quat()

        self.data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

        self.Q = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        # Pose control randomization
        if np.random.rand() < 0.5: # 50% chance to randomize pose control parameters
            self.pose_control.randomize()
        else:
            r = R.from_euler('xyz', euler).as_matrix()
            x_head = r[:2, 0]
            norm = np.linalg.norm(x_head)
            if norm > 1e-6:
                unit_vector = x_head / norm
            else:
                unit_vector= np.zeros(2)
            self.pose_control.heading_angle = unit_vector

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
        Randomize the center of mass of each body by adding a small positional offset (±2 cm).
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
        
    def _get_active_scenes(self) -> T.Optional[mujoco.MjvScene]:
        """Helper to get the scenes objects."""
        scenes = []
        renderer = mujoco.Renderer(self.model, self.data)
        # Check if renderer is initialized
        if renderer is not None:
            scenes.append(renderer.scene)

        # Check if viewer is initialized
        if self.viewer is not None:
            scenes.append(self.viewer.user_scn)

        if len(scenes) == 0:
            print("Warning: No active scenes for rendering.")
            return None

        return scenes

    def render_vector(self, origin: np.ndarray, vector: np.ndarray, color: T.List[float], scale: float = 0.2, radius: float = 0.005, offset: float = 0.0):
        """Helper to render an arrow geometry in the scene."""
        scns = self._get_active_scenes()

        if scns is None: return # No active scenes

        for scn in scns:
            if scn.ngeom >= scn.maxgeom: return # Check geom buffer space

            origin_offset = origin.copy() + np.array([0, 0, offset])
            endpoint = origin_offset + (vector * scale)
            idx = scn.ngeom
            try:
                mujoco.mjv_initGeom(scn.geoms[idx], mujoco.mjtGeom.mjGEOM_ARROW1, np.zeros(3), np.zeros(3), np.zeros(9), np.array(color, dtype=np.float32))
                mujoco.mjv_connector(scn.geoms[idx], mujoco.mjtGeom.mjGEOM_ARROW1, radius, origin_offset, endpoint)
                scn.ngeom += 1
            except IndexError:
                print("Warning: Ran out of geoms in MuJoCo scene for rendering vector.")