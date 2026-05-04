"""
Core Gymnasium environment for the self-balancing two-wheeled robot.

Responsibilities:
  - Load MuJoCo model and manage simulation stepping.
  - Apply domain randomisation at each episode reset for robustness.
  - Manage PoseControl and VelocityControl reference generators.
  - Report episode outcome to VelocityControl for curriculum advancement.

Observations and rewards are computed by the wrappers layered on top.
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

from src.env.control.pose_control import PoseControl
from src.env.control.velocity_control import VelocityControl


class SelfBalancingRobotEnv(gym.Env):
    """
    MuJoCo environment for a self-balancing two-wheeled robot.

    Physical constants (derived from the robot XML model):
        WHEEL_RADIUS  = 0.0625 m   (cylinder geom: size="0.0625 0.01")
        WHEEL_TRACK   = 0.2982 m   (|y_WheelL − y_WheelR| ≈ 0.298 m)
        MAX_CTRL      = 8.775 rad/s (actuator ctrlrange)
        MAX_LIN_VEL   = MAX_CTRL × WHEEL_RADIUS ≈ 0.548 m/s

    Sensor layout in data.sensordata (see SBRobot_Snello.xml):
        [0:3]   noisy accelerometer   [ax, ay, az]  m/s²
        [3:6]   noisy gyroscope       [wx, wy, wz]  rad/s
        [6]     left wheel position   [rad]  (absolute, from jointpos)
        [7]     right wheel position  [rad]
        [8]     ideal left wheel vel  [rad/s]
        [9]     ideal right wheel vel [rad/s]
        [10:13] ideal accelerometer
        [13:16] ideal gyroscope
        [16:19] body velocity at FRAME site [vx, vy, vz] in local frame m/s
    """

    # ---------- Physical constants ----------
    WHEEL_RADIUS: float = 0.0625          # m
    WHEEL_TRACK: float  = 0.2982          # m
    MAX_CTRL: float     = 8.775           # rad/s
    MAX_LIN_VEL: float  = MAX_CTRL * WHEEL_RADIUS  # ≈ 0.548 m/s

    # ---------- Sensor sensordata indices ----------
    IDX_ACCEL        = slice(0, 3)
    IDX_GYRO         = slice(3, 6)
    IDX_WHEEL_L_POS  = 6
    IDX_WHEEL_R_POS  = 7
    IDX_WHEEL_L_VEL  = 8    # ideal (no noise)
    IDX_WHEEL_R_VEL  = 9    # ideal (no noise)
    IDX_IDEAL_ACCEL  = slice(10, 13)
    IDX_IDEAL_GYRO   = slice(13, 16)
    IDX_BODY_VEL     = slice(16, 19)  # velocimeter at FRAME, local frame

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        environment_path: str = "./models/scene.xml",
        max_time: float = 20.0,
        max_pitch: float = 0.5,
        frame_skip: int = 10,
        initial_curriculum_phase: int = 0,
        render_mode: str = None,
        randomization_scale: float = 1.0,
        action_filter_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.randomization_scale = randomization_scale  # 0.0 = no randomization (test), 1.0 = full (train)
        # Low-pass filter on applied actions: 0.0 = off, higher = more smoothing.
        # Eliminates high-frequency trembling without reducing step-response reactivity.
        self._action_filter_alpha = action_filter_alpha
        self._filtered_action: np.ndarray = np.zeros(2)
        self.viewer = None
        self._offscreen_renderer = None

        full_path = os.path.abspath(environment_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")

        self.model: MjModel = MjModel.from_xml_path(full_path)
        self.data: MjData  = MjData(self.model)

        self.max_time: float   = max_time
        self.max_pitch: float  = max_pitch
        self.frame_skip: int   = frame_skip
        self.time_step: float  = self.model.opt.timestep * frame_skip

        # Observation space: 10 normalised features (see ObservationWrapper)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # Action space: left and right wheel velocity commands [rad/s]
        ctrl_ranges = self.model.actuator_ctrlrange
        self.action_space = gym.spaces.Box(
            low=ctrl_ranges[:, 0].astype(np.float32),
            high=ctrl_ranges[:, 1].astype(np.float32),
            dtype=np.float32,
        )

        # Madgwick quaternion state – shared with ObservationWrapper
        self.Q: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]

        # Reference generators
        self.pose_control     = PoseControl()
        self.velocity_control = VelocityControl(initial_phase=initial_curriculum_phase)
        self.training: bool   = True

        # Baseline values for domain randomisation (saved once at construction)
        self._initial_masses      = self.model.body_mass.copy()
        self._original_body_ipos  = self.model.body_ipos.copy()

        imu_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "IMU")
        if imu_id < 0:
            raise RuntimeError("IMU body not found in model. Check XML body name.")
        self._imu_id              = imu_id
        self._original_imu_pos    = self.model.body_pos[imu_id].copy()
        self._original_imu_quat   = self.model.body_quat[imu_id].copy()

    # ------------------------------------------------------------------ #
    #  Core Gymnasium API                                                  #
    # ------------------------------------------------------------------ #

    def step(
        self, action: np.ndarray
    ) -> T.Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self._action_filter_alpha > 0.0:
            self._filtered_action = (self._action_filter_alpha * self._filtered_action
                                     + (1.0 - self._action_filter_alpha) * action)
            self.data.ctrl[:] = self._filtered_action
        else:
            self.data.ctrl[:] = action

        if self.training:
            self.velocity_control.update(dt=self.time_step)
            self.pose_control.update(dt=self.time_step)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        terminated = self._is_terminated()
        truncated  = self._is_truncated()

        # Observations and reward are filled in by the wrappers
        return np.zeros(10, dtype=np.float32), 0.0, terminated, truncated, {}

    def reset(
        self,
        seed: T.Optional[int] = None,
        options: T.Optional[dict] = None,
    ) -> T.Tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)
        super().reset(seed=seed)

        # Report episode outcome to curriculum (skip the very first call)
        if self.training and self.data.time > 0:
            # True success = robot survived the full episode without falling
            success = self._is_truncated()
            self.velocity_control.report_episode_result(success)

        mujoco.mj_resetData(self.model, self.data)
        self._filtered_action[:] = 0.0
        self._initialize_random_state()

        return np.zeros(10, dtype=np.float32), {}

    # ------------------------------------------------------------------ #
    #  Termination / truncation                                            #
    # ------------------------------------------------------------------ #

    def _is_terminated(self) -> bool:
        """Robot fell over – pitch angle exceeded the safety limit."""
        return bool(abs(self._get_pitch()) > self.max_pitch)

    def _is_truncated(self) -> bool:
        """Episode time limit reached (robot survived)."""
        return self.data.time >= self.max_time

    def _get_pitch(self) -> float:
        """Pitch angle [rad] from the ideal quaternion in data.qpos."""
        q = self.data.qpos[3:7]
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        return float(r.as_euler("xyz", degrees=False)[1])

    # ------------------------------------------------------------------ #
    #  Episode initialisation & domain randomisation                       #
    # ------------------------------------------------------------------ #

    def _initialize_random_state(self) -> None:
        self._space_positioning()
        self._reset_ctrl()
        self._randomize_masses()
        self._randomize_com()
        self._randomize_imu_pose()
        self._randomize_actuator_gains()
        self._randomize_wheel_friction()
        # Reset reference generators for the new episode
        self.velocity_control.reset()
        self.pose_control.generate_random()  # fully random initial heading each episode

    def _space_positioning(self) -> None:
        """Randomise starting position, pitch and yaw."""
        self.data.qpos[:3] = [
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5),
            0.255,
        ]
        euler = [
            0.0,
            np.random.uniform(-0.1, 0.1),   # small pitch perturbation
            np.random.uniform(-np.pi, np.pi),  # random yaw
        ]
        q_xyzw = R.from_euler("xyz", euler).as_quat()
        q_mj   = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        self.data.qpos[3:7] = q_mj

        # Initialise Madgwick filter from the actual starting orientation
        self.Q = q_mj.copy()

    def _reset_ctrl(self) -> None:
        self.data.ctrl[:] = 0.0
        # Small random initial body velocity to avoid overfitting to the exact upright start
        self.data.qvel[:2] = np.random.uniform(-0.05, 0.05, size=2)
        self.data.qvel[2:] = 0.0

    def _randomize_masses(self) -> None:
        s = self.randomization_scale
        for i in range(self.model.nbody):
            lo = 1.0 - 0.20 * s
            hi = 1.0 + 0.20 * s
            self.model.body_mass[i] = self._initial_masses[i] * np.random.uniform(lo, hi)

    def _randomize_com(self) -> None:
        s = self.randomization_scale
        max_offset = 0.008 * s  # up to ±8mm at full scale, 0 at test
        for i in range(self.model.nbody):
            self.model.body_ipos[i] = (
                self._original_body_ipos[i] + np.random.uniform(-max_offset, max_offset, size=3)
            )

    def _randomize_imu_pose(self) -> None:
        s = self.randomization_scale
        pos_offset = np.random.uniform(-0.005 * s, 0.005 * s, size=3)
        self.model.body_pos[self._imu_id] = self._original_imu_pos + pos_offset

        euler_off = np.random.uniform(-0.01 * s, 0.01 * s, size=3)
        q_off     = R.from_euler("xyz", euler_off).as_quat()
        q_off_mj  = np.array([q_off[3], q_off[0], q_off[1], q_off[2]])
        new_quat  = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(new_quat, self._original_imu_quat, q_off_mj)
        self.model.body_quat[self._imu_id] = new_quat

    def _randomize_actuator_gains(self) -> None:
        s = self.randomization_scale
        for name in ("left_motor", "right_motor"):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            # Independent L/R gains to simulate real motor asymmetry
            self.model.actuator_gainprm[aid][0] = np.random.uniform(
                20.0 - 4.0 * s, 20.0 + 4.0 * s
            )

    def _randomize_wheel_friction(self) -> None:
        s = self.randomization_scale
        # Independent L/R friction to simulate floor/wheel asymmetry
        for name in ("WheelL_collision", "WheelR_collision"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.model.geom_friction[gid] = [
                np.random.uniform(0.90 - 0.09 * s, 0.90 + 0.09 * s),   # sliding
                np.random.uniform(0.050 - 0.005 * s, 0.050 + 0.005 * s),  # torsional
                np.random.uniform(0.0020 - 0.0002 * s, 0.0020 + 0.0002 * s),  # rolling
            ]

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_offscreen()
        else:
            self._render_human()

    def _render_offscreen(self) -> np.ndarray:
        if self._offscreen_renderer is None:
            self._offscreen_renderer = mujoco.Renderer(self.model, height=480, width=640)

        chassis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Chassis")
        cam = mujoco.MjvCamera()
        cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = chassis_id
        cam.distance    = 2.0
        cam.azimuth     = 135.0
        cam.elevation   = -20.0

        self._offscreen_renderer.update_scene(self.data, camera=cam)
        return self._offscreen_renderer.render()

    def _render_human(self) -> None:
        if self.viewer is None:
            self.viewer = launch_passive(self.model, self.data)

        if not self.viewer.is_running():
            return

        self.viewer.user_scn.ngeom = 0

        chassis_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Chassis")
        chassis_pos = self.data.xpos[chassis_id].copy()
        origin      = chassis_pos + np.array([0.0, 0.0, 0.2])

        self.viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = chassis_id

        q   = self.data.qpos[3:7]
        rot = R.from_quat([q[1], q[2], q[3], q[0]])
        _, _, yaw = rot.as_euler("xyz", degrees=False)
        current_fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])

        desired   = self.pose_control.heading
        desired3d = np.array([desired[0], desired[1], 0.0])

        self._render_vector(origin, desired3d,   [0.0, 1.0, 0.0, 1.0], scale=0.5)
        self._render_vector(origin, current_fwd, [1.0, 0.0, 0.0, 0.8], scale=0.5)

        vel_ratio = self.velocity_control.speed / self.MAX_LIN_VEL
        vel_vec   = current_fwd * vel_ratio * 0.4
        self._render_vector(origin, vel_vec, [0.0, 0.0, 1.0, 0.8], scale=1.0)

        self.viewer.sync()
        time.sleep(self.time_step)

    def _render_vector(
        self,
        origin: np.ndarray,
        vector: np.ndarray,
        color: T.List[float],
        scale: float = 1.0,
        radius: float = 0.02,
    ) -> None:
        if self.viewer is None:
            return
        scn = self.viewer.user_scn
        if scn.ngeom >= scn.maxgeom:
            return
        endpoint = origin + vector * scale
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3), np.zeros(3), np.zeros(9),
            np.array(color, dtype=np.float32),
        )
        mujoco.mjv_connector(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_ARROW,
            radius, origin, endpoint,
        )
        scn.ngeom += 1

    def close(self) -> None:
        if self._offscreen_renderer is not None:
            self._offscreen_renderer.close()
            self._offscreen_renderer = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
