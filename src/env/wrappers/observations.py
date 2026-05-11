"""
Observation wrapper for the SelfBalancingRobotEnv.

Converts raw MuJoCo sensor data into a clean, normalised 10-element vector
suitable for policy gradient / actor-critic algorithms.

Observation vector layout:
    [0]  pitch_norm          – Madgwick pitch / MAX_PITCH         ∈ [-1, 1] (≈)
    [1]  pitch_rate_norm     – noisy gyro Y  / GYRO_FSR_RAD      ∈ [-1, 1] (≈)
    [2]  yaw_rate_norm       – noisy gyro Z  / GYRO_FSR_RAD      ∈ [-1, 1] (≈)
    [3]  fwd_vel_norm        – avg wheel vel (m/s) / MAX_LIN_VEL  ∈ [-1, 1] (≈)
    [4]  diff_vel_norm       – diff wheel vel (m/s) / MAX_LIN_VEL ∈ [-1, 1] (≈)
    [5]  target_vel_norm     – velocity setpoint / MAX_LIN_VEL    ∈ [-1, 1]
    [6]  target_yaw_rate_norm– yaw rate setpoint / MAX_YAW_RATE   ∈ [-1, 1]
    [7]  yaw_rate_err_norm   – yaw rate error / MAX_YAW_RATE      ∈ [-1, 1] (≈)
    [8]  prev_act_L_norm     – previous left  motor cmd / MAX_CTRL ∈ [-1, 1]
    [9]  prev_act_R_norm     – previous right motor cmd / MAX_CTRL ∈ [-1, 1]

Sensor usage:
    Observations use NOISY sensors (accelerometer, gyroscope, quantised encoders)
    to simulate real hardware, making policies transferable to physical robots.
    All signals needed to compute the observation are available on real hardware
    from IMU gyroscope (pitch rate, yaw rate) and wheel encoders — no magnetometer
    or external localisation is required.

Madgwick filter:
    The filter is re-seeded from the ideal quaternion at every episode reset
    so the pitch estimate starts from a consistent state.
"""
import numpy as np
import typing as T
import gymnasium as gym
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R

from src.env.robot import SelfBalancingRobotEnv
from src.env.control.yaw_rate_control import MAX_YAW_RATE


# ---------- IMU constants (match real MPU-6050 / ICM-42688 defaults) ----------
GYRO_FSR_DEG: float = 250.0
GYRO_FSR_RAD: float = GYRO_FSR_DEG * np.pi / 180.0  # ≈ 4.363 rad/s

# Encoder quantisation (12-bit encoder → 4096 counts/revolution of the wheel)
ENCODER_RESOLUTION: float = (2.0 * np.pi) / 8192.0  # rad per tick


class ObservationWrapper(gym.Wrapper):
    """Processes raw sensor data into a normalised 10-element observation."""

    def __init__(self, env: SelfBalancingRobotEnv) -> None:
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        self._madgwick = Madgwick(
            frequency=1.0 / self.env.time_step,
            beta=0.033,
        )

        self._prev_wheel_pos: np.ndarray = np.zeros(2)
        self._prev_action: np.ndarray    = np.zeros(2)

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                       #
    # ------------------------------------------------------------------ #

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._get_obs(action)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_wrapper_state()
        obs = self._get_obs(np.zeros(2))
        return obs, info

    # ------------------------------------------------------------------ #
    #  Internal reset                                                      #
    # ------------------------------------------------------------------ #

    def _reset_wrapper_state(self) -> None:
        self.env.Q = self.env.data.qpos[3:7].copy()

        self._prev_wheel_pos[:] = self.env.data.sensordata[
            [self.env.IDX_WHEEL_L_POS, self.env.IDX_WHEEL_R_POS]
        ]

        self._prev_action[:] = 0.0

    # ------------------------------------------------------------------ #
    #  Sensor reading helpers                                              #
    # ------------------------------------------------------------------ #

    def _read_noisy_gyro(self) -> np.ndarray:
        return self.env.data.sensordata[self.env.IDX_GYRO].copy()

    def _read_noisy_accel(self) -> np.ndarray:
        return self.env.data.sensordata[self.env.IDX_ACCEL].copy()

    def _read_quantised_wheel_vel(self) -> np.ndarray:
        """
        Estimate wheel angular velocities [rad/s] from quantised encoder positions.
        """
        raw_pos = self.env.data.sensordata[
            [self.env.IDX_WHEEL_L_POS, self.env.IDX_WHEEL_R_POS]
        ]
        q_pos = np.floor(raw_pos / ENCODER_RESOLUTION) * ENCODER_RESOLUTION
        vel = (q_pos - self._prev_wheel_pos) / self.env.time_step
        self._prev_wheel_pos[:] = q_pos
        return vel

    def _update_madgwick_pitch(self, gyro: np.ndarray, accel: np.ndarray) -> float:
        self.env.Q = self._madgwick.updateIMU(
            self.env.Q,
            gyr=gyro,
            acc=accel,
        )
        q = self.env.Q
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        _, pitch, _ = r.as_euler("xyz", degrees=False)
        return float(pitch)

    # ------------------------------------------------------------------ #
    #  Observation construction                                            #
    # ------------------------------------------------------------------ #

    def _get_obs(self, action: np.ndarray) -> np.ndarray:
        base = self.env

        # --- 1. IMU data (noisy) ---
        gyro  = self._read_noisy_gyro()
        accel = self._read_noisy_accel()

        pitch      = self._update_madgwick_pitch(gyro, accel)
        pitch_rate = float(gyro[1])
        yaw_rate   = float(gyro[2])  # measured yaw rate from noisy gyro

        # --- 2. Wheel velocities (quantised encoders) ---
        wheel_vel = self._read_quantised_wheel_vel()
        fwd_wheel_vel  = (wheel_vel[0] + wheel_vel[1]) * 0.5
        diff_wheel_vel = (wheel_vel[0] - wheel_vel[1])

        fwd_vel_ms  = fwd_wheel_vel  * base.WHEEL_RADIUS
        diff_vel_ms = diff_wheel_vel * base.WHEEL_RADIUS

        # --- 3. Setpoints and errors ---
        target_vel      = base.velocity_control.speed
        target_yaw_rate = base.yaw_rate_control.rate
        yaw_rate_error  = base.yaw_rate_control.error(yaw_rate)

        # --- 4. Normalise ---
        obs = np.array([
            pitch            / base.max_pitch,           # [0] balance state
            pitch_rate       / GYRO_FSR_RAD,             # [1] pitch dynamics
            yaw_rate         / GYRO_FSR_RAD,             # [2] measured turn rate
            fwd_vel_ms       / base.MAX_LIN_VEL,         # [3] current forward speed
            diff_vel_ms      / (2.0 * base.MAX_LIN_VEL), # [4] differential (turning)
            target_vel       / base.MAX_LIN_VEL,         # [5] desired forward speed
            target_yaw_rate  / MAX_YAW_RATE,             # [6] desired yaw rate
            yaw_rate_error   / MAX_YAW_RATE,             # [7] yaw rate error
            self._prev_action[0] / base.MAX_CTRL,        # [8] previous left command
            self._prev_action[1] / base.MAX_CTRL,        # [9] previous right command
        ], dtype=np.float32)

        self._prev_action[:] = action
        return obs
