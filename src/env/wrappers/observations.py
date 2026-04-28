"""
Observation wrapper for the SelfBalancingRobotEnv.

Converts raw MuJoCo sensor data into a clean, normalised 10-element vector
suitable for policy gradient / actor-critic algorithms.

Observation vector layout:
    [0]  pitch_norm       – Madgwick pitch / MAX_PITCH      ∈ [-1, 1] (≈)
    [1]  pitch_rate_norm  – noisy gyro Y  / GYRO_FSR_RAD   ∈ [-1, 1] (≈)
    [2]  yaw_rate_norm    – noisy gyro Z  / GYRO_FSR_RAD   ∈ [-1, 1] (≈)
    [3]  fwd_vel_norm     – avg wheel vel (m/s) / MAX_LIN_VEL ∈ [-1, 1] (≈)
    [4]  diff_vel_norm    – diff wheel vel (m/s) / MAX_LIN_VEL ∈ [-1, 1] (≈)
    [5]  target_vel_norm  – velocity setpoint / MAX_LIN_VEL  ∈ [-1, 1]
    [6]  heading_err_norm – signed heading error (rad) / π   ∈ [-1, 1]
    [7]  vel_err_norm     – velocity error (m/s) / MAX_LIN_VEL ∈ [-1, 1] (≈)
    [8]  prev_act_L_norm  – previous left  motor cmd / MAX_CTRL ∈ [-1, 1]
    [9]  prev_act_R_norm  – previous right motor cmd / MAX_CTRL ∈ [-1, 1]

Sensor usage:
    Observations use NOISY sensors (accelerometer, gyroscope, quantised encoders)
    to simulate real hardware, making policies transferable to physical robots.
    Heading error uses the ideal quaternion (data.qpos[3:7]) as a proxy for a
    compass/visual-odometry heading estimate.

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


# ---------- IMU constants (match real MPU-6050 / ICM-42688 defaults) ----------
GYRO_FSR_DEG: float = 250.0                    # °/s full-scale range
GYRO_FSR_RAD: float = GYRO_FSR_DEG * np.pi / 180.0  # ≈ 4.363 rad/s

# Encoder quantisation (12-bit encoder → 4096 counts/revolution of the wheel)
ENCODER_RESOLUTION: float = (2.0 * np.pi) / 8192.0  # rad per tick


class ObservationWrapper(gym.Wrapper):
    """Processes raw sensor data into a normalised 10-element observation."""

    def __init__(self, env: SelfBalancingRobotEnv) -> None:
        super().__init__(env)

        # Override the base env observation space with the exact shape we return
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # Madgwick orientation filter (IMU fusion for pitch/roll/yaw estimate)
        self._madgwick = Madgwick(
            frequency=1.0 / self.env.time_step,  # control frequency [Hz]
            beta=0.033,                           # filter gain (gyro-trust parameter)
        )

        # Internal state
        self._prev_wheel_pos: np.ndarray = np.zeros(2)  # for encoder differentiation
        self._prev_action: np.ndarray    = np.zeros(2)  # for smoothness observation

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                       #
    # ------------------------------------------------------------------ #

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._get_obs(action)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Reset wrapper state BEFORE building the first observation
        self._reset_wrapper_state()
        obs = self._get_obs(np.zeros(2))
        return obs, info

    # ------------------------------------------------------------------ #
    #  Internal reset                                                      #
    # ------------------------------------------------------------------ #

    def _reset_wrapper_state(self) -> None:
        """Reset all per-episode wrapper state."""
        # Re-seed Madgwick from the actual starting quaternion so pitch starts clean
        self.env.Q = self.env.data.qpos[3:7].copy()  # [w, x, y, z]

        # Reset encoder baseline so the first velocity estimate is zero
        self._prev_wheel_pos[:] = self.env.data.sensordata[
            [self.env.IDX_WHEEL_L_POS, self.env.IDX_WHEEL_R_POS]
        ]

        # Reset previous action
        self._prev_action[:] = 0.0

    # ------------------------------------------------------------------ #
    #  Sensor reading helpers                                              #
    # ------------------------------------------------------------------ #

    def _read_noisy_gyro(self) -> np.ndarray:
        """Return angular velocity [wx, wy, wz] rad/s from the noisy gyroscope."""
        return self.env.data.sensordata[self.env.IDX_GYRO].copy()

    def _read_noisy_accel(self) -> np.ndarray:
        """Return specific force [ax, ay, az] m/s² from the noisy accelerometer."""
        return self.env.data.sensordata[self.env.IDX_ACCEL].copy()

    def _read_quantised_wheel_vel(self) -> np.ndarray:
        """
        Estimate wheel angular velocities [rad/s] by differentiating quantised
        encoder positions (simulates real-hardware quadrature encoder behaviour).

        Returns [left_vel, right_vel] in rad/s.
        """
        raw_pos = self.env.data.sensordata[
            [self.env.IDX_WHEEL_L_POS, self.env.IDX_WHEEL_R_POS]
        ]
        # Quantise to encoder resolution
        q_pos = np.floor(raw_pos / ENCODER_RESOLUTION) * ENCODER_RESOLUTION
        # Finite-difference velocity estimate
        vel = (q_pos - self._prev_wheel_pos) / self.env.time_step
        self._prev_wheel_pos[:] = q_pos
        return vel

    def _update_madgwick_pitch(self, gyro: np.ndarray, accel: np.ndarray) -> float:
        """
        Run one Madgwick IMU update and return the estimated pitch [rad].

        The Madgwick filter fuses gyroscope and accelerometer to give a
        drift-free pitch estimate (yaw still drifts without magnetometer,
        but pitch is well-conditioned for 10-second episodes).
        """
        self.env.Q = self._madgwick.updateIMU(
            self.env.Q,
            gyr=gyro,    # [wx, wy, wz] rad/s in body frame
            acc=accel,   # [ax, ay, az] m/s² in body frame (includes gravity)
        )
        # Convert [w, x, y, z] → scipy [x, y, z, w]
        q = self.env.Q
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        _, pitch, _ = r.as_euler("xyz", degrees=False)
        return float(pitch)

    # ------------------------------------------------------------------ #
    #  Observation construction                                            #
    # ------------------------------------------------------------------ #

    def _get_obs(self, action: np.ndarray) -> np.ndarray:
        base = self.env  # SelfBalancingRobotEnv

        # --- 1. IMU data (noisy) ---
        gyro  = self._read_noisy_gyro()   # [wx, wy, wz] rad/s
        accel = self._read_noisy_accel()  # [ax, ay, az] m/s²

        # Pitch from Madgwick filter
        pitch = self._update_madgwick_pitch(gyro, accel)

        pitch_rate = float(gyro[1])  # pitch angular rate (body Y axis)
        yaw_rate   = float(gyro[2])  # yaw angular rate  (body Z axis)

        # --- 2. Wheel velocities (quantised encoders, noisy) ---
        wheel_vel = self._read_quantised_wheel_vel()  # [rad/s, rad/s]
        fwd_wheel_vel  = (wheel_vel[0] + wheel_vel[1]) * 0.5  # avg rad/s
        diff_wheel_vel = (wheel_vel[0] - wheel_vel[1])        # differential rad/s

        # Convert to linear speed [m/s]
        fwd_vel_ms  = fwd_wheel_vel  * base.WHEEL_RADIUS
        diff_vel_ms = diff_wheel_vel * base.WHEEL_RADIUS  # max = 2 * MAX_LIN_VEL when spinning in opposite directions

        # --- 3. Setpoints and errors ---
        target_vel = base.velocity_control.speed  # [m/s]
        vel_error  = base.velocity_control.error(fwd_vel_ms)  # [m/s]

        # Heading error: use ideal quaternion (privileged in sim, replaced by
        # compass on real hardware).  Returns signed angle ∈ (-π, π].
        heading_error = base.pose_control.error_with_quaternion(
            base.data.qpos[3:7]
        )

        # --- 4. Normalise all values to approximately [-1, 1] ---
        obs = np.array([
            pitch        / base.max_pitch,          # [0] balance state
            pitch_rate   / GYRO_FSR_RAD,            # [1] pitch dynamics
            yaw_rate     / GYRO_FSR_RAD,            # [2] turning rate
            fwd_vel_ms   / base.MAX_LIN_VEL,          # [3] current forward speed
            diff_vel_ms  / (2.0 * base.MAX_LIN_VEL), # [4] differential (turning) – max ±1 when spinning opposite
            target_vel   / base.MAX_LIN_VEL,        # [5] desired speed
            heading_error / np.pi,                  # [6] signed heading error
            vel_error    / base.MAX_LIN_VEL,        # [7] velocity tracking error
            self._prev_action[0] / base.MAX_CTRL,   # [8] previous left command
            self._prev_action[1] / base.MAX_CTRL,   # [9] previous right command
        ], dtype=np.float32)

        # Store current action for next step's observation
        self._prev_action[:] = action

        return obs
