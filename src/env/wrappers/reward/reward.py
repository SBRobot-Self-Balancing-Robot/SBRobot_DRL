"""
Reward wrapper for the SelfBalancingRobotEnv.

Per-step reward structure:
    r_balance  = exp(-((pitch - K·vel_error) / σ_pitch)²)   ∈ [0, 1]
    r_velocity = exp(-(vel_error / σ_vel)²)                  ∈ [0, 1]
    r_heading  = (1 + cos(heading_error)) / 2                ∈ [0, 1]
    r_task     = r_balance × (w_b + w_v·r_velocity·r_heading + w_h·r_heading)
    r_smooth   = −w_smooth × mean((Δaction / MAX_CTRL)²)
    r_yaw      = −w_yaw × heading_aligned × (yaw_rate / GYRO_FSR)²

    reward = r_task + r_smooth + r_yaw      (fall → −20)

Key design choices:
  - Balance gates everything: no reward for tracking while tilted.
  - Pitch reference follows velocity error: robot leans forward to accelerate.
  - Cosine kernel for heading: non-zero gradient at all error magnitudes.
  - Velocity reward gated by heading alignment: robot is not rewarded for
    going fast in the wrong direction — forces turn-first behaviour.
  - Yaw penalty suppressed when heading error > 90°: robot can rotate freely
    to correct direction without being penalised.
  - Adaptive velocity sigma: tight near zero setpoint to prevent drift,
    wide during speed tracking to maintain gradient at high speeds.
"""
import numpy as np
import typing as T
import gymnasium as gym
from scipy.spatial.transform import Rotation as R

from src.env.robot import SelfBalancingRobotEnv


# ──────────────────────────────────────────────────────────────────────────────
#  Reward hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────

SIGMA_PITCH:     float = 0.15   # rad  – balance kernel width
SIGMA_VEL:       float = 0.35   # m/s  – velocity tracking kernel (wide for high speed)
SIGMA_VEL_ZERO:  float = 0.12   # m/s  – tight kernel when target ≈ 0 (prevents drift)

# Pitch equilibrium gain: robot leans forward proportionally to velocity error.
# At 0.5 m/s error → 0.12 rad (≈7°) lean.
PITCH_VEL_GAIN: float = 0.24

# Component weights
W_BALANCE:  float = 1.0
W_VELOCITY: float = 2.0
W_HEADING:  float = 2.5
W_SMOOTH:   float = 0.15
W_YAW:      float = 0.10

FALL_PENALTY: float = -20.0


# ──────────────────────────────────────────────────────────────────────────────
#  Reward wrapper
# ──────────────────────────────────────────────────────────────────────────────

class RewardWrapper(gym.Wrapper):
    """Computes shaped per-step reward and applies fall penalty."""

    def __init__(self, env) -> None:
        super().__init__(env)
        self._calculator = RewardCalculator()
        self._prev_action: np.ndarray = np.zeros(2)

    def step(self, action: np.ndarray):
        obs, _, terminated, truncated, info = self.env.step(action)

        delta_action = action - self._prev_action
        self._prev_action = action.copy()

        if terminated:
            reward = FALL_PENALTY
        else:
            reward = self._calculator.compute(self._base_env, delta_action)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action[:] = 0.0
        return obs, info

    @property
    def _base_env(self) -> SelfBalancingRobotEnv:
        e = self.env
        while hasattr(e, "env"):
            e = e.env
        return e  # type: ignore[return-value]

    @property
    def reward_calculator(self) -> "RewardCalculator":
        return self._calculator


# ──────────────────────────────────────────────────────────────────────────────
#  Reward calculator (stateless)
# ──────────────────────────────────────────────────────────────────────────────

class RewardCalculator:

    def __init__(self) -> None:
        self.w_balance:      float = W_BALANCE
        self.w_velocity:     float = W_VELOCITY
        self.w_heading:      float = W_HEADING
        self.w_smooth:       float = W_SMOOTH
        self.w_yaw:          float = W_YAW
        self.sigma_pitch:    float = SIGMA_PITCH
        self.sigma_vel:      float = SIGMA_VEL
        self.sigma_vel_zero: float = SIGMA_VEL_ZERO
        self.pitch_vel_gain: float = PITCH_VEL_GAIN

    def compute(self, env: SelfBalancingRobotEnv, delta_action: np.ndarray) -> float:
        pitch         = self._ideal_pitch(env)
        vel_error     = self._ideal_velocity_error(env)
        heading_error = self._ideal_heading_error(env)

        pitch_ref = self.pitch_vel_gain * vel_error
        r_balance = self._gaussian_kernel(pitch - pitch_ref, self.sigma_pitch)

        # Adaptive sigma: tight at zero setpoint to prevent drift
        target_vel = env.velocity_control.speed
        sigma      = self.sigma_vel_zero if abs(target_vel) < 0.05 else self.sigma_vel
        r_velocity = self._gaussian_kernel(vel_error, sigma)
        r_heading  = self._cosine_kernel(heading_error)

        # Velocity reward gated by heading: not rewarded for going fast the wrong way
        r_velocity_gated = r_velocity * r_heading

        # Smoothness penalty
        r_smooth = -self.w_smooth * float(np.mean((delta_action / env.MAX_CTRL) ** 2))

        # Yaw penalty suppressed when heading error > 90° so robot can turn freely
        heading_aligned = max(0.0, 1.0 - abs(heading_error) / (np.pi / 2))
        ideal_yaw_rate  = float(env.data.sensordata[env.IDX_IDEAL_GYRO][2])
        r_yaw = -self.w_yaw * heading_aligned * (ideal_yaw_rate / 4.363) ** 2

        r_task = r_balance * (self.w_balance + self.w_velocity * r_velocity_gated + self.w_heading * r_heading)
        return float(r_task + r_smooth + r_yaw)

    @staticmethod
    def _ideal_pitch(env: SelfBalancingRobotEnv) -> float:
        q = env.data.qpos[3:7]
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        return float(r.as_euler("xyz", degrees=False)[1])

    @staticmethod
    def _ideal_velocity_error(env: SelfBalancingRobotEnv) -> float:
        left_vel  = float(env.data.sensordata[env.IDX_WHEEL_L_VEL])
        right_vel = float(env.data.sensordata[env.IDX_WHEEL_R_VEL])
        ideal_fwd_vel = (left_vel + right_vel) * 0.5 * env.WHEEL_RADIUS
        return env.velocity_control.error(ideal_fwd_vel)

    @staticmethod
    def _ideal_heading_error(env: SelfBalancingRobotEnv) -> float:
        return env.pose_control.error_with_quaternion(env.data.qpos[3:7])

    @staticmethod
    def _gaussian_kernel(x: float, sigma: float) -> float:
        return float(np.exp(-((x / sigma) ** 2)))

    @staticmethod
    def _cosine_kernel(angle: float) -> float:
        return float((1.0 + np.cos(angle)) / 2.0)
