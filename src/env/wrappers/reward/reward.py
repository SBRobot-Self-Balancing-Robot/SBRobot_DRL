"""
Reward wrapper for the SelfBalancingRobotEnv.

Per-step reward structure:
    r_balance  = exp(-((pitch - K·vel_error) / σ_pitch)²)   ∈ [0, 1]
    r_velocity = exp(-(vel_error / σ_vel)²)                  ∈ [0, 1]
    r_yaw      = exp(-(yaw_rate_error / σ_yaw)²)             ∈ [0, 1]
    r_task     = r_balance × (w_b + w_v·r_velocity + w_ω·r_yaw)
    r_smooth   = −w_smooth × mean((Δfiltered_action / MAX_CTRL)²)

    reward = r_task + r_smooth      (fall → −20)

Key design choices:
  - Balance gates everything: no reward for tracking while tilted.
  - Pitch reference follows velocity error: robot leans forward to accelerate.
  - Yaw rate interface: policy tracks a commanded turn rate [rad/s] directly
    measurable from gyroscope — no magnetometer or external heading reference.
  - Adaptive velocity sigma: tight near zero setpoint to prevent drift.
  - Smoothness penalty on filtered (applied) actions, not raw policy output.
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
SIGMA_VEL_ZERO:  float = 0.04   # m/s  – tight kernel when target ≈ 0 (prevents drift)
SIGMA_YAW:       float = 0.5    # rad/s – yaw rate tracking kernel

# Pitch equilibrium gain: robot leans forward proportionally to velocity error.
# At 0.5 m/s error → 0.12 rad (≈7°) lean.
PITCH_VEL_GAIN: float = 0.24

# Component weights
W_BALANCE:  float = 1.0
W_VELOCITY: float = 2.0
W_YAW:      float = 2.0
W_SMOOTH:   float = 0.15

FALL_PENALTY: float = -20.0


# ──────────────────────────────────────────────────────────────────────────────
#  Reward wrapper
# ──────────────────────────────────────────────────────────────────────────────

class RewardWrapper(gym.Wrapper):
    """Computes shaped per-step reward and applies fall penalty."""

    def __init__(self, env) -> None:
        super().__init__(env)
        self._calculator = RewardCalculator()
        self._prev_filtered_action: np.ndarray = np.zeros(2)

    def step(self, action: np.ndarray):
        obs, _, terminated, truncated, info = self.env.step(action)

        # Delta on the filtered (actually applied) action, not raw policy output
        base = self._base_env
        delta_filtered = base._filtered_action - self._prev_filtered_action
        self._prev_filtered_action[:] = base._filtered_action

        if terminated:
            reward = FALL_PENALTY
        else:
            reward = self._calculator.compute(base, delta_filtered)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_filtered_action[:] = 0.0
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
        self.w_yaw:          float = W_YAW
        self.w_smooth:       float = W_SMOOTH
        self.sigma_pitch:    float = SIGMA_PITCH
        self.sigma_vel:      float = SIGMA_VEL
        self.sigma_vel_zero: float = SIGMA_VEL_ZERO
        self.sigma_yaw:      float = SIGMA_YAW
        self.pitch_vel_gain: float = PITCH_VEL_GAIN

    def compute(self, env: SelfBalancingRobotEnv, delta_filtered_action: np.ndarray) -> float:
        pitch     = self._ideal_pitch(env)
        vel_error = self._ideal_velocity_error(env)

        # Ideal yaw rate (no sensor noise) for a clean training signal
        ideal_yaw_rate = float(env.data.sensordata[env.IDX_IDEAL_GYRO][2])
        yaw_rate_error = env.yaw_rate_control.error(ideal_yaw_rate)

        pitch_ref  = self.pitch_vel_gain * vel_error
        r_balance  = self._gaussian_kernel(pitch - pitch_ref, self.sigma_pitch)

        target_vel = env.velocity_control.speed
        sigma      = self.sigma_vel_zero if abs(target_vel) < 0.05 else self.sigma_vel
        r_velocity = self._gaussian_kernel(vel_error, sigma)

        r_yaw = self._gaussian_kernel(yaw_rate_error, self.sigma_yaw)

        r_smooth = -self.w_smooth * float(
            np.mean((delta_filtered_action / env.MAX_CTRL) ** 2)
        )

        r_task = r_balance * (
            self.w_balance + self.w_velocity * r_velocity + self.w_yaw * r_yaw
        )
        return float(r_task + r_smooth)

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
    def _gaussian_kernel(x: float, sigma: float) -> float:
        return float(np.exp(-((x / sigma) ** 2)))
