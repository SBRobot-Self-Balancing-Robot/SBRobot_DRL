"""
Reward wrapper for the SelfBalancingRobotEnv.

Design philosophy (state-of-the-art for Segway-type robots):
  • Balance is the primary objective: the robot cannot execute any task if it falls.
  • Task rewards (velocity + heading) are gated by balance quality so the agent is
    not rewarded for tracking references while lying on the floor.
  • Balance and velocity use Gaussian (RBF) kernels for tight precision incentive.
  • Heading uses a COSINE kernel: (1 + cos(error)) / 2.
    Unlike a Gaussian, the cosine kernel gives a non-zero gradient at ALL heading
    errors, including large ones (90°, 180°). A Gaussian with σ=0.30 rad collapses
    to ≈0 at 45° — giving the policy zero gradient signal when far off-heading.
    The cosine kernel ensures the policy always receives a useful learning signal.
  • Ideal sensors are used for reward computation (privileged information):
    - pitch from ideal quaternion (data.qpos)
    - forward velocity from ideal wheel-velocity sensors (sensordata[8:10])
    - heading from ideal quaternion
  • A fall terminates the episode with a fixed penalty.

Per-step reward structure:
    r_balance  = exp(-(pitch / σ_p)²)                  ∈ [0, 1]
    r_velocity = exp(-(vel_error / σ_v)²)               ∈ [0, 1]
    r_heading  = (1 + cos(heading_error)) / 2           ∈ [0, 1]
    r_task     = r_balance × (w_b + w_v·r_velocity + w_h·r_heading)
    r_smooth   = -w_s × mean((Δaction / MAX_CTRL)²)    ≤ 0

    reward = r_task + r_smooth

    On fall (terminated):
        reward = FALL_PENALTY  (replaces per-step reward)

Per-step reward range: [−0.05, ~4.0] when balanced and tracking well.
Fall penalty: −20  (5× per-step maximum → strong negative incentive).

Weight breakdown at perfect tracking (r_balance=r_vel=r_heading=1):
    r_task = 1 × (1 + 1×1 + 2×1) = 4.0
    Heading only (r_vel=0):  1 × (1 + 0 + 2) = 3.0
    Velocity only (r_heading=0): 1 × (1 + 1 + 0) = 2.0
    Balance only (both=0):   1 × 1 = 1.0
"""
import numpy as np
import typing as T
import gymnasium as gym
from scipy.spatial.transform import Rotation as R

from src.env.robot import SelfBalancingRobotEnv


# ──────────────────────────────────────────────────────────────────────────────
#  Reward hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────

# Gaussian kernel widths (σ values) – only used for balance and velocity
SIGMA_PITCH: float = 0.15  # rad  – half-max at ≈ 8.6°; tight to incentivise balance
SIGMA_VEL:   float = 0.35  # m/s  – covers full phase-3 range (±0.50 m/s).
                            # σ=0.25 collapses to ~0 gradient at phase-3 max (0.50 m/s);
                            # σ=0.35 keeps gradient = -1.17 there.

# Component weights
W_BALANCE:  float = 1.0   # survival bonus: incentivises staying upright
W_VELOCITY: float = 1.5   # velocity tracking
W_HEADING:  float = 2.0   # heading tracking
W_SMOOTH:   float = 0.15  # increased from 0.05 – penalises oscillations at high speed

# Terminal reward
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
    """
    Computes the per-step shaped reward using ideal sensor data.

    Attributes can be adjusted externally for ablations / hyper-parameter sweeps.
    """

    def __init__(self) -> None:
        self.w_balance:  float = W_BALANCE
        self.w_velocity: float = W_VELOCITY
        self.w_heading:  float = W_HEADING
        self.w_smooth:   float = W_SMOOTH
        self.sigma_pitch: float = SIGMA_PITCH
        self.sigma_vel:   float = SIGMA_VEL

    def compute(self, env: SelfBalancingRobotEnv, delta_action: np.ndarray) -> float:
        pitch         = self._ideal_pitch(env)
        vel_error     = self._ideal_velocity_error(env)
        heading_error = self._ideal_heading_error(env)

        # Balance and velocity: Gaussian kernels (tight precision incentive)
        r_balance  = self._gaussian_kernel(pitch,     self.sigma_pitch)
        r_velocity = self._gaussian_kernel(vel_error, self.sigma_vel)

        # Heading: cosine kernel — gives non-zero gradient at ALL error magnitudes.
        # A Gaussian with σ=0.30 rad gives ~0 gradient beyond 45° error, making
        # the policy blind to large heading deviations. Cosine has gradient everywhere.
        r_heading = self._cosine_kernel(heading_error)

        # Action-smoothness penalty
        r_smooth = -self.w_smooth * float(np.mean((delta_action / env.MAX_CTRL) ** 2))

        # Balance gates everything: no reward for tracking while tilted
        r_task = r_balance * (self.w_balance + self.w_velocity * r_velocity + self.w_heading * r_heading)
        return float(r_task + r_smooth)

    # ------------------------------------------------------------------ #
    #  Ideal-sensor helpers                                                #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Reward kernels                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _gaussian_kernel(x: float, sigma: float) -> float:
        """Gaussian (RBF) kernel: exp(-(x/σ)²). Returns 1 at x=0, →0 for |x|≫σ."""
        return float(np.exp(-((x / sigma) ** 2)))

    @staticmethod
    def _cosine_kernel(angle: float) -> float:
        """
        Cosine kernel: (1 + cos(angle)) / 2.

        Returns 1 at angle=0, 0.5 at ±90°, 0 at ±180°.
        Unlike a Gaussian, it provides non-zero gradient at all error magnitudes,
        which is critical for heading tracking where initial errors can be ≫45°.
        """
        return float((1.0 + np.cos(angle)) / 2.0)
