"""
Yaw-rate reference generator for the self-balancing robot.

Generates target yaw-rate commands [rad/s] using a timer-based hold strategy.
The target is held for a random interval then a new target is sampled.

Replaces the absolute-heading PoseControl: instead of commanding a world-frame
compass direction (which needs a magnetometer on real hardware), we command a
turn rate directly measurable from the gyroscope Z axis — no external reference.

Physical limit: when one wheel drives full forward and the other full reverse,
  yaw_rate_max = MAX_LIN_VEL / (WHEEL_TRACK / 2) = 0.548 / 0.149 ≈ 3.68 rad/s
Practical limit set to 2.0 rad/s to leave margin for forward speed.
"""
import numpy as np
import typing as T
from src.env.control.base_control import BaseControl

MAX_YAW_RATE: float = 2.0  # rad/s – practical turning speed limit


class YawRateControl(BaseControl):
    """
    Generates target yaw-rate references in rad/s using a timer-based hold strategy.

    The target is held for [MIN_HOLD_TIME, MAX_HOLD_TIME] seconds, then a new
    target is sampled uniformly from [-MAX_YAW_RATE, +MAX_YAW_RATE].
    Positive rate = CCW (turn left), negative = CW (turn right).
    """

    ZERO_PROBABILITY: float = 0.30  # chance of commanding zero yaw rate (go straight)
    MIN_HOLD_TIME:    float = 2.0   # s
    MAX_HOLD_TIME:    float = 5.0   # s

    def __init__(self) -> None:
        super().__init__()
        self._rate: float = 0.0
        self._timer: float = 0.0
        self._time_to_switch: float = 0.0

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def rate(self) -> float:
        """Current target yaw rate [rad/s]."""
        return self._rate

    @rate.setter
    def rate(self, value: float) -> None:
        """Override rate directly (interactive / test control)."""
        self._rate = float(np.clip(value, -MAX_YAW_RATE, MAX_YAW_RATE))

    # ------------------------------------------------------------------ #
    #  Update / generation                                                 #
    # ------------------------------------------------------------------ #

    def update(self, dt: float) -> None:
        self._timer += dt
        if self._timer >= self._time_to_switch:
            self._pick_new_target()
            self._reset_timer()

    def _pick_new_target(self) -> None:
        if np.random.random() < self.ZERO_PROBABILITY:
            self._rate = 0.0
        else:
            self._rate = float(np.random.uniform(-MAX_YAW_RATE, MAX_YAW_RATE))

    def _reset_timer(self) -> None:
        self._timer = 0.0
        self._time_to_switch = float(
            np.random.uniform(self.MIN_HOLD_TIME, self.MAX_HOLD_TIME)
        )

    def generate_random(self) -> float:
        self._pick_new_target()
        self._reset_timer()
        return self._rate

    # ------------------------------------------------------------------ #
    #  Error / reset                                                       #
    # ------------------------------------------------------------------ #

    def error(self, current_rate: float) -> float:
        """Yaw-rate tracking error [rad/s]: positive = spinning too slow CCW."""
        return self._rate - current_rate

    def reset(self) -> None:
        self.generate_random()

    def __repr__(self) -> str:
        return (
            f"YawRateControl(rate={self._rate:.3f} rad/s, "
            f"timer={self._timer:.1f}/{self._time_to_switch:.1f}s)"
        )
