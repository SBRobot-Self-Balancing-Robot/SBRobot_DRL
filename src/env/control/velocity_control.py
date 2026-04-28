import typing as T
import numpy as np
from src.env.control.base_control import BaseControl

# Physical constant: wheel radius from robot XML (cylinder size="0.0625 ...")
WHEEL_RADIUS: float = 0.0625  # m
MAX_LIN_VEL: float = 8.775 * WHEEL_RADIUS  # ≈ 0.548 m/s (physical speed limit)


class VelocityControl(BaseControl):
    """
    Generates forward velocity references in m/s using a timer-based hold strategy.

    The target speed is held for a random interval [MIN_HOLD_TIME, MAX_HOLD_TIME],
    then a new target is sampled. Curriculum learning progressively expands the speed
    range as the robot demonstrates consistent episode survival.
    """

    # Curriculum speed ranges [m/s]. All values physically achievable (< MAX_LIN_VEL).
    CURRICULUM_PHASES: T.List[T.Tuple[float, float]] = [
        (-0.10, 0.10),  # Phase 0 – very slow, focus on balance with small disturbances
        (-0.20, 0.20),  # Phase 1 – slow
        (-0.35, 0.35),  # Phase 2 – medium
        (-0.50, 0.50),  # Phase 3 – fast (≈ 91% of physical maximum)
    ]

    CURRICULUM_WINDOW: int = 150              # episodes to evaluate for advancement
    CURRICULUM_SUCCESS_THRESHOLD: float = 0.75  # 75% survival rate to advance

    ZERO_PROBABILITY: float = 0.25  # chance of commanding zero speed
    MIN_HOLD_TIME: float = 2.0      # s – minimum time to hold each target
    MAX_HOLD_TIME: float = 5.0      # s – maximum time to hold each target

    def __init__(self, initial_phase: int = 0) -> None:
        super().__init__()
        self._speed: float = 0.0
        self._curriculum_phase: int = max(0, min(initial_phase, len(self.CURRICULUM_PHASES) - 1))
        self._episode_results: T.List[bool] = []
        self._timer: float = 0.0
        self._time_to_switch: float = 0.0

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def speed(self) -> float:
        """Current target forward velocity [m/s]."""
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        """Override speed directly (used for interactive / test control)."""
        self._speed = float(np.clip(value, -MAX_LIN_VEL, MAX_LIN_VEL))

    @property
    def curriculum_phase(self) -> int:
        return self._curriculum_phase

    @property
    def current_range(self) -> T.Tuple[float, float]:
        return self.CURRICULUM_PHASES[self._curriculum_phase]

    @property
    def success_rate(self) -> float:
        if not self._episode_results:
            return 0.0
        return float(sum(self._episode_results)) / len(self._episode_results)

    # ------------------------------------------------------------------ #
    #  Update / generation                                                 #
    # ------------------------------------------------------------------ #

    def update(self, dt: float) -> None:
        """Advance the internal timer; pick a new target when the hold expires."""
        self._timer += dt
        if self._timer >= self._time_to_switch:
            self._pick_new_target()
            self._reset_timer()

    def _pick_new_target(self) -> None:
        if np.random.random() < self.ZERO_PROBABILITY:
            self._speed = 0.0
        else:
            low, high = self.current_range
            self._speed = float(np.random.uniform(low, high))

    def _reset_timer(self) -> None:
        self._timer = 0.0
        self._time_to_switch = float(np.random.uniform(self.MIN_HOLD_TIME, self.MAX_HOLD_TIME))

    def generate_random(self) -> float:
        """Force an immediate target change (used on episode reset)."""
        self._pick_new_target()
        self._reset_timer()
        return self._speed

    # ------------------------------------------------------------------ #
    #  Curriculum                                                          #
    # ------------------------------------------------------------------ #

    def report_episode_result(self, success: bool) -> None:
        """
        Record whether the episode was successful (robot survived to max_time).
        Advances the curriculum phase when the recent success rate is high enough.
        """
        self._episode_results.append(success)
        if len(self._episode_results) > self.CURRICULUM_WINDOW:
            self._episode_results = self._episode_results[-self.CURRICULUM_WINDOW:]

        if len(self._episode_results) >= self.CURRICULUM_WINDOW:
            if self.success_rate >= self.CURRICULUM_SUCCESS_THRESHOLD:
                if self._curriculum_phase < len(self.CURRICULUM_PHASES) - 1:
                    self._curriculum_phase += 1
                    print(
                        f"[VelocityControl] Curriculum → phase {self._curriculum_phase} "
                        f"| range {self.current_range} m/s"
                    )
                self._episode_results.clear()

    # ------------------------------------------------------------------ #
    #  Error / reset                                                       #
    # ------------------------------------------------------------------ #

    def error(self, current_speed: float) -> float:
        """Velocity tracking error [m/s]: positive = robot too slow, negative = too fast."""
        return self._speed - current_speed

    def reset(self) -> None:
        """Reset for a new episode (curriculum phase is preserved)."""
        self.generate_random()

    def full_reset(self) -> None:
        """Reset everything including curriculum phase and episode history."""
        self._curriculum_phase = 0
        self._episode_results.clear()
        self.generate_random()

    def __repr__(self) -> str:
        return (
            f"VelocityControl(speed={self._speed:.3f} m/s, "
            f"timer={self._timer:.1f}/{self._time_to_switch:.1f}s, "
            f"phase={self._curriculum_phase} {self.current_range} m/s)"
        )
