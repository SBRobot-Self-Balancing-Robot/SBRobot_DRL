import typing as T
import numpy as np
from src.env.control.base_control import BaseControl

class VelocityControl(BaseControl):
    """
    This class handles the random generation of the desired linear velocity for the robot to follow.
    Supports curriculum learning by progressively widening the speed range during training.

    Curriculum advancement is based on **cross-episode success rate**:
    the robot must survive (reach max_time without falling) consistently
    before the velocity range is widened. This ensures Phase 0 (balance only)
    actually teaches stability before introducing velocity commands.
    """
    
    # Default ranges for random generation
    DEFAULT_SPEED_RANGE: T.Tuple[float, float] = (-3, 3) # rad/s

    # Curriculum settings
    CURRICULUM_PHASES: T.List[T.Tuple[float, float]] = [
        (0.0,  0.0),     # Phase 0: No velocity command (balance only)
        (-0.3, 0.3),     # Phase 1: Slow movements
        (-0.7, 0.7),     # Phase 2: Medium movements
        (-1.5, 1.5),     # Phase 3: Full range
    ]

    # Curriculum advancement settings (cross-episode)
    CURRICULUM_WINDOW: int = 50                          # Number of recent episodes to evaluate
    CURRICULUM_SUCCESS_THRESHOLD: float = 0.8            # Required success rate to advance (80%)

    # Incremental update settings (intra-episode)
    DEFAULT_DELTA_RANGE: T.Tuple[float, float] = (-0.15, 0.15)  # Max incremental change per update

    def __init__(self):
        super().__init__()
        self._speed = 0.0                               # Desired speed (rad/s)
        self._curriculum_phase: int = 0                  # Current curriculum phase index
        self._hold_steps: int = 0                        # Counter: steps the current target has been held
        self._min_hold_steps: int = 50                   # Minimum steps before the target can change
        self._velocity_error_threshold: float = 0.1      # Error threshold to consider target "reached"
        self._consecutive_good_steps: int = 0            # Steps with error below threshold
        self._good_steps_required: int = 20              # Consecutive good steps required to allow change
        self._episode_results: T.List[bool] = []         # Rolling window of episode outcomes (True=success)

    # ------------------------------------------------------------------ #
    #                         PROPERTIES                                  #
    # ------------------------------------------------------------------ #

    @property
    def speed(self) -> float:
        """Get the desired linear speed (rad/s)."""
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        """Set the desired linear speed (rad/s)."""
        self._speed = float(value)

    @property
    def curriculum_phase(self) -> int:
        """Get the current curriculum phase index."""
        return self._curriculum_phase

    @curriculum_phase.setter
    def curriculum_phase(self, value: int) -> None:
        """Set the curriculum phase (clamped to valid range)."""
        self._curriculum_phase = int(np.clip(value, 0, len(self.CURRICULUM_PHASES) - 1))

    @property
    def current_range(self) -> T.Tuple[float, float]:
        """Get the speed range for the current curriculum phase."""
        return self.CURRICULUM_PHASES[self._curriculum_phase]

    @property
    def hold_steps(self) -> int:
        """Get the number of steps the current target has been held."""
        return self._hold_steps

    @property
    def is_ready_to_change(self) -> bool:
        """Check if enough hold time has passed AND the target has been tracked well."""
        return (self._hold_steps >= self._min_hold_steps and 
                self._consecutive_good_steps >= self._good_steps_required)

    @property
    def success_rate(self) -> float:
        """Success rate over the recent episode window."""
        if not self._episode_results:
            return 0.0
        return sum(self._episode_results) / len(self._episode_results)

    # ------------------------------------------------------------------ #
    #                    UPDATE / GENERATION METHODS                      #
    # ------------------------------------------------------------------ #
    
    def update(self, current_speed: float = 0.0, heading_error: float = 0.0) -> None:
        """
        Incremental update of the velocity target within the current episode.

        - In Phase 0 (balance only): speed is forced to 0, no perturbations.
        - In later phases: tracks hold time and consecutive "good" steps,
          then applies small perturbations within the current phase range.
        - Curriculum advancement is NOT done here; it is driven by
          cross-episode success rate via `report_episode_result()`.

        Args:
            current_speed: The current average wheel speed (rad/s).
            heading_error: The current heading error (radians), used for coupling.
        """
        self._hold_steps += 1

        # Phase 0: balance only — force speed to 0, skip everything else
        if self._curriculum_phase == 0:
            self._speed = 0.0
            return

        # Track consecutive steps with small velocity error
        vel_error = abs(self.error(current_speed))
        if vel_error < self._velocity_error_threshold:
            self._consecutive_good_steps += 1
        else:
            self._consecutive_good_steps = 0

        # Attenuate speed when heading error is large (coupling)
        if abs(heading_error) > 0.3:
            self._speed *= 0.95

        # If ready, apply a small incremental change (no curriculum advance)
        if self.is_ready_to_change:
            self._apply_incremental_change()
            self._hold_steps = 0
            self._consecutive_good_steps = 0

    def _apply_incremental_change(self) -> None:
        """
        Apply a small random perturbation to the current speed target,
        clamped within the current curriculum phase range.
        """
        low, high = self.current_range
        # If phase 0 (balance only), keep speed at 0
        if low == 0.0 and high == 0.0:
            self._speed = 0.0
            return

        delta = float(np.random.uniform(*self.DEFAULT_DELTA_RANGE))
        new_speed = self._speed + delta
        self._speed = float(np.clip(new_speed, low, high))

    def generate_random(self,
            low: T.Optional[float] = None,
            high: T.Optional[float] = None) -> float:
        """
        Generate a random speed uniformly sampled within [low, high].
        If not specified, uses the current curriculum phase range.

        Args:
            low:  Minimum speed (default: current phase low).
            high: Maximum speed (default: current phase high).

        Returns:
            float: The new desired speed.
        """
        phase_low, phase_high = self.current_range
        low = low if low is not None else phase_low
        high = high if high is not None else phase_high
        self._speed = float(np.random.uniform(low, high))
        self._hold_steps = 0
        self._consecutive_good_steps = 0
        return self._speed

    def report_episode_result(self, success: bool) -> None:
        """
        Report whether the last episode was successful (survived full duration).
        Called by the environment at each reset.

        When the success rate over the last CURRICULUM_WINDOW episodes exceeds
        CURRICULUM_SUCCESS_THRESHOLD, the curriculum advances to the next phase
        and the episode history is cleared to start fresh evaluation.

        Args:
            success: True if the episode reached max_time without falling.
        """
        self._episode_results.append(success)
        # Keep only the latest window
        if len(self._episode_results) > self.CURRICULUM_WINDOW:
            self._episode_results = self._episode_results[-self.CURRICULUM_WINDOW:]

        # Advance only when the window is full and success rate is high enough
        if len(self._episode_results) >= self.CURRICULUM_WINDOW:
            if self.success_rate >= self.CURRICULUM_SUCCESS_THRESHOLD:
                self._advance_curriculum()
                self._episode_results.clear()  # Reset tracking for new phase

    def _advance_curriculum(self) -> int:
        """
        Advance to the next curriculum phase (if available).
        Internal method — advancement is driven by `report_episode_result()`.
        
        Returns:
            int: The new curriculum phase index.
        """
        if self._curriculum_phase < len(self.CURRICULUM_PHASES) - 1:
            self._curriculum_phase += 1
        return self._curriculum_phase

    # ------------------------------------------------------------------ #
    #                        ERROR METHODS                                #
    # ------------------------------------------------------------------ #

    def error(self, current_speed: float) -> float:
        """
        Compute the speed error between the desired speed and the current speed.

        Args:
            current_speed: The current speed of the robot (rad/s).
        Returns:
            float: The speed error (desired - current).
        """
        return self._speed - current_speed
    
    # ------------------------------------------------------------------ #
    #                        RESET                                        #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset velocity control state (keeps curriculum phase and episode history)."""
        self._speed = 0.0
        self._hold_steps = 0
        self._consecutive_good_steps = 0

    def full_reset(self) -> None:
        """Reset everything, including curriculum phase and episode history."""
        self.reset()
        self._curriculum_phase = 0
        self._episode_results.clear()
        
    # ------------------------------------------------------------------ #
    #                         DUNDER METHODS                             #
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        phase_range = self.current_range
        return (f"VelocityControl(speed={self._speed:.2f} rad/s, "
                f"phase={self._curriculum_phase} [{phase_range[0]:.1f}, {phase_range[1]:.1f}], "
                f"hold={self._hold_steps})")