import typing as T
import numpy as np
from src.env.control.base_control import BaseControl

class VelocityControl(BaseControl):
    """
    Handles the generation of the desired linear velocity for the robot.
    
    STRATEGY: "Time-based Step Function with Zero Bias"
    - Changes target based on time intervals, not performance.
    - Uses a high probability of selecting 0.0 speed to reinforce static balancing.
    - Uses Curriculum Learning to progressively widen the max speed range.
    """
    
    # Curriculum settings: Ranges for random generation [min, max]
    # NOTE: Phase 0 is no longer (0,0). We start with small movements allowed.
    # The static balance is now handled by ZERO_PROBABILITY.
    CURRICULUM_PHASES: T.List[T.Tuple[float, float]] = [
        (-0.5, 0.5),     # Phase 0: Slow movements
        (-1.0, 1.0),     # Phase 1: Medium movements
        (-2.0, 2.0),     # Phase 2: Fast movements
        (-3.0, 3.0),     # Phase 3: Full range
    ]

    # Curriculum advancement settings (cross-episode)
    CURRICULUM_WINDOW: int = 50                          
    CURRICULUM_SUCCESS_THRESHOLD: float = 0.8            

    # Update settings
    ZERO_PROBABILITY: float = 0.4  # 40% chance to pick exactly 0.0 as target
    MIN_HOLD_TIME: float = 2.0     # Min seconds to hold a target
    MAX_HOLD_TIME: float = 5.0     # Max seconds to hold a target

    def __init__(self):
        super().__init__()
        self._speed = 0.0                               
        self._curriculum_phase: int = 0                  
        self._episode_results: T.List[bool] = []         
        
        # Timer variables
        self._timer: float = 0.0
        self._time_to_switch: float = 0.0

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
    def success_rate(self) -> float:
        """Success rate over the recent episode window."""
        if not self._episode_results:
            return 0.0
        return sum(self._episode_results) / len(self._episode_results)

    # ------------------------------------------------------------------ #
    #                    UPDATE / GENERATION METHODS                      #
    # ------------------------------------------------------------------ #
    
    def update(self, dt: float, current_speed: float = 0.0, heading_error: float = 0.0) -> None:
        """
        Updates the target velocity based on a timer.
        
        Args:
            dt: Time step duration (seconds).
            current_speed: (Unused in this strategy, kept for interface compatibility)
            heading_error: (Unused in this strategy, kept for interface compatibility)
        """
        self._timer += dt

        # Time to switch target?
        if self._timer >= self._time_to_switch:
            self._generate_next_target()
            self._reset_timer()

    def _generate_next_target(self) -> None:
        """
        Generates a new target speed with a bias towards zero.
        """
        low, high = self.current_range
        
        # Roll a die to see if we should stop (Zero Bias)
        if np.random.random() < self.ZERO_PROBABILITY:
            self._speed = 0.0
        else:
            # Otherwise, pick a random speed in the current curriculum range
            self._speed = float(np.random.uniform(low, high))

    def _reset_timer(self) -> None:
        """Resets the timer with a random duration."""
        self._timer = 0.0
        self._time_to_switch = np.random.uniform(self.MIN_HOLD_TIME, self.MAX_HOLD_TIME)

    def generate_random(self,
            low: T.Optional[float] = None,
            high: T.Optional[float] = None) -> float:
        """
        Force a random target generation immediately.
        Arguments low/high are mostly ignored by internal logic in favor of curriculum,
        but kept for compatibility if manual override is needed.
        """
        if low is not None and high is not None:
             self._speed = float(np.random.uniform(low, high))
        else:
            self._generate_next_target()
            
        self._reset_timer()
        return self._speed

    # ------------------------------------------------------------------ #
    #                  CURRICULUM & RESET METHODS                         #
    # ------------------------------------------------------------------ #

    def report_episode_result(self, success: bool) -> None:
        """
        Report whether the last episode was successful.
        Updates curriculum based on success rate.
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
        """Advance to the next curriculum phase (if available)."""
        if self._curriculum_phase < len(self.CURRICULUM_PHASES) - 1:
            self._curriculum_phase += 1
            # Optional: print debug info
            # print(f"*** CURRICULUM ADVANCED TO PHASE {self._curriculum_phase}: Range {self.current_range} ***")
        return self._curriculum_phase

    def error(self, current_speed: float) -> float:
        """Compute the speed error."""
        return self._speed - current_speed
    
    def reset(self) -> None:
        """
        Reset for new episode. 
        IMPORTANT: Immediately pick a target (maybe 0, maybe moving)
        to prevent 'start-from-zero' overfitting.
        """
        self.generate_random()

    def full_reset(self) -> None:
        """Reset everything, including curriculum phase and episode history."""
        self.reset()
        self._curriculum_phase = 0
        self._episode_results.clear()
        
    def __repr__(self) -> str:
        phase_range = self.current_range
        return (f"VelocityControl(speed={self._speed:.2f}, "
                f"timer={self._timer:.1f}/{self._time_to_switch:.1f}, "
                f"phase={self._curriculum_phase} {phase_range})")