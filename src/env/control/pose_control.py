import numpy as np
import typing as T
from src.utils.math import signed_sin

class PoseControl:
    """
    This class handles the random generation of the heading vector 
    and the target point for the robot to follow.
    """

    # Default ranges for random generation
    DEFAULT_SPEED_RANGE: T.Tuple[float, float] = (-1.5, 1.5) # rad/s

    def __init__(self):
        self._heading = np.array([0.0, 0.0])          # Unit heading vector (pointing along x-axis)
        self._speed = 0.0                             # Desired speed (m/s)

    # ------------------------------------------------------------------ #
    #                         PROPERTIES                                  #
    # ------------------------------------------------------------------ #

    @property
    def heading(self) -> np.ndarray:
        """Get the heading unit vector (2D)."""
        return self._heading

    @heading.setter
    def heading(self, value: np.ndarray) -> None:
        """Set the heading vector. It will be normalised automatically."""
        value = np.asarray(value, dtype=np.float64)
        norm = np.linalg.norm(value)
        if norm < 1e-8:
            raise ValueError("Heading vector cannot be zero.")
        self._heading = value / norm

    @property
    def heading_angle(self) -> float:
        """Get the heading yaw angle in radians (atan2 of y/x component)."""
        return float(np.arctan2(self._heading[1], self._heading[0]))

    @heading_angle.setter
    def heading_angle(self, versor: np.ndarray) -> None:
        """Set the heading from a yaw angle (radians). Z component is kept at 0."""
        self._heading = np.array([versor[0], versor[1]])

    @property
    def speed(self) -> float:
        """Get the desired linear speed (m/s)."""
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        """Set the desired linear speed (m/s). Must be non-negative."""
        if value < 0.0:
            raise ValueError(f"Speed must be non-negative, got {value}.")
        self._speed = float(value)

    @property
    def velocity_vector(self) -> np.ndarray:
        """Get the velocity vector (heading * speed) in 3D."""
        return self._heading * self._speed

    # ------------------------------------------------------------------ #
    #                    RANDOM GENERATION METHODS                        #
    # ------------------------------------------------------------------ #

    def update_heading(self):
        """
        Update the heading vector based on the current heading angle.
        """
        # Update the heading vector adding or subtracting a random angle between 10 and -10 degrees wrt the current angle
        delta_angle = np.deg2rad(np.random.uniform(-10, 10))
        current_angle = self.heading_angle
        new_angle = current_angle + delta_angle
        self.heading_angle = np.array([np.cos(new_angle), np.sin(new_angle)])

    def generate_random_heading(self) -> np.ndarray:
        """
        Generate a random heading direction uniformly sampled on the unit circle (XY plane).
        
        Returns:
            np.ndarray: The new heading unit vector (2D).
        """
        yaw = np.random.uniform(-np.pi, np.pi)
        self._heading = np.array([np.cos(yaw), np.sin(yaw)])
        return self._heading.copy()

    def generate_random_speed(self,
                              low: T.Optional[float] = None,
                              high: T.Optional[float] = None) -> float:
        """
        Generate a random speed uniformly sampled within [low, high].

        Args:
            low:  Minimum speed (default: DEFAULT_SPEED_RANGE[0]).
            high: Maximum speed (default: DEFAULT_SPEED_RANGE[1]).

        Returns:
            float: The new desired speed.
        """
        low = low if low is not None else self.DEFAULT_SPEED_RANGE[0]
        high = high if high is not None else self.DEFAULT_SPEED_RANGE[1]
        self._speed = float(np.random.uniform(low, high))
        return self._speed

    def randomize(self) -> None:
        """
        Convenience method: randomize heading, speed, and target point at once.

        """
        self.generate_random_heading()
        self.generate_random_speed()

    # ------------------------------------------------------------------ #
    #                        UTILITY METHODS                              #
    # ------------------------------------------------------------------ #

    def heading_error(self, direction_vector: np.ndarray) -> float:
        """
        Compute the signed angle error between the current heading and a given direction vector.

        Args:
            direction_vector: A 2D vector representing the direction to compare with.
        Returns:
            float: Signed angle error in radians, positive if the direction is to the left of the heading.
        """
        dir_vec = np.asarray(direction_vector, dtype=np.float64).flatten()
        if dir_vec.shape[0] == 2:
            dir_vec = np.array([dir_vec[0], dir_vec[1]])
        else:
            raise ValueError(f"Direction vector must have 2 elements, got {dir_vec.shape[0]}.")
        return float(signed_sin(self._heading[:2], dir_vec[:2]))

    def velocity_error(self, current_speed: float) -> float:
        """
        Compute the speed error between the desired speed and the current speed.

        Args:
            current_speed: The current speed of the robot (m/s).
        Returns:
            float: The speed error (desired - current).
        """
        return self._speed - current_speed

    def reset(self) -> None:
        """Reset all pose control parameters to their default values."""
        self._heading = np.array([1.0, 0.0])
        self._speed = 0.0

    # ------------------------------------------------------------------ #
    #                         DUNDER METHODS                              #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"PoseControl("
            f"heading_angle={np.degrees(self.heading_angle):.1f}°, "
            f"speed={self._speed:.2f} m/s, "
        )
