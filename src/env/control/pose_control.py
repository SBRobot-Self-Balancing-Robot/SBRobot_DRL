import numpy as np
import typing as T
from scipy.spatial.transform import Rotation as R
from src.env.control.base_control import BaseControl


class PoseControl(BaseControl):
    """
    Generates heading references as 2D unit vectors with timer-based switching.

    Every [MIN_HOLD_TIME, MAX_HOLD_TIME] seconds a new target heading is sampled
    uniformly from [-π, π].  Heading error is the signed angle (atan2) from the
    target to the robot's current forward direction, in radians ∈ (-π, π].
    Positive error → current heading is CCW of target → robot must turn CW (right).
    """

    MIN_HOLD_TIME: float = 3.0  # s
    MAX_HOLD_TIME: float = 8.0  # s

    def __init__(self) -> None:
        super().__init__()
        self._heading: np.ndarray = np.array([1.0, 0.0])  # [cos θ, sin θ]
        self._timer: float = 0.0
        self._time_to_switch: float = 0.0

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def heading(self) -> np.ndarray:
        """Current target heading unit vector [cos θ, sin θ]."""
        return self._heading.copy()

    @heading.setter
    def heading(self, vec: np.ndarray) -> None:
        """Set heading from a 2D vector (normalised automatically)."""
        v = np.asarray(vec, dtype=np.float64).flatten()[:2]
        n = np.linalg.norm(v)
        self._heading = v / n if n > 1e-6 else np.array([1.0, 0.0])

    @property
    def heading_angle(self) -> float:
        """Target heading as yaw angle [rad]."""
        return float(np.arctan2(self._heading[1], self._heading[0]))

    @heading_angle.setter
    def heading_angle(self, angle: float) -> None:
        """Set heading from a yaw angle [rad] (used for interactive / test control)."""
        self._heading = np.array([np.cos(float(angle)), np.sin(float(angle))])

    # ------------------------------------------------------------------ #
    #  Update / generation                                                 #
    # ------------------------------------------------------------------ #

    def update(self, dt: float) -> None:
        """Advance the timer; switch to a new random heading when it expires."""
        self._timer += dt
        if self._timer >= self._time_to_switch:
            self._pick_new_heading()
            self._reset_timer()

    def _pick_new_heading(self) -> None:
        angle = float(np.random.uniform(-np.pi, np.pi))
        self._heading = np.array([np.cos(angle), np.sin(angle)])

    def _reset_timer(self) -> None:
        self._timer = 0.0
        self._time_to_switch = float(np.random.uniform(self.MIN_HOLD_TIME, self.MAX_HOLD_TIME))

    def generate_random(self) -> np.ndarray:
        """Force an immediate heading change (used on episode reset)."""
        self._pick_new_heading()
        self._reset_timer()
        return self._heading.copy()

    def set_from_quaternion(self, quat_mj: np.ndarray) -> None:
        """
        Align the target heading with the robot's current orientation.
        Call at episode start so the initial heading error is zero.

        Args:
            quat_mj: MuJoCo quaternion [w, x, y, z] from data.qpos[3:7].
        """
        r = R.from_quat([quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]])
        forward_xy = r.as_matrix()[:2, 0]
        norm = np.linalg.norm(forward_xy)
        self._heading = forward_xy / norm if norm > 1e-6 else np.array([1.0, 0.0])
        self._reset_timer()

    # ------------------------------------------------------------------ #
    #  Error computation                                                   #
    # ------------------------------------------------------------------ #

    def error_with_quaternion(self, quat_mj: np.ndarray) -> float:
        """
        Signed heading error from MuJoCo quaternion [w, x, y, z].

        Returns the signed angle (radians, ∈ (-π, π]) from the target heading
        to the robot's current forward direction (measured CCW in world XY plane).
        """
        r = R.from_quat([quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]])
        forward_xy = r.as_matrix()[:2, 0]
        norm = np.linalg.norm(forward_xy)
        if norm < 1e-6:
            return 0.0
        forward_xy = forward_xy / norm
        return self._signed_angle(self._heading, forward_xy)

    def error(self, direction_vector: np.ndarray) -> float:
        """
        Signed heading error from a 2D direction vector.

        Returns the signed angle (radians, ∈ (-π, π]) from the target heading
        to the given direction (measured CCW in XY plane).
        """
        d = np.asarray(direction_vector, dtype=np.float64).flatten()[:2]
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            return 0.0
        d = d / norm
        return self._signed_angle(self._heading, d)

    @staticmethod
    def _signed_angle(target: np.ndarray, current: np.ndarray) -> float:
        """
        Signed angle from target to current (2D unit vectors), using atan2.
        Unlike sin-only methods, this is unambiguous at ±180°.
        Positive = current is CCW of target.
        """
        sin_err = float(np.cross(target, current))  # sin(θ)
        cos_err = float(np.dot(target, current))     # cos(θ)
        return float(np.arctan2(sin_err, cos_err))

    # ------------------------------------------------------------------ #
    #  Reset                                                               #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset to a random heading for the new episode."""
        self.generate_random()

    def __repr__(self) -> str:
        return (
            f"PoseControl(heading={np.degrees(self.heading_angle):.1f}°, "
            f"timer={self._timer:.1f}/{self._time_to_switch:.1f}s)"
        )
