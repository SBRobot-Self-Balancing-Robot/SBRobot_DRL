import mujoco
import numpy as np
from typing import List, Optional


def _get_active_scenes(self) -> Optional[mujoco.MjvScene]:
    """Helper to get the scenes objects."""
    scenes = []

    # Check if renderer is initialized
    if self.renderer is not None:
        scenes.append(self.renderer.scene)

    # Check if viewer is initialized
    if self.viewer is not None:
        scenes.append(self.viewer.user_scn)

    if len(scenes) == 0:
        print("Warning: No active scenes for rendering.")
        return None

    return scenes

def render_vector(self, origin: np.ndarray, vector: np.ndarray, color: List[float], scale: float = 0.2, radius: float = 0.005, offset: float = 0.0):
    """Helper to render an arrow geometry in the scene."""
    scns = self._get_active_scenes()

    if scns is None: return # No active scenes

    for scn in scns:
        if scn.ngeom >= scn.maxgeom: return # Check geom buffer space

        origin_offset = origin.copy() + np.array([0, 0, offset])
        endpoint = origin_offset + (vector * scale)
        idx = scn.ngeom
        try:
            mujoco.mjv_initGeom(scn.geoms[idx], mujoco.mjtGeom.mjGEOM_ARROW1, np.zeros(3), np.zeros(3), np.zeros(9), np.array(color, dtype=np.float32))
            mujoco.mjv_connector(scn.geoms[idx], mujoco.mjtGeom.mjGEOM_ARROW1, radius, origin_offset, endpoint)
            scn.ngeom += 1
        except IndexError:
            print("Warning: Ran out of geoms in MuJoCo scene for rendering vector.")

def render_point(self, position: np.ndarray, color: List[float], radius: float = 0.01):
    """Helper to render a sphere geometry at a point."""
    scns = self._get_active_scenes()

    if scns is None: return # No active scenes

    for scn in scns:
        if scn.ngeom >= scn.maxgeom: return

        idx = scn.ngeom
        size = np.array([radius, radius, radius])
        rgba = np.array(color, dtype=np.float32)
        try:
            mujoco.mjv_initGeom(scn.geoms[idx], mujoco.mjtGeom.mjGEOM_SPHERE, size, position.astype(np.float64), np.eye(3).flatten(), rgba)
            scn.ngeom += 1
        except IndexError:
            print("Warning: Ran out of geoms in MuJoCo scene for rendering point.")
