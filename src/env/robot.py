
import typing as T
import numpy as np
import gymnasium as gym
import copy
import mujoco
from mujoco import MjModel, MjData
import os
from scipy.spatial.transform import Rotation as R

class SelfBalancingRobotEnv(gym.Env):

    def __init__(self, environment_path: str = "../../models/scene.xml"):
        """
        Initialize the SelfBalancingRobot environment.
        
        Args:
            environment_path (str): Path to the MuJoCo model XML file.
        """
        # Initialize the environment
        super().__init__()
        full_path = os.path.abspath(environment_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")
        self.model = MjModel.from_xml_path(full_path)
        self.data = MjData(self.model)

        # Action and observation spaces
        # Observation space: inclination angle, angular velocity, linear position and velocity (could be added: other axis for position and last action)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Action space: torque applied to the wheels
        self.action_limit = 10.0
        self.action_space = gym.spaces.Box(low=np.array([-self.action_limit, -self.action_limit]), high=np.array([self.action_limit, self.action_limit]), dtype=np.float32)

        # If last action is needed, uncomment the following line
        # self.last_action = [0.0, 0.0]

    def reset(self, seed: T.Optional[int] = None, options: T.Optional[dict] = None) -> np.ndarray:
        """
        Reset the environment to an initial state.
        
        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options for resetting the environment.
            
        Returns:
            T.Dict[str, T.Any]: A dictionary containing the initial observation of the environment (pitch angle, pitch velocity, x position, and x velocity).
        """
        # Seed the random number generator
        super().reset(seed=seed)

        self._initialize_random_state()
        # info = self._get_info()
        obs = self._get_obs()
        return obs

    
    def _get_obs(self) -> np.ndarray:
        """
        Get the current observation of the environment.
        
        Returns:
            np.ndarray: The observation vector containing the pitch angle, pitch velocity, x position, and x velocity.
        """
        quat_xyzw = self.data.qpos[3:7][[1, 2, 3, 0]]  # convert [w, x, y, z] → [x, y, z, w]
        euler = R.from_quat(quat_xyzw).as_euler('xyz')

        pitch = euler[1]
        pitch_vel = self.data.qvel[4] # for the moment it is useless

        x_pos = self.data.qpos[0]
        x_vel = self.data.qvel[0]

        return np.array([pitch, pitch_vel, x_pos, x_vel], dtype=np.float32)

    def _get_info(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _initialize_random_state(self):
        # Reset position and velocity
        self.data.qpos[:3] = [0.0, 0.0, 0.25]  # Initial position (x, y, z)
        self.data.qvel[:] = 0.0              # Initial speed

        # Euler angles: Roll=0, Pitch=random, Yaw=random
        euler = [
            0.0,
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-np.pi, np.pi)
        ]

        # Euler → Quaternion [x, y, z, w]
        quat_xyzw = R.from_euler('xyz', euler).as_quat()
        self.data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]



def main():
    env = SelfBalancingRobotEnv()
    env.reset()
    import mujoco
    import mujoco.viewer
    import time
    import os


    print("Avvio del visualizzatore MuJoCo. Chiudere la finestra per terminare.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Ciclo di simulazione principale.
        while viewer.is_running():
            step_start = time.time()

            # Esegui un passo della simulazione.
            mujoco.mj_step(env.model, env.data)

            # Sincronizza il visualizzatore con i dati di simulazione.
            viewer.sync()

            # Attendi per mantenere la simulazione circa in tempo reale.
            # model.opt.timestep è l'intervallo di tempo della simulazione (definito in scene.xml).
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            # Reset the environment each 10 seconds
            if env.data.time > 3.0:
                env.reset()
                env.data.time = 0.0
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("Visualizzatore chiuso. Programma terminato.")

if __name__ == "__main__":
    main()