import os
import time
import mujoco
import numpy as np
import typing as T
import gymnasium as gym
from mujoco import MjModel, MjData
from mujoco.viewer import launch_passive
from scipy.spatial.transform import Rotation as R


class SelfBalancingRobotEnv(gym.Env):
    
    def __init__(self, environment_path: str = "./models/scene.xml", max_time: float = 10.0, max_pitch: float = 1.0472, frame_skip: int = 10):
        """
        Initialize the SelfBalancingRobot environment.
        
        Args:
            environment_path (str): Path to the MuJoCo model XML file.
        """
        # Initialize the environment
        super().__init__()
        self.viewer = None
        full_path = os.path.abspath(environment_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")
        self.model = MjModel.from_xml_path(full_path)
        self.data = MjData(self.model)
        self.max_time = max_time  # Maximum time for the episode
        self.frame_skip = frame_skip  # Number of frames to skip in each step   
        # Action and observation spaces
        # Observation space: pitch, roll, yaw, body_ang_vel_x, body_ang_vel_y, linear_vel_x, linear_vel_y
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64)

        # Action space: torque applied to the wheels
        self.action_limit = 10.0
        self.action_space = gym.spaces.Box(low=np.array([-self.action_limit, -self.action_limit]), high=np.array([self.action_limit, self.action_limit]), dtype=np.float64)

        self.max_pitch = max_pitch  # Maximum pitch angle before truncation
        
        # Inizializza i pesi per la nuova reward - FOCALIZZATI SULLO STARE FERMI
        self.weight_upright = 2.0       # Più importante: stare dritto (pitch e roll)
        self.weight_ang_vel_stability = 0.5 # Ridurre oscillazioni (velocità angolari del corpo)
        self.weight_no_linear_movement = 10.0 # Molto importante: non muoversi linearmente (velocità lineare X e Y)
        self.weight_no_yaw_movement = 1.0 # Molto importante: non ruotare su se stesso (velocità angolare Z - yaw)
        self.weight_control_effort = 0.0001 # Penalità sforzo motori (per efficienza)
        self.weight_action_rate = 0.0001 # Penalità per azioni brusche (per fluidità)
        self.weight_fall_penalty = 100.0 # Grande penalità alla caduta

        
        # Variabile per tracciare l'azione precedente per il calcolo dell'action_rate
        self._last_action = np.zeros(2) # Assumiamo 2 motori/azioni
        

    def step(self, action: T.Tuple[float, float]) -> T.Tuple[np.ndarray, float, bool, dict]:
        """
        Perform a step in the environment.
        
        Args:
            action (np.ndarray or list): The action to be taken, which is a torque applied
            to the wheels of the robot.

        Returns:
            T.Tuple[np.ndarray, float, bool, dict]: A tuple containing:
                - obs (np.ndarray): The observation of the environment (pitch angle, pitch velocity, x position, and x velocity).
                - reward (float): The reward received after taking the action   
                - terminated (bool): Whether the episode has terminated.
                - truncated (bool): Whether the episode has been truncated.
                - info (dict): Additional information about the environment.
        """
        self.data.ctrl[:] = np.clip(action, -self.action_limit, self.action_limit)  # Apply action (torque to the wheels)
        for skip in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)  # Step the simulation
        obs = self._get_obs()
        reward = self._compute_reward(np.array(action)) # Passa l'azione alla reward
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        # Penalità di caduta al termine dell'episodio
        if truncated:
            reward -= self.weight_fall_penalty # Penalizza fortemente la caduta
        
        # info = self._get_info()
        return obs, reward, terminated, truncated, {}
        
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
        if seed is not None:
            np.random.seed(seed)
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)  # Reset the simulation data
        self._initialize_random_state()
        self._last_action = np.zeros(2) # Resetta anche l'ultima azione
        # info = self._get_info()
        obs = self._get_obs()
        return obs, {}
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): The mode in which to render the environment. Default is 'human'.
        """
        if self.viewer is None:
            self.viewer = launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()
            time.sleep(self.model.opt.timestep * self.frame_skip)  # Sleep for the duration of the frame skip
        else:
            raise RuntimeError("Viewer is not running. Please reset the environment or start the viewer.")

    def _get_body_orientation_angles(self) -> T.Tuple[float, float, float]:
        """
        Estrae gli angoli di roll, pitch, yaw dal corpo principale del robot.
        Questo dipende da come il tuo robot è modellato in MuJoCo.
        """
        # Converti il quaternione MuJoCo [w, x, y, z] a scipy [x, y, z, w]
        quat_wxyz = self.data.qpos[3:7]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        
        euler = R.from_quat(quat_xyzw).as_euler('xyz')
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]
        return float(roll), float(pitch), float(yaw)

    def _get_body_angular_velocities(self) -> np.ndarray:
        """
        Estrae le velocità angolari del corpo principale del robot.
        """
        # Assumiamo che le velocità angolari del corpo libero siano in data.qvel[3:6]
        return self.data.qvel[3:6]

    def _get_robot_linear_velocity(self) -> np.ndarray:
        """
        Ottiene la velocità lineare del robot (spesso del centro di massa o di un corpo specifico).
        Assumiamo che le velocità lineari del corpo libero siano in data.qvel[0:3]
        """
        return self.data.qvel[0:3]

    def _kernel(self, x: float, alpha: float) -> float:
        """
        Gaussian kernel function for reward computation.
        Args:
            x (float): The input value.
            alpha (float): The bandwidth parameter for the Gaussian kernel. 
        Returns:
            float: The value of the Gaussian kernel at x.
        """
        return np.exp(-(x**2)/alpha)

    # --- Funzione _compute_reward focalizzata sul bilanciamento e la stasi ---
    def _compute_reward(self, action: np.ndarray) -> float:
        """
        Compute the reward for the current step, focused on self-balancing and staying still.

        Args:
            action (np.ndarray): The action taken in the environment (es. comandi motori).

        Returns:
            float: The computed reward.
        """
        roll, pitch, yaw = self._get_body_orientation_angles() # Ottieni roll, pitch, yaw
        body_ang_vel = self._get_body_angular_velocities() # Velocità angolari del corpo (x, y, z)
        linear_vel = self._get_robot_linear_velocity() # Velocità lineare del robot (x, y, z)
        
        # Dati motori/azioni
        torques = self.data.ctrl # Torque effettivamente applicati o comandi motore
        torque_norm = np.linalg.norm(torques)
        
        # Calcolo dell'action_rate per penalizzare i movimenti bruschi
        action_rate = np.linalg.norm(action - self._last_action)
        self._last_action = action # Aggiorna l'azione precedente

        # --- Componenti di Reward ---

        ## 1. Mantenere la postura eretta (Bilanciamento) ⚖️
        # Penalizza l'inclinazione (pitch e roll). Più vicino a zero è meglio.
        # Usa un alpha minore per un kernel più "stretto", premiando solo angoli molto piccoli.
        upright_reward = self._kernel(float(pitch), alpha=0.01) * self._kernel(float(roll), alpha=0.01) * self._kernel(float(yaw), alpha=0.01)
        upright_reward *= self.weight_upright

        # Penalizza la velocità angolare eccessiva del corpo (per stabilizzare). Riduce le oscillazioni.
        # body_ang_vel[0] = roll rate, body_ang_vel[1] = pitch rate, body_ang_vel[2] = yaw rate
        # Norm di tutte le velocità angolari per una stabilità generica
        angular_velocity_stability_penalty = np.linalg.norm(body_ang_vel)
        # Premiamo quando le velocità angolari sono vicine a zero
        angular_velocity_stability_reward = self._kernel(float(angular_velocity_stability_penalty), alpha=0.01)
        angular_velocity_stability_reward *= self.weight_ang_vel_stability


        ## 2. Stare fermo (Assenza di movimento) 🛑
        # Penalità per la velocità lineare del robot (su X e Y). Premiamo se vicina a zero.
        # Usa la norma della velocità lineare su X e Y
        linear_speed = np.linalg.norm(linear_vel[0:2]) # Velocità su piano orizzontale
        no_linear_movement_reward = self._kernel(float(linear_speed), alpha=0.01) # Premiamo quando la velocità è 0
        no_linear_movement_reward *= self.weight_no_linear_movement

        # Penalità per la rotazione sul posto (velocità angolare di yaw). Premiamo se vicina a zero.
        yaw_rate = body_ang_vel[2] # Velocità angolare di yaw
        no_yaw_movement_reward = self._kernel(float(yaw_rate), alpha=0.01) # Premiamo quando yaw_rate è 0
        no_yaw_movement_reward *= self.weight_no_yaw_movement


        ## 3. Penalità per efficienza e fluidità ✨
        # Penalità per sforzo di controllo (torque). Rende i movimenti più efficienti.
        control_effort_penalty = torque_norm * self.weight_control_effort

        # Penalità per il tasso di azione (movimenti bruschi). Rende i movimenti più fluidi.
        action_rate_penalty = action_rate * self.weight_action_rate

        # --- Calcolo della Reward Finale ---
        reward = (
            upright_reward -
            angular_velocity_stability_reward +
            no_linear_movement_reward +
            no_yaw_movement_reward -
            control_effort_penalty -
            action_rate_penalty
        )

        return float(reward)

    # --- Le seguenti funzioni non sono più necessarie o sono state inglobate ---
    # Le ho lasciate commentate per chiarezza, non rimuoverle se vuoi mantenere il file pulito ma con storia.
    # def _pitch_reward_component(self, alpha: float) -> float:
    #     pass 

    # def _velocity_reward_component(self, alpha: float) -> float:
    #     pass 

    # def _wheels_reward_component(self, alpha: float) -> T.Tuple[float, float, float]:
    #     pass 

    # def _gyro_reward_component(self, alpha: float) -> float:
    #     pass 

        
    def _get_obs(self) -> np.ndarray:
        """
        Get the current observation of the environment.
        
        Returns:
            np.ndarray: The observation vector.
            
            Ho mantenuto l'output di 7 elementi, che ora include:
            [pitch, roll, yaw, body_ang_vel_x, body_ang_vel_y, linear_vel_x, linear_vel_y]
        """
        roll, pitch, yaw = self._get_body_orientation_angles()
        body_ang_vel = self._get_body_angular_velocities() # [gyro_x, gyro_y, gyro_z]
        linear_vel = self._get_robot_linear_velocity() # [vel_x, vel_y, vel_z]
        
        return np.array([
            pitch,          # Inclinazione avanti/indietro (fondamentale per bilanciamento)
            roll,           # Inclinazione laterale (fondamentale per bilanciamento)
            yaw,            # Orientamento sul piano orizzontale (per la stasi rotazionale)
            body_ang_vel[0], # Velocità angolare asse X del corpo (roll rate)
            body_ang_vel[1], # Velocità angolare asse Y del corpo (pitch rate)
            linear_vel[0],  # Velocità lineare in avanti (asse X) (per la stasi lineare)
            linear_vel[1]   # Velocità lineare laterale (asse Y) (per la stasi lineare)
        ], dtype=np.float64)

    def _get_info(self):
        # Questo metodo non è stato modificato in quanto non è stato specificato.
        # Potrebbe essere utile per debug o per raccogliere metriche.
        return {}

    def _is_terminated(self) -> bool:
        """
        Check if the episode is terminated.
        
        Returns:
            bool: True if the episode is terminated, False otherwise.
        """
        # Terminated quando il robot cade o raggiunge la fine del tempo massimo
        return self._is_truncated() or self.data.time >= self.max_time
    
    def _is_truncated(self) -> bool:
        # Ottieni orientamento attuale in angoli Euler
        roll, pitch, yaw = self._get_body_orientation_angles()

        # Truncate if the pitch or roll angle is too high (robot falls)
        return bool(abs(pitch) > self.max_pitch or abs(roll) > self.max_pitch) # max_pitch funge anche da max_roll


    def _initialize_random_state(self):
        # Reset position and velocity
        self.data.qpos[:3] = [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.25]  # Initial position (x, y, z)
        self.data.qvel[:] = 0.0  # Initial speed

        # Euler angles: Roll=0, Pitch=random, Yaw=random
        euler = [
            0.0, # Roll
            np.random.uniform(-0.6, 0.6), # Pitch
            np.random.uniform(-np.pi, np.pi) # Yaw
        ]

        # Euler → Quaternion [x, y, z, w]
        quat_xyzw = R.from_euler('xyz', euler).as_quat()
        self.data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]