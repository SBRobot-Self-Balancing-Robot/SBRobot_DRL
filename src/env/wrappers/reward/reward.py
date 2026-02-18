import mujoco
import numpy as np
import typing as T
import gymnasium as gym
from src.utils.math import signed_sin
from scipy.spatial.transform import Rotation as R
from src.env.robot import SelfBalancingRobotEnv

class RewardWrapper(gym.Wrapper):
    """
    Wrapper for the SelfBalancingRobotEnv to modify the reward structure.
    """
    def __init__(self, env: SelfBalancingRobotEnv):
        super().__init__(env)
        self.reward_calculator = RewardCalculator()

    def step(self, action):
        """
        Executes one step in the environment with the given action.
        
        Args:
            action: The action to take in the environment.
        Returns:
            obs: The observation after taking the action.
            reward: The modified reward after taking the action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode was truncated.
            info: Additional information from the environment.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward += self.reward_calculator.compute_reward(self.env) # type: ignore

        if truncated and not terminated:
            reward += 50
        elif terminated:
            reward -= 200 * (1 - (self.env.env.data.time / self.env.env.max_time))

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment.
        
        Args:
            **kwargs: Optional arguments for the reset.
        Returns:
            Initial observation and additional info.
        """
        return self.env.reset(**kwargs)

class RewardCalculator:
    """
    Class to compute the reward for the SelfBalancingRobotEnv.
    HEADING-FIRST VERSION: The robot must first align to the desired heading,
    then track the desired velocity. Velocity reward is gated by heading accuracy.
    """
    def __init__(self, 
                    heading_weight: float = 1.0, 
                    velocity_weight: float = 1.6, 
                    control_variety_weight: float = 0.2):
        self.heading_weight = heading_weight
        self.velocity_weight = velocity_weight
        self.control_variety_weight = control_variety_weight
    
    def _heading_error(self, env: SelfBalancingRobotEnv) -> float:
        # take the quaternion from the environment and convert it to a rotation matrix
        quat = env.env.data.qpos[3:7]  # Assuming the quaternion is in the order [w, x, y, z]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) # Rearrange to [x, y, z, w]
        rot_matrix = r.as_matrix()
        # The forward direction in the robot's local frame is typically along the x-axis
        forward_vector = rot_matrix[:2, 0]  # Get the first column of the rotation
        # Normalize the forward vector
        norm = np.linalg.norm(forward_vector)
        if norm > 1e-6:
            forward_vector /= norm
        else:
            forward_vector = np.array([0.0, 0.0])
        
        return env.env.pose_control.error(forward_vector)
    
    def _velocity_error(self, env: SelfBalancingRobotEnv) -> float:
        # 1. Recupera i dati del sensore (VELOCITÀ LOCALE)
        vel_id = mujoco.mj_name2id(env.env.model, mujoco.mjtObj.mjOBJ_SENSOR, "body_vel") 
        vel_adr = env.env.model.sensor_adr[vel_id]
        vel_local = env.env.data.sensordata[vel_adr : vel_adr + 3] # [vx_local, vy_local, vz_local]
        
        # 2. Recupera l'orientamento (Quaternione)
        quat = env.env.data.qpos[3:7]  # [w, x, y, z]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) # Scipy vuole [x, y, z, w]
        
        # 3. Trasforma la velocità da LOCALE a GLOBALE (World Frame)
        # Questa operazione applica la rotazione del robot al vettore velocità
        vel_world = r.apply(vel_local) # Ora abbiamo [vx_world, vy_world, vz_world]
        
        # 4. Calcoliamo la direzione in cui il robot sta guardando (Heading)
        # Prendiamo l'asse X locale (1, 0, 0) e lo ruotiamo nel mondo
        # Nota: corrisponde alla prima colonna della matrice di rotazione
        rot_matrix = r.as_matrix()
        robot_forward_vector = rot_matrix[:, 0] # Vettore direzione nel mondo 3D
        
        # Proiettiamo sul piano orizzontale (Z=0) per ignorare il beccheggio (pitch)
        forward_ground_vector = robot_forward_vector[:2]
        norm = np.linalg.norm(forward_ground_vector)
        if norm > 1e-6:
            forward_ground_vector /= norm
        else:
            forward_ground_vector = np.zeros(2)

        # 5. Calcoliamo la velocità effettiva di avanzamento AL SUOLO
        # Facciamo il prodotto scalare tra la velocità World (xy) e la direzione Heading (xy)
        current_ground_speed = np.dot(vel_world[:2], forward_ground_vector)

        # Restituiamo l'errore rispetto al setpoint
        return env.env.velocity_control.error(current_ground_speed)
    
    def _pitch(self, env: SelfBalancingRobotEnv) -> float:
        quat = env.env.data.qpos[3:7]  # Assuming the quaternion is in the order [w, x, y, z]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) # Rearrange to [x, y, z, w]
        return r.as_euler('xyz', degrees=False)[1]  # Return the pitch angle in radians
    
    def compute_reward(self, env: SelfBalancingRobotEnv) -> float:
        """
        Compute the reward based on the current state of the environment.
        Heading has strict priority over velocity: the velocity reward is
        gated (scaled) by how well the robot is currently tracking the heading.
        
        Args:
            env: The environment instance to compute the reward for.
        Returns:
            float: The computed reward.
        """
        heading_error = self._heading_error(env)
        velocity_error = self._velocity_error(env)
        ctrl_variation = env.ctrl_variation
        ctrl = env.ctrl
        pitch = self._pitch(env)

        # --- 1. Heading reward (always active, high priority) ---
        heading_reward = self.heading_weight * (1.0 - abs(heading_error))

        # --- 2. Heading gate: [0, 1] factor that suppresses velocity reward when heading is poor ---
        #    gate ≈ 1 when |heading_error| ≈ 0, gate → 0 when |heading_error| is large
        heading_gate = self._kernel(heading_error, 0.15)

        # --- 3. Velocity reward (gated by heading accuracy) ---
        velocity_reward = self.velocity_weight * (1.0 - abs(velocity_error))

        # --- 4. Control smoothness penalty ---
        smoothness_penalty = -self.control_variety_weight * np.linalg.norm(ctrl_variation)

        # --- 5. Precision bonuses ---
        # Heading-only bonus: strong incentive for near-perfect heading alignment
        heading_bonus = self._kernel(heading_error, 0.01)

        # --- 6. Balance bonus: reward for maintaining low pitch angles (stability) ---
        balance_bonus = self._kernel(pitch, 0.05)

        # --- 7. Speed bonus ---
        speed_bonus = self._kernel(velocity_error, 0.01)

        # --- 8. Control Difference Bonus ---
        ctrl_difference = ctrl[0] - ctrl[1]  # Assuming ctrl[0] is left wheel and ctrl[1] is right wheel
        ctrl_difference_bonus = self._kernel(ctrl_difference, 0.05)

        # Combined bonus: rewards simultaneous precision on heading + velocity + low ctrl
        combined_bonus = self._kernel(np.linalg.norm(ctrl), 0.5) \
                       * self._kernel(heading_error, 0.005) \
                       * self._kernel(velocity_error, 0.005)

        reward = heading_reward \
               + velocity_reward \
               + smoothness_penalty \
               + heading_bonus * speed_bonus
        
        return reward # type: ignore

    def _kernel(self, x: float, alpha: float) -> float:
        return np.exp(-(x**2)/alpha)