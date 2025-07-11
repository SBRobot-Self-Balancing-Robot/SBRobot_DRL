import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import uuid
import src.env.self_balancing_robot_env.self_balancing_robot_env
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env():
    """
    Crea un'istanza dell'ambiente SelfBalancingRobotEnv.
    """
    def _init():
        env = gym.make("SelfBalancingRobot-v0")
        env = Monitor(env)
        check_env(env, warn=True)
        return env
    return _init

if __name__ == "__main__":
    # Wrappa per l'algoritmo
    vec_env = SubprocVecEnv([make_env() for _ in range(70)])

    # Load model if it exists
    try:
        model = SAC.load("ppo_self_balancing", env=vec_env)
        print("Model loaded successfully.")
    except FileNotFoundError:
        model = SAC("MlpPolicy", vec_env, verbose=1)
        model.learn(total_timesteps=5_00_000, progress_bar=True)
        file_uuid = str(uuid.uuid4())
        model.save("./recordings/ppo_self_balancing_" + file_uuid)

    # Test
    env = make_env()()
    env.reset()
    obs, _ = env.reset()
    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
