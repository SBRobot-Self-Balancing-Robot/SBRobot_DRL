import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from src.env.self_balancing_robot_env.self_balancing_robot_env import SelfBalancingRobotEnv  # Assicura il registry dell'env

def make_env():
    """
    Crea un'istanza dell'ambiente SelfBalancingRobotEnv con rendering.
    """
    env = SelfBalancingRobotEnv(environment_path="./models/scene.xml", max_time=float("inf"))
    env = Monitor(env)
    return env

if __name__ == "__main__":
    env = make_env()

    # Carica il modello addestrato
    model_path = "./recordings/new_reward_0"
    model = SAC.load(model_path, env=env)
    print(f"Modello caricato da: {model_path}")

    # Esecuzione del test
    obs, _ = env.reset()
    for _ in range(10_000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
