"""
Test script for the self-balancing robot environment using a trained SAC model.
"""
import argparse
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from src.env.self_balancing_robot_env.self_balancing_robot_env import (SelfBalancingRobotEnv)

def parse_arguments():
    """
    Parsing degli argomenti della riga di comando per il test.
    """
    parser = argparse.ArgumentParser(
        description="Test the self-balancing robot with a trained model"
    )
    
    parser.add_argument("--path", type=str,
                       default="./policies/new_reward_2025-07-14_09-28-06",
                       help="Path to the model to test")
    
    parser.add_argument("--environment-path", type=str, default="./models/scene.xml",
                       help="Path to the environment XML file (default: ./models/scene.xml)")

    parser.add_argument("--max-time", type=float, default=float("inf"),
                       help="Maximum simulation time (default: infinite)")

    parser.add_argument("--test-steps", type=int, default=10_000,
                       help="Number of test steps (default: 10000)")
    
    return parser.parse_args()

def make_env(environment_path="./models/scene.xml", max_time=float("inf")):
    """
    Crea un'istanza dell'ambiente SelfBalancingRobotEnv con rendering.
    """
    env = SelfBalancingRobotEnv(environment_path=environment_path, max_time=max_time)
    env = Monitor(env)
    return env

if __name__ == "__main__":
    # Parse degli argomenti della riga di comando
    args = parse_arguments()
    POLICY = args.path
    ENV_PATH = args.environment_path
    MAX_TIME = args.max_time
    STEPS = args.test_steps
    
    print("Test configuration:")
    print(f"  - Model: {POLICY}")
    print(f"  - Environment: {ENV_PATH}")
    print(f"  - Max time: {MAX_TIME}")
    print(f"  - Test steps: {STEPS}")
    print()
    
    env = make_env(environment_path=ENV_PATH, max_time=MAX_TIME)

    model = SAC.load(POLICY, env=env)
    print(f"Loaded model: {POLICY}")

    obs, _ = env.reset()
    for _ in range(args.test_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()