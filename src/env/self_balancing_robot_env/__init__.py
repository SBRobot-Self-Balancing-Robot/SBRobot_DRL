from .self_balancing_robot_env import SelfBalancingRobotEnv
import gymnasium as gym

gym.register(
    id="SelfBalancingRobot-v0",
    entry_point="src.env.self_balancing_robot_env.self_balancing_robot_env:SelfBalancingRobotEnv",
    #max_episode_steps=300,
)