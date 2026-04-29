import argparse
from stable_baselines3 import PPO, TD3, A2C, DDPG, SAC


def parse_train_arguments():
    """
    Command line argument parser for training the Self-Balancing Robot environment.
    """
    parser = argparse.ArgumentParser(
        description="Training the self-balancing robot with Deep Reinforcement Learning"
    )

    parser.add_argument("--xml-file", type=str, default="./models/scene.xml",
                        help="Path to the XML file defining the robot and environment (default: ./models/scene.xml)")

    parser.add_argument("--model", type=str, default="PPO",
                        help="Model to use for training (default: PPO). Other options: PPO, TD3, A2C, DDPG")

    parser.add_argument("--policies-folder", type=str, default="./policies",
                        help="Folder where models are saved/loaded (default: ./policies)")

    parser.add_argument("--file-prefix", type=str, default=None,
                        help="Base name for the saved files")

    parser.add_argument("--folder-prefix", type=str, default=None,
                        help="Base folder name for the saved files (default: None, uses timestamp)")

    parser.add_argument("--policy", type=str, default=None,
                        help="Name of the model file to load")

    parser.add_argument("--iterations", type=int, default=1_000_000,
                        help="Number of training iterations (default: 1000000)")

    parser.add_argument("--processes", type=int, default=50,
                        help="Number of parallel processes for training (default: 50)")

    # Add rollout size parameters
    parser.add_argument("--n-steps", type=int, default=1024,
                        help="Number of steps per environment before update (for PPO/A2C, default: 1024)")

    parser.add_argument("--buffer-size", type=int, default=100000,
                        help="Replay buffer size (for SAC/TD3/DDPG, default: 100000)")

    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for training (default: cpu). Other options: cpu, cuda, mps")

    parser.add_argument("--register-dataset", action="store_true",
                        help="Register the dataset after training (default: False)", default=False)

    parser.add_argument("--test-steps", type=int, default=10_000,
                        help="Number of test steps (default: 10000)")

    parser.add_argument("--headless", action="store_true",
                        help="Run the environment in headless mode (default: False)", default=False)

    parser.add_argument("--policy-scp", action="store_true",
                        help="Copy the policy to the remote server (default: False)", default=False)
    parser.add_argument("--wandb", action="store_true",
                        help="Use Weights & Biases for experiment tracking (default: False)", default=False)

    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4). Halve to 1.5e-4 if in plateau.")

    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="Entropy coefficient for PPO/A2C (default: 0.0). "
                             "Use 0.005 to prevent std collapse in hard curriculum phases.")

    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99). Lower (e.g. 0.97) makes "
                             "the policy more reactive to immediate rewards.")

    parser.add_argument("--curriculum-phase", type=int, default=0,
                        help="Initial curriculum phase (0-3). Use >0 to resume from a "
                             "mid-curriculum checkpoint without repeating early phases. "
                             "Phase 0=±0.10 m/s, 1=±0.20, 2=±0.35, 3=±0.50 m/s")

    return parser.parse_args()


def parse_test_arguments():
    """
    Parsing degli argomenti della riga di comando per il test.
    """
    parser = argparse.ArgumentParser(
        description="Test the self-balancing robot with a trained model"
    )

    parser.add_argument("--path", type=str,
                        default=None,
                        help="Path to the model to test")

    # parser.add_argument("--environment-path", type=str, default="./models/scene.xml",
    #                    help="Path to the environment XML file (default: ./models/scene.xml)")

    parser.add_argument("--max-time", type=float, default=float("inf"),
                        help="Maximum simulation time (default: infinite)")

    parser.add_argument("--test-steps", type=int, default=10_000,
                        help="Number of test steps (default: 10000)")

    parser.add_argument("--interactive", action="store_true",
                        help="Enable interactive mode (default: False)")

    return parser.parse_args()


def parse_model(model_name: str):
    """
    Parses the model name to return the corresponding Stable Baselines3 model class.
    """
    models = {
        "SAC": SAC,
        "PPO": PPO,
        "TD3": TD3,
        "A2C": A2C,
        "DDPG": DDPG
    }
    if model_name in models:
        return models[model_name]
    else:
        return models["SAC"]  # Default to SAC if the model is not recognized
