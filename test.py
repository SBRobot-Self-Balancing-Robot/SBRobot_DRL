"""
Test script for the self-balancing robot environment using a trained SAC model.
"""
import os
import json
import shutil
import tarfile
import argparse
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC, PPO, TD3, A2C, DDPG
from src.env.wrappers.observations import ObservationWrapper
from src.env.robot import SelfBalancingRobotEnv
from src.utils.files import compress_and_remove
from src.utils.parser import parse_test_arguments, parse_model


def make_env(environment_path="./models/scene.xml", max_time=float("inf")):
    """
    Crea un'istanza dell'ambiente SelfBalancingRobotEnv con rendering.
    """
    env = SelfBalancingRobotEnv(
        environment_path=environment_path, max_time=max_time)
    env = ObservationWrapper(env)
    env = Monitor(env)
    return env


def read_joystick_input(joystick):
    rpt = joystick.read(64)
    if rpt:
        lx, ly, rx, ry = map(normalize, rpt[1:5])
        # Return as a numpy array
        return np.array([lx, ly, rx, ry], dtype=np.float32)
    return None  # Return None if no input is read


def extract(folder_path: str) -> str:
    """
    Decompress a .tar.gz folder.

    Return: path to the decompressed folder.
    """
    # Check if the folder_path is already decompressed
    if os.path.exists(folder_path):
        return folder_path

    if (os.path.exists(f"{folder_path}.tar.gz") or
        os.path.exists(f"{folder_path}.tgz") or
        os.path.exists(f"{folder_path}.tar") or
        os.path.exists(f"{folder_path}.zip") or
            os.path.exists(f"{folder_path}.gz")):
        with tarfile.open(f"{folder_path}.tar.gz", "r:gz") as tar:
            tar.extractall(path=os.path.dirname(folder_path))
        print(
            f"Decompressed folder: {os.path.splitext(os.path.splitext(folder_path)[0])[0]}")
        return os.path.splitext(os.path.splitext(folder_path)[0])[0]


if __name__ == "__main__":
    # Parse degli argomenti della riga di comando
    args = parse_test_arguments()
    POLICY = args.path
    MAX_TIME = args.max_time
    STEPS = args.test_steps
    INTERACTIVE = args.interactive

    # Load the json configuration
    if POLICY is None:
        raise ValueError(
            "Please provide the path to the model using --path argument.")
    compressed_path = f"./policies/{POLICY}"

    POLICY_FOLDER_PATH = extract(compressed_path)  # policies/POLICY
    # policies/POLICY/config.json
    CONFIG_PATH = f"{POLICY_FOLDER_PATH}/config.json"
    POLICY_PATH = f"{POLICY_FOLDER_PATH}/policy"  # policies/POLICY/policy
    ENV_PATH = f"{POLICY_FOLDER_PATH}/scene.xml"  # policies/POLICY/scene.xml

    if not os.path.exists(ENV_PATH):
        ENV_PATH = "./models/scene.xml"
    if not os.path.exists(CONFIG_PATH) and not os.path.exists(f"{POLICY_FOLDER_PATH}/{POLICY}.json"):
        raise FileNotFoundError(
            f"Configuration json file does not exist in {POLICY_FOLDER_PATH}.")
    if not os.path.exists(f"{POLICY_PATH}.zip") and not os.path.exists(f"{POLICY_FOLDER_PATH}/{POLICY}.zip"):
        raise FileNotFoundError(
            f"Model file policy.zip does not exist in {POLICY_FOLDER_PATH}.")

    # Rename the files with the standard names if necessary
    if not os.path.exists(POLICY_PATH) and os.path.exists(f"{POLICY_FOLDER_PATH}/{POLICY}.zip"):
        os.rename(f"{POLICY_FOLDER_PATH}/{POLICY}.zip", f"{POLICY_PATH}.zip")
    if not os.path.exists(CONFIG_PATH) and os.path.exists(f"{POLICY_FOLDER_PATH}/{POLICY}.json"):
        os.rename(f"{POLICY_FOLDER_PATH}/{POLICY}.json", CONFIG_PATH)

    if INTERACTIVE:
        import hid
        # Attempt to open the joystick device
        try:
            joystick = hid.device()
            joystick.open(VENDOR_ID, PRODUCT_ID)
            print("Joystick opened successfully.")
        except Exception as e:
            print(f"Error opening joystick: {e}")
            INTERACTIVE = False

    print("Test configuration:")
    print(f"  - Model: {POLICY}")
    print(f"  - Environment: {ENV_PATH}")
    print(f"  - Max time: {MAX_TIME}")
    print(f"  - Test steps: {STEPS}")
    print(f"  - Interactive: {INTERACTIVE}")
    print()

    # Get the file from the path
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    print("Configuration loaded:")
    print(json.dumps(config, indent=4))

    MODEL = _parse_model(config.get("model", "PPO"))

    env = make_env(environment_path=ENV_PATH, max_time=MAX_TIME)

    model = MODEL.load(POLICY_PATH, env=env)
    print(f"Loaded model: {POLICY_PATH} ")

    obs, _ = env.reset()
    for _ in range(args.test_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        try:
            env.render()
        except Exception as e:
            env.close()
            break
        if terminated or truncated:
            obs, _ = env.reset()

    compress_and_remove(POLICY_FOLDER_PATH)
