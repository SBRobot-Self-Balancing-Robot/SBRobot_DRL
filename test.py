"""
Test script for the self-balancing robot environment using a trained SAC model.
"""
import os
import json
import tarfile
import numpy as np
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
from src.env.wrappers.observations import ObservationWrapper
from src.env.robot import SelfBalancingRobotEnv
from src.utils.files import compress_and_remove
from src.utils.parser import parse_test_arguments, parse_model
from src.env.control.pose_control import PoseControl
from src.env.control.velocity_control import VelocityControl


def make_env(environment_path="./models/scene.xml", max_time=float("inf")):
    """
    Crea un'istanza dell'ambiente SelfBalancingRobotEnv con rendering.
    """
    env = SelfBalancingRobotEnv(
        environment_path=environment_path,
        max_time=max_time,
        randomization_scale=0.0,
        action_filter_alpha=0.5,
    )
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
    CONFIG_PATH = f"{POLICY_FOLDER_PATH}/config.json"
    POLICY_PATH = f"{POLICY_FOLDER_PATH}/policy"  # policies/POLICY/policy
    ENV_PATH = f"{POLICY_FOLDER_PATH}/scene.xml"  # policies/POLICY/scene.xml

    if not os.path.exists(ENV_PATH):
        ENV_PATH = "./models/scene.xml"

    # Rename legacy files if necessary
    if not os.path.exists(POLICY_PATH) and os.path.exists(f"{POLICY_FOLDER_PATH}/{POLICY}.zip"):
        os.rename(f"{POLICY_FOLDER_PATH}/{POLICY}.zip", f"{POLICY_PATH}.zip")
    if not os.path.exists(CONFIG_PATH) and os.path.exists(f"{POLICY_FOLDER_PATH}/{POLICY}.json"):
        os.rename(f"{POLICY_FOLDER_PATH}/{POLICY}.json", CONFIG_PATH)

    if not os.path.exists(f"{POLICY_PATH}.zip"):
        raise FileNotFoundError(f"Model file policy.zip does not exist in {POLICY_FOLDER_PATH}.")

    if INTERACTIVE:
        from pynput import keyboard

        keys_pressed: set = set()

        def _on_press(key):
            try:
                keys_pressed.add(key.char.lower())
            except AttributeError:
                keys_pressed.add(key)  # special keys (arrows, etc.)

        def _on_release(key):
            try:
                keys_pressed.discard(key.char.lower())
            except AttributeError:
                keys_pressed.discard(key)

        _kb_listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
        _kb_listener.daemon = True
        _kb_listener.start()

        # Tuning parameters for arrow-key control
        TURN_RATE  = np.deg2rad(3)   # rotation per step when Left/Right is held
        SPEED_STEP = 0.02            # speed increment per step when Up/Down is held
        MAX_SPEED  = 0.50            # m/s – phase-3 curriculum limit (physical max = 0.548)

        print("Arrow-key control enabled:")
        print("  Up / Down    = increase / decrease speed")
        print("  Left / Right = turn left / right")
        print("  R            = reset heading & speed")

    print("Test configuration:")
    print(f"  - Model: {POLICY}")
    print(f"  - Environment: {ENV_PATH}")
    print(f"  - Max time: {MAX_TIME}")
    print(f"  - Test steps: {STEPS}")
    print(f"  - Interactive: {INTERACTIVE}")
    print()

    # Load config if available, otherwise use defaults
    IS_LSTM = False
    if args.model is not None:
        MODEL = parse_model(args.model)
        print(f"Model override from --model flag: {args.model}")
    elif os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        print("Configuration loaded:")
        print(json.dumps(config, indent=4))
        MODEL = parse_model(config.get("model", "PPO"))
        IS_LSTM = config.get("policy_type", "MlpPolicy") == "MlpLstmPolicy"
    else:
        print("No config.json found (training still in progress?), using PPO defaults. Use --model to override.")
        MODEL = parse_model("PPO")

    if IS_LSTM:
        MODEL = RecurrentPPO

    env = make_env(environment_path=ENV_PATH, max_time=MAX_TIME)

    model = MODEL.load(POLICY_PATH, env=env)
    print(f"Loaded model: {POLICY_PATH} ({'LSTM' if IS_LSTM else 'MLP'})")

    # Access the unwrapped environment to get pose_control
    base_env: SelfBalancingRobotEnv = env.unwrapped  # type: ignore
    pose_control: PoseControl = base_env.pose_control
    velocity_control: VelocityControl = base_env.velocity_control

    # In non-interactive mode: let the timer-based updates in step() handle
    # heading and velocity changes automatically (pose_control switches every
    # MIN_HOLD_TIME–MAX_HOLD_TIME seconds, velocity_control switches every
    # MIN_HOLD_TIME–MAX_HOLD_TIME seconds).
    # In interactive mode: disable automatic updates and use keyboard control.

    # Start curriculum at a meaningful phase so the test is representative.
    # Phase 2 = ±0.35 m/s. If the policy was trained through the curriculum it
    # should handle this; if it only saw phase 0 the slowness will be visible.
    velocity_control._curriculum_phase = min(2, len(velocity_control.CURRICULUM_PHASES) - 1)

    obs, _ = env.reset()
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    for step in range(args.test_steps):
        # ---- Reference update ----
        if INTERACTIVE:
            base_env.training = False  # disables timer-based updates in step()
            if keyboard.Key.left in keys_pressed:
                pose_control.heading_angle = pose_control.heading_angle + TURN_RATE
            if keyboard.Key.right in keys_pressed:
                pose_control.heading_angle = pose_control.heading_angle - TURN_RATE
            if keyboard.Key.up in keys_pressed:
                velocity_control.speed = min(velocity_control.speed + SPEED_STEP, MAX_SPEED)
            if keyboard.Key.down in keys_pressed:
                velocity_control.speed = max(velocity_control.speed - SPEED_STEP, -MAX_SPEED)
            if 'r' in keys_pressed:
                pose_control.generate_random()
                velocity_control.speed = 0.0
        # Non-interactive: timer-based updates fire automatically inside step()

        if IS_LSTM:
            action, lstm_states = model.predict(obs, state=lstm_states,
                                                episode_start=episode_start,
                                                deterministic=True)
            episode_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)
        try:
            env.render()
        except Exception as e:
            print(f"Rendering error: {e}")
            env.close()
            break
        if terminated or truncated:
            obs, _ = env.reset()
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
            pose_control.generate_random()

    # Only recompress if the folder was extracted from an archive (not a live training folder)
    archive_existed = (
        os.path.exists(f"{compressed_path}.tar.gz") or
        os.path.exists(f"{compressed_path}.tgz") or
        os.path.exists(f"{compressed_path}.tar") or
        os.path.exists(f"{compressed_path}.zip") or
        os.path.exists(f"{compressed_path}.gz")
    )
    if archive_existed:
        compress_and_remove(POLICY_FOLDER_PATH)
