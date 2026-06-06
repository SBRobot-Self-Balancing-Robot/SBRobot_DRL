"""
Test script for the self-balancing robot environment using a trained SAC model.

Besides the interactive / qualitative viewer, this script can collect quantitative
performance metrics (pitch oscillation about the equilibrium, forward-velocity
tracking error and yaw-rate / heading tracking error, together with the shaped
reward and its individual components) and dump them to a CSV file via
``--metrics-csv``. Those CSVs are the input of ``plot.py``, which produces every
figure used in the Experimental Analysis chapter.
"""
import os
import csv
import json
import tarfile
import numpy as np
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
from src.env.wrappers.observations import ObservationWrapper
from src.env.wrappers.reward.reward import RewardCalculator
from src.env.robot import SelfBalancingRobotEnv
from src.utils.files import compress_and_remove
from src.utils.parser import parse_test_arguments, parse_model
from src.env.control.yaw_rate_control import YawRateControl, MAX_YAW_RATE
from src.env.control.velocity_control import VelocityControl


def make_env(environment_path="./models/scene.xml", max_time=float("inf"),
             randomization_scale=0.0):
    """
    Crea un'istanza dell'ambiente SelfBalancingRobotEnv con rendering.
    """
    env = SelfBalancingRobotEnv(
        environment_path=environment_path,
        max_time=max_time,
        randomization_scale=randomization_scale,
        action_filter_alpha=0.5,
    )
    env = ObservationWrapper(env)
    env = Monitor(env)
    return env


# Columns written to the metrics CSV (one row per simulation step).
METRIC_FIELDS = [
    "tag", "phase", "episode", "step", "t",
    "pitch_rad", "pitch_deg", "pitch_rate",
    "target_vel", "meas_vel", "vel_err",
    "target_yaw", "meas_yaw", "yaw_err",
    "act_L", "act_R",
    "r_balance", "r_velocity", "r_yaw", "r_smooth", "reward",
    "terminated",
]


def compute_metrics(calc: RewardCalculator, base_env: SelfBalancingRobotEnv,
                    delta_filtered: np.ndarray) -> dict:
    """
    Compute the per-step analysis metrics from the *ideal* (noise-free) simulator
    state, mirroring the quantities used by the reward function so that the logged
    reward matches what the agent actually optimised during training.
    """
    pitch = calc._ideal_pitch(base_env)
    vel_error = calc._ideal_velocity_error(base_env)

    # Ideal (noise-free) forward velocity reconstructed from wheel sensors.
    left_vel = float(base_env.data.sensordata[base_env.IDX_WHEEL_L_VEL])
    right_vel = float(base_env.data.sensordata[base_env.IDX_WHEEL_R_VEL])
    meas_vel = (left_vel + right_vel) * 0.5 * base_env.WHEEL_RADIUS

    ideal_yaw_rate = float(base_env.data.sensordata[base_env.IDX_IDEAL_GYRO][2])
    yaw_rate_error = base_env.yaw_rate_control.error(ideal_yaw_rate)

    pitch_ref = calc.pitch_vel_gain * vel_error
    r_balance = calc._gaussian_kernel(pitch - pitch_ref, calc.sigma_pitch)

    target_vel = base_env.velocity_control.speed
    sigma = calc.sigma_vel_zero if abs(target_vel) < 0.05 else calc.sigma_vel
    r_velocity = calc._gaussian_kernel(vel_error, sigma)

    r_yaw = calc._gaussian_kernel(yaw_rate_error, calc.sigma_yaw)
    r_smooth = -calc.w_smooth * float(
        np.mean((delta_filtered / base_env.MAX_CTRL) ** 2))
    r_task = r_balance * (
        calc.w_balance + calc.w_velocity * r_velocity + calc.w_yaw * r_yaw)

    pitch_rate = float(base_env.data.sensordata[base_env.IDX_IDEAL_GYRO][1])

    return {
        "pitch_rad": pitch,
        "pitch_deg": np.degrees(pitch),
        "pitch_rate": pitch_rate,
        "target_vel": target_vel,
        "meas_vel": meas_vel,
        "vel_err": vel_error,
        "target_yaw": base_env.yaw_rate_control.rate,
        "meas_yaw": ideal_yaw_rate,
        "yaw_err": yaw_rate_error,
        "r_balance": r_balance,
        "r_velocity": r_velocity,
        "r_yaw": r_yaw,
        "r_smooth": r_smooth,
        "reward": r_task + r_smooth,
    }


def collect_metrics(env, model, base_env, *, is_lstm, tag, episodes,
                    phases, render, csv_path):
    """
    Roll out the deterministic policy and log per-step metrics to ``csv_path``.

    For every requested curriculum ``phase`` the velocity controller is locked to
    that phase and ``episodes`` complete episodes are recorded, each row tagged
    with the phase so that plot.py can build the curriculum-learning comparison.
    """
    calc = RewardCalculator()
    velocity_control = base_env.velocity_control

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    n_rows = 0
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_FIELDS)
        writer.writeheader()

        for phase in phases:
            velocity_control._curriculum_phase = phase
            print(f"\n[metrics] Phase {phase} {velocity_control.current_range} m/s "
                  f"— {episodes} episodes")

            for ep in range(episodes):
                obs, _ = env.reset()
                # Phase can be reset inside env.reset(); re-lock it.
                velocity_control._curriculum_phase = phase
                lstm_states = None
                episode_start = np.ones((1,), dtype=bool)
                prev_filtered = np.zeros(2)
                step = 0

                while True:
                    if is_lstm:
                        action, lstm_states = model.predict(
                            obs, state=lstm_states,
                            episode_start=episode_start, deterministic=True)
                        episode_start = np.zeros((1,), dtype=bool)
                    else:
                        action, _ = model.predict(obs, deterministic=True)

                    obs, _, terminated, truncated, _ = env.step(action)

                    delta_filtered = base_env._filtered_action - prev_filtered
                    prev_filtered = base_env._filtered_action.copy()

                    row = compute_metrics(calc, base_env, delta_filtered)
                    row.update(tag=tag, phase=phase, episode=ep, step=step,
                               t=base_env.data.time,
                               act_L=float(base_env._filtered_action[0]),
                               act_R=float(base_env._filtered_action[1]),
                               terminated=int(terminated))
                    writer.writerow(row)
                    n_rows += 1
                    step += 1

                    if render:
                        try:
                            env.render()
                        except Exception as e:
                            print(f"Rendering error: {e}")
                            render = False

                    if terminated or truncated:
                        break

                print(f"  episode {ep + 1:>3}/{episodes}: {step:>4} steps "
                      f"({'fell' if terminated else 'survived'})")

    print(f"\n[metrics] Wrote {n_rows} rows to {csv_path}")


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
    METRICS_CSV = args.metrics_csv

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

        SPEED_STEP = 0.02   # m/s increment per step when Up/Down is held
        MAX_SPEED  = 0.50   # m/s – phase-3 curriculum limit

        print("Arrow-key control enabled:")
        print("  Up / Down    = increase / decrease speed")
        print("  Left / Right = turn left / right (hold for continuous rotation)")
        print("  R            = reset speed to 0, yaw rate to 0")

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

    # When collecting metrics we need finite episodes so they truncate cleanly;
    # fall back to the standard 20 s horizon if the user left max_time infinite.
    if METRICS_CSV is not None and not np.isfinite(MAX_TIME):
        MAX_TIME = 20.0

    env = make_env(environment_path=ENV_PATH, max_time=MAX_TIME,
                   randomization_scale=args.randomization_scale)

    model = MODEL.load(POLICY_PATH, env=env)
    print(f"Loaded model: {POLICY_PATH} ({'LSTM' if IS_LSTM else 'MLP'})")

    # Access the unwrapped environment to get yaw_rate_control
    base_env: SelfBalancingRobotEnv = env.unwrapped  # type: ignore
    yaw_rate_control: YawRateControl = base_env.yaw_rate_control
    velocity_control: VelocityControl = base_env.velocity_control

    # In non-interactive mode: timer-based updates in step() handle velocity and
    # yaw rate changes automatically.
    # In interactive mode: disable automatic updates and use keyboard control.

    # Start curriculum at a meaningful phase so the test is representative.
    # Phase 2 = ±0.35 m/s. If the policy was trained through the curriculum it
    # should handle this; if it only saw phase 0 the slowness will be visible.
    velocity_control._curriculum_phase = min(2, len(velocity_control.CURRICULUM_PHASES) - 1)

    # ------------------------------------------------------------------ #
    #  Metrics-acquisition path (writes a CSV, then exits)                 #
    # ------------------------------------------------------------------ #
    if METRICS_CSV is not None:
        # Keep timer-based setpoint updates ON so velocity / yaw references vary
        # within each episode (richer tracking time-series); the curriculum phase
        # only advances on reset and is re-locked there, so it stays fixed.
        base_env.training = True
        n_phases = len(velocity_control.CURRICULUM_PHASES)
        phases = list(range(n_phases)) if args.sweep_phases else [
            max(0, min(args.curriculum_phase, n_phases - 1))]
        tag = args.tag if args.tag is not None else os.path.splitext(POLICY)[0]

        collect_metrics(
            env, model, base_env,
            is_lstm=IS_LSTM,
            tag=tag,
            episodes=args.episodes,
            phases=phases,
            render=args.render,
            csv_path=METRICS_CSV,
        )
        env.close()
        raise SystemExit(0)

    obs, _ = env.reset()
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    for step in range(args.test_steps):
        # ---- Reference update ----
        if INTERACTIVE:
            base_env.training = False  # disables timer-based updates in step()
            # Yaw rate: commanded while key held, zero when released
            if keyboard.Key.left in keys_pressed:
                yaw_rate_control.rate = MAX_YAW_RATE
            elif keyboard.Key.right in keys_pressed:
                yaw_rate_control.rate = -MAX_YAW_RATE
            else:
                yaw_rate_control.rate = 0.0
            if keyboard.Key.up in keys_pressed:
                velocity_control.speed = min(velocity_control.speed + SPEED_STEP, MAX_SPEED)
            if keyboard.Key.down in keys_pressed:
                velocity_control.speed = max(velocity_control.speed - SPEED_STEP, -MAX_SPEED)
            if 'r' in keys_pressed:
                velocity_control.speed = 0.0
                yaw_rate_control.rate  = 0.0
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
