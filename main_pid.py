"""
PID cascade controller for the self-balancing robot.

Outer loop : velocity error (m/s)  →  pitch setpoint offset (deg)
Inner loop : pitch error    (deg)  →  wheel speed command   (rad/s)

Keyboard controls:
  Up / Down    — increase / decrease forward speed
  Left / Right — turn left / right
  R            — reset speed and heading to zero
  Q            — quit
"""
import os
import time
import numpy as np
from pynput import keyboard as kb
from scipy.spatial.transform import Rotation as R

import mujoco
from src.env.robot import SelfBalancingRobotEnv

# ── Physical constants ────────────────────────────────────────────────────────
WHEEL_RADIUS = 0.0625       # m
WHEEL_TRACK  = 0.2982       # m  (distance between wheels)
MAX_CTRL     = 8.775        # rad/s  (actuator limit)
MAX_LIN_VEL  = MAX_CTRL * WHEEL_RADIUS   # ≈ 0.548 m/s

# ── Tunable PID gains ─────────────────────────────────────────────────────────
# Inner loop: pitch error (deg) → wheel speed (rad/s).
# Kp < 0 : lean forward → spin wheels forward to recover.
KP_BALANCE   = -2.0
KI_BALANCE   = -1.5
KD_BALANCE   = -0.05
MAX_INTEGRAL =  5.0          # anti-windup limit (deg·s)
TILT_LIMIT   = 35.0          # deg – stop motors if tilt exceeds this

# Outer loop: velocity error (m/s) → pitch setpoint offset (deg).
# Kp > 0 : too slow → lean forward (positive pitch offset) to accelerate.
KP_SPEED     = 10.0          # deg / (m/s) — equals ~0.18 rad / (m/s) from DRL
KD_SPEED     = 0.5
MAX_PITCH_SP = 3.0           # deg maximum pitch offset from outer loop

# Rotation
TURN_GAIN    = 1.5           # scales js_x to wheel speed differential (rad/s)


# ── PID inner loop ────────────────────────────────────────────────────────────

class BalancePID:
    """Pitch error (deg) → wheel speed setpoint (rad/s)."""

    def __init__(self):
        self._integral  = 0.0
        self._last_err  = 0.0

    def reset(self):
        self._integral = 0.0
        self._last_err = 0.0

    def step(self, pitch_deg: float, setpoint_deg: float, dt: float) -> float:
        if abs(pitch_deg) > TILT_LIMIT:
            self.reset()
            return 0.0

        err              = setpoint_deg - pitch_deg
        self._integral   = np.clip(self._integral + err * dt,
                                   -MAX_INTEGRAL, MAX_INTEGRAL)
        d_err            = (err - self._last_err) / dt
        self._last_err   = err
        out              = KP_BALANCE * err + KI_BALANCE * self._integral + KD_BALANCE * d_err
        return float(np.clip(out, -MAX_CTRL, MAX_CTRL))


# ── PD outer loop ─────────────────────────────────────────────────────────────

class SpeedPD:
    """Velocity error (m/s) → pitch setpoint offset (deg)."""

    def __init__(self):
        self._last_err = 0.0

    def reset(self):
        self._last_err = 0.0

    def step(self, vel_error_ms: float, dt: float) -> float:
        d_err          = (vel_error_ms - self._last_err) / dt
        self._last_err = vel_error_ms
        out = KP_SPEED * vel_error_ms + KD_SPEED * d_err
        return float(np.clip(out, -MAX_PITCH_SP, MAX_PITCH_SP))


# ── State reader (ideal sensors, no noise) ────────────────────────────────────

def read_state(env: SelfBalancingRobotEnv):
    """Return (pitch_deg, fwd_vel_ms) from ideal MuJoCo sensors.

    Pitch is computed in the robot's LOCAL frame (tilt in the forward direction
    regardless of yaw).  as_euler("xyz")[1] gives world-frame Y rotation which
    is only correct at yaw=0; this formula works at any heading.
    """
    q   = env.data.qpos[3:7]                          # [w, x, y, z]
    rot = R.from_quat([q[1], q[2], q[3], q[0]])       # scipy [x, y, z, w]

    # Gravity vector in body frame: positive X component = leaning forward
    gravity_body = rot.inv().apply([0.0, 0.0, -1.0])
    pitch_rad    = float(np.arctan2(gravity_body[0], -gravity_body[2]))
    pitch_deg    = float(np.degrees(pitch_rad))

    left_vel   = float(env.data.sensordata[env.IDX_WHEEL_L_VEL])   # rad/s
    right_vel  = float(env.data.sensordata[env.IDX_WHEEL_R_VEL])   # rad/s
    fwd_vel_ms = (left_vel + right_vel) * 0.5 * WHEEL_RADIUS

    return pitch_deg, fwd_vel_ms


# ── Keyboard handler ──────────────────────────────────────────────────────────

SPEED_STEP = 0.05    # m/s per step when key held
MAX_SPEED  = MAX_LIN_VEL
TURN_STEP  = 0.05    # dimensionless

keys_pressed: set   = set()
quit_flag: list     = [False]

def _on_press(key):
    try:
        ch = key.char.lower() if hasattr(key, 'char') and key.char else None
        if ch == 'q':
            quit_flag[0] = True
        keys_pressed.add(ch if ch else key)
    except Exception:
        keys_pressed.add(key)

def _on_release(key):
    try:
        ch = key.char.lower() if hasattr(key, 'char') and key.char else None
        keys_pressed.discard(ch if ch else key)
    except Exception:
        keys_pressed.discard(key)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = SelfBalancingRobotEnv(
        environment_path="./models/scene.xml",
        max_time=float("inf"),
        randomization_scale=0.0,
    )
    env.training = False   # disable automatic velocity/heading reference updates

    kb_listener = kb.Listener(on_press=_on_press, on_release=_on_release)
    kb_listener.daemon = True
    kb_listener.start()

    balance_pid = BalancePID()
    speed_pd    = SpeedPD()

    target_speed_ms = 0.0   # m/s
    js_x            = 0.0   # rotation input ∈ [-1, 1]

    obs, _ = env.reset()
    balance_pid.reset()
    speed_pd.reset()

    dt = env.time_step
    print(f"PID running at {1/dt:.0f} Hz  (dt={dt*1000:.1f} ms)")
    print("Controls: ↑↓ speed | ←→ turn | R reset | Q quit\n")

    for step in range(500_000):
        if quit_flag[0]:
            break

        # ── Keyboard ──────────────────────────────────────────────────────
        if kb.Key.up    in keys_pressed:
            target_speed_ms = min(target_speed_ms + SPEED_STEP, MAX_SPEED)
        if kb.Key.down  in keys_pressed:
            target_speed_ms = max(target_speed_ms - SPEED_STEP, -MAX_SPEED)
        if kb.Key.left  in keys_pressed:
            js_x = min(js_x + TURN_STEP, 1.0)
        if kb.Key.right in keys_pressed:
            js_x = max(js_x - TURN_STEP, -1.0)
        if 'r' in keys_pressed:
            target_speed_ms = 0.0
            js_x = 0.0
            balance_pid.reset()
            speed_pd.reset()

        # ── Read state ────────────────────────────────────────────────────
        pitch_deg, fwd_vel_ms = read_state(env)

        # ── Outer loop: speed → pitch setpoint ───────────────────────────
        vel_error   = target_speed_ms - fwd_vel_ms
        pitch_sp_deg = speed_pd.step(vel_error, dt)

        # ── Inner loop: pitch → wheel speed ───────────────────────────────
        wheel_cmd   = balance_pid.step(pitch_deg, pitch_sp_deg, dt)

        # ── Differential for turning ──────────────────────────────────────
        turn_delta  = js_x * TURN_GAIN
        action      = np.array([wheel_cmd - turn_delta,
                                 wheel_cmd + turn_delta], dtype=np.float32)
        action      = np.clip(action, -MAX_CTRL, MAX_CTRL)

        # ── Step environment ──────────────────────────────────────────────
        _, _, terminated, truncated, _ = env.step(action)
        env.render()

        if step % 100 == 0:
            print(f"step {step:6d} | pitch {pitch_deg:+6.2f}° | "
                  f"fwd {fwd_vel_ms:+5.3f} m/s  target {target_speed_ms:+5.3f} | "
                  f"pitch_sp {pitch_sp_deg:+5.2f}° | "
                  f"cmd [{action[0]:+5.2f}, {action[1]:+5.2f}]")

        if terminated or truncated:
            print(f"  → reset at step {step}  (pitch={pitch_deg:+.1f}°)")
            obs, _ = env.reset()
            balance_pid.reset()
            speed_pd.reset()

    env.close()
    print("Done.")
