# Write a PID controller using @main.py as reference
import mujoco
import mujoco.viewer
import time
import os
import numpy as np

# Il file XML della scena che include il robot e l'ambiente.
xml_file = './models/scene.xml'
xml_path = os.path.join(os.path.dirname(__file__), xml_file)

# Carica il modello MuJoCo dal file XML.
try:
    model = mujoco.MjModel.from_xml_path(xml_path)
except Exception as e:
    print(f"Errore durante il caricamento del modello: {e}")
    exit()

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

from src.env.self_balancing_robot_env.self_balancing_robot_env import SelfBalancingRobotEnv
from src.env.wrappers.reward import RewardWrapper
from stable_baselines3.common.monitor import Monitor

# Crea un'istanza dei dati di simulazione.
data = mujoco.MjData(model)

def make_env():
    """
    Creates an instance of the SelfBalancingRobotEnv environment wrapped with RewardWrapper.
    """
    def _init():
        environment = SelfBalancingRobotEnv()
        environment = RewardWrapper(environment)
        environment = Monitor(environment)
        check_env(environment, warn=True)
        return environment
    return _init

env = make_env()()

# Controller Class
class PIDController:
    def __init__(self):
        # PID gains for angle control
        self.Kp = -1.85
        self.Ki = -4.8
        self.Kd = -0.007
        
        # PID gains for speed control
        self.Kp_speed = 0.73
        self.Kd_speed = 0.0045
        
        # Setpoints
        self.base_angle_sp = 0.0  # Base angle setpoint (degrees)
        self.speed_sp = 0.0  # Speed setpoint (rad/s)
        self.angle_sp = 0.0  # Angle setpoint (degrees)
        
        # Limits
        self.max_angle_offset = 2.0  # degrees
        self.max_speed = 5.0  # rad/s
        self.tilt_angle_limit = 45.0  # degrees
        
        # State variables
        self.integral_error = 0.0
        self.last_error = 0.0
        self.last_speed_err = 0.0
        
        # Joystick inputs (simulated)
        self.js_speed_sp = 0.0
        self.js_y = 0.0  # Forward/backward
        self.js_x = 0.0  # Rotation
        self.js_multiplier = 1.0
        self.js_multiplier_sp = 1.0
        
        # Filter parameter
        self.alpha = 0.1
        
        # Robot parameters
        self.WHEEL_AXIS_MIDPOINT = 0.298  # Distance between wheels in meters
        self.MAX_CTRL_FREQUENCY = 8.775  # Max control frequency (rad/s)
        
        # Control outputs
        self.V_l_cmd = 0.0
        self.V_r_cmd = 0.0
        
    def reset(self):
        """Reset controller state"""
        self.integral_error = 0.0
        self.last_error = 0.0
        self.last_speed_err = 0.0
        self.speed_sp = 0.0
        self.angle_sp = self.base_angle_sp
        self.V_l_cmd = 0.0
        self.V_r_cmd = 0.0
        
    def setpoint_speed(self, current_speed, dt):
        """Update speed setpoint and calculate angle offset"""
        # Filter speed setpoint
        self.speed_sp = self.alpha * self.js_speed_sp + (1.0 - self.alpha) * self.speed_sp
        
        # Calculate speed error
        speed_err = self.speed_sp - current_speed
        
        # Derivative of speed error
        derivative_speed_err = (speed_err - self.last_speed_err) / dt
        
        # Calculate angle offset
        angle_offset = (self.Kp_speed * speed_err + 
                       self.Kd_speed * derivative_speed_err)
        
        # Limit angle offset
        angle_offset = np.clip(angle_offset, -self.max_angle_offset, self.max_angle_offset)
        
        # Update angle setpoint
        self.angle_sp = self.base_angle_sp + angle_offset
        
        # Save last speed error
        self.last_speed_err = speed_err
        
    def differential_drive_kinematics(self, speed_setpoint):
        """Calculate wheel speeds using differential drive kinematics"""
        V = self.js_y * self.js_multiplier
        omega = -self.js_x * 0.05
        
        if abs(omega) < 1e-6:
            # Only forward/backward
            V_r = V
            V_l = V
        elif abs(V) < 1e-6:
            # Only rotation
            V_r = omega * (self.WHEEL_AXIS_MIDPOINT / 2.0)
            V_l = -omega * (self.WHEEL_AXIS_MIDPOINT / 2.0)
        else:
            # Invert rotation direction if going backward
            omega *= (-1.0 if self.js_y < -0.05 else 1.0)
            
            # Forward and rotation
            R = V / omega
            if abs(R) > 1e6 or np.isnan(R) or np.isinf(R):
                V_r = V
                V_l = V
            else:
                V_r = omega * (R + self.WHEEL_AXIS_MIDPOINT / 2.0)
                V_l = omega * (R - self.WHEEL_AXIS_MIDPOINT / 2.0)
        
        # Add speed setpoint
        V_l_cmd = V_l + speed_setpoint
        V_r_cmd = V_r + speed_setpoint
        
        # Avoid non-valid values
        if np.isnan(V_l_cmd) or np.isinf(V_l_cmd):
            V_l_cmd = 0.0
        if np.isnan(V_r_cmd) or np.isinf(V_r_cmd):
            V_r_cmd = 0.0
        
        # Clip to max control frequency
        V_l_cmd = np.clip(V_l_cmd, -self.MAX_CTRL_FREQUENCY, self.MAX_CTRL_FREQUENCY)
        V_r_cmd = np.clip(V_r_cmd, -self.MAX_CTRL_FREQUENCY, self.MAX_CTRL_FREQUENCY)
        
        self.V_l_cmd = V_l_cmd
        self.V_r_cmd = V_r_cmd
    
    def setpoint_angle(self, current_angle_rad, dt):
        """Calculate control output based on angle error"""
        # Convert current angle from radians to degrees
        current_angle_deg = np.degrees(current_angle_rad)
        
        # Calculate error (in degrees)
        error = self.angle_sp - current_angle_deg
        
        # Check if tilt angle exceeds limit - DON'T RESET, just stop motors
        if abs(current_angle_deg) > self.tilt_angle_limit:
            self.V_l_cmd = 0.0
            self.V_r_cmd = 0.0
            # Keep integral error but don't accumulate more
            return
        
        # PID control
        self.integral_error += error * dt
        
        # Anti-windup: limit integral term
        max_integral = 10.0
        self.integral_error = np.clip(self.integral_error, -max_integral, max_integral)
        
        derivative_error = (error - self.last_error) / dt
        
        speed_setpoint = (self.Kp * error +
                         self.Ki * self.integral_error +
                         self.Kd * derivative_error)
        
        # Filter multiplier
        self.js_multiplier = (self.alpha * self.js_multiplier_sp + 
                             (1.0 - self.alpha) * self.js_multiplier)
        
        # Determine control mode
        if self.js_multiplier > 0.95:
            # Simple mode: same speed for both wheels
            self.V_l_cmd = speed_setpoint
            self.V_r_cmd = speed_setpoint
        else:
            # Differential drive mode
            self.differential_drive_kinematics(speed_setpoint)
        
        # Save last error
        self.last_error = error
    
    def get_control_output(self):
        """Return the control output for left and right motors"""
        return np.array([self.V_l_cmd, self.V_r_cmd], dtype=np.float32)

# Create controller instance
controller = PIDController()

# Simulation loop
obs, _ = env.reset()
controller.reset()

print("Starting PID control simulation...")
print(f"Max time: {env.unwrapped.max_time}s")
print(f"Time step: {env.unwrapped.time_step}s")
print(f"Action space: [{env.action_space.low[0]:.3f}, {env.action_space.high[0]:.3f}]")

# Debug: print raw observation
print(f"Initial obs shape: {obs.shape}")
print(f"Initial obs values: {obs}")

for step in range(10000):
    dt = env.unwrapped.time_step
    
    # Extract current state from observation
    # IMPORTANT: Check what obs[0] actually contains!
    # It might already be in radians, not normalized
    
    # Try direct interpretation first (obs[0] might already be pitch in radians)
    current_angle = obs[0]  # Assume it's already in radians
    
    # Denormalize wheel velocities (obs[4] and obs[5])
    MAX_WHEEL_SPEED = 8.775
    wheel_left_vel = obs[4] * MAX_WHEEL_SPEED
    wheel_right_vel = obs[5] * MAX_WHEEL_SPEED
    current_speed = (wheel_left_vel + wheel_right_vel) / 2.0
    
    # Update speed setpoint (this would normally come from joystick)
    controller.setpoint_speed(current_speed, dt)
    
    # Calculate control action
    controller.setpoint_angle(current_angle, dt)
    
    # Get control output
    action = controller.get_control_output()
    
    # Execute action through environment
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    
    # Print status every 100 steps
    if step % 100 == 0:
        print(f"Step {step}: Pitch={np.degrees(current_angle):.2f}°, "
              f"Speed={current_speed:.2f} rad/s, "
              f"Action=[{action[0]:.3f}, {action[1]:.3f}], "
              f"Integral={controller.integral_error:.2f}")
    
    if terminated or truncated:
        print(f"Episode ended at step {step}, Pitch={np.degrees(current_angle):.2f}°")
        obs, _ = env.reset()
        controller.reset()

print("Simulation completed!")


