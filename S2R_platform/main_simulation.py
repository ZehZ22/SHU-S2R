import torch
import numpy as np
import matplotlib.pyplot as plt
from simulator.test_env import USVSimulator
from simulator.path_manager import generate_circular_path, generate_rectangle_path, generate_straight_path, PathManager
from models.mariner import mariner
from models.mariner1 import mariner1
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 定义 SAC 策略网络（保持你的习惯）
class ModelAction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_state = torch.nn.Sequential(
            torch.nn.Linear(3, 128),
            torch.nn.ReLU(),
        )
        self.fc_mu = torch.nn.Linear(128, 1)
        self.fc_std = torch.nn.Sequential(
            torch.nn.Linear(128, 1),
            torch.nn.Softplus(),
        )

    def forward(self, state):
        state = self.fc_state(state)
        mu = self.fc_mu(state)
        std = self.fc_std(state).clamp(min=1e-6)
        dist = torch.distributions.Normal(mu, std)
        sample = dist.rsample()
        action = torch.tanh(sample)
        return action * 35  # 舵角范围 [-35, 35] 度


# 仿真函数
def simulate_and_collect_trajectory(model_action, env, max_steps=5000):
    state = env.x.copy()  # 初始状态
    trajectory, rudder_angles, heading_error, speeds, cross_track_errors, u_values, v_values = [], [], [], [], [], [], []

    done = False
    step_count = 0

    while not done and step_count < max_steps:
        input_tensor = torch.FloatTensor(state[:3]).unsqueeze(0)
        with torch.no_grad():
            action = model_action(input_tensor).item()

        next_state, reward, done, _ = env.step([action])
        hdg_err = env.heading_error
        cte = env.cross_track_error

        full_state = env.x.copy()

        trajectory.append([full_state[3], full_state[4]])
        rudder_angles.append(np.degrees(full_state[6]))
        heading_error.append(np.degrees(hdg_err))
        speeds.append(env.ship_params.U0)  # 恒定速度模型
        cross_track_errors.append(cte)
        u_values.append(full_state[0])
        v_values.append(full_state[1])

        state = next_state
        step_count += 1

        if env.path_manager.is_finished(full_state[3], full_state[4]):
            print(f"Simulation finished at step {step_count}")
            break

    if not done:
        print(f"Simulation reached max steps ({max_steps}) without finishing the trajectory.")

    return (trajectory, rudder_angles, heading_error, speeds, cross_track_errors,
            env.path_manager.path, u_values, v_values)


# 可视化函数（轨迹）
def plot_trajectory(trajectory, waypoints):
    traj = np.array(trajectory)
    wps = np.array(waypoints)
    plt.figure(figsize=(8, 6))
    plt.plot(wps[:, 0], wps[:, 1], 'g--', label='Target Path')
    plt.plot(traj[:, 0], traj[:, 1], 'r-', label='USV Path')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.title('Trajectory Tracking')
    plt.grid(True)
    plt.show()


# 可视化函数（横向误差）
def plot_cross_track_error(cte_list):
    plt.figure(figsize=(8, 6))
    plt.plot(cte_list, 'b-')
    plt.xlabel('Step')
    plt.ylabel('Cross Track Error (m)')
    plt.title('Cross Track Error Over Time')
    plt.grid(True)
    plt.show()


def main():
    # === 路径类型配置 ===
    path_type = 'straight'

    if path_type == 'straight':
        waypoints = generate_straight_path(length=3000, angle_deg=0, interval=1)
    elif path_type == 'rectangle':
        waypoints = generate_rectangle_path(length=1500, width=800, interval=1)
    elif path_type == 'circle':
        waypoints = generate_circular_path(radius=1000, interval_angle_deg=5)
    else:
        raise ValueError(f"Unsupported path type: {path_type}")

    path_manager = PathManager(waypoints)

    # === 环境模型选择开关 === ✅ ✅ ✅
    enable_wind = True
    enable_wave = True
    enable_current = True

    # === 仿真配置 ===
    wind_config = {
        'method': 'func2' if enable_wind else None,
        'speed': 10.0 if enable_wind else 0.0,
        'direction': 45.0
    }

    wave_config = {
        'method': 'func1' if enable_wave else None,
        'wave_a': 1.0,
        'wave_beta': 75.0,
        'wave_T0': 8.0,
        'wave_zeta4': 0.2,
        'wave_T4': 6.0
    }

    current_config = {
        'method': 'default' if enable_current else None
    }

    # === 加载模型 ===
    model_action = ModelAction()
    model_action.load_state_dict(torch.load('./policys/model_action.pth'))
    model_action.eval()

    # === 初始化环境 ===
    env = USVSimulator(
        path_manager=path_manager,
        ship_model='mariner',
        wind_config=wind_config,
        wave_config=wave_config,
        current_config=current_config,
        dt=0.1
    )

    # === 执行仿真 ===
    trajectory, rudder_angles, heading_error, speeds, cross_track_errors, waypoints, u_values, v_values = \
        simulate_and_collect_trajectory(model_action, env)
    print(f"[CONFIG] Wind Model: {wind_config}")
    print(f"[CONFIG] Wave Model: {wave_config}")
    print(f"[CONFIG] Current Model: {current_config}")

    # === 可视化 ===
    plot_trajectory(trajectory, waypoints)
    plot_cross_track_error(cross_track_errors)


if __name__ == '__main__':
    main()
