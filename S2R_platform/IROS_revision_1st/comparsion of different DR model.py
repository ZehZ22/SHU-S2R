import torch
import numpy as np
import matplotlib.pyplot as plt
from USVState import USVState, generate_path
from sim1 import marinerwind

# 设置全局字体大小
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20
})


# 定义 SAC 模型类
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
        return action * 35  # 舵角范围为 [-35, 35] 度


# 单独绘制图例图像
def save_legend_only_image():
    legend_items_simple = {
        'multiple-factor': {'color': 'k', 'linestyle': '-', 'label': 'multiple-factor'},
        'wind speed': {'color': 'r', 'linestyle': (0, (1, 1)), 'label': 'wind speed'},
        'wind direction': {'color': 'g', 'linestyle': '--', 'label': 'wind direction'},
        'current speed': {'color': 'b', 'linestyle': '-.', 'label': 'current speed'},
        'current direction': {'color': 'c', 'linestyle': ':', 'label': 'current direction'},
        'wave height': {'color': 'm', 'linestyle': (0, (5, 2)), 'label': 'wave height'},
        'wave direction': {'color': 'y', 'linestyle': (0, (3, 1, 1, 1)), 'label': 'wave direction'},
    }

    fig, ax = plt.subplots(figsize=(10, 2))
    for name, style in legend_items_simple.items():
        ax.plot([], [], color=style['color'], linestyle=style['linestyle'], label=style['label'])

    ax.legend(loc='center', ncol=4, fontsize=10)
    ax.axis('off')
    plt.savefig("legend_only_clean.png", dpi=300, bbox_inches='tight')
    plt.close()


# 仿真函数
def simulate_and_collect_trajectory(model_action, model_func, wind_mode="fixed", wind_speed=3):
    angle_deg = 30
    length = 3000
    interval = 1
    waypoints = generate_path(angle_deg, length, interval)

    env = USVState(
        waypoints=waypoints,
        current_index=1,
        x=np.array([0.0, 0.0, 0.0, 300.0, 0.0, 20, 0.0], dtype=np.float32),
        ui=0.0,
        model_func=model_func,
        wind_mode=wind_mode,
        wind_speed=wind_speed
    )

    state = env.reset()
    trajectory, rudder_angles, heading_error, speeds = [], [], [], []
    cross_track_errors, u_values, v_values = [], [], []
    done = False
    step_count = 0
    max_steps = 4000

    while not done and step_count < max_steps:
        action = model_action(torch.FloatTensor(state).reshape(1, 3)).item()
        next_state, _, done, info = env.step([action])
        full_state = env.get_full_state()
        trajectory.append([full_state[3], full_state[4]])
        rudder_angles.append(np.degrees(full_state[2]))
        heading_error.append(np.degrees(full_state[0]))
        speeds.append(full_state[6])
        cross_track_errors.append(full_state[1])
        u_values.append(full_state[7])
        v_values.append(full_state[8])
        state = next_state
        step_count += 1
        if env.check_done():
            done = True
            print(f"Simulation finished at step {step_count}")
    return trajectory, rudder_angles, heading_error, speeds, cross_track_errors, waypoints, u_values, v_values


# 主流程
def load_and_compare_models():
    model_files = {
        'N': 'model_actionN.pth',
        'L': 'model_actionL.pth',
        'H': 'model_actionH.pth',
        'D': 'model_actionD.pth',
        'W': 'model_actionW.pth',
        'S': 'model_actionS.pth',
        'F': 'model_actionF.pth'
    }
    plot_order = ['F', 'N', 'L', 'H', 'D', 'W', 'S']
    plot_styles = {
        'N': {'label': 'wind speed', 'color': 'r', 'linestyle': (0, (1, 1))},
        'L': {'label': 'wind direction', 'color': 'g', 'linestyle': '--'},
        'H': {'label': 'current speed', 'color': 'b', 'linestyle': '-.'},
        'D': {'label': 'current direction', 'color': 'c', 'linestyle': ':'},
        'W': {'label': 'wave height', 'color': 'm', 'linestyle': (0, (5, 2))},
        'S': {'label': 'wave direction', 'color': 'y', 'linestyle': (0, (3, 1, 1, 1))},
        'F': {'label': 'multiple-factor', 'color': 'k', 'linestyle': '-'}
    }

    models = {}
    for name in plot_order:
        model = ModelAction()
        model.load_state_dict(torch.load(model_files[name]))
        model.eval()
        models[name] = model

    results = {}
    for name in plot_order:
        results[name] = simulate_and_collect_trajectory(
            model_action=models[name],
            model_func=marinerwind,
            wind_mode="fixed",
            wind_speed=4
        )

    start_idx = 250
    end_idx = 2500
    time_steps = np.arange(start_idx, end_idx)

    def plot_metric(index, ylabel, save_name):
        plt.figure(figsize=(8, 6))
        for name in plot_order:
            data = results[name][index]
            style = plot_styles[name]
            plt.plot(time_steps, data[start_idx:end_idx],
                     color=style['color'],
                     linestyle=style['linestyle'])

        plt.xlabel('Time Step')
        plt.ylabel(ylabel)

        # 设置 Y 轴范围和刻度
        if 'ce' in save_name.lower():
            plt.ylim(-50, 50)
            plt.yticks(np.arange(-50, 55, 10))

        if 'he' in save_name.lower():
            plt.ylim(-20, 20)
            plt.yticks(np.arange(-20, 25, 5))

        plt.grid(True)
        plt.savefig(f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    plot_metric(6, 'Lateral Speed u (m/s)', 'lateral_speed_u')
    plot_metric(7, 'Longitudinal Speed v (m/s)', 'longitudinal_speed_v')
    plot_metric(1, 'Rudder Angle (deg)', 'rudder_angle')
    plot_metric(2, 'Heading Error (deg)', 'he')
    plot_metric(3, 'Speed (m/s)', 'speed')
    plot_metric(4, 'Cross Track Error (m)', 'ce')

    # Trajectory comparison (without legend)
    plt.figure(figsize=(8, 6))
    waypoints = np.array(results['N'][5])[start_idx:end_idx]
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'g--')
    for name in plot_order:
        traj = np.array(results[name][0])[start_idx:end_idx]
        style = plot_styles[name]
        plt.plot(traj[:, 0], traj[:, 1],
                 color=style['color'],
                 linestyle=style['linestyle'])
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.grid(True)
    plt.savefig("trajectory_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 保存图例图像
    save_legend_only_image()


# 执行主函数
load_and_compare_models()
