import torch
import numpy as np
import matplotlib.pyplot as plt
from USVState import USVState, generate_path
from sim1 import marinerwind


# 设置全局字体大小
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 16
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


def simulate_and_collect_trajectory(model_action, model_func, wind_mode="fixed", wind_speed=3):
    # 设置环境
    angle_deg = 30
    length = 3000  # 总路径长度
    interval = 1  # 路径点间隔
    waypoints = generate_path(angle_deg, length, interval)

    env = USVState(
        waypoints=waypoints,
        current_index=1,
        x=np.array([0.0, 0.0, 0.0, 300.0, 0.0, np.radians(20), 0.0], dtype=np.float32),
        ui=0.0,
        model_func=model_func,
        wind_mode=wind_mode,
        wind_speed=wind_speed
    )

    state = env.reset()
    trajectory = []
    rudder_angles = []
    heading_error = []
    speeds = []
    cross_track_errors = []
    u_values = []  # 横摇速度
    v_values = []  # 纵摇速度
    done = False
    step_count = 0
    max_steps = 4000  # 仿真最大步数

    while not done and step_count < max_steps:
        action = model_action(torch.FloatTensor(state).reshape(1, 3)).item()
        next_state, _, done, info = env.step([action])
        full_state = env.get_full_state()

        trajectory.append([full_state[3], full_state[4]])
        rudder_angles.append(np.degrees(full_state[2]))
        heading_error.append(np.degrees(full_state[0]))
        speeds.append(full_state[6])
        cross_track_errors.append(full_state[1])
        u_values.append(full_state[7])  # 提取横摇速度 u
        v_values.append(full_state[8])  # 提取纵摇速度 v

        state = next_state
        step_count += 1

        # 如果到达最后一个航点且距离阈值满足要求
        if env.check_done():
            done = True
            print(f"Simulation finished at step {step_count}: USV has reached the final waypoint.")

    # 如果仿真达到最大步数但未完成
    if not done:
        print(f"Simulation reached max steps ({max_steps}) without finishing the trajectory.")

    return trajectory, rudder_angles, heading_error, speeds, cross_track_errors, waypoints, u_values, v_values


def load_and_compare_models():
    # 加载(NDR)
    model_actionN = ModelAction()
    model_actionN.load_state_dict(torch.load('model_actionN.pth'))
    model_actionN.eval()

    # 加载(LDR)
    model_actionL = ModelAction()
    model_actionL.load_state_dict(torch.load('model_actionL.pth'))
    model_actionL.eval()

    # 加载(HDR)
    model_actionH = ModelAction()
    model_actionH.load_state_dict(torch.load('model_actionH.pth'))
    model_actionH.eval()


    # 运行 无域随机模型 (NDR)
    NDR_trajectory, NDR_rudder_angles, NDR_heading_errors, NDR_speeds, NDR_cross_track_errors, NDR_waypoints, NDR_u, NDR_v = \
        simulate_and_collect_trajectory(model_actionN, model_func=marinerwind, wind_mode="fixed", wind_speed=4)

    # 运行 低域随机模型 (LDR)
    LDR_trajectory, LDR_rudder_angles, LDR_heading_errors, LDR_speeds, LDR_cross_track_errors, LDR_waypoints, LDR_u, LDR_v = \
        simulate_and_collect_trajectory(model_actionL, model_func=marinerwind, wind_mode="fixed", wind_speed=4)

    # 运行 高域随机模型 (HDR)
    HDR_trajectory, HDR_rudder_angles, HDR_heading_errors, HDR_speeds, HDR_cross_track_errors, HDR_waypoints, HDR_u, HDR_v = \
        simulate_and_collect_trajectory(model_actionH, model_func=marinerwind, wind_mode="fixed", wind_speed=4)


    # 绘制横摇速度（u）对比图
    plt.figure(figsize=(8, 6))
    time_steps = np.arange(len(NDR_u))
    plt.plot(time_steps, NDR_u, 'r-', label='NDR Model u')
    plt.plot(time_steps, LDR_u, 'g-', label='LDR Model u')
    plt.plot(time_steps, HDR_u, 'y-', label='HDR Model u')
 
    plt.xlabel('Time Step')
    plt.ylabel('Lateral Speed u (m/s)')
    plt.title('Comparison of Lateral Speed u (Roll Motion)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制纵摇速度（v）对比图
    plt.figure(figsize=(8, 6))
    plt.plot(time_steps, NDR_v, 'r-', label='NDR Model v')
    plt.plot(time_steps, LDR_v, 'g-', label='LDR Model v')
    plt.plot(time_steps, HDR_v, 'y-', label='HDR Model v')
    plt.xlabel('Time Step')
    plt.ylabel('Longitudinal Speed v (m/s)')
    plt.title('Comparison of Longitudinal Speed v (Pitch Motion)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 转为 NumPy 数组
    NDR_traj = np.array(NDR_trajectory)
    LDR_traj = np.array(LDR_trajectory)
    HDR_traj = np.array(HDR_trajectory)
    NDR_waypoints = np.array(NDR_waypoints)

    # 创建图像
    plt.figure(figsize=(8, 6))

    # 绘制目标路径
    plt.plot(NDR_waypoints[:, 0], NDR_waypoints[:, 1], 'b--', label='Target Path')

    # 绘制轨迹，按照 LDR → HDR → NDR 顺序添加图例
    plt.plot(LDR_traj[:, 0], LDR_traj[:, 1], color='k', linestyle='-', label='LDR Path')
    plt.plot(HDR_traj[:, 0], HDR_traj[:, 1], color='y', linestyle='-.', label='HDR Path')
    plt.plot(NDR_traj[:, 0], NDR_traj[:, 1], color='r', linestyle='--', label='NDR Path')

    # 坐标轴与图例
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.ylim(0, 1000)
    plt.yticks(np.arange(0, 1000, 100))
    plt.xlim(0,2000)
    plt.legend()
    plt.grid(True)

    # ✅ 保存轨迹图像
    plt.savefig("con0", dpi=300, bbox_inches='tight')
    plt.show()

    # 舵角比较
    plt.figure(figsize=(8, 6))
    plt.plot(NDR_rudder_angles, 'r-', label='NDR Model Rudder Angle (deg)')
    plt.plot(LDR_rudder_angles, 'g-', label='LDR Model Rudder Angle (deg)')
    plt.plot(HDR_rudder_angles, 'y-', label='HDR Model Rudder Angle (deg)')
    plt.title('Rudder Angle Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Rudder Angle (deg)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 航向角比较
    plt.figure(figsize=(8, 6))
    plt.plot(NDR_heading_errors, 'r-', label='NDR Model Heading Error (deg)')
    plt.plot(LDR_heading_errors, 'g-', label='LDR Model Heading Error (deg)')
    plt.plot(HDR_heading_errors, 'y-', label='HDR Model Heading Error (deg)')
    plt.title('Heading Error Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Heading Error (deg)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 速度比较
    plt.figure(figsize=(8, 6))
    plt.plot(NDR_speeds, 'r-', label='NDR Model Speed (m/s)')
    plt.plot(LDR_speeds, 'g-', label='LDR Model Speed (m/s)')
    plt.plot(HDR_speeds, 'y-', label='HDR Model Speed (m/s)')
    plt.title('Speed Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 横向误差比较
    plt.figure(figsize=(8, 6))
    plt.plot(NDR_cross_track_errors, 'r-', label='NDR Model Cross Track Error (m)')
    plt.plot(LDR_cross_track_errors, 'g-', label='LDR Model Cross Track Error (m)')
    plt.plot(HDR_cross_track_errors, 'y-', label='HDR Model Cross Track Error (m)')
    plt.title('Cross Track Error Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Cross Track Error (m)')
    plt.legend()
    plt.grid(True)
    plt.show()


# 调用函数加载模型并显示比较
if __name__ == '__main__':
    load_and_compare_models()
