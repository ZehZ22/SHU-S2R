import random
import numpy as np
import torch
from gym import spaces
from isherwood72 import isherwood72
from sim2 import marinerwind
from ship_params import ship_params
from wave_irregular import waveforce_irregular


def set_seed(seeds):
    random.seed(seeds)
    np.random.seed(seeds)
    torch.manual_seed(seeds)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seeds)
        torch.cuda.manual_seed_all(seeds)


# 设置随机种子
seed = 20
set_seed(seed)

# 创建独立的随机生成器，不受全局随机种子的影响
no_seed_random = random.Random()


def current(x, y, U0):
    """
    动态计算洋流速度和方向。
    """
    V_c = 1.5 / U0  # 动态速度，例如随位置变化
    V_angle = np.radians(155)  # 动态方向角
    return V_c, V_angle


def decompose_current(beta_c, V_c, psi, U0):
    x = np.cos(beta_c) * V_c * U0
    y = np.sin(beta_c) * V_c * U0
    u_c = (np.cos(psi) * x - np.sin(psi) * y) / U0
    v_c = (np.sin(psi) * x + np.cos(psi) * y) / U0
    return u_c, v_c


class USVState:
    def __init__(self, waypoints, current_index, x, ui, model_func=marinerwind, wind_speed=4.0, wind_direction=60.0,
                 current_speed=2.5, current_direction=100.0,  # 洋流参数
                 wave_direction=np.radians(75), wave_height=2,
                 wind_mode="fixed", randomize_params=True, k1=1.0, k2=1.0, k3=1.0, w_chi=0.4, w_ey=0.5,
                 w_sigma_delta=0.0, U0=7.7175, mass=798e-5):

        """
        初始化 USVState，支持风速、洋流速度和船舶参数的域随机化。
        """
        # 初始化洋流参数
        self.current_speed = current_speed
        self.current_direction = current_direction
        self.V_c = current_speed / U0  # 无量纲化洋流速度
        self.V_angle = np.radians(current_direction) 
        self.V_c, self.V_angle = current(x[3], x[4], U0)
        # 初始化波浪参数
        self.wave_direction = np.radians(wave_direction) 
        self.wave_height = wave_height

        self.waypoints = waypoints
        self.current_index = current_index
        self.x = x
        self.ui = ui
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.w_chi = w_chi
        self.w_ey = w_ey
        self.w_sigma_delta = w_sigma_delta
        self.delta_history = []
        self.model_func = model_func

        # 风力参数
        self.wind_speed = wind_speed
        self.wind_direction = np.radians(wind_direction) 
        self.wind_mode = wind_mode

        self.randomize_params = randomize_params
        self.np_rng = np.random.default_rng(seed)  

        # 随机化参数
        if randomize_params:
            self.wind_speed = self.randomize_value(2.6, 3.6, no_seed_random)  # 风速范围 [5,7]knot
            self.wind_direction = np.radians(random.choice([0.0, 45.0, 90.0, 135.0]))  # 风向范围离散
            self.current_speed = self.randomize_value(1.0, 1.5, no_seed_random)  # 洋流速度范围 [2,3] knot
            self.current_direction = random.choice([0.0, 45.0, 90.0, 135.0])  # 洋流方向范围离散
            self.wave_height = self.randomize_value(3.0, 4.0, no_seed_random)  # 波高范围 [3,4] m
            self.wave_direction = np.radians(random.choice([0.0, 45.0, 90.0, 135.0]))

        self.U0 = U0  # 静态速度
        self.prev_waypoint = waypoints[current_index - 1]
        self.current_waypoint = waypoints[current_index]

        self.path_angle, self.waypoint_angle, self.distance_to_waypoint = self.calculate_path_parameters()
        self.heading_error = self.calculate_heading_error()
        self.cross_track_error = self.calculate_cross_track_error()

        self.action_space = spaces.Box(
            low=np.array([-35.0], dtype=np.float32),
            high=np.array([35.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -100.0, -35.0], dtype=np.float32),
            high=np.array([np.pi, 100.0, 35.0], dtype=np.float32),
            dtype=np.float32
        )
    
        self.time = 0.0  # 初始化时间为0

    def randomize_value(self, min_value, max_value, rng):
        """
        在指定的最小值和最大值之间生成随机数。
        :param min_value: 随机化的最小值
        :param max_value: 随机化的最大值
        :param rng: 独立的随机生成器
        :return: 随机化后的值
        """
        return rng.uniform(min_value, max_value)

    def get_full_state(self):
        """
        获取完整的船舶状态，包括航向误差、横向误差、舵角等信息。
        :return: 当前船舶状态（包含航向误差、横向误差、舵角等）
        """
        speed = np.sqrt((self.U0 + self.x[0]) ** 2 + self.x[1] ** 2)  
        return np.array(
            [self.heading_error, self.cross_track_error, self.x[6], self.x[3], self.x[4], self.x[5], speed, self.x[0],
             self.x[1]])

    def calculate_heading_error(self):
        self.path_angle = np.arctan2(self.current_waypoint[1] - self.prev_waypoint[1],
                                     self.current_waypoint[0] - self.prev_waypoint[0])
        heading_error = self.path_angle - self.x[5]
        # 将误差归一化到 [-π, π]
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        return heading_error

    def calculate_cross_track_error(self):
        d_wk_minus_1 = np.sqrt((self.x[4] - self.prev_waypoint[1]) ** 2 + (self.x[3] - self.prev_waypoint[0]) ** 2)
        chi_wk_minus_1 = np.arctan2(self.x[4] - self.prev_waypoint[1], self.x[3] - self.prev_waypoint[0])

        cross_track_error = np.sin(self.path_angle - chi_wk_minus_1) * d_wk_minus_1
        return cross_track_error

    def calculate_path_parameters(self):
        path_angle = np.arctan2(self.current_waypoint[1] - self.prev_waypoint[1],
                                self.current_waypoint[0] - self.prev_waypoint[0])
        waypoint_angle = np.arctan2(self.x[4] - self.prev_waypoint[1],
                                    self.x[3] - self.prev_waypoint[0])
        distance_to_waypoint = np.sqrt(
            (self.x[4] - self.prev_waypoint[1]) ** 2 + (self.x[3] - self.prev_waypoint[0]) ** 2)
        return path_angle, waypoint_angle, distance_to_waypoint

    def step(self, ui):
        dt = 0.5
        self.ui = ui[0]
        # 更新洋流参数
        self.V_c = self.current_speed / self.U0
        self.V_angle = np.radians(self.current_direction)

        # 计算洋流干扰
        u_c, v_c = decompose_current(self.V_angle, self.V_c, self.x[5], self.U0)
        scaling_factor = 0.2
        self.x[0] -= u_c * scaling_factor * dt
        self.x[1] -= v_c * scaling_factor * dt

        # 计算波浪扰动
        beta_r = self.wave_direction - self.x[5]  # 遭遇角
        beta_r = (beta_r + np.pi) % (2 * np.pi) - np.pi

        w = self.np_rng.uniform(0.1, 1.5, 30)
        fai = self.np_rng.uniform(0, 2 * np.pi, 30)

        tau_wave = waveforce_irregular(self.time, L=160.93, h=self.wave_height, T=3.0,
                                       beta_r=beta_r, w=w, fai=fai, U=self.U0)

        # 更新
        xdot, _ = self.model_func(self.x, self.ui,
                                  wind_speed=self.wind_speed,
                                  wind_direction=self.wind_direction,
                                  tau_wave=tau_wave)

        self.x += xdot * dt

        self.heading_error = self.calculate_heading_error()
        self.cross_track_error = self.calculate_cross_track_error()

        self.delta_history.append(self.x[6])
        if len(self.delta_history) > 20:
            self.delta_history.pop(0)

        heading_error_scalar = self.heading_error
        cross_track_error_scalar = self.cross_track_error
        rudder_angle = self.x[6]

        self.state = np.array([heading_error_scalar, cross_track_error_scalar, rudder_angle], dtype=np.float32)
        reward = self.calculate_reward()
        done = self.check_done()

        if not done and self.distance_to_waypoint < 10:
            self.current_index += 1
            self.prev_waypoint = self.current_waypoint
            self.current_waypoint = self.waypoints[self.current_index]
            self.path_angle, self.waypoint_angle, self.distance_to_waypoint = self.calculate_path_parameters()

        next_state = np.array([heading_error_scalar, cross_track_error_scalar, rudder_angle])
        if np.isnan(xdot).any() or np.isinf(xdot).any() or (np.abs(xdot) > 1e6).any():
            print("⚠️ xdot contains NaN, Inf, or extreme values!", xdot)
            xdot = np.clip(xdot, -1e6, 1e6)  # 限制 `xdot` 绝对值最大为 1e6

        self.time += dt  

        return next_state, reward, done, {}

    def calculate_reward(self):
        r_chi = -self.k1 * np.abs(self.heading_error)
        r_ey = -self.k2 * np.abs(self.cross_track_error)

        if len(self.delta_history) > 1:
            sigma_delta = np.std(self.delta_history)
        else:
            sigma_delta = 0
        r_sigma_delta = -self.k3 * sigma_delta

        reward = self.w_chi * r_chi + self.w_ey * r_ey + self.w_sigma_delta * r_sigma_delta
        return reward

    def check_done(self):
        if self.current_index >= len(self.waypoints) - 1:
            return True
        return False

    def reset(self):
        self.current_index = 1
        self.x = np.array([0.0, 0.0, 0.0, 300, 0, np.radians(20), 0.0])
        self.ui = 0.0
        self.delta_history = []

        # 随机化其他参数，每次 reset 时都会触发
        if self.randomize_params:
            self.wind_speed = self.randomize_value(2.6, 3.6, no_seed_random)  # 风速范围 [5,7]knot
            self.wind_direction = np.radians(random.choice([0.0, 45.0, 90.0, 135.0]))  # 风向范围离散
            self.current_speed = self.randomize_value(1.0, 1.5, no_seed_random)  # 洋流速度范围 [2,3] knot
            self.current_direction = random.choice([0.0, 45.0, 90.0, 135.0])  # 洋流方向范围离散
            self.wave_height = self.randomize_value(3.0, 4.0, no_seed_random)  # 波高范围 [3,4] m
            self.wave_direction = np.radians(random.choice([0.0, 45.0, 90.0, 135.0]))

        self.prev_waypoint = self.waypoints[self.current_index - 1]
        self.current_waypoint = self.waypoints[self.current_index]

        self.path_angle, self.waypoint_angle, self.distance_to_waypoint = self.calculate_path_parameters()

        self.heading_error = self.calculate_heading_error()
        self.cross_track_error = self.calculate_cross_track_error()
        rudder_angle = self.x[6]

        self.state = np.array([self.heading_error, self.cross_track_error, rudder_angle], dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        heading_error, cross_track_error, rudder_angle = self.state
        return np.array([heading_error, cross_track_error, rudder_angle], dtype=np.float32)


def generate_path(angle_deg, length, interval):
    angle_rad = np.radians(angle_deg)
    num_points = int(length // interval) + 1

    waypoints = []
    for i in range(num_points):
        x = i * interval * np.cos(angle_rad)
        y = i * interval * np.sin(angle_rad)
        waypoints.append((x, y))

    waypoints.append((length * np.cos(angle_rad), length * np.sin(angle_rad)))

    return waypoints
