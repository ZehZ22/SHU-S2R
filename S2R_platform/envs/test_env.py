import numpy as np
from disturbances.wave import wave_model
from disturbances.wind import wind_model
from disturbances.current import current, decompose_current
from ship_params import ShipParams
from importlib import import_module


class USVSimulator:
    def __init__(self, path_manager, ship_model,
                 wind_config, wave_config, current_config, dt=0.1, random_start=False):
        self.path_manager = path_manager
        self.dt = dt
        self.ship_params = ShipParams()
        self.random_start = random_start

        # ==== 环境参数 ====
        self.wind_method = wind_config.get('method', None)
        self.wind_speed = wind_config.get('speed', 0.0)
        self.wind_direction = np.radians(wind_config.get('direction', 0.0))

        self.wave_method = wave_config.get('method', None)
        self.wave_params = wave_config
        self.wave_state = (0.0, 0.0)
        self.z_0 = 19.4
        self.gau_noise = np.zeros(3)

        self.current_method = current_config.get('method', None)
        self.ship_model = getattr(import_module(f"models.{ship_model}"), ship_model)

        # 初始化路径
        self.waypoints = path_manager.path
        self.current_index = 1
        self.prev_waypoint = self.waypoints[0]
        self.current_waypoint = self.waypoints[1]

        self.reset()

    def reset(self):
        self.current_index = 1
        self.prev_waypoint = self.waypoints[0]
        self.current_waypoint = self.waypoints[1]
        self.time = 0.0
        self.ui = 0.0
        self.wave_state = (0.0, 0.0)

        if self.random_start:
            x0 = np.random.uniform(200, 400)
            y0 = np.random.uniform(-50, 50)
            psi0 = np.radians(np.random.uniform(10, 30))
        else:
            x0 = 0.0
            y0 = -0.0
            psi0 = np.radians(20)

        self.x = np.array([
            0.0, 0.0, 0.0,  # u, v, r
            x0, y0, psi0,   # x, y, psi
            0.0             # rudder
        ], dtype=np.float32)

        self.delta_history = []
        self.update_errors()
        return self._get_obs()

    def step(self, ui):
        self.ui = ui[0]
        psi = self.x[5]
        U0 = self.ship_params.U

        if self.current_method:
            V_c, V_angle = current(self.x[3], self.x[4], U0)
            u_c, v_c = decompose_current(V_angle, V_c, psi, U0)
            self.x[0] -= u_c * 0.2 * self.dt
            self.x[1] -= v_c * 0.2 * self.dt

        if self.wind_method:
            wind_force, *_ = wind_model(
                self.wind_method, self.time, self.wind_speed, self.wind_direction,
                self.x[3:6], self.x[0:2], ship=self.ship_params
            )
        else:
            wind_force = np.zeros(3)

        if self.wave_method:
            if self.wave_method == 'func1':
                wave_args = {
                    't': self.time,
                    'a': self.wave_params['wave_a'],
                    'beta': np.radians(self.wave_params['wave_beta']),
                    'T_0': self.wave_params['wave_T0'],
                    'zeta4': self.wave_params['wave_zeta4'],
                    'T4': self.wave_params['wave_T4'],
                    'GMT': self.ship_params.GMT,
                    'Cb': self.ship_params.Cb,
                    'U': self.ship_params.U,
                    'L': self.ship_params.Loa,
                    'B': self.ship_params.B,
                    'T': self.ship_params.T,
                    'rho_water': self.ship_params.rho_water,
                }
            else:
                wave_args = {
                    't': self.time,
                    'dt': self.dt,
                    'wave_state': self.wave_state,
                    'wind_speed': self.wind_speed,
                    'z_0': self.z_0,
                    'eta': self.x[3:6],
                    'nu': self.x[0:2],
                    'Psi_wind': self.wind_direction,
                    'gau_noise': self.gau_noise,
                    'ship': self.ship_params,
                    'wave_a': self.wave_params['wave_a'],
                    'wave_beta': np.radians(self.wave_params['wave_beta']),
                    'wave_T0': self.wave_params['wave_T0'],
                    'wave_zeta4': self.wave_params['wave_zeta4'],
                    'wave_T4': self.wave_params['wave_T4']
                }
            z_heave, phi_roll_deg, theta_pitch_deg, self.wave_state = wave_model(self.wave_method, **wave_args)
        else:
            z_heave = phi_roll_deg = theta_pitch_deg = 0.0

        self.x[0] += z_heave / self.ship_params.Loa
        self.x[2] += np.radians(phi_roll_deg) / self.ship_params.Loa

        xdot = self.ship_model(self.x, self.ui, wind_force=wind_force)
        self.x += xdot * self.dt
        self.time += self.dt

        self.update_errors()
        done = self.path_manager.is_finished(self.x[3], self.x[4])
        reward = self.get_reward()

        return self._get_obs(), reward, done, {}

    def calculate_heading_error(self):
        return self.path_manager.calculate_heading_error(self.x[3:6])

    def calculate_cross_track_error(self):
        return self.path_manager.calculate_cross_track_error(self.x[3:6])

    def update_errors(self):
        self.heading_error = self.calculate_heading_error()
        self.cross_track_error = self.calculate_cross_track_error()

    def get_reward(self):
        if not hasattr(self, 'delta_history'):
            self.delta_history = []
        self.delta_history.append(self.x[6])
        sigma_delta = np.std(self.delta_history) if len(self.delta_history) > 1 else 0.0

        k1, k2, k3 = 1.0, 1.0, 1.0
        w_chi, w_ey, w_sigma_delta = 0.4, 0.5, 0.0

        r_chi = -k1 * abs(self.heading_error)
        r_ey = -k2 * abs(self.cross_track_error)
        r_sigma_delta = -k3 * sigma_delta

        return w_chi * r_chi + w_ey * r_ey + w_sigma_delta * r_sigma_delta

    def _get_obs(self):
        return np.array([self.heading_error, self.cross_track_error, self.x[6]], dtype=np.float32)

    def get_full_state(self):
        return self.x.copy()