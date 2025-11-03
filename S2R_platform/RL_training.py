import math
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch

from vessels.kcs import KCS_ode, L, d_em, rho, U_des
from utils.path_generator import (
    generate_straight_path,
    generate_random_line_path,
    generate_s_curve_path,
    generate_sine_path,
    PathManager,
)
from utils.LOS import ILOSpsi
from main_turning_circle import disturbance_func as build_ext_force, make_current_func


# =========================
# Environment
# =========================


@dataclass
class EnvConfig:
    # Time step is nondimensional (L/U_des)
    dt: float = 0.1
    max_steps: int = 5000
    rudder_limit_deg: float = 35.0
    with_disturbance: bool = False
    path_type: str = 'random_line'  # 'random_line', 'line', 'S_curve', 'sine'
    # Path params in nondimensional L units
    line_length: float = 40.0
    line_angle_deg: float = 0.0
    line_interval: float = 4.0
    # Random straight path sampling annulus radii (nd)
    r_min: float = 8.0
    r_max: float = 18.0
    # Path manager thresholds (nd)
    wp_switch_threshold: float = 1.6  # same magnitude as R_switch
    finish_tol: float = 1.8           # finished when within 0.5L of final waypoint
    # ILOS guidance parameters (nd)
    los_delta: float = 2.0            # lookahead distance 2L
    los_rswitch: float = 1.6          # switching radius 1.6L
    # Initial nondimensional surge speed (U/U_des)
    up0: float = 1.0


class KCSPathTrackingEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        # Build initial path (nondimensional waypoints)
        waypoints = self._make_waypoints()
        self.path_manager = PathManager(waypoints)
        self.wpt = {'x': np.array([p[0] for p in waypoints], dtype=float),
                    'y': np.array([p[1] for p in waypoints], dtype=float)}
        # LOS parameters (fixed multiples of L)
        self.kappa = 0.1
        self._set_los_params_from_path()

        # Disturbances
        if cfg.with_disturbance:
            wind_conf = dict(V_wind=10.0, Psi_wind_deg=30.0)
            wave_conf = dict(H=1.0, T=8.0, beta_deg=120.0, phase=0.0)
            self.ext_force = build_ext_force(wind_conf=wind_conf, wave_conf=wave_conf, ship=None)
            self.current_func = make_current_func(Vc_mps=0.4, beta_c_deg=150.0)
        else:
            self.ext_force = None
            self.current_func = None

        # Internal state
        self.t = 0.0
        self.x = np.zeros(7, dtype=float)
        self.x[0] = 1.0  # up = 1 (U_des)
        # Initialize heading aligned with first path segment
        self.x[5] = float(self.path_manager.start_psi)
        self.state = None
        self.step_count = 0

    def reset(self) -> np.ndarray:
        # If random path type, resample a new path each episode
        if self.cfg.path_type == 'random_line':
            waypoints = self._make_waypoints()
            self.path_manager = PathManager(waypoints)
            self.wpt = {'x': np.array([p[0] for p in waypoints], dtype=float),
                        'y': np.array([p[1] for p in waypoints], dtype=float)}
            self._set_los_params_from_path()

        # Reset ILOS persistent states to avoid index overflow across episodes
        if hasattr(ILOSpsi, 'k'):
            ILOSpsi.k = None
        if hasattr(ILOSpsi, 'y_int'):
            ILOSpsi.y_int = 0

        self.path_manager.current_index = 1
        self.t = 0.0
        self.step_count = 0
        self.x[:] = 0.0
        self.x[0] = float(self.cfg.up0)
        # Align initial heading with first path segment
        self.x[5] = float(self.path_manager.start_psi)
        self.x[6] = 0.0
        return self._obs()

    def _errors(self) -> Tuple[float, float]:
        # Cross-track error y1 via geometry
        cross_track_error = self.path_manager.calculate_cross_track_error(self.x[3:6])
        # Desired heading via ILOS guidance, then heading error y2 = psi_d - psi
        psi_d = ILOSpsi(
            x=self.x[3], y=self.x[4], Delta=self.Delta, kappa=self.kappa,
            h=self.cfg.dt, U=1.0, R_switch=self.R_switch, wpt=self.wpt, psi=self.x[5]
        )[0]
        # Wrap to [-pi, pi]
        y2 = (psi_d - self.x[5] + np.pi) % (2 * np.pi) - np.pi
        return cross_track_error, float(y2)

    def _obs(self) -> np.ndarray:
        y1, y2 = self._errors()
        delta = self.x[6]
        return np.array([y1, y2, delta], dtype=np.float32)

    def _reward(self, y1: float, y2: float) -> float:
        # r = 2*exp(-|y1|) - 1 + cos(y2)
        return 2.0 * math.exp(-abs(y1)) - 1.0 + math.cos(y2)

    def step(self, rudder_cmd_deg: float) -> Tuple[np.ndarray, float, bool, dict]:
        # Saturate command in deg, convert to rad
        lim = self.cfg.rudder_limit_deg
        rudder_cmd_deg = float(np.clip(rudder_cmd_deg, -lim, lim))
        delta_c = math.radians(rudder_cmd_deg)

        # Integrate dynamics (RK4 on nondim ODE)
        def f(t, v):
            return KCS_ode(t, v, delta_c, ext_force=self.ext_force, current_func=self.current_func)

        h = self.cfg.dt
        k1 = f(self.t, self.x)
        k2 = f(self.t + 0.5 * h, self.x + 0.5 * h * k1)
        k3 = f(self.t + 0.5 * h, self.x + 0.5 * h * k2)
        k4 = f(self.t + h, self.x + h * k3)
        self.x = self.x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.t += h
        self.step_count += 1

        # Update path manager waypoint based on current position
        self.path_manager.update_waypoint(self.x[3], self.x[4], threshold=self.cfg.wp_switch_threshold)

        # Compute reward and termination
        y1, y2 = self._errors()
        reward = self._reward(y1, y2)
        done = self.path_manager.is_finished(self.x[3], self.x[4], tolerance=self.cfg.finish_tol) \
               or (self.step_count >= self.cfg.max_steps)
        return self._obs(), reward, done, {
            'y1': y1,
            'y2': y2,
            'psi': self.x[5],
            'x': self.x[3],
            'y': self.x[4],
        }

    def get_full_state(self) -> np.ndarray:
        return self.x.copy()

    def _make_waypoints(self):
        cfg = self.cfg
        if cfg.path_type == 'line':
            return generate_straight_path(
                length=cfg.line_length, angle_deg=cfg.line_angle_deg, interval=cfg.line_interval
            )
        elif cfg.path_type == 'random_line':
            return generate_random_line_path(r_min=cfg.r_min, r_max=cfg.r_max, interval=cfg.line_interval)
        elif cfg.path_type == 'S_curve':
            return generate_s_curve_path()
        elif cfg.path_type == 'sine':
            return generate_sine_path()
        else:
            raise ValueError(f"Unsupported path_type: {cfg.path_type}")

    def _set_los_params_from_path(self):
        # Fixed lookahead and switching radius in nd units
        self.Delta = self.cfg.los_delta
        self.R_switch = self.cfg.los_rswitch


# =========================
# SAC components
# =========================


class Actor(torch.nn.Module):
    def __init__(self, obs_dim=3, hidden=128, act_limit_deg=35.0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
        )
        self.mu = torch.nn.Linear(hidden, 1)
        self.log_std = torch.nn.Parameter(torch.zeros(1))
        self.act_limit = act_limit_deg

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, obs):
        z = self.net(obs)
        mu = self.mu(z)
        std = torch.nn.functional.softplus(self.log_std) + 1e-6
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)  # in [-1,1]
        a_deg = a * self.act_limit
        # log-prob with tanh correction
        logp = dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-7)
        return a_deg, logp


class Critic(torch.nn.Module):
    def __init__(self, obs_dim=3, hidden=128):
        super().__init__()
        self.q = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + 1, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, obs, act_deg):
        x = torch.cat([obs, act_deg], dim=1)
        return self.q(x)


class ReplayBuffer:
    def __init__(self, size=100000):
        self.size = size
        self.buf = []

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))
        if len(self.buf) > self.size:
            self.buf.pop(0)

    def sample(self, batch=64):
        idx = np.random.choice(len(self.buf), size=min(batch, len(self.buf)), replace=False)
        s, a, r, s2, d = zip(*[self.buf[i] for i in idx])
        return (
            torch.as_tensor(np.array(s), dtype=torch.float32),
            torch.as_tensor(np.array(a), dtype=torch.float32).view(-1, 1),
            torch.as_tensor(np.array(r), dtype=torch.float32).view(-1, 1),
            torch.as_tensor(np.array(s2), dtype=torch.float32),
            torch.as_tensor(np.array(d), dtype=torch.float32).view(-1, 1),
        )


# =========================
# Training
# =========================


def train(with_disturbance=False, path_type='line', epochs=100, steps_per_epoch=2000, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = EnvConfig(with_disturbance=with_disturbance, path_type=path_type)
    env = KCSPathTrackingEnv(cfg)

    actor = Actor(act_limit_deg=cfg.rudder_limit_deg)
    q1 = Critic(); q2 = Critic()
    q1_t = Critic(); q2_t = Critic()
    q1_t.load_state_dict(q1.state_dict()); q2_t.load_state_dict(q2.state_dict())

    pi_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    q1_opt = torch.optim.Adam(q1.parameters(), lr=1e-3)
    q2_opt = torch.optim.Adam(q2.parameters(), lr=1e-3)
    log_alpha = torch.tensor(math.log(0.1), requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=1e-4)
    target_entropy = -1.0  # for 1D action

    buf = ReplayBuffer(size=200000)
    gamma = 0.99
    tau = 0.01

    def soft_update(src, dst, tau_):
        for p, p_t in zip(src.parameters(), dst.parameters()):
            p_t.data.mul_(1 - tau_).add_(tau_ * p.data)

    returns = []
    for ep in range(epochs):
        s = env.reset()
        ep_ret = 0.0
        for t in range(steps_per_epoch):
            with torch.no_grad():
                a_deg, _ = actor(torch.as_tensor(s).view(1, -1))
                a_deg = a_deg.item()
            s2, r, d, info = env.step(a_deg)
            buf.push(s, a_deg, r, s2, float(d))
            s = s2
            ep_ret += r

            # Updates
            if len(buf.buf) >= 1024:
                bs, ba, br, bs2, bd = buf.sample(256)
                with torch.no_grad():
                    a2, logp2 = actor(bs2)
                    q1_targ = q1_t(bs2, a2)
                    q2_targ = q2_t(bs2, a2)
                    q_targ = torch.min(q1_targ, q2_targ) - logp2.exp() * torch.exp(log_alpha)
                    y = br + (1 - bd) * gamma * q_targ

                q1_loss = torch.nn.functional.mse_loss(q1(bs, ba), y)
                q2_loss = torch.nn.functional.mse_loss(q2(bs, ba), y)
                q1_opt.zero_grad(); q1_loss.backward(); q1_opt.step()
                q2_opt.zero_grad(); q2_loss.backward(); q2_opt.step()

                # Policy loss
                a_pi, logp_pi = actor(bs)
                q1_pi = q1(bs, a_pi)
                q2_pi = q2(bs, a_pi)
                q_pi = torch.min(q1_pi, q2_pi)
                pi_loss = (logp_pi.exp() * torch.exp(log_alpha) - q_pi).mean()
                pi_opt.zero_grad(); pi_loss.backward(); pi_opt.step()

                # Temperature loss
                alpha_loss = (-(logp_pi + target_entropy).detach() * torch.exp(log_alpha)).mean()
                alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

                # Targets soft update
                soft_update(q1, q1_t, tau)
                soft_update(q2, q2_t, tau)

            if d:
                break

        returns.append(ep_ret)
        print(f"Epoch {ep+1}/{epochs} Return: {ep_ret:.3f}")

    # Save models
    os.makedirs('results', exist_ok=True)
    torch.save(actor.state_dict(), os.path.join('results', 'actor_kcs.pth'))
    torch.save(q1.state_dict(), os.path.join('results', 'critic1_kcs.pth'))
    torch.save(q2.state_dict(), os.path.join('results', 'critic2_kcs.pth'))


if __name__ == '__main__':
    # Example: training with random straight paths per episode, no disturbances
    train(with_disturbance=False, path_type='random_line', epochs=200, steps_per_epoch=5000)
