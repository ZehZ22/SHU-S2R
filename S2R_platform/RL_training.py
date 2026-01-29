import math
import os
import csv
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from vessels.kcs import KCS_ode, L, d_em, rho, U_des
from utils.path_generator import (
    PathManager,
    generate_random_line_path,
    generate_s_curve_path,
    generate_sine_path,
    generate_straight_path,
)
from utils.LOS import ILOSpsi
from utils.domain_randomizer import (
    DomainRandomizer,
    CurriculumDomainRandomizer,
    LDRDomainRandomizer,
    HDRDomainRandomizer,
    DisturbanceSample,
    KNOT_TO_MPS,
)
from disturbances.wind import isherwood72
from disturbances.wave import waveforce_irregular
from disturbances.current import decompose_current
from ship_params import ShipParams


# =========================
# Environment
# =========================


@dataclass
class EnvConfig:
    # Time step is nondimensional (L/U_des)
    dt: float = 0.1
    max_steps: int = 2000
    rudder_limit_deg: float = 35.0
    delta_rate_weight: float = 50.0  # penalty on rudder rate to smooth control
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


def build_ext_force(wind_conf=None, wave_conf=None, ship=None, rng_seed: int = 123):
    """Construct external force callback using wind/wave models and ship parameters."""
    scale_F = 0.5 * rho * L * d_em * (U_des ** 2)
    scale_N = 0.5 * rho * (L ** 2) * d_em * (U_des ** 2)

    def _ext_force(t_nd, v, _):
        t_sec = t_nd * (L / U_des)
        psi = v[5]
        up = v[0]
        U_ship = up * U_des

        X_SI = 0.0
        Y_SI = 0.0
        N_SI = 0.0

        # Wind load (requires ship params)
        if wind_conf is not None and ship is not None:
            V_wind = wind_conf.get('V_wind', 0.0)
            Psi_wind = math.radians(wind_conf.get('Psi_wind_deg', 0.0))
            gamma_r = Psi_wind - psi
            tauW, _, _, _ = isherwood72(
                gamma_r=gamma_r,
                V_r=V_wind,
                Loa=ship.Loa,
                B=ship.B,
                ALw=ship.ALw,
                AFw=ship.AFw,
                A_SS=ship.A_SS,
                S=ship.S,
                C=ship.C,
                M=ship.M,
                rho_air=ship.rho_air,
            )
            X_SI += tauW[0]
            Y_SI += tauW[1]
            N_SI += tauW[2]

        # Wave load
        if wave_conf is not None:
            h = wave_conf.get('H', 0.0)
            T = wave_conf.get('T', 10.0)
            beta_wave = math.radians(wave_conf.get('beta_deg', 0.0))
            w0 = 2 * math.pi / max(T, 1e-6)
            Nw = 21
            w = np.linspace(0.8 * w0, 1.2 * w0, Nw)
            rng = np.random.default_rng(rng_seed)
            fai = rng.uniform(0, 2 * math.pi, size=Nw)
            beta_r = beta_wave - psi
            tau_wave = waveforce_irregular(
                t=t_sec, L=L, h=h, T=T, beta_r=beta_r, w=w, fai=fai, U=U_ship,
            )
            X_SI += tau_wave[0]
            Y_SI += tau_wave[1]
            N_SI += tau_wave[2]

        Xp = X_SI / scale_F
        Yp = Y_SI / scale_F
        Np = N_SI / scale_N
        return np.array([Xp, Yp, Np])

    return _ext_force


def make_current_func(Vc_mps=0.0, beta_c_deg=0.0):
    """Ambient current components in body frame (nondimensional)."""
    def _cur(t_nd, v, _):
        psi = v[5]
        beta_c = math.radians(beta_c_deg)
        V_c_nd = Vc_mps / U_des
        u_c, v_c = decompose_current(beta_c=beta_c, V_c=V_c_nd, psi=psi, U0=U_des)
        return u_c, v_c
    return _cur


class KCSPathTrackingEnv:
    def __init__(self, cfg: EnvConfig, domain_randomizer: Optional[DomainRandomizer] = None):
        self.cfg = cfg
        self.domain_randomizer = domain_randomizer
        # Build initial path (nondimensional waypoints)
        waypoints = self._make_waypoints()
        self.path_manager = PathManager(waypoints)
        self.wpt = {'x': np.array([p[0] for p in waypoints], dtype=float),
                    'y': np.array([p[1] for p in waypoints], dtype=float)}
        # LOS parameters (fixed multiples of L)
        self.kappa = 0.1
        self._set_los_params_from_path()

        # Disturbances
        self.ext_force = None
        self.current_func = None
        self._last_disturbance_sample: Optional[DisturbanceSample] = None
        self._configure_disturbances()

        # Internal state
        self.t = 0.0
        self.x = np.zeros(7, dtype=float)
        self.x[0] = 1.0  # up = 1 (U_des)
        # Initialize heading aligned with first path segment
        self.x[5] = float(self.path_manager.start_psi)
        self.prev_delta = float(self.x[6])
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
        if self.cfg.with_disturbance and self.domain_randomizer is not None:
            self._configure_disturbances()

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
        self.prev_delta = float(self.x[6])
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

    def _reward(self, y1: float, y2: float, delta_rate: float) -> float:
        """Reward with rudder-rate penalty to discourage high-frequency actuation."""
        smooth_penalty = self.cfg.delta_rate_weight * (delta_rate ** 2)
        return 2.0 * math.exp(-abs(y1)) - 1.0 + math.cos(y2) - smooth_penalty

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
        delta_rate = float(self.x[6] - self.prev_delta)
        self.prev_delta = float(self.x[6])

        # Update path manager waypoint based on current position
        self.path_manager.update_waypoint(self.x[3], self.x[4], threshold=self.cfg.wp_switch_threshold)

        # Compute reward and termination
        y1, y2 = self._errors()
        reward = self._reward(y1, y2, delta_rate)
        done = self.path_manager.is_finished(self.x[3], self.x[4], tolerance=self.cfg.finish_tol) \
               or (self.step_count >= self.cfg.max_steps)
        return self._obs(), reward, done, {
            'y1': y1,
            'y2': y2,
            'psi': self.x[5],
            'x': self.x[3],
            'y': self.x[4],
            'disturbance': self._last_disturbance_sample,
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

    def _configure_disturbances(self) -> None:
        if not self.cfg.with_disturbance:
            self.ext_force = None
            self.current_func = None
            self._last_disturbance_sample = None
            return

        ship = ShipParams()
        if self.domain_randomizer is not None:
            sample = self.domain_randomizer.sample()
            wind_conf = sample.wind_conf
            wave_conf = sample.wave_conf
            current_conf = sample.current_conf
            self._last_disturbance_sample = sample
        else:
            # Velocities are defined in knots here and converted to m/s for the disturbance models.
            wind_conf = dict(V_wind=4.0 * KNOT_TO_MPS, Psi_wind_deg=60.0)
            wave_conf = dict(H=4.0, T=8.0, beta_deg=70.0, phase=0.0)
            current_conf = dict(Vc_mps=2.5 * KNOT_TO_MPS, beta_c_deg=100.0)
            self._last_disturbance_sample = None

        self.ext_force = build_ext_force(wind_conf=wind_conf, wave_conf=wave_conf, ship=ship)
        self.current_func = make_current_func(
            Vc_mps=current_conf.get('Vc_mps', 0.0),
            beta_c_deg=current_conf.get('beta_c_deg', 0.0),
        )


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


def train(
    with_disturbance: bool = False,
    path_type: str = 'line',
    epochs: int = 100,
    steps_per_epoch: int = 2000,
    seed: int = 0,
    domain_randomization: bool = False,
    domain_randomizer_kwargs: Optional[dict] = None,
    curriculum_domain_randomization: bool = False,
    curriculum_kwargs: Optional[dict] = None,
    ldr_domain_randomization: bool = False,
    hdr_domain_randomization: bool = False,
    log_to_csv: bool = True,
    log_dir: str = 'logs',
    log_name: str = 'training_log.csv',
    log_every: int = 1,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = EnvConfig(with_disturbance=with_disturbance, path_type=path_type)
    domain_randomizer = None
    if curriculum_domain_randomization:
        params = curriculum_kwargs or {}
        domain_randomizer = CurriculumDomainRandomizer(seed=seed, **params)
        cfg.with_disturbance = True
    elif ldr_domain_randomization:
        domain_randomizer = LDRDomainRandomizer(seed=seed)
        cfg.with_disturbance = True
    elif hdr_domain_randomization:
        domain_randomizer = HDRDomainRandomizer(seed=seed)
        cfg.with_disturbance = True
    elif domain_randomization:
        params = domain_randomizer_kwargs or {}
        domain_randomizer = DomainRandomizer(seed=seed, **params)
        cfg.with_disturbance = True
    env = KCSPathTrackingEnv(cfg, domain_randomizer=domain_randomizer)

    log_file = None
    log_writer = None
    log_path = None
    if log_to_csv:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_name)
        file_exists = os.path.exists(log_path)
        log_file = open(log_path, 'a', newline='', encoding='utf-8')
        log_writer = csv.writer(log_file)
        if (not file_exists) or (os.path.getsize(log_path) == 0):
            log_writer.writerow([
                'episode', 'steps', 'ep_return',
                'avg_abs_y1', 'avg_abs_y2', 'progress',
                'wind_speed_mps', 'wind_dir_deg',
                'wave_height_m', 'wave_period_s', 'wave_dir_deg',
                'current_speed_mps', 'current_dir_deg'
            ])

    actor = Actor(act_limit_deg=cfg.rudder_limit_deg)
    q1 = Critic(); q2 = Critic()
    q1_t = Critic(); q2_t = Critic()
    q1_t.load_state_dict(q1.state_dict()); q2_t.load_state_dict(q2.state_dict())

    pi_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    q1_opt = torch.optim.Adam(q1.parameters(), lr=1e-3)
    q2_opt = torch.optim.Adam(q2.parameters(), lr=1e-3)
    log_alpha = torch.tensor(math.log(0.1), requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=1e-5)
    target_entropy = -1.0  # for 1D action

    buf = ReplayBuffer(size=200000)
    gamma = 0.99
    tau = 0.01

    def soft_update(src, dst, tau_):
        for p, p_t in zip(src.parameters(), dst.parameters()):
            p_t.data.mul_(1 - tau_).add_(tau_ * p.data)

    returns = []
    for ep in range(epochs):
        progress = ep / max(1, epochs - 1)
        if hasattr(domain_randomizer, "update"):
            domain_randomizer.update(progress)
        s = env.reset()
        ep_ret = 0.0
        sum_abs_y1 = 0.0
        sum_abs_y2 = 0.0
        steps = 0
        for t in range(steps_per_epoch):
            with torch.no_grad():
                a_deg, _ = actor(torch.as_tensor(s).view(1, -1))
                a_deg = a_deg.item()
            s2, r, d, info = env.step(a_deg)
            sum_abs_y1 += abs(info.get('y1', 0.0))
            sum_abs_y2 += abs(info.get('y2', 0.0))
            steps += 1
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

        avg_abs_y1 = sum_abs_y1 / max(1, steps)
        avg_abs_y2 = sum_abs_y2 / max(1, steps)

        returns.append(ep_ret)
        print(f"Episode {ep+1} cumulative reward: {ep_ret:.3f}")
        if log_writer and (ep % max(1, log_every) == 0):
            sample = env._last_disturbance_sample
            wind = sample.wind_conf if sample is not None else {}
            wave = sample.wave_conf if sample is not None else {}
            current = sample.current_conf if sample is not None else {}
            log_writer.writerow([
                ep + 1, steps, float(ep_ret),
                float(avg_abs_y1), float(avg_abs_y2), float(progress),
                wind.get('V_wind'), wind.get('Psi_wind_deg'),
                wave.get('H'), wave.get('T'), wave.get('beta_deg'),
                current.get('Vc_mps'), current.get('beta_c_deg')
            ])
            log_file.flush()

    # Save models
    os.makedirs('policys', exist_ok=True)
    torch.save(actor.state_dict(), os.path.join('policys', 'actor_kcs.pth'))
    torch.save(q1.state_dict(), os.path.join('policys', 'critic1_kcs.pth'))
    torch.save(q2.state_dict(), os.path.join('policys', 'critic2_kcs.pth'))

    if log_file is not None:
        log_file.close()

    return returns


def plot_training_returns(returns):
    if not returns:
        print("No reward data to visualize.")
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Failed to plot reward curve: {exc}")
        return

    epochs = np.arange(1, len(returns) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, returns, marker='o', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Training Reward Curve')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Example: random straight paths per episode with domain randomization enabled
    training_returns = train(
        with_disturbance=True,
        path_type='random_line',
        epochs=500,
        steps_per_epoch=5000,
        curriculum_domain_randomization=True, curriculum_kwargs={'schedule': 'quadratic'},
        ldr_domain_randomization=False,
        hdr_domain_randomization=False,
    )
    plot_training_returns(training_returns)
