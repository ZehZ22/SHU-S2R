import math
from typing import Tuple, Dict

import numpy as np

from RL_training import EnvConfig, KCSPathTrackingEnv
from utils.LOS import ILOSpsi
import matplotlib.pyplot as plt
from disturbances.wind import isherwood72
from disturbances.wave import waveforce_irregular
from disturbances.current import decompose_current
from utils.domain_randomizer import KNOT_TO_MPS
from ship_params import ShipParams
from vessels.kcs import L, d_em, rho, U_des


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
            V_wind = wind_conf.get("V_wind", 0.0)
            Psi_wind = math.radians(wind_conf.get("Psi_wind_deg", 0.0))
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
            h = wave_conf.get("H", 0.0)
            T = wave_conf.get("T", 10.0)
            beta_wave = math.radians(wave_conf.get("beta_deg", 0.0))
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


def pd_controller(
    state: np.ndarray,
    waypoints: dict,
    *,
    Kp: float = 5.0,
    Td: float = 2.0,
    Delta: float = 2.0,
    kappa: float = 0.1,
    U: float = 1.0,
    R_switch: float = 1.6,
    h: float = 0.1,
):
    """Compute rudder command (rad) using a simple PD law with ILOS heading."""
    r = float(state[2])
    psi = float(state[5])
    x_pos = float(state[3])
    y_pos = float(state[4])

    psi_d = ILOSpsi(x_pos, y_pos, Delta, kappa, h, U, R_switch, waypoints)
    angle_error = (psi - psi_d + math.pi) % (2 * math.pi) - math.pi
    delta_rad = -Kp * (angle_error + Td * r)
    return delta_rad, psi_d


def run_pd_episode(
    steps: int = 2000,
    kp: float = 1.0,
    td: float = 1.0,
    path_type: str = "random_line",
    with_disturbance: bool = True,
    wind_conf: Dict = None,
    wave_conf: Dict = None,
    current_conf: Dict = None,
    x0: float = None,
    y0: float = None,
    psi0_deg: float = 20,
    up0: float = None,
    r0: float = None,
    delta0_deg: float = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], KCSPathTrackingEnv]:
    """Run a single PD-controlled episode and collect trajectory plus control/error logs."""
    cfg = EnvConfig(path_type=path_type, with_disturbance=with_disturbance)
    cfg.max_steps = steps
    env = KCSPathTrackingEnv(cfg)

    # Configure disturbances if requested
    if with_disturbance:
        ship = ShipParams()
        if wind_conf or wave_conf or current_conf:
            wind_cfg = wind_conf or dict(V_wind=4.0 * KNOT_TO_MPS, Psi_wind_deg=60.0)
            wave_cfg = wave_conf or dict(H=4.0, T=8.0, beta_deg=70.0, phase=0.0)
            current_cfg = current_conf or dict(Vc_mps=2.5 * KNOT_TO_MPS, beta_c_deg=100.0)
            env.ext_force = build_ext_force(wind_conf=wind_cfg, wave_conf=wave_cfg, ship=ship)
            env.current_func = make_current_func(
                Vc_mps=current_cfg.get("Vc_mps", 0.0),
                beta_c_deg=current_cfg.get("beta_c_deg", 0.0),
            )
        else:
            # Fallback to default disturbance configuration with real ship params for wind force
            default_wind = dict(V_wind=4.0 * KNOT_TO_MPS, Psi_wind_deg=60.0)
            default_wave = dict(H=4.0, T=8.0, beta_deg=70.0, phase=0.0)
            default_current = dict(Vc_mps=2.5 * KNOT_TO_MPS, beta_c_deg=100.0)
            env.ext_force = build_ext_force(wind_conf=default_wind, wave_conf=default_wave, ship=ship)
            env.current_func = make_current_func(
                Vc_mps=default_current.get("Vc_mps", 0.0),
                beta_c_deg=default_current.get("beta_c_deg", 0.0),
            )

    # Reset ILOS persistent states to avoid leakage across runs
    if hasattr(ILOSpsi, "k"):
        ILOSpsi.k = None
    if hasattr(ILOSpsi, "y_int"):
        ILOSpsi.y_int = 0

    env.reset()
    # Optional manual initialization (positions/heading/velocities/rudder)
    if x0 is not None:
        env.x[3] = float(x0)
    if y0 is not None:
        env.x[4] = float(y0)
    if psi0_deg is not None:
        env.x[5] = math.radians(psi0_deg)
    if up0 is not None:
        env.x[0] = float(up0)
    if r0 is not None:
        env.x[2] = float(r0)
    if delta0_deg is not None:
        env.x[6] = math.radians(delta0_deg)
        env.prev_delta = env.x[6]

    xs, ys = [], []
    deltas_deg, heading_err_deg, cross_track_err, ts = [], [], [], []
    t_now = 0.0
    for t in range(cfg.max_steps):
        state = env.get_full_state()
        delta_rad, psi_d = pd_controller(
            state,
            env.wpt,
            Kp=kp,
            Td=td,
            Delta=env.Delta,
            kappa=env.kappa,
            U=1.0,
            R_switch=env.R_switch,
            h=cfg.dt,
        )
        delta_deg = math.degrees(delta_rad)
        delta_deg = float(np.clip(delta_deg, -cfg.rudder_limit_deg, cfg.rudder_limit_deg))
        _, _, done, info = env.step(delta_deg)
        xs.append(info["x"])
        ys.append(info["y"])
        deltas_deg.append(delta_deg)
        heading_err_deg.append(math.degrees(info["y2"]))
        cross_track_err.append(info["y1"])
        ts.append(t_now)
        t_now += cfg.dt
        if done:
            break
    print(f"Episode finished: steps={t + 1}")

    traj = np.column_stack((np.array(xs), np.array(ys))) if xs else np.empty((0, 2))
    log = {
        "t": np.array(ts),
        "delta_deg": np.array(deltas_deg),
        "heading_error_deg": np.array(heading_err_deg),
        "cross_track_error": np.array(cross_track_err),
    }
    return traj, log, env


def plot_pd_results(env: KCSPathTrackingEnv, traj: np.ndarray, log: Dict[str, np.ndarray]):
    """Plot trajectory and time histories in the same style as utils/visualize.py."""
    # Trajectory
    wps = np.array(env.path_manager.path)
    plt.figure(figsize=(8, 6))
    plt.plot(wps[:, 0], wps[:, 1], "k--", label="Path")
    if len(traj) > 0:
        plt.plot(traj[:, 0], traj[:, 1], "b", label="PD track")
    plt.axis("equal")
    plt.xlabel("x (nd)")
    plt.ylabel("y (nd)")
    plt.title("PD Path Tracking Trajectory")
    plt.legend()

    # Time histories
    t = log["t"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, log["delta_deg"])
    axes[0].set_ylabel("delta (deg)")
    axes[0].set_title("Rudder angle")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(t, log["cross_track_error"])
    axes[1].set_ylabel("y1 (nd)")
    axes[1].set_title("Cross-track error")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(t, log["heading_error_deg"])
    axes[2].set_ylabel("y2 (deg)")
    axes[2].set_xlabel("t (nd)")
    axes[2].set_title("Heading error")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    traj, log, env = run_pd_episode(
        steps=2000,
        kp=5.0,
        td=1.0,
        path_type="S_curve",
        with_disturbance=True,
        wind_conf=dict(V_wind=400.0 * KNOT_TO_MPS, Psi_wind_deg=160.0),
        wave_conf=dict(H=4.0, T=8.0, beta_deg=70.0, phase=0.0),
        current_conf=dict(Vc_mps=2.5 * KNOT_TO_MPS, beta_c_deg=100.0),
    )
    plot_pd_results(env, traj, log)
