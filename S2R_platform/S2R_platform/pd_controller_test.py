import math
from typing import Tuple, Dict

import numpy as np

from RL_training import EnvConfig, KCSPathTrackingEnv
from utils.LOS import ILOSpsi
import matplotlib.pyplot as plt


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
    x0: float = None,
    y0: float = None,
    psi0_deg: float = 20,
    up0: float = None,
    r0: float = None,
    delta0_deg: float = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], KCSPathTrackingEnv]:
    """Run a single PD-controlled episode and collect trajectory plus control/error logs."""
    cfg = EnvConfig(path_type=path_type)
    cfg.max_steps = steps
    env = KCSPathTrackingEnv(cfg)

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
        steps=10000,
        kp=5.0,
        td=1.0,
        path_type="S_curve",
    )
    plot_pd_results(env, traj, log)
