import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path when running as a script
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from RL_training import EnvConfig, KCSPathTrackingEnv, build_ext_force, make_current_func
from ship_params import ShipParams
from utils.domain_randomizer import KNOT_TO_MPS


@dataclass
class SeaState:
    name: str
    wind_kn: Tuple[float, float]
    wave_m: Tuple[float, float]
    current_kn: Tuple[float, float]


SEA_STATES = [
    SeaState("SS1", wind_kn=(0.0, 0.0), wave_m=(0.0, 0.0), current_kn=(0.0, 0.0)),
    SeaState("SS2", wind_kn=(4.0, 7.0), wave_m=(0.5, 1.0), current_kn=(0.5, 1.0)),
    SeaState("SS3", wind_kn=(8.0, 11.0), wave_m=(1.5, 2.0), current_kn=(1.5, 2.0)),
    SeaState("SS4", wind_kn=(12.0, 15.0), wave_m=(2.5, 3.0), current_kn=(2.5, 3.0)),
]

DIR_SET = (0.0, 45.0, 90.0, 135.0)


def pd_control(error: float, yaw_rate: float, kp: float, td: float) -> float:
    """PD control: delta = -Kp * (error + Td * r)."""
    return -kp * (error + td * yaw_rate)


def sample_disturbance(rng: np.random.Generator, sea: SeaState) -> Dict:
    wind_speed = rng.uniform(sea.wind_kn[0], sea.wind_kn[1]) * KNOT_TO_MPS
    current_speed = rng.uniform(sea.current_kn[0], sea.current_kn[1]) * KNOT_TO_MPS
    wave_height = rng.uniform(sea.wave_m[0], sea.wave_m[1])

    wind_dir = rng.choice(DIR_SET) if wind_speed > 0 else 0.0
    current_dir = rng.choice(DIR_SET) if current_speed > 0 else 0.0
    wave_dir = rng.choice(DIR_SET) if wave_height > 0 else 0.0

    return dict(
        wind=dict(V_wind=wind_speed, Psi_wind_deg=wind_dir),
        wave=dict(H=wave_height, T=8.0, beta_deg=wave_dir, phase=0.0),
        current=dict(Vc_mps=current_speed, beta_c_deg=current_dir),
    )


def run_pd_episode(
    *,
    steps: int,
    kp: float,
    td: float,
    wind_conf: Dict,
    wave_conf: Dict,
    current_conf: Dict,
    path_type: str = "S_curve",
) -> Tuple[np.ndarray, Dict[str, np.ndarray], KCSPathTrackingEnv]:
    cfg = EnvConfig(path_type=path_type, with_disturbance=True)
    cfg.max_steps = steps
    env = KCSPathTrackingEnv(cfg)

    ship = ShipParams()
    env.ext_force = build_ext_force(wind_conf=wind_conf, wave_conf=wave_conf, ship=ship)
    env.current_func = make_current_func(
        Vc_mps=current_conf.get("Vc_mps", 0.0),
        beta_c_deg=current_conf.get("beta_c_deg", 0.0),
    )

    obs = env.reset()
    y2_cur = float(obs[1])

    xs, ys = [], []
    deltas_deg, heading_err_rad, cross_track_err, ts = [], [], [], []
    t_now = 0.0
    completed = False
    for _ in range(cfg.max_steps):
        state = env.get_full_state()
        r = float(state[2])
        angle_error = -y2_cur
        delta_rad = pd_control(angle_error, r, kp=kp, td=td)
        delta_cmd_deg = math.degrees(delta_rad)
        delta_cmd_deg = float(np.clip(delta_cmd_deg, -cfg.rudder_limit_deg, cfg.rudder_limit_deg))

        _, _, done, info = env.step(delta_cmd_deg)
        y2_cur = float(info["y2"])
        xs.append(info["x"])
        ys.append(info["y"])
        # Keep the metric definition consistent with evaluate_sea_states.py.
        deltas_deg.append(np.degrees(env.get_full_state()[6]))
        heading_err_rad.append(y2_cur)
        cross_track_err.append(float(info["y1"]))
        ts.append(t_now)
        t_now += cfg.dt
        if done:
            completed = env.path_manager.is_finished(
                env.x[3], env.x[4], tolerance=env.cfg.finish_tol
            )
            break

    traj = np.column_stack((np.array(xs), np.array(ys))) if xs else np.empty((0, 2))
    log = {
        "t": np.array(ts),
        "delta_deg": np.array(deltas_deg),
        "heading_error_rad": np.array(heading_err_rad),
        "cross_track_error": np.array(cross_track_err),
        "completed": completed,
    }
    return traj, log, env


def plot_pd_results(
    env: KCSPathTrackingEnv, traj: np.ndarray, log: Dict[str, np.ndarray], sea_name: str
):
    wps = np.array(env.path_manager.path)
    plt.figure(figsize=(8, 6))
    plt.plot(wps[:, 0], wps[:, 1], "k--", label="Path")
    if len(traj) > 0:
        plt.plot(traj[:, 0], traj[:, 1], "b", label="PD track")
    plt.axis("equal")
    plt.xlabel("x (nd)")
    plt.ylabel("y (nd)")
    plt.title(f"PD Path Tracking Trajectory ({sea_name})")
    plt.legend()

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

    axes[2].plot(t, np.degrees(log["heading_error_rad"]))
    axes[2].set_ylabel("y2 (deg)")
    axes[2].set_xlabel("t (nd)")
    axes[2].set_title("Heading error")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    plt.show()


def _get_sea_state(name: str) -> SeaState:
    for sea in SEA_STATES:
        if sea.name == name:
            return sea
    supported = ", ".join(s.name for s in SEA_STATES)
    raise ValueError(f"Unsupported sea state '{name}'. Supported: {supported}")


def mae_and_std(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(np.abs(values))), float(np.std(values))


def evaluate_pd(
    *,
    sea_name: str = "SS4",
    trials: int = 100,
    steps: int = 2000,
    kp: float = 5.0,
    td: float = 1.0,
    path_type: str = "S_curve",
    seed: int = 0,
    plot_first_trial: bool = True,
) -> Dict[str, float]:
    sea = _get_sea_state(sea_name)
    rng = np.random.default_rng(seed)
    y1_all: List[float] = []
    y2_all: List[float] = []
    delta_all: List[float] = []
    n_completed = 0

    for i in range(trials):
        disturb = sample_disturbance(rng, sea)
        traj, log, env = run_pd_episode(
            steps=steps,
            kp=kp,
            td=td,
            wind_conf=disturb["wind"],
            wave_conf=disturb["wave"],
            current_conf=disturb["current"],
            path_type=path_type,
        )

        if plot_first_trial and i == 0:
            plot_pd_results(env, traj, log, sea_name=sea.name)

        y1_all.extend(log["cross_track_error"].tolist())
        y2_all.extend(log["heading_error_rad"].tolist())
        delta_all.extend(log["delta_deg"].tolist())
        if log["completed"]:
            n_completed += 1

    y1_arr = np.array(y1_all)
    y2_arr = np.array(y2_all)
    d_arr = np.array(delta_all)
    y1_mae, y1_std = mae_and_std(y1_arr)
    y2_mae, y2_std = mae_and_std(y2_arr)
    d_mae, d_std = mae_and_std(d_arr)
    y1_max = float(np.max(y1_arr)) if y1_arr.size > 0 else 0.0
    y2_max = float(np.max(y2_arr)) if y2_arr.size > 0 else 0.0
    y1_abs_max = float(np.max(np.abs(y1_arr))) if y1_arr.size > 0 else 0.0
    y2_abs_max = float(np.max(np.abs(y2_arr))) if y2_arr.size > 0 else 0.0

    print("SeaState,Policy,y1_MAE,y1_STD,y2_MAE(rad),y2_STD(rad),delta_MAE(deg),delta_STD(deg)")
    print(f"{sea.name},PD,{y1_mae:.4f},{y1_std:.4f},{y2_mae:.4f},{y2_std:.4f},{d_mae:.4f},{d_std:.4f}")
    print(f"  y1 max = {y1_max:.4f}, |y1| max = {y1_abs_max:.4f}")
    print(f"  y2 max (rad) = {y2_max:.4f}, |y2| max (rad) = {y2_abs_max:.4f}")
    print(f"  completion rate = {n_completed}/{trials} ({100 * n_completed / trials:.1f}%)")

    return {
        "sea_name": sea.name,
        "y1_mae": y1_mae,
        "y1_std": y1_std,
        "y2_mae_rad": y2_mae,
        "y2_std_rad": y2_std,
        "y1_max": y1_max,
        "y2_max_rad": y2_max,
        "y1_abs_max": y1_abs_max,
        "y2_abs_max_rad": y2_abs_max,
        "delta_mae_deg": d_mae,
        "delta_std_deg": d_std,
        "completion_rate": n_completed / max(1, trials),
    }


# Backward-compatible alias.
def evaluate_pd_ss3(
    *,
    trials: int = 100,
    steps: int = 2000,
    kp: float = 5.0,
    td: float = 1.0,
    path_type: str = "S_curve",
    seed: int = 0,
):
    return evaluate_pd(
        sea_name="SS3",
        trials=trials,
        steps=steps,
        kp=kp,
        td=td,
        path_type=path_type,
        seed=seed,
        plot_first_trial=True,
    )


if __name__ == "__main__":
    evaluate_pd(sea_name="SS1", trials=100, steps=2000, kp=5.0, td=1.0, path_type="S_curve", seed=0)
