import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is on sys.path when running as a script
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from RL_training import Actor, EnvConfig, KCSPathTrackingEnv
from utils.path_generator import PathManager, generate_circular_path
from vessels.kcs import L as L_ship


DEFAULT_MODELS = {
    "NDR": "policys/NDR/actor_kcs.pth",
    "DR": "policys/DR/actor_kcs.pth",
    "CRDR": "policys/CRDR/actor_kcs.pth",
    "LDR": "policys/LDR/actor_kcs.pth",
    "HDR": "policys/HDR/actor_kcs.pth",
}


def _migrate_old_state_dict(state: dict) -> dict:
    """Map legacy Actor keys (mu, log_std scalar) to current architecture."""
    if "mu.weight" in state and "mu_head.weight" not in state:
        state["mu_head.weight"] = state.pop("mu.weight")
        state["mu_head.bias"] = state.pop("mu.bias")
    if "log_std" in state and "log_std_head.weight" not in state:
        old_log_std = state.pop("log_std")
        hidden_dim = state["mu_head.weight"].shape[1]
        state["log_std_head.weight"] = torch.zeros(1, hidden_dim)
        state["log_std_head.bias"] = old_log_std.view(1)
    return state


def load_actor(model_path: str, act_limit_deg: float) -> Actor:
    actor = Actor(act_limit_deg=act_limit_deg)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    state = _migrate_old_state_dict(state)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def _set_turning_circle_path(env: KCSPathTrackingEnv, radius_nd: float, interval_angle_deg: float) -> None:
    waypoints = generate_circular_path(radius=radius_nd, interval_angle_deg=interval_angle_deg)
    env.path_manager = PathManager(waypoints)
    env.wpt = {
        "x": np.array([p[0] for p in waypoints], dtype=float),
        "y": np.array([p[1] for p in waypoints], dtype=float),
    }
    env._set_los_params_from_path()


def reset_for_turning(env: KCSPathTrackingEnv, radius_nd: float, interval_angle_deg: float) -> np.ndarray:
    env.reset()
    _set_turning_circle_path(env, radius_nd=radius_nd, interval_angle_deg=interval_angle_deg)

    env.path_manager.current_index = 1
    env.t = 0.0
    env.step_count = 0
    env.x[:] = 0.0
    env.x[0] = float(env.cfg.up0)
    env.x[3] = float(env.path_manager.path[0][0])
    env.x[4] = float(env.path_manager.path[0][1])
    env.x[5] = float(env.path_manager.start_psi)
    env.x[6] = 0.0
    env.prev_delta = 0.0
    env._cached_errors = None
    return env._obs()


def rollout_turning(env: KCSPathTrackingEnv, actor: Actor, steps: int, radius_nd: float, interval_angle_deg: float):
    obs = reset_for_turning(env, radius_nd=radius_nd, interval_angle_deg=interval_angle_deg)
    traj = []
    hist = {"t": [], "up": [], "vp": [], "rp": [], "delta_deg": [], "y1": [], "y2_deg": []}
    t_now = 0.0

    for _ in range(steps):
        with torch.no_grad():
            z = actor.net(torch.as_tensor(obs).view(1, -1))
            mu = actor.mu_head(z)
            rudder_deg = torch.tanh(mu) * actor.act_limit
            rudder_deg = float(rudder_deg.item())

        obs, _, done, info = env.step(rudder_deg)
        state = env.get_full_state()
        traj.append((info["x"], info["y"]))
        hist["t"].append(t_now)
        hist["up"].append(state[0])
        hist["vp"].append(state[1])
        hist["rp"].append(state[2])
        hist["delta_deg"].append(np.degrees(state[6]))
        hist["y1"].append(info["y1"])
        hist["y2_deg"].append(np.degrees(info["y2"]))
        t_now += env.cfg.dt

        if done:
            break

    return np.array(traj), {k: np.array(v) for k, v in hist.items()}


def plot_turning_result(env: KCSPathTrackingEnv, traj: np.ndarray, hist: dict, title_suffix: str):
    wps = np.array(env.path_manager.path)
    plt.figure(figsize=(8, 6))
    plt.plot(wps[:, 0], wps[:, 1], "k--", label="Turning circle target path")
    if len(traj) > 0:
        plt.plot(traj[:, 0], traj[:, 1], "r-", label="USV trajectory")
        plt.plot(traj[0, 0], traj[0, 1], "go", label="Start")
        plt.plot(traj[-1, 0], traj[-1, 1], "bo", label="End")
    plt.axis("equal")
    plt.xlabel("x (nd)")
    plt.ylabel("y (nd)")
    plt.title(f"Turning Experiment Trajectory ({title_suffix})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    t = hist["t"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    ax = axes.ravel()
    ax[0].plot(t, hist["up"])
    ax[0].set_ylabel("u (nd)")
    ax[0].set_title("Surge speed")
    ax[1].plot(t, hist["vp"])
    ax[1].set_ylabel("v (nd)")
    ax[1].set_title("Sway speed")
    ax[2].plot(t, hist["rp"])
    ax[2].set_ylabel("r (nd)")
    ax[2].set_title("Yaw rate")
    ax[3].plot(t, hist["delta_deg"])
    ax[3].set_ylabel("delta (deg)")
    ax[3].set_title("Rudder angle")
    ax[4].plot(t, hist["y1"])
    ax[4].set_ylabel("y1 (nd)")
    ax[4].set_title("Cross-track error")
    ax[5].plot(t, hist["y2_deg"])
    ax[5].set_ylabel("y2 (deg)")
    ax[5].set_title("Heading error")
    ax[4].set_xlabel("t (nd)")
    ax[5].set_xlabel("t (nd)")
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Turning experiment visualization with selectable policy")
    parser.add_argument(
        "--model-name",
        type=str,
        default="NDR",
        choices=list(DEFAULT_MODELS.keys()),
        help="Model preset to load (default: NDR)",
    )
    parser.add_argument("--model-path", type=str, default=None, help="Optional custom model path, overrides --model-name")
    parser.add_argument("--disturb", action=argparse.BooleanOptionalAction, default=False, help="Enable disturbances")
    parser.add_argument("--dt", type=float, default=0.1, help="Nondimensional time step")
    parser.add_argument("--steps", type=int, default=3000, help="Max rollout steps")
    parser.add_argument("--radius", type=float, default=12.0, help="Turning circle radius")
    parser.add_argument("--interval-angle", type=float, default=5.0, help="Waypoint angular interval (deg)")
    parser.add_argument("--units", type=str, default="nd", choices=["nd", "m"], help="Input units for --radius")
    args = parser.parse_args()

    model_path = args.model_path or DEFAULT_MODELS[args.model_name]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    radius_nd = args.radius / L_ship if args.units == "m" else args.radius

    cfg = EnvConfig(
        dt=args.dt,
        max_steps=args.steps,
        with_disturbance=args.disturb,
        path_type="line",
    )
    env = KCSPathTrackingEnv(cfg)
    actor = load_actor(model_path, act_limit_deg=cfg.rudder_limit_deg)
    traj, hist = rollout_turning(
        env=env,
        actor=actor,
        steps=args.steps,
        radius_nd=radius_nd,
        interval_angle_deg=args.interval_angle,
    )

    print(f"[CONFIG] model_name={args.model_name}, model_path={model_path}")
    print(f"[CONFIG] disturbance={'ON' if args.disturb else 'OFF'}")
    print(f"[CONFIG] radius_nd={radius_nd:.3f}, interval_angle={args.interval_angle:.1f} deg")
    if len(traj) > 0:
        print(f"[RESULT] final_position=({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f}), steps={len(traj)}")

    plot_turning_result(
        env=env,
        traj=traj,
        hist=hist,
        title_suffix=f"{args.model_name}, disturb={'ON' if args.disturb else 'OFF'}",
    )


if __name__ == "__main__":
    main()
