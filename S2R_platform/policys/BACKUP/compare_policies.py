import argparse
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure project root is on sys.path when running as a script
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from RL_training import Actor, KCSPathTrackingEnv, EnvConfig
from vessels.kcs import L as L_ship, U_des


def load_actor(model_path: str, act_limit_deg: float) -> Actor:
    actor = Actor(act_limit_deg=act_limit_deg)
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def rollout(
    env: KCSPathTrackingEnv,
    actor: Actor,
    steps: int,
    seed: int | None = None,
) -> Tuple[np.ndarray, dict]:
    if seed is not None and env.cfg.path_type == 'random_line':
        np.random.seed(seed)
    obs = env.reset()
    traj = []
    hist = {'t': [], 'y1': [], 'y2': [], 'delta_deg': []}
    t_now = 0.0
    for _ in range(steps):
        with torch.no_grad():
            a_deg, _ = actor(torch.as_tensor(obs).view(1, -1))
            a_deg = float(a_deg.item())
        obs, _, done, info = env.step(a_deg)
        traj.append((info['x'], info['y']))
        hist['t'].append(t_now)
        hist['y1'].append(info.get('y1', 0.0))
        hist['y2'].append(info.get('y2', 0.0))
        hist['delta_deg'].append(np.degrees(env.get_full_state()[6]))
        t_now += env.cfg.dt
        if done:
            break
    return np.array(traj), {k: np.array(v) for k, v in hist.items()}


def plot_compare(path, trajectories: Dict[str, Tuple[np.ndarray, dict]]):
    plt.figure(figsize=(8, 6))
    wps = np.array(path)
    plt.plot(wps[:, 0], wps[:, 1], 'k--', label='Path')
    for label, (traj, style) in trajectories.items():
        if len(traj) == 0:
            continue
        plt.plot(traj[:, 0], traj[:, 1], label=label, **style)
    plt.axis('equal')
    plt.xlabel('x (nd)')
    plt.ylabel('y (nd)')
    plt.title('Policy Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_time_histories(histories: Dict[str, Tuple[dict, dict]]):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for label, (hist, style) in histories.items():
        t = hist['t']
        axes[0].plot(t, hist['y1'], label=label, **style)
        axes[1].plot(t, np.degrees(hist['y2']), label=label, **style)
        axes[2].plot(t, hist['delta_deg'], label=label, **style)
    axes[0].set_ylabel('y1 (nd)')
    axes[0].set_title('Cross-track error')
    axes[1].set_ylabel('y2 (deg)')
    axes[1].set_title('Heading error')
    axes[2].set_ylabel('delta (deg)')
    axes[2].set_title('Rudder angle')
    axes[2].set_xlabel('t (nd)')
    axes[0].legend()
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare HDR/LDR/CRDR policies on one plot')
    parser.add_argument('--hdr', type=str, default='policys/HDR/actor_kcs.pth')
    parser.add_argument('--ldr', type=str, default='policys/LDR/actor_kcs.pth')
    parser.add_argument('--crdr', type=str, default='policys/CRDR/actor_kcs.pth')
    parser.add_argument('--disturb', action='store_true', help='Enable wind/wave/current disturbances')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--path', type=str, default='S_curve', help='Path type (S_curve, random_line, line)')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random_line path reproducibility')
    parser.add_argument('--units', type=str, default='nd', choices=['nd', 'm'])
    parser.add_argument('--r-min', type=float, default=8.0)
    parser.add_argument('--r-max', type=float, default=18.0)
    parser.add_argument('--line-length', type=float, default=20.0)
    parser.add_argument('--line-angle', type=float, default=30.0)
    parser.add_argument('--line-interval', type=float, default=4.0)
    args = parser.parse_args()

    # Convert input to nondimensional values (divide by L if user specified meters)
    if args.units == 'm':
        r_min = args.r_min / L_ship
        r_max = args.r_max / L_ship
        line_length = args.line_length / L_ship
        line_interval = args.line_interval / L_ship
    else:
        r_min = args.r_min
        r_max = args.r_max
        line_length = args.line_length
        line_interval = args.line_interval

    cfg = EnvConfig(
        dt=args.dt,
        with_disturbance=args.disturb,
        path_type=args.path,
        line_length=line_length,
        line_angle_deg=args.line_angle,
        line_interval=line_interval,
        r_min=r_min,
        r_max=r_max,
    )

    # Load policies
    policies = {
        'HDR': args.hdr,
        'LDR': args.ldr,
        'CRDR': args.crdr,
    }
    styles = {
        'HDR': dict(color='tab:red', linestyle='-'),
        'LDR': dict(color='tab:blue', linestyle='--'),
        'CRDR': dict(color='tab:green', linestyle='-.'),
    }

    trajectories = {}
    histories = {}
    base_env = KCSPathTrackingEnv(cfg)
    if args.path == 'random_line':
        np.random.seed(args.seed)
    base_env.reset()
    path = base_env.path_manager.path

    for label, model_path in policies.items():
        if not os.path.exists(model_path):
            print(f"[WARN] Missing model: {model_path}")
            continue
        env = KCSPathTrackingEnv(cfg)
        actor = load_actor(model_path, act_limit_deg=cfg.rudder_limit_deg)
        traj, hist = rollout(env, actor, steps=args.steps, seed=args.seed)
        trajectories[label] = (traj, styles[label])
        histories[label] = (hist, styles[label])

    plot_compare(path, trajectories)
    plot_time_histories(histories)


if __name__ == '__main__':
    # Set to True/False to control disturbance without CLI flags
    WITH_DISTURBANCE = False
    sys.argv.extend(['--disturb'] if WITH_DISTURBANCE else [])
    main()
