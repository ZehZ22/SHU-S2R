import argparse
import os
import sys
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


def rollout(env: KCSPathTrackingEnv, actor: Actor, steps: int, x0: float = None, y0: float = None):
    obs = env.reset()
    # Optionally override initial position (nondimensional units)
    if x0 is not None:
        env.x[3] = float(x0)
    if y0 is not None:
        env.x[4] = float(y0)
    if (x0 is not None) or (y0 is not None):
        # Recompute observation after changing position
        obs = env._obs()
    traj = []  # (x, y)
    hist = {
        't': [], 'up': [], 'vp': [], 'rp': [], 'delta_deg': [], 'y1': [], 'y2_deg': []
    }
    t = 0.0
    for i in range(steps):
        with torch.no_grad():
            a_deg, _ = actor(torch.as_tensor(obs).view(1, -1))
            a_deg = float(a_deg.item())
        obs, reward, done, info = env.step(a_deg)

        xfull = env.get_full_state()
        traj.append((info['x'], info['y']))
        hist['t'].append(t)
        hist['up'].append(xfull[0])
        hist['vp'].append(xfull[1])
        hist['rp'].append(xfull[2])
        hist['delta_deg'].append(np.degrees(xfull[6]))
        hist['y1'].append(info['y1'])
        hist['y2_deg'].append(np.degrees(info['y2']))
        t += env.cfg.dt
        if done:
            break
    return np.array(traj), {k: np.array(v) for k, v in hist.items()}


def plot_results(env: KCSPathTrackingEnv, traj: np.ndarray, hist: dict):
    # Trajectory
    wps = np.array(env.path_manager.path)
    plt.figure(figsize=(8, 6))
    plt.plot(wps[:, 0], wps[:, 1], 'k--', label='Path')
    if len(traj) > 0:
        plt.plot(traj[:, 0], traj[:, 1], 'r', label='Track')
    plt.axis('equal')
    plt.xlabel('x (nd)')
    plt.ylabel('y (nd)')
    plt.title('Path Tracking Trajectory')
    plt.legend()

    # Time histories
    t = hist['t']
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    ax = axes.ravel()
    ax[0].plot(t, hist['up']); ax[0].set_ylabel('u (nd)'); ax[0].set_title('Surge speed')
    ax[1].plot(t, hist['vp']); ax[1].set_ylabel('v (nd)'); ax[1].set_title('Sway speed')
    ax[2].plot(t, hist['rp']); ax[2].set_ylabel('r (nd)'); ax[2].set_title('Yaw rate')
    ax[3].plot(t, hist['delta_deg']); ax[3].set_ylabel('delta (deg)'); ax[3].set_title('Rudder angle')
    ax[4].plot(t, hist['y1']); ax[4].set_ylabel('y1 (m, nd)'); ax[4].set_title('Cross-track error')
    ax[5].plot(t, hist['y2_deg']); ax[5].set_ylabel('y2 (deg)'); ax[5].set_title('Heading error')
    ax[4].set_xlabel('t (nd)'); ax[5].set_xlabel('t (nd)')
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize KCS path tracking with trained SAC actor')
    parser.add_argument('--model', type=str, default='policys/actor_kcs.pth', help='Path to actor .pth')    
    parser.add_argument('--disturb', action='store_true', help='Enable wind/wave/current disturbances')
    parser.add_argument('--dt', type=float, default=0.1, help='Nondimensional time step')
    parser.add_argument('--steps', type=int, default=2000, help='Max rollout steps')
    parser.add_argument('--prefer-steps', action='store_true', help='If set, do not override steps even if --time-s is given')
    parser.add_argument('--time-s', type=float, default=None, help='Desired physical simulation time in seconds')
    parser.add_argument('--path', type=str, default='S_curve', help='Path type (S_curve, random_line, line)')
    parser.add_argument('--units', type=str, default='nd', choices=['nd','m'], help='Input units for distances: nd=L units (default) or m')
    parser.add_argument('--r-min', type=float, default=8.0)
    parser.add_argument('--r-max', type=float, default=18.0)
    parser.add_argument('--line-length', type=float, default=20.0)
    parser.add_argument('--line-angle', type=float, default=30.0)
    parser.add_argument('--line-interval', type=float, default=4.0)
    parser.add_argument('--speed-kn', type=float, default=None, help='Initial speed in knots; overrides up0 if set')
    parser.add_argument('--x0', type=float, default=None, help='Initial x position (nd or m per --units)')
    parser.add_argument('--y0', type=float, default=None, help='Initial y position (nd or m per --units)')
    args = parser.parse_args()

    # Convert input to nondimensional values (divide by L if user specified meters)
    if args.units == 'm':
        r_min = args.r_min / L_ship
        r_max = args.r_max / L_ship
        line_length = args.line_length / L_ship
        line_interval = args.line_interval / L_ship
        x0_nd = None if args.x0 is None else (args.x0 / L_ship)
        y0_nd = None if args.y0 is None else (args.y0 / L_ship)
    else:
        r_min = args.r_min
        r_max = args.r_max
        line_length = args.line_length
        line_interval = args.line_interval
        x0_nd = args.x0
        y0_nd = args.y0

    # If a physical duration is given, convert to steps in nd time
    if args.time_s is not None and args.time_s > 0:
        # nd_time = t_s / (L / U_des)
        nd_time = args.time_s / (L_ship / U_des)
        args.steps = int(np.ceil(nd_time / args.dt))

    # Optional: set initial speed via up0 based on desired knots
    # Provide --speed-kn to match a target speed; default is up0=1.0
    up0 = 1.0
    if args.speed_kn is not None:
        speed_mps = args.speed_kn * 0.514444
        up0 = float(speed_mps / U_des)

    cfg = EnvConfig(
        dt=args.dt,
        with_disturbance=args.disturb,
        path_type=args.path,
        line_length=line_length,
        line_angle_deg=args.line_angle,
        line_interval=line_interval,
        r_min=r_min,
        r_max=r_max,
        up0=up0,
    )
    env = KCSPathTrackingEnv(cfg)
    actor = load_actor(args.model, act_limit_deg=cfg.rudder_limit_deg)
    rollout_kwargs = {}
    if x0_nd is not None:
        rollout_kwargs['x0'] = x0_nd
    if y0_nd is not None:
        rollout_kwargs['y0'] = y0_nd
    traj, hist = rollout(env, actor, steps=args.steps, **rollout_kwargs)
    plot_results(env, traj, hist)


if __name__ == '__main__':
    main()


