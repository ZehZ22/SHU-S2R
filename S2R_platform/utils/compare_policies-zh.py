import argparse
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 与 visualize_seastate_policy_metrics_m_split.py 一致：中文宋体（SimSun）、数值刻度 Times New Roman；字号与图尺寸统一
FIG_SIZE = (7.2, 5.2)
# 多子图纵向堆叠：宽度与单图一致，高度按行数放大
FIG_SIZE_TIME_3ROW = (FIG_SIZE[0], FIG_SIZE[1] * 2)
BASE_FONT_PT = 11
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["SimSun", "NSimSun", "Times New Roman", "DejaVu Sans"],
        "font.size": BASE_FONT_PT,
        "axes.labelsize": BASE_FONT_PT,
        "xtick.labelsize": BASE_FONT_PT,
        "ytick.labelsize": BASE_FONT_PT,
        "legend.fontsize": BASE_FONT_PT,
        "axes.unicode_minus": False,
        "figure.dpi": 220,
        "savefig.dpi": 300,
        "figure.figsize": FIG_SIZE,
    }
)
# Default legend switch; can be overridden in __main__
SHOW_LEGEND = True


def _style_axis_cn_ticks_tnr(ax) -> None:
    """轴标题宋体；数值刻度 Times New Roman（与 visualize_seastate_policy_metrics_m_split 一致）。"""
    ax.xaxis.label.set_fontfamily("SimSun")
    ax.yaxis.label.set_fontfamily("SimSun")
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily("Times New Roman")
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily("Times New Roman")


def _style_legend_simsun(leg) -> None:
    for t in leg.get_texts():
        t.set_fontfamily("SimSun")

# Ensure project root is on sys.path when running as a script
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from RL_training import (
    Actor,
    KCSPathTrackingEnv,
    EnvConfig,
    build_ext_force,
    make_current_func,
)
from ship_params import ShipParams
from utils.domain_randomizer import KNOT_TO_MPS
from vessels.kcs import L as L_ship, U_des


def _migrate_old_state_dict(state: dict) -> dict:
    """Map legacy Actor keys (mu, log_std scalar) to current architecture."""
    if 'mu.weight' in state and 'mu_head.weight' not in state:
        state['mu_head.weight'] = state.pop('mu.weight')
        state['mu_head.bias'] = state.pop('mu.bias')
    if 'log_std' in state and 'log_std_head.weight' not in state:
        old_log_std = state.pop('log_std')
        hidden_dim = state['mu_head.weight'].shape[1]
        state['log_std_head.weight'] = torch.zeros(1, hidden_dim)
        state['log_std_head.bias'] = old_log_std.view(1)
    return state


def load_actor(model_path: str, act_limit_deg: float) -> Actor:
    actor = Actor(act_limit_deg=act_limit_deg)
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    state = _migrate_old_state_dict(state)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def apply_manual_disturbance_config(
    env: KCSPathTrackingEnv,
    *,
    wind_conf: dict | None,
    wave_conf: dict | None,
    current_conf: dict | None,
) -> None:
    """Override default wind/wave/current settings for compare_policies-zh only."""
    if not env.cfg.with_disturbance:
        env.ext_force = None
        env.current_func = None
        return

    ship = ShipParams()
    env.ext_force = build_ext_force(wind_conf=wind_conf, wave_conf=wave_conf, ship=ship)
    env.current_func = make_current_func(
        Vc_mps=0.0 if current_conf is None else current_conf.get('Vc_mps', 0.0),
        beta_c_deg=0.0 if current_conf is None else current_conf.get('beta_c_deg', 0.0),
    )


def rollout(
    env: KCSPathTrackingEnv,
    actor: Actor,
    steps: int,
    seed: int | None = None,
) -> Tuple[np.ndarray, dict]:
    if seed is not None and env.cfg.path_type == 'random_line':
        np.random.seed(seed)
    t_scale = L_ship / U_des  # nd time -> seconds
    obs = env.reset()
    traj = []
    hist = {'t': [], 'y1': [], 'y2': [], 'delta_deg': []}
    t_now = 0.0
    for _ in range(steps):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs).view(1, -1)
            z = actor.net(obs_t)
            mu = actor.mu_head(z)
            a_deg = torch.tanh(mu) * actor.act_limit
            a_deg = float(a_deg.item())
        obs, _, done, info = env.step(a_deg)
        traj.append((info['x'] * L_ship, info['y'] * L_ship))
        hist['t'].append(t_now * t_scale)
        hist['y1'].append(info.get('y1', 0.0) * L_ship)
        hist['y2'].append(info.get('y2', 0.0))
        hist['delta_deg'].append(np.degrees(env.get_full_state()[6]))
        t_now += env.cfg.dt
        if done:
            break
    return np.array(traj), {k: np.array(v) for k, v in hist.items()}


def plot_compare(path, trajectories: Dict[str, Tuple[np.ndarray, dict]], show_legend: bool = True):
    plt.figure(figsize=FIG_SIZE, dpi=220)
    wps = np.array(path) * L_ship
    plt.plot(wps[:, 0], wps[:, 1], 'k--', linewidth=1.8, label='路径')
    for label, (traj, style) in trajectories.items():
        if len(traj) == 0:
            continue
        plt.plot(traj[:, 0], traj[:, 1], linewidth=2.0, label=label, **style)
    ax = plt.gca()
    plt.axis('equal')
    ax.set_xlabel('距离 (m)')
    ax.set_ylabel('距离 (m)')
    ax.yaxis.set_major_locator(MultipleLocator(2000))
    _style_axis_cn_ticks_tnr(ax)
    if show_legend:
        leg = ax.legend(loc='best')
        _style_legend_simsun(leg)
    plt.tight_layout()
    plt.show()


def _plot_single_time_history(
    histories: Dict[str, Tuple[dict, dict]],
    key: str,
    ylabel: str,
    *,
    transform=None,
    show_legend: bool = True,
    y_locator: MultipleLocator | None = None,
):
    plt.figure(figsize=FIG_SIZE, dpi=220)
    ax = plt.gca()
    for label, (hist, style) in histories.items():
        t = hist['t']
        y = hist[key] if transform is None else transform(hist[key])
        ax.plot(t, y, linewidth=2.0, label=label, **style)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('时间 (s)')
    ax.grid(axis='y', alpha=0.25)
    if y_locator is not None:
        ax.yaxis.set_major_locator(y_locator)
    if show_legend:
        leg = ax.legend(loc='best')
        _style_legend_simsun(leg)
    _style_axis_cn_ticks_tnr(ax)
    plt.tight_layout()
    plt.show()


def plot_time_histories(histories: Dict[str, Tuple[dict, dict]], show_legend: bool = True):
    _plot_single_time_history(
        histories,
        'y1',
        '侧偏距 (m)',
        show_legend=show_legend,
        y_locator=MultipleLocator(100),
    )
    _plot_single_time_history(
        histories,
        'y2',
        'y2 (deg)',
        transform=np.degrees,
        show_legend=show_legend,
    )
    _plot_single_time_history(
        histories,
        'delta_deg',
        'delta (deg)',
        show_legend=show_legend,
    )


def main():
    parser = argparse.ArgumentParser(description='Compare NDR/HDR/CRDR policies on one plot')
    parser.add_argument('--ndr', type=str, default='policys/NDR/actor_kcs.pth')
    parser.add_argument('--dr', type=str, default='policys/DR/actor_kcs.pth')
    parser.add_argument('--crdr', type=str, default='policys/CRDR/actor_kcs.pth')
    parser.add_argument('--disturb', action='store_true', help='Enable wind/wave/current disturbances')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--path', type=str, default='line', help='Path type (S_curve, random_line, line)')
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
        '无域随机化': args.ndr,
        '多因子域随机化': args.dr,
        '课程学习式域随机化': args.crdr,
    }
    styles = {
        '无域随机化': dict(color='tab:red', linestyle='-'),
        '多因子域随机化': dict(color='tab:blue', linestyle='--'),
        '课程学习式域随机化': dict(color='tab:green', linestyle='-.'),
    }

    trajectories = {}
    histories = {}
    base_env = KCSPathTrackingEnv(cfg)
    apply_manual_disturbance_config(
        base_env,
        wind_conf=MANUAL_WIND_CONF,
        wave_conf=MANUAL_WAVE_CONF,
        current_conf=MANUAL_CURRENT_CONF,
    )
    if args.path == 'random_line':
        np.random.seed(args.seed)
    base_env.reset()
    path = base_env.path_manager.path

    for label, model_path in policies.items():
        if not os.path.exists(model_path):
            print(f"[WARN] Missing model: {model_path}")
            continue
        env = KCSPathTrackingEnv(cfg)
        apply_manual_disturbance_config(
            env,
            wind_conf=MANUAL_WIND_CONF,
            wave_conf=MANUAL_WAVE_CONF,
            current_conf=MANUAL_CURRENT_CONF,
        )
        actor = load_actor(model_path, act_limit_deg=cfg.rudder_limit_deg)
        traj, hist = rollout(env, actor, steps=args.steps, seed=args.seed)
        trajectories[label] = (traj, styles[label])
        histories[label] = (hist, styles[label])

    plot_compare(path, trajectories, show_legend=SHOW_LEGEND)
    plot_time_histories(histories, show_legend=SHOW_LEGEND)


if __name__ == '__main__':
    # Set to True/False to control disturbance without CLI flags
    WITH_DISTURBANCE = True
    # Set to True/False to control legend visibility in all plots
    SHOW_LEGEND = True

    # Manual disturbance configuration for compare_policies-zh.
    # Modify these three dictionaries directly to change the simulation case.
    # Set one of them to None to disable that disturbance source.
    MANUAL_WIND_CONF = dict(V_wind=14.0 * KNOT_TO_MPS, Psi_wind_deg=60.0)
    MANUAL_WAVE_CONF = dict(H=2.6, T=8.0, beta_deg=70.0, phase=0.0)
    MANUAL_CURRENT_CONF = dict(Vc_mps=2.6 * KNOT_TO_MPS, beta_c_deg=100.0)

    print(f"[CONFIG] WITH_DISTURBANCE = {WITH_DISTURBANCE}")
    print(f"[CONFIG] wind_conf = {MANUAL_WIND_CONF}")
    print(f"[CONFIG] wave_conf = {MANUAL_WAVE_CONF}")
    print(f"[CONFIG] current_conf = {MANUAL_CURRENT_CONF}")

    sys.argv.extend(['--disturb'] if WITH_DISTURBANCE else [])
    main()
