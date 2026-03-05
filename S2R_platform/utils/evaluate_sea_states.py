import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure project root is on sys.path when running as a script
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from RL_training import Actor, KCSPathTrackingEnv, EnvConfig, build_ext_force, make_current_func
from ship_params import ShipParams
from utils.domain_randomizer import KNOT_TO_MPS


@dataclass
class SeaState:
    name: str
    wind_kn: Tuple[float, float]
    wave_m: Tuple[float, float]
    current_kn: Tuple[float, float]


SEA_STATES = [
    SeaState('SS1', wind_kn=(0.0, 0.0), wave_m=(0.0, 0.0), current_kn=(0.0, 0.0)),
    SeaState('SS2', wind_kn=(4.0, 7.0), wave_m=(0.5, 1.0), current_kn=(0.5, 1.0)),
    SeaState('SS3', wind_kn=(8.0, 11.0), wave_m=(1.5, 2.0), current_kn=(1.5, 2.0)),
    SeaState('SS4', wind_kn=(12.0, 15.0), wave_m=(2.5, 3.0), current_kn=(2.5, 3.0)),
]

DIR_SET = (0.0, 45.0, 90.0, 135.0)


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


def deterministic_action(actor: Actor, obs: np.ndarray) -> float:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs).view(1, -1)
        z = actor.net(obs_t)
        mu = actor.mu_head(z)
        a_deg = torch.tanh(mu) * actor.act_limit
        return float(a_deg.item())


def sample_disturbance(rng: np.random.Generator, sea: SeaState) -> Dict:
    wind_speed = rng.uniform(sea.wind_kn[0], sea.wind_kn[1]) * KNOT_TO_MPS
    current_speed = rng.uniform(sea.current_kn[0], sea.current_kn[1]) * KNOT_TO_MPS
    wave_height = rng.uniform(sea.wave_m[0], sea.wave_m[1])
    if wind_speed > 0:
        wind_dir = rng.choice(DIR_SET)
    else:
        wind_dir = 0.0
    if current_speed > 0:
        current_dir = rng.choice(DIR_SET)
    else:
        current_dir = 0.0
    if wave_height > 0:
        wave_dir = rng.choice(DIR_SET)
    else:
        wave_dir = 0.0
    return dict(
        wind=dict(V_wind=wind_speed, Psi_wind_deg=wind_dir),
        wave=dict(H=wave_height, T=8.0, beta_deg=wave_dir, phase=0.0),
        current=dict(Vc_mps=current_speed, beta_c_deg=current_dir),
    )


def rollout_episode(env: KCSPathTrackingEnv, actor: Actor, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = env.reset()
    y1_list: List[float] = []
    y2_list: List[float] = []
    delta_list: List[float] = []
    for _ in range(steps):
        a_deg = deterministic_action(actor, obs)
        obs, _, done, info = env.step(a_deg)
        y1_list.append(float(info.get('y1', 0.0)))
        y2_list.append(float(info.get('y2', 0.0)))
        delta_list.append(np.degrees(env.get_full_state()[6]))
        if done:
            break
    return np.array(y1_list), np.array(y2_list), np.array(delta_list)


def mae_and_std(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(np.abs(values))), float(np.std(values))


def evaluate_policy(
    model_path: str,
    sea: SeaState,
    trials: int,
    steps: int,
    seed: int,
    path_type: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    cfg = EnvConfig(with_disturbance=True, path_type=path_type)
    env = KCSPathTrackingEnv(cfg)
    actor = load_actor(model_path, act_limit_deg=cfg.rudder_limit_deg)
    ship = ShipParams()

    all_y1: List[float] = []
    all_y2: List[float] = []
    all_delta: List[float] = []

    sum_abs_y1 = np.zeros(steps, dtype=float)
    sum_abs_y2 = np.zeros(steps, dtype=float)
    sum_abs_delta = np.zeros(steps, dtype=float)
    count = np.zeros(steps, dtype=float)

    for _ in range(trials):
        disturb = sample_disturbance(rng, sea)
        env.ext_force = build_ext_force(
            wind_conf=disturb['wind'],
            wave_conf=disturb['wave'],
            ship=ship,
        )
        cur = disturb['current']
        env.current_func = make_current_func(
            Vc_mps=cur.get('Vc_mps', 0.0),
            beta_c_deg=cur.get('beta_c_deg', 0.0),
        )

        y1, y2, delta = rollout_episode(env, actor, steps=steps)
        all_y1.extend(y1.tolist())
        all_y2.extend(y2.tolist())
        all_delta.extend(delta.tolist())

        n = len(y1)
        if n > 0:
            sum_abs_y1[:n] += np.abs(y1)
            sum_abs_y2[:n] += np.abs(y2) * (180.0 / np.pi)
            sum_abs_delta[:n] += np.abs(delta)
            count[:n] += 1.0

    y1_arr = np.array(all_y1)
    y2_arr = np.array(all_y2)
    d_arr = np.array(all_delta)
    y1_mae, y1_std = mae_and_std(y1_arr)
    y2_mae, y2_std = mae_and_std(y2_arr)
    d_mae, d_std = mae_and_std(d_arr)
    y1_abs_max = float(np.max(np.abs(y1_arr))) if y1_arr.size > 0 else 0.0
    y2_abs_max = float(np.max(np.abs(y2_arr))) if y2_arr.size > 0 else 0.0
    d_abs_max = float(np.max(np.abs(d_arr))) if d_arr.size > 0 else 0.0
    stats = {
        'y1': {'mae': y1_mae, 'std': y1_std, 'abs_max': y1_abs_max},
        'y2': {'mae': y2_mae, 'std': y2_std, 'abs_max': y2_abs_max},
        'delta': {'mae': d_mae, 'std': d_std, 'abs_max': d_abs_max},
    }
    mae_t = {
        't': np.arange(steps, dtype=float) * env.cfg.dt,
        'y1': np.divide(sum_abs_y1, np.maximum(count, 1.0)),
        'y2_deg': np.divide(sum_abs_y2, np.maximum(count, 1.0)),
        'delta_deg': np.divide(sum_abs_delta, np.maximum(count, 1.0)),
    }
    return stats, mae_t


def main():
    parser = argparse.ArgumentParser(description='Evaluate policies under multiple sea states')
    parser.add_argument('--crdr', type=str, default='policys/CRDR/actor_kcs.pth')
    parser.add_argument('--dr', type=str, default='policys/DR/actor_kcs.pth')
    parser.add_argument('--ndr', type=str, default='policys/NDR/actor_kcs.pth')
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--path', type=str, default='S_curve', help='Path type (S_curve, random_line, line)')
    args = parser.parse_args()

    policies = {
        'CRDR': args.crdr,
        'DR': args.dr,
        'NDR': args.ndr,
    }

    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    mae_time: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    print('SeaState,Policy,y1_MAE,y1_STD,y1_MAX_ABS,y2_MAE(rad),y2_STD(rad),y2_MAX_ABS(rad),delta_MAE(deg),delta_STD(deg),delta_MAX_ABS(deg)')
    for sea in SEA_STATES:
        results[sea.name] = {}
        mae_time[sea.name] = {}
        for label, model_path in policies.items():
            if not os.path.exists(model_path):
                print(f"{sea.name},{label},MISSING_MODEL")
                continue
            stats, mae_t = evaluate_policy(
                model_path=model_path,
                sea=sea,
                trials=args.trials,
                steps=args.steps,
                seed=args.seed,
                path_type=args.path,
            )
            y1_mae = stats['y1']['mae']
            y1_std = stats['y1']['std']
            y1_max = stats['y1']['abs_max']
            y2_mae = stats['y2']['mae']
            y2_std = stats['y2']['std']
            y2_max = stats['y2']['abs_max']
            d_mae = stats['delta']['mae']
            d_std = stats['delta']['std']
            d_max = stats['delta']['abs_max']
            print(f"{sea.name},{label},{y1_mae:.4f},{y1_std:.4f},{y1_max:.4f},{y2_mae:.4f},{y2_std:.4f},{y2_max:.4f},{d_mae:.4f},{d_std:.4f},{d_max:.4f}")
            results[sea.name][label] = stats
            mae_time[sea.name][label] = mae_t

    styles = {
        'CRDR': dict(color='tab:green', linestyle='-.'),
        'DR': dict(color='tab:blue', linestyle='--'),
        'NDR': dict(color='tab:red', linestyle='-'),
    }

    for sea in SEA_STATES:
        if not mae_time[sea.name]:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for label, style in styles.items():
            if label not in mae_time[sea.name]:
                continue
            t = mae_time[sea.name][label]['t']
            axes[0].plot(t, mae_time[sea.name][label]['y1'], label=label, **style)
            axes[1].plot(t, mae_time[sea.name][label]['y2_deg'], label=label, **style)
            axes[2].plot(t, mae_time[sea.name][label]['delta_deg'], label=label, **style)
        axes[0].set_ylabel('y1 MAE (nd)')
        axes[0].set_title(f'Cross-track error MAE ({sea.name})')
        axes[1].set_ylabel('y2 MAE (deg)')
        axes[1].set_title(f'Heading error MAE ({sea.name})')
        axes[2].set_ylabel('delta MAE (deg)')
        axes[2].set_title(f'Rudder angle MAE ({sea.name})')
        axes[2].set_xlabel('t (nd)')
        axes[0].legend()
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
