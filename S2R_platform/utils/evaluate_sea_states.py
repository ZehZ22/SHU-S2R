import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

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


def load_actor(model_path: str, act_limit_deg: float) -> Actor:
    actor = Actor(act_limit_deg=act_limit_deg)
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def deterministic_action(actor: Actor, obs: np.ndarray) -> float:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs).view(1, -1)
        z = actor.net(obs_t)
        mu = actor.mu(z)
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


def evaluate_policy(model_path: str, sea: SeaState, trials: int, steps: int, seed: int, path_type: str) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    cfg = EnvConfig(with_disturbance=True, path_type=path_type)
    env = KCSPathTrackingEnv(cfg)
    actor = load_actor(model_path, act_limit_deg=cfg.rudder_limit_deg)
    ship = ShipParams()

    all_y1: List[float] = []
    all_y2: List[float] = []
    all_delta: List[float] = []

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

    y1_mae, y1_std = mae_and_std(np.array(all_y1))
    y2_mae, y2_std = mae_and_std(np.array(all_y2))
    d_mae, d_std = mae_and_std(np.array(all_delta))
    return {
        'y1': (y1_mae, y1_std),
        'y2': (y2_mae, y2_std),
        'delta': (d_mae, d_std),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate policies under multiple sea states')
    parser.add_argument('--crdr', type=str, default='policys/CRDR/actor_kcs.pth')
    parser.add_argument('--dr', type=str, default='policys/DR/actor_kcs.pth')
    parser.add_argument('--ndr', type=str, default='policys/NDR/actor_kcs.pth')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--path', type=str, default='S_curve', help='Path type (S_curve, random_line, line)')
    args = parser.parse_args()

    policies = {
        'CRDR': args.crdr,
        'DR': args.dr,
        'NDR': args.ndr,
    }

    print('SeaState,Policy,y1_MAE,y1_STD,y2_MAE(rad),y2_STD(rad),delta_MAE(deg),delta_STD(deg)')
    for sea in SEA_STATES:
        for label, model_path in policies.items():
            if not os.path.exists(model_path):
                print(f"{sea.name},{label},MISSING_MODEL")
                continue
            stats = evaluate_policy(
                model_path=model_path,
                sea=sea,
                trials=args.trials,
                steps=args.steps,
                seed=args.seed,
                path_type=args.path,
            )
            y1_mae, y1_std = stats['y1']
            y2_mae, y2_std = stats['y2']
            d_mae, d_std = stats['delta']
            print(f"{sea.name},{label},{y1_mae:.4f},{y1_std:.4f},{y2_mae:.4f},{y2_std:.4f},{d_mae:.4f},{d_std:.4f}")


if __name__ == '__main__':
    main()
