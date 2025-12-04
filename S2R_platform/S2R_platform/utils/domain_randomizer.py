"""
Utilities for sampling randomized disturbance parameters for RL training.

The domain randomizer exposes configurable parameter ranges for the different
disturbance sources (wind, wave, and current). Each call to ``sample`` draws a
fresh set of parameters within those ranges so that every training episode can
interact with a different domain realization.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

KNOT_TO_MPS = 0.514444


@dataclass(frozen=True)
class DisturbanceSample:
    """Container that groups a single draw of disturbance parameters."""

    wind_conf: Dict[str, float]
    wave_conf: Dict[str, float]
    current_conf: Dict[str, float]


class DomainRandomizer:
    """
    Domain randomization helper used by RL training to sample disturbances.

    Parameters are specified as ranges (for continuous values) or enumerations
    (for categorical values). The defaults follow the snippet shared by the
    user but they can be overridden through the constructor arguments.
    """

    def __init__(
        self,
        seed: int | None = None,
        wind_speed_kn: Tuple[float, float] = (5.0, 7.0),
        wind_dir_deg: Sequence[float] = (0.0, 45.0, 90.0, 135.0),
        current_speed_kn: Tuple[float, float] = (2.0, 3.0),
        current_dir_deg: Sequence[float] = (0.0, 45.0, 90.0, 135.0),
        wave_height_m: Tuple[float, float] = (3.0, 4.0),
        wave_dir_deg: Sequence[float] = (0.0, 45.0, 90.0, 135.0),
        wave_period_s: float = 8.0,
        wave_phase_rad: float = 0.0,
    ) -> None:
        self._rng = random.Random(seed)
        self.wind_speed_kn = wind_speed_kn
        self.wind_dir_deg = tuple(wind_dir_deg)
        self.current_speed_kn = current_speed_kn
        self.current_dir_deg = tuple(current_dir_deg)
        self.wave_height_m = wave_height_m
        self.wave_dir_deg = tuple(wave_dir_deg)
        self.wave_period_s = float(wave_period_s)
        self.wave_phase_rad = float(wave_phase_rad)

    def _select_rng(self, no_seed_random: bool) -> random.Random:
        """Return the RNG to use for a draw."""
        return random.Random() if no_seed_random else self._rng

    def randomize_value(
        self, min_value: float, max_value: float, rng: random.Random
    ) -> float:
        """
        在指定区间[min_value, max_value]内生成均匀分布随机数。
        通过传入的 rng 控制是否复现或重采样。
        """
        return rng.uniform(min_value, max_value)

    def randomize_choice(
        self, options: Sequence[float], rng: random.Random
    ) -> float:
        """从给定的离散集合中抽取一个样本。"""
        if not options:
            raise ValueError("options must be non-empty")
        return rng.choice(tuple(options))

    def sample(self, no_seed_random: bool = False) -> DisturbanceSample:
        """Sample a new disturbance realization for the six targeted factors."""
        rng = self._select_rng(no_seed_random)
        wind_speed = (
            self.randomize_value(self.wind_speed_kn[0], self.wind_speed_kn[1], rng)
            * KNOT_TO_MPS
        )
        wind_dir = self.randomize_choice(self.wind_dir_deg, rng)
        current_speed = (
            self.randomize_value(self.current_speed_kn[0], self.current_speed_kn[1], rng)
            * KNOT_TO_MPS
        )
        current_dir = self.randomize_choice(self.current_dir_deg, rng)
        wave_height = self.randomize_value(self.wave_height_m[0], self.wave_height_m[1], rng)
        wave_dir = self.randomize_choice(self.wave_dir_deg, rng)

        wind_conf = dict(V_wind=wind_speed, Psi_wind_deg=wind_dir)
        wave_conf = dict(
            H=wave_height,
            T=self.wave_period_s,
            beta_deg=wave_dir,
            phase=self.wave_phase_rad,
        )
        current_conf = dict(Vc_mps=current_speed, beta_c_deg=current_dir)

        return DisturbanceSample(
            wind_conf=wind_conf, wave_conf=wave_conf, current_conf=current_conf
        )
