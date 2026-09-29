"""Wave fitness: per-zombie stats (P1.2 contract) and their weighted score (P4.5).

Zombie updates its FitnessStats live during a wave (see game/zombie.py).
compute_fitness turns them into one scalar, higher = better:

    fitness = W_DAMAGE * damage + W_PROXIMITY * proximity
            + W_SURVIVAL * survival + W_REACHED * reached

Every term is normalised to [0, 1] (references and weights live in config.py),
so with weights summing to 1 the fitness is in [0, 1] and no single raw unit
(HP vs pixels vs seconds) dominates by accident.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from config import (
    FITNESS_DAMAGE_REF,
    FITNESS_DISTANCE_REF,
    FITNESS_W_DAMAGE,
    FITNESS_W_PROXIMITY,
    FITNESS_W_REACHED,
    FITNESS_W_SURVIVAL,
    WAVE_TIME_LIMIT,
)


@dataclass
class FitnessStats:
    """Per-zombie stats accumulated over one wave."""

    damage_dealt: float = 0.0  # HP taken off the player
    time_alive: float = 0.0  # seconds
    min_dist_to_player: float = float("inf")  # px, smallest gap between the bodies (0 = touching)
    reached_player: bool = False  # ever in contact with the player


def _unit(value: float, reference: float) -> float:
    """value / reference clipped to [0, 1]."""
    return min(max(value / reference, 0.0), 1.0)


def compute_fitness(stats: FitnessStats) -> float:
    """Weighted combination of FitnessStats into a single scalar in [0, sum of weights]."""
    for name in ("damage_dealt", "time_alive", "min_dist_to_player"):
        value = getattr(stats, name)
        if math.isnan(value):
            raise ValueError(f"{name} is NaN")
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    damage = _unit(stats.damage_dealt, FITNESS_DAMAGE_REF)
    proximity = 1.0 - _unit(stats.min_dist_to_player, FITNESS_DISTANCE_REF)  # inf -> 0
    survival = _unit(stats.time_alive, WAVE_TIME_LIMIT)
    reached = 1.0 if stats.reached_player else 0.0
    return (
        FITNESS_W_DAMAGE * damage
        + FITNESS_W_PROXIMITY * proximity
        + FITNESS_W_SURVIVAL * survival
        + FITNESS_W_REACHED * reached
    )
