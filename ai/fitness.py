"""Frozen data contract for wave fitness tracking (P1.2).

Weighting logic implemented in P4.5; stats are updated live by Zombie during
a wave (see game/zombie.py, P3.4).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FitnessStats:
    """Per-zombie stats accumulated over one wave."""

    damage_dealt: float = 0.0
    time_alive: float = 0.0
    min_dist_to_player: float = float("inf")
    reached_player: bool = False


def compute_fitness(stats: FitnessStats) -> float:
    """Weighted combination of FitnessStats into a single scalar. Implemented in P4.5."""
    raise NotImplementedError
