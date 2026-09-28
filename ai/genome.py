"""Zombie genomes: frozen contract (P1.2), decode (P3.4), random + repair (P4.1).

decode() maps genes onto real stats using the ranges in config.py:

    speed          -> speed          ZOMBIE_SPEED_RANGE
    health         -> max_hp         ZOMBIE_HP_RANGE
    vision         -> vision_radius  ZOMBIE_VISION_RANGE
    aggression     -> give_up_time   ZOMBIE_GIVE_UP_RANGE (more aggressive = searches longer)
    path_strategy  -> path_strategy  [0,1/3) DIRECT, [1/3,2/3) GREEDY, [2/3,1] ASTAR
    swarm          -> swarm_weight   0 .. ZOMBIE_SWARM_WEIGHT_MAX

repair() makes a genome valid: every gene clipped to [0, 1] and the three
budget genes (speed, health, vision) rescaled so they sum to exactly
STAT_BUDGET, keeping their ratios. Scaling down never leaves [0, 1]; scaling up
caps genes at 1 and hands the excess to the others (so budget 3 = all maxed).
If every budget gene is 0 there are no ratios to keep, so the budget is split
evenly. aggression / path_strategy / swarm are only clipped: they are
behaviours, not stats, and don't cost budget.

random() draws each gene uniformly from [0, 1) with the given Generator and
then repairs, so the same seed always yields the same genome.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from config import (
    STAT_BUDGET,
    ZOMBIE_GIVE_UP_RANGE,
    ZOMBIE_HP_RANGE,
    ZOMBIE_SPEED_RANGE,
    ZOMBIE_SWARM_WEIGHT_MAX,
    ZOMBIE_VISION_RANGE,
)

GENE_NAMES = ("speed", "health", "vision", "aggression", "path_strategy", "swarm")
NUM_GENES = len(GENE_NAMES)
BUDGET_GENES = ("speed", "health", "vision")
_BUDGET_IDX = np.array([GENE_NAMES.index(name) for name in BUDGET_GENES])


class PathStrategy(Enum):
    """Movement strategy a zombie's path_strategy gene decodes to."""

    DIRECT = "direct"
    GREEDY = "greedy"
    ASTAR = "astar"


_STRATEGY_ORDER = (PathStrategy.DIRECT, PathStrategy.GREEDY, PathStrategy.ASTAR)


def strategy_from_gene(value: float) -> PathStrategy:
    """Split [0, 1] into three equal bins."""
    index = min(int(float(np.clip(value, 0.0, 1.0)) * len(_STRATEGY_ORDER)), len(_STRATEGY_ORDER) - 1)
    return _STRATEGY_ORDER[index]


def strategy_gene(strategy: PathStrategy) -> float:
    """Centre of the gene bin that decodes to `strategy` (inverse of strategy_from_gene)."""
    return (_STRATEGY_ORDER.index(strategy) + 0.5) / len(_STRATEGY_ORDER)


def _fit_budget(stats: np.ndarray, budget: float) -> np.ndarray:
    """Rescale values in [0, 1] to sum to `budget`, keeping ratios, capping at 1."""
    stats = stats.copy()
    total = stats.sum()
    if total >= budget:
        return stats * (budget / total) if total > 0.0 else stats  # total 0 => budget 0
    # Scale up. Genes that hit 1 are fixed; the rest share what's left. Each round
    # fixes at least one more gene, so this ends within len(stats) rounds.
    capped = np.zeros(len(stats), dtype=bool)
    while True:
        free = ~capped
        if not free.any():
            return stats  # budget == len(stats): everything maxed
        remaining = budget - capped.sum()
        free_total = stats[free].sum()
        if free_total > 0.0:
            stats[free] *= remaining / free_total
        else:
            stats[free] = remaining / free.sum()
        over = free & (stats >= 1.0)
        if not over.any():
            return stats
        stats[over] = 1.0
        capped |= over


def _lerp(bounds: tuple, t: float) -> float:
    lo, hi = bounds
    return lo + (hi - lo) * t


@dataclass(frozen=True)
class ZombieStats:
    """Decoded, real-valued stats a Zombie is built from. Output of Genome.decode()."""

    speed: float
    max_hp: float
    vision_radius: float
    give_up_time: float
    path_strategy: PathStrategy
    swarm_weight: float


@dataclass
class Genome:
    """6 genes, each in [0, 1]. Order fixed by GENE_NAMES."""

    genes: np.ndarray

    def __post_init__(self) -> None:
        self.genes = np.asarray(self.genes, dtype=np.float64)
        if self.genes.shape != (NUM_GENES,):
            raise ValueError(f"Genome expects shape ({NUM_GENES},), got {self.genes.shape}")

    @classmethod
    def of(cls, **genes) -> "Genome":
        """Build a genome by gene name; unspecified genes are 0.5. `path_strategy`
        may also be a PathStrategy. E.g. Genome.of(speed=1.0, path_strategy=PathStrategy.ASTAR)."""
        unknown = set(genes) - set(GENE_NAMES)
        if unknown:
            raise ValueError(f"unknown gene(s): {sorted(unknown)}")
        if isinstance(genes.get("path_strategy"), PathStrategy):
            genes["path_strategy"] = strategy_gene(genes["path_strategy"])
        return cls(np.array([genes.get(name, 0.5) for name in GENE_NAMES]))

    def __getitem__(self, name: str) -> float:
        return float(self.genes[GENE_NAMES.index(name)])

    @classmethod
    def random(cls, rng: np.random.Generator, budget: float = STAT_BUDGET) -> "Genome":
        """Uniform random genome, repaired to `budget`. Deterministic for a given rng state."""
        return cls(rng.random(NUM_GENES)).repair(budget)

    def repair(self, budget: float = STAT_BUDGET) -> "Genome":
        """New genome with genes clipped to [0, 1] and speed + health + vision
        summing to `budget` (see module docstring). Does not modify self."""
        if not 0.0 <= budget <= len(BUDGET_GENES):
            raise ValueError(f"budget must be in [0, {len(BUDGET_GENES)}], got {budget}")
        if np.isnan(self.genes).any():
            raise ValueError(f"genome contains NaN: {self.genes}")
        genes = np.clip(self.genes, 0.0, 1.0)
        genes[_BUDGET_IDX] = _fit_budget(genes[_BUDGET_IDX], budget)
        return Genome(genes)

    def budget_used(self) -> float:
        """speed + health + vision genes."""
        return float(self.genes[_BUDGET_IDX].sum())

    def decode(self) -> ZombieStats:
        """Map genes to real-valued ZombieStats. Genes are clipped to [0, 1] first,
        so decode never produces out-of-range stats even for an unrepaired genome."""
        g = {name: float(v) for name, v in zip(GENE_NAMES, np.clip(self.genes, 0.0, 1.0))}
        return ZombieStats(
            speed=_lerp(ZOMBIE_SPEED_RANGE, g["speed"]),
            max_hp=_lerp(ZOMBIE_HP_RANGE, g["health"]),
            vision_radius=_lerp(ZOMBIE_VISION_RANGE, g["vision"]),
            give_up_time=_lerp(ZOMBIE_GIVE_UP_RANGE, g["aggression"]),
            path_strategy=strategy_from_gene(g["path_strategy"]),
            swarm_weight=g["swarm"] * ZOMBIE_SWARM_WEIGHT_MAX,
        )
