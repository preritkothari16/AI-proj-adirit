"""Frozen data contracts for zombie genomes (P1.2).

Genome logic — random generation, budget repair, decoding — is implemented
in P4.1. This module only fixes the shapes/types so game/ and ai/ can be
built in parallel against a stable interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

GENE_NAMES = ("speed", "health", "vision", "aggression", "path_strategy", "swarm")
NUM_GENES = len(GENE_NAMES)


class PathStrategy(Enum):
    """Movement strategy a zombie's path_strategy gene decodes to."""

    DIRECT = "direct"
    GREEDY = "greedy"
    ASTAR = "astar"


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
    def random(cls, rng: np.random.Generator) -> "Genome":
        """Uniform random genome, budget-repaired. Implemented in P4.1."""
        raise NotImplementedError

    def repair(self) -> "Genome":
        """Clip genes to [0, 1] and rescale speed/health/vision to STAT_BUDGET. Implemented in P4.1."""
        raise NotImplementedError

    def decode(self) -> ZombieStats:
        """Map genes to real-valued ZombieStats. Implemented in P4.1."""
        raise NotImplementedError
