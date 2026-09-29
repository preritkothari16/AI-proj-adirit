"""Hill Climbing baseline (P4.4).

population_size independent stochastic hill climbers run side by side, so one
generation costs the same number of evaluations as the GA and RandomSearch.

Each climber i keeps a current genome and its fitness. Every generation:
  1. the population just evaluated holds one candidate per climber; candidate i
     replaces climber i's current genome only if its fitness is strictly higher
     (the first evaluated population simply becomes the starting points),
  2. each climber proposes a neighbour: every gene of its current genome gets
     N(0, sigma) added (the GA's mutate() with rate 1), then Genome.repair().
No crossover, no information shared between climbers: a local search that can
get stuck, which is exactly what it is compared against.

The optimizer remembers its climbers between calls, so next_generation must be
given back the population it returned last time, evaluated (as the harness
does). Everything random goes through the Generator passed in. Pure module:
never imports game/ or pygame.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ai.genetic import _check_fitnesses, mutate
from ai.genome import Genome
from config import MUTATION_SIGMA, POPULATION_SIZE, STAT_BUDGET


class HillClimber:
    def __init__(
        self,
        rng: np.random.Generator,
        population_size: int = POPULATION_SIZE,
        sigma: float = MUTATION_SIGMA,
        budget: float = STAT_BUDGET,
    ):
        if population_size < 1 or sigma <= 0:
            raise ValueError("need population_size >= 1 and sigma > 0")
        self.rng = rng
        self.population_size = population_size
        self.sigma = sigma
        self.budget = budget
        self.current: List[Genome] = []  # each climber's accepted genome (empty until first next_generation)
        self.current_fitnesses = np.empty(0)

    def initial_population(self) -> List[Genome]:
        self.current, self.current_fitnesses = [], np.empty(0)
        return [Genome.random(self.rng, self.budget) for _ in range(self.population_size)]

    def neighbour(self, genome: Genome) -> Genome:
        """Local move: Gaussian noise on every gene, then repair."""
        return mutate(genome, self.rng, rate=1.0, sigma=self.sigma).repair(self.budget)

    def next_generation(self, genomes: Sequence[Genome], fitnesses: Sequence[float]) -> List[Genome]:
        """Accept strictly better candidates, then one neighbour per climber (same size as `genomes`)."""
        fit = _check_fitnesses(genomes, fitnesses)
        if not self.current:
            self.current = [Genome(g.genes.copy()) for g in genomes]
            self.current_fitnesses = fit.copy()
        else:
            if len(genomes) != len(self.current):
                raise ValueError(f"expected {len(self.current)} genomes (one per climber), got {len(genomes)}")
            for i in np.flatnonzero(fit > self.current_fitnesses):
                self.current[i] = Genome(genomes[i].genes.copy())
                self.current_fitnesses[i] = fit[i]
        return [self.neighbour(g) for g in self.current]
