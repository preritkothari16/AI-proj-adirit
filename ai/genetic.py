"""Genetic Algorithm operators and optimiser (P4.2), RandomSearch baseline (P4.3).

One generation (GeneticAlgorithm.next_generation):
  1. elitism    the ELITISM_COUNT fittest genomes are copied over unchanged
                (ties -> earlier index).
  2. selection  each parent is the fittest of TOURNAMENT_SIZE distinct genomes
                drawn at random (ties -> earlier index).
  3. crossover  with probability CROSSOVER_RATE, uniform crossover: each gene
                comes from either parent with p = 0.5, and the second child gets
                the complementary genes. Otherwise the children are parent copies.
  4. mutation   each gene independently, with probability MUTATION_RATE, gets
                N(0, MUTATION_SIGMA) added.
  5. repair     Genome.repair(): clip to [0, 1], restore the stat budget.
Children are added until the new population is as large as the old one.

RandomSearch ignores fitness and samples a fresh Genome.random() population
every generation: same interface and evaluations per generation as the GA, no
learning. It is the "is the GA actually doing anything?" baseline.

Everything random goes through the Generator passed in, so a run is
reproducible from its seed. Pure module: never imports game/ or pygame.
"""
from __future__ import annotations

from typing import List, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from ai.genome import NUM_GENES, Genome
from config import (
    CROSSOVER_RATE,
    ELITISM_COUNT,
    MUTATION_RATE,
    MUTATION_SIGMA,
    POPULATION_SIZE,
    STAT_BUDGET,
    TOURNAMENT_SIZE,
)


@runtime_checkable
class Optimizer(Protocol):
    """Common interface for GA / Random Search / Hill Climbing (fair comparison)."""

    def initial_population(self) -> List[Genome]: ...

    def next_generation(self, genomes: Sequence[Genome], fitnesses: Sequence[float]) -> List[Genome]: ...


# --- operators (pure apart from the rng) ---


def _check_fitnesses(genomes: Sequence[Genome], fitnesses: Sequence[float]) -> np.ndarray:
    fit = np.asarray(fitnesses, dtype=np.float64)
    if fit.shape != (len(genomes),):
        raise ValueError(f"need one fitness per genome: {len(genomes)} genomes, fitness shape {fit.shape}")
    if len(genomes) == 0:
        raise ValueError("empty population")
    if np.isnan(fit).any():
        raise ValueError("fitness contains NaN")
    return fit


def elite_indices(fitnesses: Sequence[float], count: int) -> List[int]:
    """Indices of the `count` highest fitnesses, best first; ties keep the earlier index."""
    order = np.argsort(-np.asarray(fitnesses, dtype=np.float64), kind="stable")
    return [int(i) for i in order[:count]]


def tournament_select(fitnesses: Sequence[float], rng: np.random.Generator, size: int = TOURNAMENT_SIZE) -> int:
    """Index of the fittest of `size` distinct random contestants (all of them if fewer)."""
    fit = np.asarray(fitnesses, dtype=np.float64)
    contestants = np.sort(rng.choice(len(fit), size=min(size, len(fit)), replace=False))
    return int(contestants[np.argmax(fit[contestants])])  # sorted, so ties -> earlier index


def uniform_crossover(
    a: Genome, b: Genome, rng: np.random.Generator, rate: float = CROSSOVER_RATE
) -> Tuple[Genome, Genome]:
    """Two children. With probability `rate` genes are swapped by a fair coin per
    gene (child 2 is the complement of child 1); otherwise plain copies."""
    if rng.random() >= rate:
        return Genome(a.genes.copy()), Genome(b.genes.copy())
    from_a = rng.random(NUM_GENES) < 0.5
    return Genome(np.where(from_a, a.genes, b.genes)), Genome(np.where(from_a, b.genes, a.genes))


def mutate(
    genome: Genome,
    rng: np.random.Generator,
    rate: float = MUTATION_RATE,
    sigma: float = MUTATION_SIGMA,
) -> Genome:
    """Add N(0, sigma) to each gene with probability `rate`. Not repaired:
    the result may leave [0, 1] or the stat budget (next_generation repairs)."""
    hit = rng.random(NUM_GENES) < rate
    noise = rng.normal(0.0, sigma, NUM_GENES)
    return Genome(genome.genes + np.where(hit, noise, 0.0))


# --- optimiser ---


class GeneticAlgorithm:
    def __init__(
        self,
        rng: np.random.Generator,
        population_size: int = POPULATION_SIZE,
        tournament_size: int = TOURNAMENT_SIZE,
        elitism: int = ELITISM_COUNT,
        crossover_rate: float = CROSSOVER_RATE,
        mutation_rate: float = MUTATION_RATE,
        mutation_sigma: float = MUTATION_SIGMA,
        budget: float = STAT_BUDGET,
    ):
        if population_size < 1 or tournament_size < 1 or not 0 <= elitism <= population_size:
            raise ValueError("need population_size >= 1, tournament_size >= 1, 0 <= elitism <= population_size")
        self.rng = rng
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.budget = budget

    def initial_population(self) -> List[Genome]:
        return [Genome.random(self.rng, self.budget) for _ in range(self.population_size)]

    def next_generation(self, genomes: Sequence[Genome], fitnesses: Sequence[float]) -> List[Genome]:
        """Same size as `genomes`. Elites first (best first, unchanged), then repaired children."""
        fit = _check_fitnesses(genomes, fitnesses)
        n = len(genomes)
        new = [Genome(genomes[i].genes.copy()) for i in elite_indices(fit, min(self.elitism, n))]
        while len(new) < n:
            a = genomes[tournament_select(fit, self.rng, self.tournament_size)]
            b = genomes[tournament_select(fit, self.rng, self.tournament_size)]
            for child in uniform_crossover(a, b, self.rng, self.crossover_rate):
                if len(new) < n:  # odd slot count: second child dropped
                    child = mutate(child, self.rng, self.mutation_rate, self.mutation_sigma)
                    new.append(child.repair(self.budget))
        return new


class RandomSearch:
    """Baseline: every generation is a brand-new uniformly random (repaired) population."""

    def __init__(self, rng: np.random.Generator, population_size: int = POPULATION_SIZE, budget: float = STAT_BUDGET):
        if population_size < 1:
            raise ValueError("need population_size >= 1")
        self.rng = rng
        self.population_size = population_size
        self.budget = budget

    def initial_population(self) -> List[Genome]:
        return [Genome.random(self.rng, self.budget) for _ in range(self.population_size)]

    def next_generation(self, genomes: Sequence[Genome], fitnesses: Sequence[float]) -> List[Genome]:
        """Same size as `genomes`; fitness is validated but otherwise ignored."""
        _check_fitnesses(genomes, fitnesses)
        return [Genome.random(self.rng, self.budget) for _ in range(len(genomes))]
