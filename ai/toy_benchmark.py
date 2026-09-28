"""Optimizer sanity benchmark on a toy problem (P4.3). Not a game experiment.

Toy fitness = -(Euclidean distance between a genome's genes and a fixed target
genome). The target is itself a repaired genome, so the optimum (fitness 0) is
reachable under the stat budget. Higher is better, like the real fitness.

Comparable conditions: for each seed, GA and RandomSearch get
  - their own Generator built from the same seed, so their initial populations
    are identical (both start with population_size Genome.random() calls),
  - the same population size and number of generations = same evaluations.
The reported number is the best fitness found so far (over every genome ever
evaluated), recorded after each generation's evaluation.

Run:  python -m ai.toy_benchmark [--seeds 10] [--generations 50]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

from ai.genetic import GeneticAlgorithm, Optimizer, RandomSearch
from ai.genome import Genome

# On the budget (0.8 + 0.2 + 0.5 = 1.5) and away from the [0, 1] edges.
TOY_TARGET = Genome(np.array([0.8, 0.2, 0.5, 0.9, 0.1, 0.7]))

FitnessFn = Callable[[Genome], float]
OptimizerFactory = Callable[[np.random.Generator], Optimizer]

OPTIMIZERS: Dict[str, OptimizerFactory] = {
    "GA": lambda rng: GeneticAlgorithm(rng),
    "Random": lambda rng: RandomSearch(rng),
}


def toy_fitness(target: Genome = TOY_TARGET) -> FitnessFn:
    """Fitness function: negative distance to `target` (0 = perfect)."""
    goal = target.genes.copy()

    def fitness(genome: Genome) -> float:
        return -float(np.linalg.norm(genome.genes - goal))

    return fitness


@dataclass
class RunResult:
    best_so_far: List[float]  # after evaluating generation 0, 1, ..., generations-1
    mean_per_generation: List[float]
    best_genome: Genome
    evaluations: int


def run(optimizer: Optimizer, fitness: FitnessFn, generations: int) -> RunResult:
    """Evaluate `generations` populations; next_generation is called between them."""
    population = optimizer.initial_population()
    best, best_genome = -np.inf, population[0]
    best_so_far, means, evaluations = [], [], 0
    for gen in range(generations):
        scores = [fitness(g) for g in population]
        evaluations += len(scores)
        i = int(np.argmax(scores))
        if scores[i] > best:
            best, best_genome = scores[i], population[i]
        best_so_far.append(best)
        means.append(float(np.mean(scores)))
        if gen < generations - 1:
            population = optimizer.next_generation(population, scores)
    return RunResult(best_so_far, means, best_genome, evaluations)


def compare(
    seeds: Sequence[int], generations: int, fitness: FitnessFn | None = None
) -> Dict[str, List[RunResult]]:
    """Every optimizer in OPTIMIZERS on every seed, same conditions."""
    fitness = fitness or toy_fitness()
    return {
        name: [run(make(np.random.default_rng(seed)), fitness, generations) for seed in seeds]
        for name, make in OPTIMIZERS.items()
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GA vs RandomSearch on the toy target-genome problem")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--generations", type=int, default=50)
    args = parser.parse_args(argv)

    results = compare(range(args.seeds), args.generations)
    checkpoints = sorted({0, 4, 9, 19, args.generations - 1} & set(range(args.generations)))
    print(f"Toy benchmark: distance to target, best so far (mean over {args.seeds} seeds; lower is better)")
    print(f"{'generation':>12}" + "".join(f"{name:>10}" for name in results))
    for gen in checkpoints:
        row = "".join(f"{-np.mean([r.best_so_far[gen] for r in runs]):>10.4f}" for runs in results.values())
        print(f"{gen + 1:>12}{row}")
    evals = next(iter(results.values()))[0].evaluations
    print(f"evaluations per run: {evals}")
    ga, rs = results["GA"], results["Random"]
    wins = sum(g.best_so_far[-1] > r.best_so_far[-1] for g, r in zip(ga, rs))
    print(f"GA better than Random on {wins}/{args.seeds} seeds")


if __name__ == "__main__":
    main()
