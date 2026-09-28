"""P4.2 — GA operators: tournament selection, elitism, uniform crossover, mutation, repair, next_generation."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from ai.genetic import (
    GeneticAlgorithm,
    Optimizer,
    elite_indices,
    mutate,
    tournament_select,
    uniform_crossover,
)
from ai.genome import BUDGET_GENES, GENE_NAMES, NUM_GENES, Genome
from config import (
    CROSSOVER_RATE,
    ELITISM_COUNT,
    MUTATION_RATE,
    MUTATION_SIGMA,
    POPULATION_SIZE,
    STAT_BUDGET,
    TOURNAMENT_SIZE,
)

FREE_IDX = [i for i, name in enumerate(GENE_NAMES) if name not in BUDGET_GENES]


def population(seed: int = 0, n: int = POPULATION_SIZE):
    return GeneticAlgorithm(np.random.default_rng(seed), population_size=n).initial_population()


def distinct_fitnesses(n: int = POPULATION_SIZE, seed: int = 0):
    return list(np.random.default_rng(seed + 100).permutation(n).astype(float))


def assert_valid(genome: Genome, budget: float = STAT_BUDGET):
    assert np.all((genome.genes >= 0.0) & (genome.genes <= 1.0))
    assert genome.budget_used() == pytest.approx(budget)


# --- config / interface ---


def test_uses_config_defaults():
    ga = GeneticAlgorithm(np.random.default_rng(0))
    assert (ga.population_size, ga.tournament_size, ga.elitism) == (POPULATION_SIZE, TOURNAMENT_SIZE, ELITISM_COUNT)
    assert (ga.crossover_rate, ga.mutation_rate, ga.mutation_sigma) == (CROSSOVER_RATE, MUTATION_RATE, MUTATION_SIGMA)
    assert ga.budget == STAT_BUDGET
    assert ELITISM_COUNT == 2


def test_ga_satisfies_optimizer_protocol():
    assert isinstance(GeneticAlgorithm(np.random.default_rng(0)), Optimizer)


def test_initial_population_valid_and_seeded():
    pop = population(3)
    assert len(pop) == POPULATION_SIZE
    for g in pop:
        assert_valid(g)
    assert [g.genes.tolist() for g in pop] == [g.genes.tolist() for g in population(3)]
    assert len({tuple(g.genes) for g in pop}) == POPULATION_SIZE


@pytest.mark.parametrize("kwargs", [{"population_size": 0}, {"tournament_size": 0}, {"elitism": -1}, {"elitism": 21}])
def test_bad_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        GeneticAlgorithm(np.random.default_rng(0), **kwargs)


# --- elitism ---


def test_elite_indices_best_first_ties_to_earlier():
    assert elite_indices([1.0, 5.0, 3.0, 5.0, 0.0], 2) == [1, 3]
    assert elite_indices([2.0, 2.0, 2.0], 2) == [0, 1]
    assert elite_indices([-3.0, -1.0, -2.0], 1) == [1]


def test_elites_survive_unchanged():
    ga = GeneticAlgorithm(np.random.default_rng(0))
    pop = ga.initial_population()
    fit = distinct_fitnesses()
    best, second = int(np.argmax(fit)), int(np.argsort(fit)[-2])
    new = ga.next_generation(pop, fit)
    assert np.array_equal(new[0].genes, pop[best].genes)
    assert np.array_equal(new[1].genes, pop[second].genes)


def test_elites_are_copies_not_aliases():
    ga = GeneticAlgorithm(np.random.default_rng(0))
    pop = ga.initial_population()
    new = ga.next_generation(pop, distinct_fitnesses())
    assert all(new_g is not old for new_g in new for old in pop)
    assert all(not np.shares_memory(new_g.genes, old.genes) for new_g in new for old in pop)


def test_elites_survive_many_generations():
    """Best fitness never gets lost when fitness is a fixed function of the genes."""
    ga = GeneticAlgorithm(np.random.default_rng(1))
    pop = ga.initial_population()
    best_so_far = -np.inf
    for _ in range(30):
        fit = [float(g["aggression"] + g["swarm"]) for g in pop]
        assert max(fit) >= best_so_far - 1e-12
        best_so_far = max(fit)
        pop = ga.next_generation(pop, fit)


def test_elites_unchanged_even_with_certain_mutation():
    ga = GeneticAlgorithm(np.random.default_rng(0), mutation_rate=1.0, mutation_sigma=0.5)
    pop = ga.initial_population()
    fit = distinct_fitnesses()
    new = ga.next_generation(pop, fit)
    for slot, idx in enumerate(elite_indices(fit, 2)):
        assert np.array_equal(new[slot].genes, pop[idx].genes)


def test_zero_elitism():
    ga = GeneticAlgorithm(np.random.default_rng(0), elitism=0, mutation_rate=1.0)
    pop = ga.initial_population()
    new = ga.next_generation(pop, distinct_fitnesses())
    assert len(new) == len(pop)
    assert not any(np.array_equal(a.genes, b.genes) for a in new for b in pop)


# --- population size ---


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 20, 21, 50])
def test_population_size_constant(n):
    ga = GeneticAlgorithm(np.random.default_rng(n), population_size=n, elitism=min(2, n))
    pop = ga.initial_population()
    for gen in range(10):
        pop = ga.next_generation(pop, distinct_fitnesses(n, gen))
        assert len(pop) == n


def test_input_population_not_modified():
    ga = GeneticAlgorithm(np.random.default_rng(0))
    pop = ga.initial_population()
    before = [g.genes.copy() for g in pop]
    ga.next_generation(pop, distinct_fitnesses())
    assert all(np.array_equal(g.genes, b) for g, b in zip(pop, before))


@pytest.mark.parametrize("fit", [[1.0, 2.0], [1.0] * 21, [float("nan")] + [1.0] * 19])
def test_bad_fitness_rejected(fit):
    ga = GeneticAlgorithm(np.random.default_rng(0))
    with pytest.raises(ValueError):
        ga.next_generation(ga.initial_population(), fit)


def test_empty_population_rejected():
    with pytest.raises(ValueError):
        GeneticAlgorithm(np.random.default_rng(0)).next_generation([], [])


# --- next_generation output ---


def test_children_are_repaired():
    ga = GeneticAlgorithm(np.random.default_rng(0), mutation_rate=1.0, mutation_sigma=1.0)
    pop = ga.initial_population()
    for gen in range(20):
        pop = ga.next_generation(pop, distinct_fitnesses(seed=gen))
        for g in pop:
            assert_valid(g)


def test_custom_budget_respected():
    ga = GeneticAlgorithm(np.random.default_rng(0), budget=2.4)
    pop = ga.initial_population()
    for gen in range(5):
        pop = ga.next_generation(pop, distinct_fitnesses(seed=gen))
        for g in pop:
            assert_valid(g, 2.4)


def test_next_generation_deterministic_for_seed():
    def run(seed):
        ga = GeneticAlgorithm(np.random.default_rng(seed))
        pop = ga.initial_population()
        for gen in range(5):
            pop = ga.next_generation(pop, distinct_fitnesses(seed=gen))
        return np.array([g.genes for g in pop])

    assert np.array_equal(run(4), run(4))
    assert not np.array_equal(run(4), run(5))


def test_selection_pressure_improves_population():
    """Toy objective: maximise the swarm gene. Mean rises well above random."""
    ga = GeneticAlgorithm(np.random.default_rng(0))
    pop = ga.initial_population()
    start = np.mean([g["swarm"] for g in pop])
    for _ in range(25):
        pop = ga.next_generation(pop, [g["swarm"] for g in pop])
    end = np.mean([g["swarm"] for g in pop])
    assert end > start + 0.25
    assert end > 0.85


# --- tournament selection ---


def test_tournament_picks_best_contestant():
    fit = np.arange(10, dtype=float)
    rng = np.random.default_rng(0)
    for _ in range(200):
        state = rng.bit_generator.state
        winner = tournament_select(fit, rng, 3)
        rng2 = np.random.default_rng()
        rng2.bit_generator.state = state
        contestants = rng2.choice(10, size=3, replace=False)
        assert winner == max(contestants)


def test_tournament_full_size_always_picks_best():
    rng = np.random.default_rng(0)
    fit = [3.0, 9.0, 1.0, 9.0, 2.0]
    assert all(tournament_select(fit, rng, 5) == 1 for _ in range(50))  # tie -> earlier
    assert all(tournament_select(fit, rng, 50) == 1 for _ in range(50))  # size capped at population


def test_tournament_size_one_is_uniform():
    rng = np.random.default_rng(0)
    counts = np.bincount([tournament_select(np.arange(5.0), rng, 1) for _ in range(10000)], minlength=5)
    assert np.all(np.abs(counts / 10000 - 0.2) < 0.02)


def test_tournament_never_picks_worst_with_distinct_contestants():
    rng = np.random.default_rng(0)
    fit = np.arange(20, dtype=float)
    picks = [tournament_select(fit, rng, 3) for _ in range(5000)]
    assert min(picks) >= 2  # needs 2 others below it
    # P(win) for rank r (0 = worst) = C(r, 2) / C(20, 3): best wins 3/20 of the time
    assert np.mean(np.array(picks) == 19) == pytest.approx(3 / 20, abs=0.015)


# --- uniform crossover ---


A = Genome(np.zeros(NUM_GENES))
B = Genome(np.ones(NUM_GENES))


def test_crossover_known_result_for_fixed_seed():
    rng = np.random.default_rng(12345)
    c1, c2 = uniform_crossover(A, B, rng, rate=1.0)
    # replay the same draws: one for the rate check, then the per-gene coin flips
    replay = np.random.default_rng(12345)
    replay.random()
    from_a = replay.random(NUM_GENES) < 0.5
    assert np.array_equal(c1.genes, np.where(from_a, 0.0, 1.0))
    assert np.array_equal(c2.genes, np.where(from_a, 1.0, 0.0))
    again = uniform_crossover(A, B, np.random.default_rng(12345), rate=1.0)
    assert np.array_equal(again[0].genes, c1.genes) and np.array_equal(again[1].genes, c2.genes)


def test_crossover_children_are_complementary():
    rng = np.random.default_rng(0)
    a, b = Genome(rng.random(NUM_GENES)), Genome(rng.random(NUM_GENES))
    for _ in range(200):
        c1, c2 = uniform_crossover(a, b, rng, rate=1.0)
        for i in range(NUM_GENES):
            assert {c1.genes[i], c2.genes[i]} == {a.genes[i], b.genes[i]}
        assert np.allclose(c1.genes + c2.genes, a.genes + b.genes)


def test_crossover_takes_each_parent_half_the_time():
    rng = np.random.default_rng(0)
    from_b = np.array([uniform_crossover(A, B, rng, rate=1.0)[0].genes for _ in range(5000)])
    assert np.all(np.abs(from_b.mean(axis=0) - 0.5) < 0.03)  # per gene


def test_crossover_rate_zero_copies_parents():
    rng = np.random.default_rng(0)
    for _ in range(50):
        c1, c2 = uniform_crossover(A, B, rng, rate=0.0)
        assert np.array_equal(c1.genes, A.genes) and np.array_equal(c2.genes, B.genes)
        assert c1 is not A and not np.shares_memory(c1.genes, A.genes)


def test_crossover_rate_frequency():
    rng = np.random.default_rng(0)
    n = 5000
    mixed = 0
    for _ in range(n):
        c1, _ = uniform_crossover(A, B, rng, rate=CROSSOVER_RATE)
        # an actual crossover leaves c1 == A only if every coin says A (1/64)
        mixed += not np.array_equal(c1.genes, A.genes)
    expected = CROSSOVER_RATE * (1 - 0.5**NUM_GENES)
    assert mixed / n == pytest.approx(expected, abs=0.015)


def test_crossover_of_identical_parents_is_identity():
    g = Genome(np.random.default_rng(0).random(NUM_GENES))
    c1, c2 = uniform_crossover(g, g, np.random.default_rng(1), rate=1.0)
    assert np.array_equal(c1.genes, g.genes) and np.array_equal(c2.genes, g.genes)


# --- mutation ---


def test_mutation_frequency_matches_rate():
    rng = np.random.default_rng(0)
    base = Genome(np.full(NUM_GENES, 0.5))
    n = 20000
    changed = np.array([mutate(base, rng).genes != 0.5 for _ in range(n)])
    per_gene = changed.mean(axis=0)
    assert np.all(np.abs(per_gene - MUTATION_RATE) < 0.01)  # each gene ~10 %
    assert changed.mean() == pytest.approx(MUTATION_RATE, abs=0.004)
    # genes mutate independently: P(no gene touched) = 0.9 ** 6
    assert np.mean(~changed.any(axis=1)) == pytest.approx((1 - MUTATION_RATE) ** NUM_GENES, abs=0.01)


def test_mutation_frequency_survives_repair_on_free_genes():
    """Through the whole GA path (mutate + repair), non-budget genes still change ~rate."""
    rng = np.random.default_rng(1)
    base = Genome.of(aggression=0.5, path_strategy=0.5, swarm=0.5).repair()
    n = 20000
    changed = np.array([mutate(base, rng).repair().genes[FREE_IDX] != base.genes[FREE_IDX] for _ in range(n)])
    assert changed.mean() == pytest.approx(MUTATION_RATE, abs=0.005)


def test_mutation_step_size_matches_sigma():
    rng = np.random.default_rng(0)
    base = Genome(np.full(NUM_GENES, 0.5))
    deltas = np.concatenate([mutate(base, rng).genes - 0.5 for _ in range(30000)])
    deltas = deltas[deltas != 0.0]
    assert deltas.std() == pytest.approx(MUTATION_SIGMA, rel=0.05)
    assert abs(deltas.mean()) < 0.005


@pytest.mark.parametrize("rate, expected", [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5)])
def test_mutation_rate_extremes(rate, expected):
    rng = np.random.default_rng(0)
    base = Genome(np.full(NUM_GENES, 0.5))
    changed = np.mean([mutate(base, rng, rate=rate).genes != 0.5 for _ in range(4000)])
    assert changed == pytest.approx(expected, abs=0.02)


def test_mutation_does_not_modify_input_and_is_seeded():
    base = Genome(np.full(NUM_GENES, 0.5))
    a = mutate(base, np.random.default_rng(9), rate=1.0)
    b = mutate(base, np.random.default_rng(9), rate=1.0)
    assert np.array_equal(base.genes, np.full(NUM_GENES, 0.5))
    assert np.array_equal(a.genes, b.genes)


def test_mutation_can_leave_range_and_repair_fixes_it():
    rng = np.random.default_rng(0)
    raw = [mutate(Genome(np.full(NUM_GENES, 0.98)), rng, rate=1.0, sigma=0.2) for _ in range(200)]
    assert any(np.any(g.genes > 1.0) for g in raw)  # mutate alone doesn't repair
    for g in raw:
        assert_valid(g.repair())


# --- module hygiene ---


def test_does_not_import_game_or_pygame():
    code = "import sys, ai.genetic; sys.exit(any(m == 'pygame' or m.startswith('game') for m in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
