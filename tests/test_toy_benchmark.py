"""P4.3 — RandomSearch baseline and GA-vs-Random sanity benchmark on a toy problem (no game)."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from ai.genetic import GeneticAlgorithm, Optimizer, RandomSearch
from ai.genome import NUM_GENES, Genome
from ai.toy_benchmark import TOY_TARGET, compare, main, run, toy_fitness
from config import POPULATION_SIZE, STAT_BUDGET

SEEDS = range(5)


# --- RandomSearch ---


def test_random_search_satisfies_optimizer_protocol():
    assert isinstance(RandomSearch(np.random.default_rng(0)), Optimizer)


def test_random_search_population_size_and_validity():
    rs = RandomSearch(np.random.default_rng(0))
    pop = rs.initial_population()
    assert len(pop) == POPULATION_SIZE
    for _ in range(5):
        pop = rs.next_generation(pop, [0.0] * len(pop))
        assert len(pop) == POPULATION_SIZE
        for g in pop:
            assert np.all((g.genes >= 0) & (g.genes <= 1))
            assert g.budget_used() == pytest.approx(STAT_BUDGET)


def test_random_search_ignores_fitness():
    """Same rng state -> same next population, whatever the fitness says."""
    pop = RandomSearch(np.random.default_rng(0)).initial_population()
    a = RandomSearch(np.random.default_rng(1)).next_generation(pop, list(range(len(pop))))
    b = RandomSearch(np.random.default_rng(1)).next_generation(pop, list(range(len(pop)))[::-1])
    assert all(np.array_equal(x.genes, y.genes) for x, y in zip(a, b))


def test_random_search_keeps_nothing_from_previous_generation():
    rs = RandomSearch(np.random.default_rng(0))
    pop = rs.initial_population()
    new = rs.next_generation(pop, [1.0] * len(pop))
    assert not any(np.array_equal(x.genes, y.genes) for x in new for y in pop)


def test_random_search_validates_fitness():
    rs = RandomSearch(np.random.default_rng(0))
    with pytest.raises(ValueError):
        rs.next_generation(rs.initial_population(), [1.0])
    with pytest.raises(ValueError):
        RandomSearch(np.random.default_rng(0), population_size=0)


def test_ga_and_random_start_from_same_population():
    ga = GeneticAlgorithm(np.random.default_rng(7)).initial_population()
    rs = RandomSearch(np.random.default_rng(7)).initial_population()
    assert all(np.array_equal(a.genes, b.genes) for a, b in zip(ga, rs))


# --- toy fitness ---


def test_toy_target_is_a_valid_genome():
    assert np.array_equal(TOY_TARGET.repair().genes, TOY_TARGET.genes)


def test_toy_fitness_zero_at_target_negative_elsewhere():
    fit = toy_fitness()
    assert fit(TOY_TARGET) == 0.0
    other = Genome(np.full(NUM_GENES, 0.5))
    assert fit(other) == pytest.approx(-np.linalg.norm(other.genes - TOY_TARGET.genes))
    assert fit(other) < 0


def test_toy_fitness_closer_is_better():
    fit = toy_fitness()
    far = Genome(np.zeros(NUM_GENES))
    near = Genome((TOY_TARGET.genes + far.genes) / 2)
    assert fit(far) < fit(near) < fit(TOY_TARGET)


def test_toy_fitness_custom_target_unaffected_by_later_changes():
    target = Genome(np.full(NUM_GENES, 0.5))
    fit = toy_fitness(target)
    target.genes[:] = 0.0
    assert fit(Genome(np.full(NUM_GENES, 0.5))) == 0.0


# --- run / compare harness ---


def test_run_records_every_generation_and_counts_evaluations():
    result = run(GeneticAlgorithm(np.random.default_rng(0)), toy_fitness(), 12)
    assert len(result.best_so_far) == len(result.mean_per_generation) == 12
    assert result.evaluations == 12 * POPULATION_SIZE
    assert result.best_so_far[-1] == toy_fitness()(result.best_genome)


@pytest.mark.parametrize("name", ["GA", "Random"])
def test_best_so_far_never_decreases(name):
    results = compare(SEEDS, 30)[name]
    for r in results:
        assert all(b >= a for a, b in zip(r.best_so_far, r.best_so_far[1:]))


def test_compare_is_deterministic_and_fair():
    a, b = compare([3], 15), compare([3], 15)
    for name in a:
        assert a[name][0].best_so_far == b[name][0].best_so_far
    assert a["GA"][0].evaluations == a["Random"][0].evaluations
    assert a["GA"][0].best_so_far[0] == a["Random"][0].best_so_far[0]  # identical generation 0


# --- the sanity check itself ---


def test_ga_converges_on_toy_problem():
    for r in compare(SEEDS, 50)["GA"]:
        distance = -r.best_so_far[-1]
        assert distance < 0.03, distance
        assert np.all(np.abs(r.best_genome.genes - TOY_TARGET.genes) < 0.03)


def test_ga_population_converges_not_just_one_lucky_genome():
    for r in compare(SEEDS, 50)["GA"]:
        assert -r.mean_per_generation[-1] < 0.25  # whole population near target
        assert r.mean_per_generation[-1] > r.mean_per_generation[0] + 0.2


def test_random_search_does_not_converge():
    """Random improves only by luck; its population mean never moves."""
    for r in compare(SEEDS, 50)["Random"]:
        assert -r.best_so_far[-1] > 0.05
        early, late = np.mean(r.mean_per_generation[:10]), np.mean(r.mean_per_generation[-10:])
        assert abs(late - early) < 0.1


def test_ga_beats_random_on_every_seed():
    results = compare(SEEDS, 50)
    for ga, rs in zip(results["GA"], results["Random"]):
        assert ga.best_so_far[-1] > rs.best_so_far[-1]


@pytest.mark.slow
def test_ga_beats_random_statistically_over_30_seeds():
    results = compare(range(30), 100)
    ga = np.array([-r.best_so_far[-1] for r in results["GA"]])
    rs = np.array([-r.best_so_far[-1] for r in results["Random"]])
    assert np.all(ga < rs)
    assert ga.mean() < 0.01
    assert rs.mean() > 10 * ga.mean()
    # GA is already ahead early on (by generation 10) on average
    ga10 = np.mean([-r.best_so_far[9] for r in results["GA"]])
    rs10 = np.mean([-r.best_so_far[9] for r in results["Random"]])
    assert ga10 < rs10


def test_cli_prints_table(capsys):
    main(["--seeds", "2", "--generations", "10"])
    out = capsys.readouterr().out
    assert "GA" in out and "Random" in out
    assert "evaluations per run: 200" in out


def test_module_is_pure():
    code = "import sys, ai.toy_benchmark; sys.exit(any(m == 'pygame' or m.startswith('game') for m in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
