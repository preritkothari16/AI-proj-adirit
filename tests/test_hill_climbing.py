"""P4.4 — HillClimber: parallel stochastic hill climbers behind the Optimizer interface (toy problem only)."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from ai.genetic import Optimizer, RandomSearch
from ai.genome import Genome
from ai.hill_climbing import HillClimber
from ai.toy_benchmark import TOY_TARGET, compare, main, run, toy_fitness
from config import POPULATION_SIZE, STAT_BUDGET

SEEDS = range(5)


def _valid(g: Genome, budget: float = STAT_BUDGET) -> bool:
    return bool(((g.genes >= 0) & (g.genes <= 1)).all()) and np.isclose(g.budget_used(), budget)


def _toy_scores(pop):
    f = toy_fitness()
    return [f(g) for g in pop]


def _climb(seed: int, generations: int, fitness=None):
    """Run a HillClimber by hand; returns it plus each climber's fitness after every generation."""
    fitness = fitness or toy_fitness()
    hc = HillClimber(np.random.default_rng(seed))
    pop = hc.initial_population()
    history = []
    for _ in range(generations):
        pop = hc.next_generation(pop, [fitness(g) for g in pop])
        history.append(hc.current_fitnesses.copy())
    return hc, np.array(history)


# --- interface ---


def test_satisfies_optimizer_protocol():
    assert isinstance(HillClimber(np.random.default_rng(0)), Optimizer)


def test_population_size_and_validity():
    hc = HillClimber(np.random.default_rng(0))
    pop = hc.initial_population()
    assert len(pop) == POPULATION_SIZE
    for _ in range(10):
        pop = hc.next_generation(pop, [0.0] * len(pop))
        assert len(pop) == POPULATION_SIZE
        assert all(_valid(g) for g in pop)


def test_custom_budget_respected():
    hc = HillClimber(np.random.default_rng(0), population_size=8, budget=0.9)
    pop = hc.initial_population()
    for _ in range(5):
        pop = hc.next_generation(pop, list(range(len(pop))))
    assert all(_valid(g, 0.9) for g in pop + hc.current)


@pytest.mark.parametrize("kwargs", [{"population_size": 0}, {"sigma": 0.0}, {"sigma": -0.1}])
def test_bad_params_rejected(kwargs):
    with pytest.raises(ValueError):
        HillClimber(np.random.default_rng(0), **kwargs)


def test_bad_fitness_rejected():
    hc = HillClimber(np.random.default_rng(0), population_size=4)
    pop = hc.initial_population()
    with pytest.raises(ValueError):
        hc.next_generation(pop, [0.0] * 3)
    with pytest.raises(ValueError):
        hc.next_generation(pop, [0.0, np.nan, 0.0, 0.0])


def test_population_size_change_rejected():
    hc = HillClimber(np.random.default_rng(0), population_size=4)
    pop = hc.next_generation(hc.initial_population(), [0.0] * 4)
    with pytest.raises(ValueError):
        hc.next_generation(pop[:3], [0.0] * 3)


def test_same_initial_population_as_ga_and_random():
    """Same seed -> same gen 0 as the other optimizers (fair comparison)."""
    a = HillClimber(np.random.default_rng(3)).initial_population()
    b = RandomSearch(np.random.default_rng(3)).initial_population()
    assert all(np.array_equal(x.genes, y.genes) for x, y in zip(a, b))


def test_deterministic_per_seed():
    pops = []
    for _ in range(2):
        hc = HillClimber(np.random.default_rng(11))
        pop = hc.initial_population()
        for _ in range(5):
            pop = hc.next_generation(pop, _toy_scores(pop))
        pops.append(pop)
    assert all(np.array_equal(x.genes, y.genes) for x, y in zip(*pops))


def test_input_not_modified():
    hc = HillClimber(np.random.default_rng(0))
    pop = hc.initial_population()
    before = [g.genes.copy() for g in pop]
    hc.next_generation(pop, _toy_scores(pop))
    assert all(np.array_equal(g.genes, b) for g, b in zip(pop, before))


def test_initial_population_resets_climbers():
    hc = HillClimber(np.random.default_rng(0), population_size=4)
    hc.next_generation(hc.initial_population(), [1.0] * 4)
    hc.initial_population()
    assert hc.current == [] and len(hc.current_fitnesses) == 0


# --- local search rule ---


def test_first_generation_becomes_starting_points():
    hc = HillClimber(np.random.default_rng(0), population_size=5)
    pop = hc.initial_population()
    fit = [3.0, 1.0, 4.0, 1.0, 5.0]
    hc.next_generation(pop, fit)
    assert all(np.array_equal(c.genes, g.genes) for c, g in zip(hc.current, pop))
    assert hc.current_fitnesses.tolist() == fit


def test_neighbours_are_local():
    """Each proposal is a small step from its own climber's genome, not a random restart."""
    hc = HillClimber(np.random.default_rng(0), sigma=0.05)
    proposals = hc.next_generation(hc.initial_population(), [0.0] * POPULATION_SIZE)
    steps = [np.abs(p.genes - c.genes).max() for p, c in zip(proposals, hc.current)]
    assert max(steps) < 0.5
    assert all(s > 0 for s in steps)  # something always changes


def test_accepts_only_strict_improvement():
    hc = HillClimber(np.random.default_rng(0), population_size=3)
    start = hc.initial_population()
    hc.next_generation(start, [1.0, 1.0, 1.0])
    cand = [Genome.random(np.random.default_rng(i + 100)) for i in range(3)]
    hc.next_generation(cand, [2.0, 1.0, 0.5])  # better / tie / worse
    assert np.array_equal(hc.current[0].genes, cand[0].genes)
    assert np.array_equal(hc.current[1].genes, start[1].genes)  # tie: keep old
    assert np.array_equal(hc.current[2].genes, start[2].genes)  # worse: keep old
    assert hc.current_fitnesses.tolist() == [2.0, 1.0, 1.0]


def test_accepted_genome_is_a_copy():
    hc = HillClimber(np.random.default_rng(0), population_size=1)
    hc.next_generation(hc.initial_population(), [0.0])
    cand = Genome.random(np.random.default_rng(5))
    hc.next_generation([cand], [1.0])
    cand.genes[:] = 0.0
    assert hc.current[0].genes.sum() > 0


def test_climbers_are_independent():
    """A worse candidate for one climber never affects another."""
    hc = HillClimber(np.random.default_rng(0), population_size=2)
    hc.next_generation(hc.initial_population(), [0.0, 0.0])
    kept = hc.current[1].genes.copy()
    cand = [Genome.random(np.random.default_rng(i)) for i in range(2)]
    hc.next_generation(cand, [10.0, -10.0])
    assert np.array_equal(hc.current[1].genes, kept)


# --- toy problem: never worse, and improves ---


@pytest.mark.parametrize("seed", SEEDS)
def test_each_climber_fitness_never_decreases(seed):
    _, history = _climb(seed, 60)
    assert (np.diff(history, axis=0) >= 0).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_best_so_far_never_decreases(seed):
    r = run(HillClimber(np.random.default_rng(seed)), toy_fitness(), 60)
    assert all(b >= a for a, b in zip(r.best_so_far, r.best_so_far[1:]))


@pytest.mark.parametrize("seed", SEEDS)
def test_improves_over_iterations(seed):
    """Every climber gets better, and the best one ends close to the target."""
    _, history = _climb(seed, 50)
    assert (history[-1] > history[0]).mean() >= 0.9
    assert history[-1].mean() > history[0].mean() + 0.2
    assert -history[-1].max() < 0.12


def test_improvement_keeps_going():
    """Not a one-off jump: the mean climber is still better at 10, 25 and 50 generations."""
    _, history = _climb(0, 50)
    means = history.mean(axis=1)
    assert means[0] < means[9] < means[24] < means[49]


def test_climbs_to_target_given_time():
    hc, _ = _climb(0, 300)
    best = hc.current[int(np.argmax(hc.current_fitnesses))]
    assert np.linalg.norm(best.genes - TOY_TARGET.genes) < 0.05


def test_beats_random_search_every_seed():
    results = compare(SEEDS, 50)
    for hc, rs in zip(results["HC"], results["Random"]):
        assert hc.best_so_far[-1] > rs.best_so_far[-1]


def test_same_evaluations_as_ga_and_random():
    results = compare([0], 10)
    assert {r[0].evaluations for r in results.values()} == {10 * POPULATION_SIZE}


def test_cli_lists_hill_climber(capsys):
    main(["--seeds", "2", "--generations", "10"])
    out = capsys.readouterr().out
    assert "HC" in out and "HC better than Random" in out


def test_module_pure():
    code = "import sys, ai.hill_climbing; print('pygame' in sys.modules, any(m.startswith('game') for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["False", "False"]
