"""P4.1 — genome operations: random generation, repair (ranges + stat budget), decode."""
from __future__ import annotations

import numpy as np
import pytest

from ai.genome import (
    BUDGET_GENES,
    GENE_NAMES,
    NUM_GENES,
    Genome,
    PathStrategy,
    strategy_from_gene,
)
from config import (
    STAT_BUDGET,
    ZOMBIE_GIVE_UP_RANGE,
    ZOMBIE_HP_RANGE,
    ZOMBIE_SPEED_RANGE,
    ZOMBIE_SWARM_WEIGHT_MAX,
    ZOMBIE_VISION_RANGE,
)

EPS = 1e-9
BUDGET_IDX = [GENE_NAMES.index(name) for name in BUDGET_GENES]
FREE_IDX = [i for i in range(NUM_GENES) if i not in BUDGET_IDX]


def stats_of(genome: Genome) -> np.ndarray:
    return genome.genes[BUDGET_IDX]


def assert_valid(genome: Genome, budget: float = STAT_BUDGET) -> None:
    assert genome.genes.shape == (NUM_GENES,)
    assert np.all(genome.genes >= 0.0) and np.all(genome.genes <= 1.0), genome.genes
    assert genome.budget_used() == pytest.approx(budget, abs=EPS)


def random_raw(rng: np.random.Generator, low: float = -1.0, high: float = 2.0) -> Genome:
    return Genome(rng.uniform(low, high, NUM_GENES))


# --- random() ---


def test_random_is_valid():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        assert_valid(Genome.random(rng))


def test_random_is_deterministic_for_a_seed():
    a = [Genome.random(np.random.default_rng(42)).genes for _ in range(3)]
    assert all(np.array_equal(a[0], x) for x in a)
    rng1, rng2 = np.random.default_rng(7), np.random.default_rng(7)
    for _ in range(50):
        assert np.array_equal(Genome.random(rng1).genes, Genome.random(rng2).genes)


def test_random_differs_across_seeds_and_calls():
    rng = np.random.default_rng(0)
    first, second = Genome.random(rng), Genome.random(rng)
    assert not np.array_equal(first.genes, second.genes)
    assert not np.array_equal(Genome.random(np.random.default_rng(1)).genes, Genome.random(np.random.default_rng(2)).genes)


def test_random_uses_only_the_given_rng():
    """Global numpy state must not affect or be affected by random()."""
    np.random.seed(0)
    a = Genome.random(np.random.default_rng(3)).genes
    np.random.seed(999)
    b = Genome.random(np.random.default_rng(3)).genes
    assert np.array_equal(a, b)


def test_random_covers_all_strategies_evenly():
    rng = np.random.default_rng(0)
    counts = {s: 0 for s in PathStrategy}
    n = 3000
    for _ in range(n):
        counts[Genome.random(rng).decode().path_strategy] += 1
    for strategy, count in counts.items():
        assert abs(count / n - 1 / 3) < 0.04, (strategy, count)


def test_random_spreads_non_budget_genes_over_unit_interval():
    rng = np.random.default_rng(0)
    genes = np.array([Genome.random(rng).genes for _ in range(2000)])
    for i in FREE_IDX:
        assert genes[:, i].min() < 0.02 and genes[:, i].max() > 0.98, GENE_NAMES[i]
        assert genes[:, i].mean() == pytest.approx(0.5, abs=0.03)


def test_random_respects_custom_budget():
    rng = np.random.default_rng(0)
    for budget in (0.0, 0.5, 1.0, 2.0, 2.9, 3.0):
        for _ in range(100):
            assert_valid(Genome.random(rng, budget), budget)


# --- repair(): gene ranges ---


@pytest.mark.parametrize("gene", [n for n in GENE_NAMES if n not in BUDGET_GENES])
@pytest.mark.parametrize("value, expected", [(-5.0, 0.0), (-1e-12, 0.0), (0.0, 0.0), (0.3, 0.3), (1.0, 1.0), (1.0 + 1e-12, 1.0), (9.0, 1.0)])
def test_repair_clips_non_budget_genes(gene, value, expected):
    repaired = Genome.of(**{gene: value}).repair()
    assert repaired[gene] == expected


def test_repair_clips_infinities():
    genes = np.array([np.inf, -np.inf, 0.5, np.inf, -np.inf, 0.2])
    repaired = Genome(genes).repair()
    assert_valid(repaired)
    assert repaired["aggression"] == 1.0 and repaired["path_strategy"] == 0.0


def test_repair_rejects_nan():
    genes = np.full(NUM_GENES, 0.5)
    genes[3] = np.nan
    with pytest.raises(ValueError):
        Genome(genes).repair()


def test_repair_leaves_non_budget_genes_alone_when_in_range():
    rng = np.random.default_rng(0)
    for _ in range(200):
        g = Genome(rng.random(NUM_GENES))
        assert np.array_equal(g.repair().genes[FREE_IDX], g.genes[FREE_IDX])


# --- repair(): stat budget ---


def test_repair_over_budget_scales_down_keeping_ratios():
    g = Genome.of(speed=1.0, health=0.5, vision=0.5).repair()  # sum 2.0 -> 1.5
    assert stats_of(g) == pytest.approx([0.75, 0.375, 0.375])
    assert_valid(g)


def test_repair_under_budget_scales_up_keeping_ratios():
    g = Genome.of(speed=0.2, health=0.1, vision=0.2).repair()  # sum 0.5 -> 1.5
    assert stats_of(g) == pytest.approx([0.6, 0.3, 0.6])
    assert_valid(g)


def test_repair_scale_up_caps_at_one_and_redistributes():
    g = Genome.of(speed=0.6, health=0.1, vision=0.1).repair()  # x1.875 would give speed 1.125
    speed, health, vision = stats_of(g)
    assert speed == 1.0
    assert health == pytest.approx(0.25) and vision == pytest.approx(0.25)  # remaining 0.5, 1:1
    assert_valid(g)


def test_repair_cascading_caps():
    g = Genome.of(speed=0.5, health=0.45, vision=0.05).repair(2.5)
    assert stats_of(g) == pytest.approx([1.0, 1.0, 0.5])
    assert_valid(g, 2.5)


def test_repair_all_zero_stats_split_evenly():
    g = Genome.of(speed=0.0, health=0.0, vision=0.0).repair()
    assert stats_of(g) == pytest.approx([0.5, 0.5, 0.5])


def test_repair_zero_stats_stay_zero_when_others_can_absorb():
    g = Genome.of(speed=0.3, health=0.0, vision=0.3).repair()
    assert stats_of(g) == pytest.approx([0.75, 0.0, 0.75])


def test_repair_one_nonzero_stat_takes_all_it_can():
    g = Genome.of(speed=0.1, health=0.0, vision=0.0).repair()
    assert stats_of(g) == pytest.approx([1.0, 0.25, 0.25])  # capped, rest split evenly
    assert_valid(g)


def test_repair_exactly_on_budget_is_unchanged():
    g = Genome.of(speed=0.9, health=0.4, vision=0.2)
    assert np.allclose(g.repair().genes, g.genes)


def test_repair_clips_budget_genes_before_rescaling():
    g = Genome.of(speed=5.0, health=-3.0, vision=1.0).repair()  # -> (1, 0, 1), sum 2 -> 1.5
    assert stats_of(g) == pytest.approx([0.75, 0.0, 0.75])


@pytest.mark.parametrize(
    "budget, expected",
    [(0.0, [0.0, 0.0, 0.0]), (3.0, [1.0, 1.0, 1.0])],
)
def test_repair_extreme_budgets(budget, expected):
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert stats_of(random_raw(rng).repair(budget)) == pytest.approx(expected)


@pytest.mark.parametrize("budget", [-0.1, 3.01, 10.0])
def test_repair_rejects_impossible_budget(budget):
    with pytest.raises(ValueError):
        Genome.of().repair(budget)
    with pytest.raises(ValueError):
        Genome.random(np.random.default_rng(0), budget)


def test_repair_random_raw_genomes_always_valid():
    rng = np.random.default_rng(1)
    for budget in (0.25, STAT_BUDGET, 2.75):
        for _ in range(2000):
            assert_valid(random_raw(rng, -3.0, 4.0).repair(budget), budget)


def test_repair_is_idempotent():
    rng = np.random.default_rng(2)
    for _ in range(500):
        once = random_raw(rng).repair()
        assert np.allclose(once.repair().genes, once.genes, atol=1e-12)


def test_repair_does_not_modify_original():
    genes = np.array([2.0, 0.1, 0.1, -1.0, 0.5, 3.0])
    g = Genome(genes.copy())
    repaired = g.repair()
    assert repaired is not g
    assert np.array_equal(g.genes, genes)


def test_repair_preserves_stat_order():
    """Rescaling never swaps which stat is biggest."""
    rng = np.random.default_rng(3)
    for _ in range(1000):
        raw = np.clip(random_raw(rng).genes, 0.0, 1.0)
        fixed = Genome(raw).repair().genes
        before, after = raw[BUDGET_IDX], fixed[BUDGET_IDX]
        for i in range(3):
            for j in range(3):
                if before[i] > before[j]:
                    assert after[i] >= after[j] - EPS


def test_budget_forces_tradeoff():
    """Maxing one stat after repair costs the others."""
    fast = Genome.of(speed=1.0, health=0.1, vision=0.1).repair().decode()
    tank = Genome.of(speed=0.1, health=1.0, vision=0.1).repair().decode()
    assert fast.speed > tank.speed and tank.max_hp > fast.max_hp
    greedy = Genome.of(speed=1.0, health=1.0, vision=1.0).repair()
    assert stats_of(greedy) == pytest.approx([0.5, 0.5, 0.5])  # can't have everything


# --- decode() ---


@pytest.mark.parametrize(
    "value, strategy",
    [
        (0.0, PathStrategy.DIRECT),
        (1 / 3 - 1e-9, PathStrategy.DIRECT),
        (1 / 3, PathStrategy.GREEDY),
        (2 / 3 - 1e-9, PathStrategy.GREEDY),
        (2 / 3, PathStrategy.ASTAR),
        (1.0, PathStrategy.ASTAR),
        (-0.5, PathStrategy.DIRECT),
        (1.5, PathStrategy.ASTAR),
    ],
)
def test_path_strategy_boundaries(value, strategy):
    assert strategy_from_gene(value) is strategy
    assert Genome.of(path_strategy=value).decode().path_strategy is strategy


def test_decode_ranges_at_gene_bounds():
    low = Genome(np.zeros(NUM_GENES)).decode()
    high = Genome(np.ones(NUM_GENES)).decode()
    assert (low.speed, low.max_hp, low.vision_radius) == (ZOMBIE_SPEED_RANGE[0], ZOMBIE_HP_RANGE[0], ZOMBIE_VISION_RANGE[0])
    assert (high.speed, high.max_hp, high.vision_radius) == (ZOMBIE_SPEED_RANGE[1], ZOMBIE_HP_RANGE[1], ZOMBIE_VISION_RANGE[1])
    assert (low.give_up_time, high.give_up_time) == ZOMBIE_GIVE_UP_RANGE
    assert (low.swarm_weight, high.swarm_weight) == (0.0, ZOMBIE_SWARM_WEIGHT_MAX)


def test_decoded_random_genomes_are_in_range_and_typed():
    rng = np.random.default_rng(4)
    for _ in range(500):
        s = Genome.random(rng).decode()
        assert ZOMBIE_SPEED_RANGE[0] <= s.speed <= ZOMBIE_SPEED_RANGE[1]
        assert ZOMBIE_HP_RANGE[0] <= s.max_hp <= ZOMBIE_HP_RANGE[1]
        assert ZOMBIE_VISION_RANGE[0] <= s.vision_radius <= ZOMBIE_VISION_RANGE[1]
        assert ZOMBIE_GIVE_UP_RANGE[0] <= s.give_up_time <= ZOMBIE_GIVE_UP_RANGE[1]
        assert 0.0 <= s.swarm_weight <= ZOMBIE_SWARM_WEIGHT_MAX
        assert isinstance(s.path_strategy, PathStrategy)
        assert all(type(v) is float for v in (s.speed, s.max_hp, s.vision_radius, s.give_up_time, s.swarm_weight))


def test_decode_is_pure():
    g = Genome.random(np.random.default_rng(5))
    before = g.genes.copy()
    assert g.decode() == g.decode()
    assert np.array_equal(g.genes, before)


def test_repair_does_not_change_strategy_or_behaviour_genes_decoding():
    rng = np.random.default_rng(6)
    for _ in range(300):
        g = Genome(rng.random(NUM_GENES))
        a, b = g.decode(), g.repair().decode()
        assert (a.path_strategy, a.give_up_time, a.swarm_weight) == (b.path_strategy, b.give_up_time, b.swarm_weight)
