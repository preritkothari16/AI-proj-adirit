"""P4.5 — fitness tracking on Zombie and the weighted score in ai.fitness."""
from __future__ import annotations

import math

import numpy as np
import pytest

import config
from ai.fitness import FitnessStats, compute_fitness
from ai.genome import Genome, PathStrategy
from config import (
    FITNESS_DAMAGE_REF,
    FITNESS_DISTANCE_REF,
    FITNESS_W_DAMAGE,
    FITNESS_W_PROXIMITY,
    FITNESS_W_REACHED,
    FITNESS_W_SURVIVAL,
    FIXED_DT,
    WAVE_TIME_LIMIT,
    ZOMBIE_ATTACK_DAMAGE,
)
from game.engine import World
from game.map import TileMap
from game.player import Action

TS = 32
DUMMY = Genome.of(path_strategy=PathStrategy.DIRECT, swarm=0.0)

ROOM = "\n".join(
    [
        "#########",
        "#S......#",
        "#...P...#",
        "#.......#",
        "#########",
    ]
)

WEIGHTS = (FITNESS_W_DAMAGE, FITNESS_W_PROXIMITY, FITNESS_W_SURVIVAL, FITNESS_W_REACHED)


def make_world(tmp_path) -> World:
    path = tmp_path / "map.txt"
    path.write_text(ROOM)
    return World(TileMap.load(path, tile_size=TS), np.random.default_rng(0))


# --- config / weights ---


def test_weights_positive_and_sum_to_one():
    assert all(w > 0 for w in WEIGHTS)
    assert sum(WEIGHTS) == pytest.approx(1.0)


def test_damage_reference_kills_player():
    assert FITNESS_DAMAGE_REF == config.PLAYER_MAX_HP


# --- compute_fitness: normalisation ---


def test_worst_stats_score_zero():
    assert compute_fitness(FitnessStats()) == 0.0  # defaults: no damage, never near, not alive, inf distance


def test_best_stats_score_one():
    best = FitnessStats(FITNESS_DAMAGE_REF, WAVE_TIME_LIMIT, 0.0, True)
    assert compute_fitness(best) == pytest.approx(sum(WEIGHTS))


def test_output_is_float_in_unit_interval_over_random_stats():
    rng = np.random.default_rng(1)
    for _ in range(2000):
        s = FitnessStats(
            damage_dealt=float(rng.uniform(0, 3 * FITNESS_DAMAGE_REF)),
            time_alive=float(rng.uniform(0, 3 * WAVE_TIME_LIMIT)),
            min_dist_to_player=float(rng.choice([rng.uniform(0, 3 * FITNESS_DISTANCE_REF), math.inf])),
            reached_player=bool(rng.integers(2)),
        )
        f = compute_fitness(s)
        assert isinstance(f, float)
        assert 0.0 <= f <= 1.0 + 1e-12


def test_each_term_capped_at_reference():
    base = FitnessStats()
    assert compute_fitness(FitnessStats(damage_dealt=FITNESS_DAMAGE_REF)) == compute_fitness(
        FitnessStats(damage_dealt=10 * FITNESS_DAMAGE_REF)
    )
    assert compute_fitness(FitnessStats(time_alive=WAVE_TIME_LIMIT)) == compute_fitness(
        FitnessStats(time_alive=5 * WAVE_TIME_LIMIT)
    )
    # gap at/over the reference = no proximity credit = same as never approaching
    assert compute_fitness(FitnessStats(min_dist_to_player=FITNESS_DISTANCE_REF)) == compute_fitness(base)
    assert compute_fitness(FitnessStats(min_dist_to_player=5 * FITNESS_DISTANCE_REF)) == compute_fitness(base)


def test_terms_contribute_exactly_their_weight():
    assert compute_fitness(FitnessStats(damage_dealt=FITNESS_DAMAGE_REF)) == pytest.approx(FITNESS_W_DAMAGE)
    assert compute_fitness(FitnessStats(min_dist_to_player=0.0)) == pytest.approx(FITNESS_W_PROXIMITY)
    assert compute_fitness(FitnessStats(time_alive=WAVE_TIME_LIMIT)) == pytest.approx(FITNESS_W_SURVIVAL)
    assert compute_fitness(FitnessStats(reached_player=True)) == pytest.approx(FITNESS_W_REACHED)


def test_half_values_give_half_credit():
    assert compute_fitness(FitnessStats(damage_dealt=FITNESS_DAMAGE_REF / 2)) == pytest.approx(FITNESS_W_DAMAGE / 2)
    assert compute_fitness(FitnessStats(time_alive=WAVE_TIME_LIMIT / 2)) == pytest.approx(FITNESS_W_SURVIVAL / 2)
    assert compute_fitness(FitnessStats(min_dist_to_player=FITNESS_DISTANCE_REF / 2)) == pytest.approx(
        FITNESS_W_PROXIMITY / 2
    )


# --- compute_fitness: monotonic behaviour ---


def test_more_damage_improves_fitness():
    scores = [compute_fitness(FitnessStats(damage_dealt=d)) for d in (0, 10, 30, 60, FITNESS_DAMAGE_REF)]
    assert scores == sorted(scores) and len(set(scores)) == len(scores)


def test_closer_approach_improves_fitness():
    gaps = [FITNESS_DISTANCE_REF * 0.9, 300.0, 100.0, 20.0, 0.0]
    scores = [compute_fitness(FitnessStats(min_dist_to_player=g)) for g in gaps]
    assert scores == sorted(scores) and len(set(scores)) == len(scores)


def test_never_approaching_is_worst_proximity():
    assert compute_fitness(FitnessStats(min_dist_to_player=math.inf)) < compute_fitness(
        FitnessStats(min_dist_to_player=FITNESS_DISTANCE_REF * 0.99)
    )


def test_longer_survival_improves_fitness_up_to_wave_limit():
    scores = [compute_fitness(FitnessStats(time_alive=t)) for t in (0, 5, 20, 45, WAVE_TIME_LIMIT)]
    assert scores == sorted(scores) and len(set(scores)) == len(scores)
    assert compute_fitness(FitnessStats(time_alive=2 * WAVE_TIME_LIMIT)) == scores[-1]


def test_survival_is_the_weakest_signal():
    """Living a full wave without ever fighting must not beat dealing real damage."""
    idle_survivor = compute_fitness(FitnessStats(time_alive=WAVE_TIME_LIMIT))
    one_hit = compute_fitness(FitnessStats(damage_dealt=ZOMBIE_ATTACK_DAMAGE, time_alive=1.0, min_dist_to_player=0.0, reached_player=True))
    assert one_hit > idle_survivor


def test_reaching_player_improves_fitness():
    assert compute_fitness(FitnessStats(reached_player=True)) > compute_fitness(FitnessStats())


def test_fighter_beats_wanderer():
    fighter = FitnessStats(damage_dealt=50.0, time_alive=30.0, min_dist_to_player=0.0, reached_player=True)
    wanderer = FitnessStats(damage_dealt=0.0, time_alive=60.0, min_dist_to_player=400.0)
    assert compute_fitness(fighter) > compute_fitness(wanderer)


def test_fitness_is_pure_and_does_not_mutate():
    s = FitnessStats(12.0, 7.0, 55.0, True)
    a = compute_fitness(s)
    assert compute_fitness(s) == a
    assert s == FitnessStats(12.0, 7.0, 55.0, True)


@pytest.mark.parametrize(
    "bad",
    [
        FitnessStats(damage_dealt=-1.0),
        FitnessStats(time_alive=-0.1),
        FitnessStats(min_dist_to_player=-5.0),
        FitnessStats(damage_dealt=math.nan),
        FitnessStats(time_alive=math.nan),
        FitnessStats(min_dist_to_player=math.nan),
    ],
)
def test_invalid_stats_rejected(bad):
    with pytest.raises(ValueError):
        compute_fitness(bad)


def test_fitness_usable_as_optimizer_input():
    """Scores are plain floats the GA/HC/Random optimizers accept."""
    from ai.genetic import GeneticAlgorithm

    rng = np.random.default_rng(0)
    ga = GeneticAlgorithm(rng)
    pop = ga.initial_population()
    fit = [compute_fitness(FitnessStats(damage_dealt=float(i), time_alive=1.0)) for i in range(len(pop))]
    assert len(ga.next_generation(pop, fit)) == len(pop)


# --- Zombie tracking ---


def test_new_zombie_has_default_stats(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((1, 1), DUMMY)
    assert z.fitness_stats == FitnessStats()
    assert z.damage_dealt == 0.0


def test_time_alive_counts_ticks(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((1, 1), DUMMY)
    for _ in range(100):  # short: a longer run lets the zombie kill the player and freeze the world
        world.step(Action())
    assert z.fitness_stats.time_alive == pytest.approx(100 * FIXED_DT)


def test_time_alive_stops_at_death(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((1, 1), DUMMY)
    for _ in range(30):
        world.step(Action())
    z.take_damage(1e9)
    world.step(Action())  # dead zombie is dropped from the world, not updated
    for _ in range(30):
        world.step(Action())
    assert z.fitness_stats.time_alive == pytest.approx(30 * FIXED_DT)
    z.update(FIXED_DT, world)  # even if updated directly
    assert z.fitness_stats.time_alive == pytest.approx(30 * FIXED_DT)


def test_min_distance_is_gap_and_never_increases(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((1, 2), DUMMY)
    p = world.player
    seen = []
    for _ in range(40):
        world.step(Action())
        seen.append(z.fitness_stats.min_dist_to_player)
    assert all(b <= a for a, b in zip(seen, seen[1:]))
    assert seen[-1] == pytest.approx(max(0.0, z.gap_to(p)))
    assert seen[-1] < seen[0]


def test_min_distance_remembers_closest_approach(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((1, 2), DUMMY)
    for _ in range(40):
        world.step(Action())
    closest = z.fitness_stats.min_dist_to_player
    z.x -= 100.0  # knocked away; stat keeps the best
    world.step(Action())
    assert z.fitness_stats.min_dist_to_player <= closest + 1e-9


def test_contact_marks_reached_and_zero_gap(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((1, 2), DUMMY)
    assert not z.fitness_stats.reached_player
    for _ in range(300):
        world.step(Action())
    fs = z.fitness_stats
    assert fs.reached_player
    assert fs.min_dist_to_player <= config.ZOMBIE_ATTACK_REACH


def test_no_contact_no_reached_flag(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((1, 1), Genome.of(path_strategy=PathStrategy.DIRECT, swarm=0.0, vision=0.0))
    z.sees_player = lambda w: False
    for _ in range(5):
        world.step(Action())
    assert not z.fitness_stats.reached_player
    assert math.isfinite(z.fitness_stats.min_dist_to_player)


def test_damage_dealt_accumulates_in_fitness_stats(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((1, 2), DUMMY)
    for _ in range(300):
        world.step(Action())
    assert z.fitness_stats.damage_dealt > 0.0
    assert z.fitness_stats.damage_dealt == z.damage_dealt
    assert z.fitness_stats.damage_dealt == pytest.approx(config.PLAYER_MAX_HP - world.player.hp)


def test_zombies_tracked_independently(tmp_path):
    world = make_world(tmp_path)
    near = world.spawn_zombie((3, 2), DUMMY)
    far = world.spawn_zombie((1, 1), DUMMY)
    for _ in range(20):
        world.step(Action())
    assert near.fitness_stats is not far.fitness_stats
    assert near.fitness_stats.min_dist_to_player < far.fitness_stats.min_dist_to_player


def test_end_to_end_attacker_outscores_idle_zombie(tmp_path):
    """Same genome, one sees the player and one never does: the fighter must score higher."""
    world = make_world(tmp_path)
    fighter = world.spawn_zombie((1, 2), DUMMY)
    idle = world.spawn_zombie((7, 3), DUMMY)
    idle.sees_player = lambda w: False
    idle.update = lambda dt, w: None  # stays put and untracked; compare against baseline stats
    for _ in range(300):
        world.step(Action())
    assert compute_fitness(fighter.fitness_stats) > compute_fitness(idle.fitness_stats)
