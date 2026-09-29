"""P4.6 — WaveManager: spawning, timing, end conditions, results (all headless)."""
from __future__ import annotations

import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

from ai.fitness import FitnessStats, compute_fitness
from ai.genome import Genome, PathStrategy
from config import FIXED_DT, POPULATION_SIZE, WAVE_TIME_LIMIT
from game.engine import World
from game.map import TileMap
from game.player import Action
from game.wave_manager import WaveEnd, WaveManager

TS = 32
IDLE = Action()
DUMMY = Genome.of(path_strategy=PathStrategy.DIRECT, swarm=0.0)

# Two spawns far from the player, walls around; zombies can't see the player from spawn (wall column).
ROOM = "\n".join(
    [
        "###########",
        "#S.#.....S#",
        "#..#..P...#",
        "#S.#......#",
        "###########",
    ]
)


def make_world(tmp_path, text: str = ROOM, seed: int = 0) -> World:
    path = tmp_path / "map.txt"
    path.write_text(text)
    return World(TileMap.load(path, tile_size=TS), np.random.default_rng(seed))


def level1_world(seed: int = 0) -> World:
    return World(TileMap.load("maps/level1.txt"), np.random.default_rng(seed))


def population(n: int = POPULATION_SIZE, seed: int = 1):
    rng = np.random.default_rng(seed)
    return [Genome.random(rng) for _ in range(n)]


def run_until_over(world: World, wm: WaveManager, max_ticks: int = 10_000) -> int:
    n = 0
    while not wm.is_over() and n < max_ticks:
        world.step(IDLE)
        wm.update(world)
        n += 1
    return n


def blind(z):
    """Zombie never notices the player, so it just wanders (keeps the wave from ending early)."""
    z.sees_player = lambda w: False
    return z


# --- start_wave / spawning ---


def test_start_wave_spawns_one_zombie_per_genome(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world)
    genomes = population()
    zs = wm.start_wave(genomes)
    assert len(zs) == len(world.zombies) == len(genomes)
    assert [z.genome for z in zs] == genomes  # same order as given


def test_zombies_start_on_spawn_points_only(tmp_path):
    world = make_world(tmp_path)
    WaveManager(world).start_wave(population())
    spawns = set(world.tilemap.spawn_points)
    assert {world.tilemap.world_to_tile(z.pos) for z in world.zombies} <= spawns


def test_spawn_spread_is_even(tmp_path):
    world = make_world(tmp_path)  # 3 spawns
    WaveManager(world).start_wave(population(20))
    counts = Counter(world.tilemap.world_to_tile(z.pos) for z in world.zombies)
    assert len(counts) == 3
    assert max(counts.values()) - min(counts.values()) <= 1
    assert sum(counts.values()) == 20


def test_level1_uses_every_spawn_point():
    world = level1_world()
    WaveManager(world).start_wave(population(20))
    used = {world.tilemap.world_to_tile(z.pos) for z in world.zombies}
    assert used == set(world.tilemap.spawn_points)
    counts = Counter(world.tilemap.world_to_tile(z.pos) for z in world.zombies)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_fewer_genomes_than_spawns(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world)
    wm.start_wave(population(2))
    assert len(world.zombies) == 2
    assert len(wm.results()) == 2


def test_assignment_is_deterministic_per_seed_and_varies_across_seeds(tmp_path):
    def layout(seed):
        w = make_world(tmp_path, seed=seed)
        WaveManager(w).start_wave(population(20))
        return [w.tilemap.world_to_tile(z.pos) for z in w.zombies]

    assert layout(5) == layout(5)
    assert any(layout(s) != layout(5) for s in range(6, 12))


def test_genome_position_does_not_fix_spawn(tmp_path):
    """First genome (the GA elite) must not always land on the same spawn point."""
    firsts = set()
    for seed in range(20):
        w = make_world(tmp_path, seed=seed)
        WaveManager(w).start_wave(population(20))
        firsts.add(w.tilemap.world_to_tile(w.zombies[0].pos))
    assert len(firsts) > 1


def test_start_wave_validates_input(tmp_path):
    wm = WaveManager(make_world(tmp_path))
    with pytest.raises(ValueError):
        wm.start_wave([])
    with pytest.raises(TypeError):
        wm.start_wave([np.zeros(6)])


def test_cannot_start_wave_with_dead_player(tmp_path):
    world = make_world(tmp_path)
    world.player.take_damage(1e9)
    with pytest.raises(ValueError):
        WaveManager(world).start_wave(population(3))


def test_start_wave_clears_previous_wave(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world)
    wm.start_wave(population(20))
    for _ in range(10):
        world.step(IDLE)
    wm.start_wave(population(5))
    assert len(world.zombies) == 5
    assert len(wm.results()) == 5
    assert wm.end_reason is None
    assert wm.ticks_elapsed == 0


def test_bad_time_limit():
    with pytest.raises(ValueError):
        WaveManager(None, time_limit=0)  # type: ignore[arg-type]


def test_use_before_start_raises(tmp_path):
    wm = WaveManager(make_world(tmp_path))
    for call in (wm.is_over, wm.results, wm.update):
        with pytest.raises(RuntimeError):
            call()


def test_update_rejects_other_world(tmp_path):
    wm = WaveManager(make_world(tmp_path))
    wm.start_wave(population(3))
    with pytest.raises(ValueError):
        wm.update(make_world(tmp_path))


# --- timing ---


def test_not_over_at_start(tmp_path):
    wm = WaveManager(make_world(tmp_path))
    wm.start_wave(population(3))
    assert not wm.is_over()
    assert wm.time_left == pytest.approx(WAVE_TIME_LIMIT)
    assert wm.time_elapsed == 0.0


def test_time_limit_ends_wave_exactly(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world, time_limit=2.0)
    for z in wm.start_wave(population(4)):
        blind(z)
    n = run_until_over(world, wm)
    assert n == round(2.0 / FIXED_DT)
    assert wm.end_reason is WaveEnd.TIME_UP
    assert wm.time_elapsed == pytest.approx(2.0)
    assert wm.time_left == 0.0


def test_one_tick_before_limit_still_running(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world, time_limit=1.0)
    for z in wm.start_wave(population(3)):
        blind(z)
    for _ in range(round(1.0 / FIXED_DT) - 1):
        world.step(IDLE)
        wm.update(world)
    assert not wm.is_over()
    assert wm.time_left == pytest.approx(FIXED_DT)


def test_default_limit_is_wave_time_limit(tmp_path):
    wm = WaveManager(make_world(tmp_path))
    assert wm.time_limit == WAVE_TIME_LIMIT
    assert wm.limit_ticks == round(WAVE_TIME_LIMIT / FIXED_DT)


def test_timer_counts_from_wave_start_not_world_start(tmp_path):
    world = make_world(tmp_path)
    for _ in range(100):
        world.step(IDLE)
    wm = WaveManager(world, time_limit=1.0)
    for z in wm.start_wave(population(3)):
        blind(z)
    assert wm.ticks_elapsed == 0
    assert run_until_over(world, wm) == round(1.0 / FIXED_DT)


# --- end conditions ---


def test_all_zombies_dead_ends_wave(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world)
    zs = wm.start_wave(population(5))
    for z in zs:
        z.take_damage(1e9)
    world.step(IDLE)
    wm.update(world)
    assert wm.is_over() and wm.end_reason is WaveEnd.ALL_DEAD


def test_one_survivor_keeps_wave_running(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world)
    zs = wm.start_wave(population(5))
    for z in zs[1:]:
        z.take_damage(1e9)
    blind(zs[0])
    for _ in range(30):
        world.step(IDLE)
        wm.update(world)
    assert not wm.is_over()
    assert wm.alive_count == 1


def test_player_death_ends_wave(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world)
    for z in wm.start_wave(population(3)):
        blind(z)
    world.player.take_damage(1e9)
    world.step(IDLE)
    wm.update(world)
    assert wm.end_reason is WaveEnd.PLAYER_DEAD


def test_player_killed_by_zombie_ends_wave(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world)
    z = wm.start_wave([DUMMY])[0]
    z.x, z.y = world.player.x - 21.0, world.player.y  # touching
    run_until_over(world, wm)
    assert wm.end_reason is WaveEnd.PLAYER_DEAD
    assert z.fitness_stats.damage_dealt >= world.player.max_hp


def test_player_death_beats_all_dead_and_time_up(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world, time_limit=1.0)
    zs = wm.start_wave(population(3))
    for z in zs:
        z.take_damage(1e9)
    world.player.take_damage(1e9)
    wm.update(world)
    assert wm.end_reason is WaveEnd.PLAYER_DEAD


def test_all_dead_beats_time_up(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world, time_limit=1.0)
    for z in wm.start_wave(population(3)):
        z.take_damage(1e9)
    for _ in range(round(1.0 / FIXED_DT)):
        world.step(IDLE)
    wm.update(world)
    assert wm.end_reason is WaveEnd.ALL_DEAD


def test_end_reason_is_latched(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world, time_limit=1.0)
    zs = wm.start_wave(population(3))
    for z in zs:
        z.take_damage(1e9)
    wm.update(world)
    assert wm.end_reason is WaveEnd.ALL_DEAD
    world.player.take_damage(1e9)
    wm.update(world)
    assert wm.end_reason is WaveEnd.ALL_DEAD  # unchanged


def test_shot_zombies_all_dead_end_to_end(tmp_path):
    """Real kills via bullets (dead zombies leave world.zombies) still end the wave."""
    world = make_world(tmp_path)
    wm = WaveManager(world, time_limit=30.0)
    genome = Genome.of(path_strategy=PathStrategy.DIRECT, swarm=0.0, health=0.0)
    z = wm.start_wave([genome])[0]
    z.x, z.y = world.player.x + 100.0, world.player.y
    z.sees_player = lambda w: False
    for _ in range(600):
        if wm.is_over():
            break
        world.player.cooldown_remaining = 0
        world.step(Action(shoot=True, aim=z.pos))
        wm.update(world)
    assert wm.end_reason is WaveEnd.ALL_DEAD
    assert z not in world.zombies


# --- results ---


def test_results_cover_whole_population_in_order(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world)
    genomes = population(20)
    wm.start_wave(genomes)
    run_until_over(world, wm)
    res = wm.results()
    assert len(res) == 20
    assert [g for g, _ in res] == genomes
    assert all(isinstance(s, FitnessStats) for _, s in res)


def test_results_include_dead_zombies_removed_from_world(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world, time_limit=2.0)
    zs = wm.start_wave(population(6))
    for z in zs[:3]:
        z.take_damage(1e9)
    for z in zs[3:]:
        blind(z)
    run_until_over(world, wm)
    assert len(world.zombies) == 3  # dead ones dropped from the world ...
    assert len(wm.results()) == 6  # ... but still reported


def test_dead_zombie_time_alive_frozen_survivors_full(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world, time_limit=2.0)
    zs = wm.start_wave(population(3))
    for z in zs:
        blind(z)
    for _ in range(30):
        world.step(IDLE)
        wm.update(world)
    zs[0].take_damage(1e9)
    run_until_over(world, wm)
    res = wm.results()
    assert res[0][1].time_alive == pytest.approx(30 * FIXED_DT)
    assert res[1][1].time_alive == pytest.approx(2.0)
    assert res[2][1].time_alive == pytest.approx(2.0)


def test_results_are_snapshots(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world, time_limit=1.0)
    for z in wm.start_wave(population(2)):
        blind(z)
    run_until_over(world, wm)
    res = wm.results()
    res[0][1].damage_dealt = 999.0
    assert wm.results()[0][1].damage_dealt == 0.0
    assert wm.wave_zombies[0].fitness_stats.damage_dealt == 0.0


def test_results_score_with_compute_fitness_and_feed_optimizer(tmp_path):
    from ai.genetic import GeneticAlgorithm

    world = make_world(tmp_path)
    wm = WaveManager(world, time_limit=3.0)
    genomes = population(20)
    wm.start_wave(genomes)
    run_until_over(world, wm)
    genomes_out, stats = zip(*wm.results())
    fitness = [compute_fitness(s) for s in stats]
    assert all(0.0 <= f <= 1.0 for f in fitness)
    nxt = GeneticAlgorithm(np.random.default_rng(0)).next_generation(list(genomes_out), fitness)
    assert len(nxt) == 20


def test_results_mid_wave_available(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world)
    wm.start_wave(population(4))
    world.step(IDLE)
    assert len(wm.results()) == 4


def test_fighter_gets_damage_and_reached_in_results(tmp_path):
    world = make_world(tmp_path)
    wm = WaveManager(world, time_limit=5.0)
    z = wm.start_wave([DUMMY])[0]
    z.x, z.y = world.player.x - 21.0, world.player.y
    run_until_over(world, wm)
    (genome, stats), = wm.results()
    assert genome is DUMMY
    assert stats.damage_dealt > 0 and stats.reached_player
    assert stats.min_dist_to_player <= 4.0


# --- full-size headless wave on the real map ---


@pytest.mark.slow
def test_full_wave_on_level1_terminates_with_all_results():
    world = level1_world(seed=3)
    wm = WaveManager(world)
    genomes = population(20, seed=3)
    wm.start_wave(genomes)
    run_until_over(world, wm)
    assert wm.is_over()
    assert wm.ticks_elapsed <= wm.limit_ticks
    res = wm.results()
    assert [g for g, _ in res] == genomes
    assert all(s.time_alive <= WAVE_TIME_LIMIT + 1e-9 for _, s in res)


def test_headless_no_pygame_import():
    code = "import sys; import game.wave_manager; sys.exit(1 if 'pygame' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
