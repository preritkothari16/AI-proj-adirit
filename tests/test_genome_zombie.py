"""P3.4 — genome-driven zombies: decode, stats on the Zombie, swarm force, behaviour differences."""
from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from ai.genome import GENE_NAMES, NUM_GENES, Genome, PathStrategy, strategy_from_gene, strategy_gene
from config import (
    BULLET_DAMAGE,
    FIXED_DT,
    ZOMBIE_GIVE_UP_RANGE,
    ZOMBIE_HP_RANGE,
    ZOMBIE_SPEED_RANGE,
    ZOMBIE_SWARM_RADIUS,
    ZOMBIE_SWARM_WEIGHT_MAX,
    ZOMBIE_VISION_RANGE,
)
from game.engine import World
from game.map import TileMap
from game.player import Action
from game.states import ZombieState

TS = 32
LEVEL1_PATH = Path(__file__).resolve().parent.parent / "maps" / "level1.txt"

# 16 x 7 open room, player on the right.
ROOM = "\n".join(
    [
        "################",
        "#S.............#",
        "#..............#",
        "#...........P..#",
        "#..............#",
        "#..............#",
        "################",
    ]
)

# Wall column at x=3 between the left pocket and the player; gap along the bottom row.
DETOUR = "\n".join(
    [
        "#########",
        "#..#....#",
        "#..#.P..#",
        "#..#....#",
        "#.......#",
        "#S......#",
        "#########",
    ]
)

# gene -> (ZombieStats field, range it decodes into)
NUMERIC = {
    "speed": ("speed", ZOMBIE_SPEED_RANGE),
    "health": ("max_hp", ZOMBIE_HP_RANGE),
    "vision": ("vision_radius", ZOMBIE_VISION_RANGE),
    "aggression": ("give_up_time", ZOMBIE_GIVE_UP_RANGE),
    "swarm": ("swarm_weight", (0.0, ZOMBIE_SWARM_WEIGHT_MAX)),
}


def g(strategy=PathStrategy.DIRECT, **genes) -> Genome:
    """Genome with all-0.5 stats, the given strategy and no swarm unless overridden."""
    genes.setdefault("swarm", 0.0)
    return Genome.of(path_strategy=strategy, **genes)


def make_world(tmp_path, text: str = ROOM, seed: int = 0) -> World:
    path = tmp_path / "map.txt"
    path.write_text(text)
    world = World(TileMap.load(path, tile_size=TS), np.random.default_rng(seed))
    world.player.hp = world.player.max_hp = 1e9  # behaviour tests must not end in game over
    return world


def put(body, world: World, tile) -> None:
    body.x, body.y = world.tilemap.tile_to_world(tile)


def ticks_until(world: World, condition, max_ticks: int = 3600):
    for tick in range(max_ticks):
        if condition():
            return tick
        world.step(Action())
    return None


# --- Decoding: every gene moves its own stat ---


@pytest.mark.parametrize("gene", list(NUMERIC))
def test_each_gene_changes_only_its_stat(gene):
    field, (lo, hi) = NUMERIC[gene]
    low, mid, high = (Genome.of(**{gene: v}).decode() for v in (0.0, 0.5, 1.0))
    assert getattr(low, field) == pytest.approx(lo)
    assert getattr(high, field) == pytest.approx(hi)
    assert getattr(low, field) < getattr(mid, field) < getattr(high, field)
    for other, (other_field, _) in NUMERIC.items():
        if other != gene:
            assert getattr(low, other_field) == getattr(high, other_field), other
    assert low.path_strategy is high.path_strategy


def test_decode_is_monotonic_in_each_gene():
    values = np.linspace(0.0, 1.0, 11)
    for gene, (field, _) in NUMERIC.items():
        decoded = [getattr(Genome.of(**{gene: v}).decode(), field) for v in values]
        assert all(a < b for a, b in zip(decoded, decoded[1:])), gene


@pytest.mark.parametrize(
    "value, strategy",
    [
        (0.0, PathStrategy.DIRECT),
        (0.33, PathStrategy.DIRECT),
        (0.34, PathStrategy.GREEDY),
        (0.66, PathStrategy.GREEDY),
        (0.67, PathStrategy.ASTAR),
        (1.0, PathStrategy.ASTAR),
    ],
)
def test_path_strategy_gene_bins(value, strategy):
    assert strategy_from_gene(value) is strategy
    assert Genome.of(path_strategy=value).decode().path_strategy is strategy


def test_strategy_gene_round_trips():
    for strategy in PathStrategy:
        assert strategy_from_gene(strategy_gene(strategy)) is strategy
        assert Genome.of(path_strategy=strategy).decode().path_strategy is strategy


def test_out_of_range_genes_are_clipped():
    wild = Genome(np.array([-1.0, 2.0, -0.5, 7.0, 3.0, -2.0])).decode()
    assert wild.speed == ZOMBIE_SPEED_RANGE[0]
    assert wild.max_hp == ZOMBIE_HP_RANGE[1]
    assert wild.vision_radius == ZOMBIE_VISION_RANGE[0]
    assert wild.give_up_time == ZOMBIE_GIVE_UP_RANGE[1]
    assert wild.path_strategy is PathStrategy.ASTAR
    assert wild.swarm_weight == 0.0


def test_neutral_genome_matches_old_fixed_zombie():
    stats = Genome.of().decode()
    assert (stats.speed, stats.max_hp, stats.vision_radius, stats.give_up_time) == (90.0, 50.0, 256.0, 3.0)


def test_genome_of_rejects_unknown_gene():
    with pytest.raises(ValueError):
        Genome.of(sped=1.0)


def test_genome_indexing_by_name():
    genome = Genome(np.linspace(0.0, 1.0, NUM_GENES))
    assert [genome[name] for name in GENE_NAMES] == pytest.approx(list(genome.genes))


# --- The Zombie carries its decoded stats ---


def test_zombie_takes_stats_from_genome(tmp_path):
    world = make_world(tmp_path)
    genome = Genome(np.array([0.2, 0.9, 0.4, 0.7, 0.95, 0.3]))
    stats = genome.decode()
    z = world.spawn_zombie((1, 1), genome)
    assert z.genome is genome
    assert z.stats == stats
    assert z.speed == stats.speed
    assert z.max_hp == z.hp == stats.max_hp
    assert z.vision_radius == stats.vision_radius
    assert z.give_up_time == stats.give_up_time
    assert z.path_strategy is stats.path_strategy is PathStrategy.ASTAR
    assert z.swarm_weight == stats.swarm_weight


def test_zombie_ids_are_unique_and_sequential(tmp_path):
    world = make_world(tmp_path)
    ids = [world.spawn_zombie((1, 1 + i), g()).zid for i in range(4)]
    assert ids == [0, 1, 2, 3]


# --- Swarm force ---


def test_swarm_zero_means_no_force_even_in_a_crowd(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((5, 3), g(swarm=0.0))
    for tile in [(4, 2), (6, 2), (5, 4), (4, 4)]:
        world.spawn_zombie(tile, g(swarm=1.0))
    assert z.swarm_force(world) == (0.0, 0.0)


def test_swarm_zero_moves_exactly_like_a_lone_zombie(tmp_path):
    def trajectory(with_crowd: bool):
        world = make_world(tmp_path)
        z = world.spawn_zombie((2, 3), g(PathStrategy.ASTAR, swarm=0.0))
        z.sees_player = lambda _w: True
        if with_crowd:
            for tile in [(2, 1), (2, 5), (4, 1), (4, 5)]:
                other = world.spawn_zombie(tile, g(swarm=1.0))
                other.sees_player = lambda _w: False
        out = []
        for _ in range(120):
            world.step(Action())
            out.append(z.pos)
        return out

    assert trajectory(True) == trajectory(False)


def test_swarm_force_points_at_allies_centroid(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((5, 3), g(swarm=0.8))
    world.spawn_zombie((7, 3), g())  # +2 tiles x
    world.spawn_zombie((5, 1), g())  # -2 tiles y  -> centroid direction (1, -1)
    fx, fy = z.swarm_force(world)
    assert math.hypot(fx, fy) == pytest.approx(0.8 * ZOMBIE_SWARM_WEIGHT_MAX)
    assert fx == pytest.approx(-fy) and fx > 0


def test_swarm_ignores_far_and_dead_allies(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((2, 3), g(swarm=1.0))
    assert z.swarm_force(world) == (0.0, 0.0)  # alone
    far = world.spawn_zombie((12, 3), g())
    assert math.dist(z.pos, far.pos) > ZOMBIE_SWARM_RADIUS
    assert z.swarm_force(world) == (0.0, 0.0)
    dead = world.spawn_zombie((3, 3), g())
    dead.take_damage(dead.max_hp)
    assert z.swarm_force(world) == (0.0, 0.0)


def test_swarm_pulls_wanderers_together(tmp_path):
    def mean_spread(swarm: float, seed: int) -> float:
        world = make_world(tmp_path, seed=seed)
        put(world.player, world, (14, 5))
        zombies = [world.spawn_zombie(t, g(PathStrategy.ASTAR, swarm=swarm)) for t in [(2, 2), (4, 2), (3, 4), (5, 4)]]
        for z in zombies:
            z.sees_player = lambda _w: False  # everyone wanders
        total = 0.0
        for _ in range(20 * 60):
            world.step(Action())
            total += np.mean([math.dist(a.pos, b.pos) for a, b in combinations(zombies, 2)])
        return total / (20 * 60)

    for seed in range(3):
        assert mean_spread(1.0, seed) < 0.8 * mean_spread(0.0, seed), seed


@pytest.mark.parametrize("strategy", [PathStrategy.GREEDY, PathStrategy.ASTAR])
def test_full_swarm_still_follows_paths_to_player(tmp_path, strategy):
    """A pack of max-swarm planners on the real map still reaches the player."""
    world = make_world(tmp_path, LEVEL1_PATH.read_text())
    pack = [world.spawn_zombie(t, g(strategy, swarm=1.0)) for t in [(2, 2), (3, 2), (2, 3)]]
    for z in pack:
        z.sees_player = lambda _w: True
    assert ticks_until(world, lambda: all(z.in_contact(world.player) or z.state is ZombieState.ATTACK for z in pack), 30 * 60) is not None


# --- Different genomes behave differently ---


def test_speed_gene_changes_time_to_reach_player(tmp_path):
    def reach(speed: float) -> int:
        world = make_world(tmp_path)
        z = world.spawn_zombie((2, 3), g(speed=speed))
        return ticks_until(world, lambda: z.in_contact(world.player))

    slow, fast = reach(0.1), reach(0.9)
    assert fast is not None and slow is not None
    assert fast * 2 < slow


def test_health_gene_changes_bullets_to_kill(tmp_path):
    def shots_to_kill(health: float) -> int:
        world = make_world(tmp_path)
        z = world.spawn_zombie((6, 3), g(speed=0.0, health=health))
        z.sees_player = lambda _w: False
        shots = 0
        while z in world.zombies:
            world.step(Action(shoot=True, aim=z.pos))
            for _ in range(20):
                world.step(Action())
            shots += 1
        return shots

    assert shots_to_kill(0.0) == math.ceil(ZOMBIE_HP_RANGE[0] / BULLET_DAMAGE) == 1
    assert shots_to_kill(1.0) == math.ceil(ZOMBIE_HP_RANGE[1] / BULLET_DAMAGE) == 4


def test_vision_gene_changes_what_zombie_sees(tmp_path):
    world = make_world(tmp_path)
    put(world.player, world, (14, 3))
    keen = world.spawn_zombie((6, 3), g(vision=1.0))  # 8 tiles = 256 px away
    blind = world.spawn_zombie((6, 2), g(vision=0.0))
    assert keen.sees_player(world)
    assert not blind.sees_player(world)
    world.step(Action())
    assert keen.state is ZombieState.CHASE
    assert blind.state is ZombieState.WANDER


def test_aggression_gene_changes_how_long_zombie_searches(tmp_path):
    def search_ticks(aggression: float) -> int:
        world = make_world(tmp_path, DETOUR)
        z = world.spawn_zombie((6, 1), g(PathStrategy.ASTAR, aggression=aggression))
        world.step(Action())
        assert z.state is ZombieState.CHASE
        put(world.player, world, (1, 1))  # vanish behind the wall
        world.step(Action())
        assert z.state is ZombieState.SEARCH
        return ticks_until(world, lambda: z.state is ZombieState.WANDER)

    timid, aggressive = search_ticks(0.0), search_ticks(1.0)
    assert timid == pytest.approx(ZOMBIE_GIVE_UP_RANGE[0] / FIXED_DT, abs=2)
    assert aggressive == pytest.approx(ZOMBIE_GIVE_UP_RANGE[1] / FIXED_DT, abs=2)


def test_path_strategy_gene_changes_whether_zombie_gets_around_walls(tmp_path):
    def arrives(gene: float) -> bool:
        world = make_world(tmp_path, DETOUR)
        z = world.spawn_zombie((1, 2), Genome.of(path_strategy=gene, swarm=0.0))
        z.sees_player = lambda _w: True
        return ticks_until(world, lambda: z.in_contact(world.player), 600) is not None

    assert not arrives(0.1)  # DIRECT: stuck against the wall
    assert arrives(0.5)  # GREEDY
    assert arrives(0.9)  # ASTAR


def test_random_genomes_give_varied_zombies(tmp_path):
    rng = np.random.default_rng(0)
    world = make_world(tmp_path)
    zombies = [world.spawn_zombie((1, 1), Genome(rng.random(NUM_GENES))) for _ in range(30)]
    assert len({round(z.speed) for z in zombies}) > 10
    assert len({round(z.max_hp) for z in zombies}) > 10
    assert {z.path_strategy for z in zombies} == set(PathStrategy)
