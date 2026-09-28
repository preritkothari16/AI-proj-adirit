"""P3.2 — zombie movement strategies (DIRECT / GREEDY / ASTAR), headless.

These tests isolate navigation from perception: spawn_chaser() gives the zombie
x-ray sight so it is always CHASEing (vision/LOS are tested in test_perception.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ai.genome import Genome, PathStrategy
from ai.pathfinding import astar, greedy_best_first
from config import FIXED_DT, ZOMBIE_REPATH_INTERVAL
from game.engine import World
from game.map import TileMap
from game.player import Action

TS = 32
LEVEL1_PATH = Path(__file__).resolve().parent.parent / "maps" / "level1.txt"
REPATH_TICKS = round(ZOMBIE_REPATH_INTERVAL / FIXED_DT)
PLANNERS = [PathStrategy.GREEDY, PathStrategy.ASTAR]

# Wall column at x=3 (rows 1-3) between zombie (1,2) and player (5,2); gap below.
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
DETOUR_ZOMBIE = (1, 2)

# Found by random search: from (1,3) to the player at (7,3), A* = 11 tiles, Greedy = 17.
GREEDY_TRAP = "\n".join(
    [
        "#########",
        "#S......#",
        "#...#..##",
        "#..#..#P#",
        "#.#..##.#",
        "#.......#",
        "#########",
    ]
)
TRAP_ZOMBIE = (1, 3)

# Two routes from the left room to the player: top row or bottom row.
TWO_ROUTES = "\n".join(
    [
        "#########",
        "#.......#",
        "#.#####.#",
        "#S#...#P#",
        "#.#####.#",
        "#.......#",
        "#########",
    ]
)


def make_world(tmp_path, text: str) -> World:
    path = tmp_path / "map.txt"
    path.write_text(text)
    return World(TileMap.load(path, tile_size=TS), np.random.default_rng(0))


def spawn_chaser(world: World, tile, strategy):
    """Zombie that always sees the player, so it chases from the first tick."""
    z = world.spawn_zombie(tile, Genome.of(path_strategy=strategy, swarm=0.0))
    z.sees_player = lambda _world: True
    return z


def ticks_to_contact(world: World, zombie, max_ticks: int = 1200):
    """Step with an idle player until the zombie touches them; None if it never does."""
    for tick in range(max_ticks):
        if zombie.in_contact(world.player):
            return tick
        world.step(Action())
    return None


# --- The three behaviours around an obstacle ---


def test_direct_gets_stuck_behind_wall(tmp_path):
    world = make_world(tmp_path, DETOUR)
    z = spawn_chaser(world, DETOUR_ZOMBIE, PathStrategy.DIRECT)
    assert ticks_to_contact(world, z, 600) is None
    assert z.x <= 3 * TS - z.radius + 1e-6  # pressed against the wall
    assert z.nodes_expanded == 0 and z.repaths == 0  # never plans


@pytest.mark.parametrize("strategy", PLANNERS)
def test_planners_route_around_wall(tmp_path, strategy):
    world = make_world(tmp_path, DETOUR)
    z = spawn_chaser(world, DETOUR_ZOMBIE, strategy)
    assert ticks_to_contact(world, z, 600) is not None
    assert z.nodes_expanded > 0


def test_direct_is_fine_in_the_open(tmp_path):
    world = make_world(tmp_path, DETOUR)
    z = spawn_chaser(world, (7, 2), PathStrategy.DIRECT)  # same side of the wall as the player
    assert ticks_to_contact(world, z, 300) is not None


def test_astar_beats_greedy_on_trap_map(tmp_path):
    graph = make_world(tmp_path, GREEDY_TRAP)
    goal = graph.tilemap.player_start
    assert len(astar(graph, TRAP_ZOMBIE, goal)) < len(greedy_best_first(graph, TRAP_ZOMBIE, goal))

    times = {}
    for strategy in PLANNERS:
        world = make_world(tmp_path, GREEDY_TRAP)
        z = spawn_chaser(world, TRAP_ZOMBIE, strategy)
        times[strategy] = ticks_to_contact(world, z)
        assert times[strategy] is not None, strategy
    assert times[PathStrategy.ASTAR] < times[PathStrategy.GREEDY]


def test_level1_planners_reach_player_direct_does_not(tmp_path):
    """On the real map every spawn needs to get into the central room."""
    for spawn in TileMap.load(LEVEL1_PATH).spawn_points:
        for strategy, should_arrive in [
            (PathStrategy.ASTAR, True),
            (PathStrategy.GREEDY, True),
            (PathStrategy.DIRECT, False),
        ]:
            world = make_world(tmp_path, LEVEL1_PATH.read_text())
            z = spawn_chaser(world, spawn, strategy)
            arrived = ticks_to_contact(world, z, 20 * 60) is not None
            assert arrived == should_arrive, (spawn, strategy)


# --- Following tile centres ---


@pytest.mark.parametrize("strategy", PLANNERS)
def test_path_is_tile_centres_and_never_overlaps_walls(tmp_path, strategy):
    world = make_world(tmp_path, LEVEL1_PATH.read_text())
    z = spawn_chaser(world, (2, 2), strategy)
    reached_centres = 0
    for _ in range(20 * 60):
        before, plans = list(z.path), z.repaths
        world.step(Action())
        assert not any(world.is_blocked(t) for t in z.occupied_tiles())
        assert all(not world.is_blocked(t) for t in z.path)
        if z.repaths == plans and before and z.path == before[1:]:  # popped a waypoint: on its centre
            cx, cy = world.tilemap.tile_to_world(before[0])
            assert abs(z.x - cx) <= 0.5 and abs(z.y - cy) <= 0.5
            reached_centres += 1
        if z.in_contact(world.player):
            break
    assert z.in_contact(world.player)
    assert reached_centres > 10


def test_consecutive_waypoints_are_4_neighbours(tmp_path):
    world = make_world(tmp_path, DETOUR)
    z = spawn_chaser(world, DETOUR_ZOMBIE, PathStrategy.ASTAR)
    world.step(Action())
    full = [world.tilemap.world_to_tile(z.pos)] + z.path
    for (ax, ay), (bx, by) in zip(full, full[1:]):
        assert abs(ax - bx) + abs(ay - by) == 1


# --- Re-planning ---


@pytest.mark.parametrize("strategy", PLANNERS)
def test_repaths_on_interval(tmp_path, strategy):
    world = make_world(tmp_path, LEVEL1_PATH.read_text())
    z = spawn_chaser(world, (2, 2), strategy)  # far away; won't arrive in 3 intervals
    world.step(Action())
    assert z.repaths == 1  # plans on its first update
    for _ in range(3 * REPATH_TICKS):
        world.step(Action())
    assert z.repaths == 4


def test_repaths_when_player_changes_tile(tmp_path):
    world = make_world(tmp_path, LEVEL1_PATH.read_text())
    z = spawn_chaser(world, (2, 2), PathStrategy.ASTAR)
    world.step(Action())
    assert z.path_goal == world.tilemap.player_start
    world.player.x += TS  # teleport one tile right, well before the interval is up
    world.step(Action())
    new_tile = world.tilemap.world_to_tile(world.player.pos)
    assert z.repaths == 2
    assert z.path_goal == new_tile
    assert z.path[-1] == new_tile


def test_no_repath_while_player_stays_on_same_tile(tmp_path):
    world = make_world(tmp_path, LEVEL1_PATH.read_text())
    z = spawn_chaser(world, (2, 2), PathStrategy.ASTAR)
    world.step(Action())
    for _ in range(REPATH_TICKS - 5):
        world.step(Action(move=(0.0, 0.0)))
    assert z.repaths == 1


def test_barricade_on_path_is_avoided_after_repath(tmp_path):
    world = make_world(tmp_path, TWO_ROUTES)
    z = spawn_chaser(world, (1, 3), PathStrategy.ASTAR)
    world.step(Action())
    route_tile = next(t for t in z.path if t[1] in (1, 5) and 2 <= t[0] <= 6)  # a tile on its chosen route
    assert world.place_barricade(route_tile).name == "OK"
    for _ in range(REPATH_TICKS):
        world.step(Action())
    assert route_tile not in z.path
    assert ticks_to_contact(world, z) is not None


def test_zombies_do_not_plan_while_attacking(tmp_path):
    world = make_world(tmp_path, DETOUR)
    z = spawn_chaser(world, (6, 2), PathStrategy.ASTAR)
    assert ticks_to_contact(world, z, 120) is not None
    count = z.repaths
    for _ in range(3 * REPATH_TICKS):
        world.step(Action())
    assert z.repaths == count
