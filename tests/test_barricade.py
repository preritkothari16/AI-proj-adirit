"""P2.5 — barricade placement validity, sealing prevention, pathfinding effect."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ai.genome import Genome, PathStrategy
from ai.pathfinding import astar, greedy_best_first
from config import MAX_BARRICADES
from game.barricade import PlacementResult, all_spawns_connected, check_placement
from game.engine import World
from game.map import TileMap
from game.player import Action

TS = 32
LEVEL1_PATH = Path(__file__).resolve().parent.parent / "maps" / "level1.txt"

# 9 x 5 open room, spawn top-left, player centre.
ROOM = "\n".join(
    [
        "#########",
        "#S......#",
        "#...P...#",
        "#.......#",
        "#########",
    ]
)

# Spawn connects to the player only through the 1-tile corridor at x=1..5, y=1.
CORRIDOR = "\n".join(
    [
        "#########",
        "#S....#.#",
        "#####.#.#",
        "#...P...#",
        "#########",
    ]
)

# Spawn reaches the player by two routes: down the left (1,2) or the right (7,2).
TWO_ROUTES = "\n".join(
    [
        "#########",
        "#S......#",
        "#.#####.#",
        "#...P...#",
        "#########",
    ]
)


def make_world(tmp_path, text: str) -> World:
    path = tmp_path / "map.txt"
    path.write_text(text)
    return World(TileMap.load(path, tile_size=TS), np.random.default_rng(0))


# --- Valid placement ---


def test_valid_placement_is_accepted(tmp_path):
    world = make_world(tmp_path, ROOM)
    assert world.place_barricade((6, 1)) is PlacementResult.OK
    assert (6, 1) in world.barricades
    assert world.last_placement is PlacementResult.OK


def test_check_placement_does_not_modify_world(tmp_path):
    world = make_world(tmp_path, ROOM)
    assert check_placement(world, (6, 1)) is PlacementResult.OK
    assert world.barricades == set()


def test_step_places_barricade_from_action(tmp_path):
    world = make_world(tmp_path, ROOM)
    world.step(Action(barricade=(6, 1)))
    assert (6, 1) in world.barricades


def test_step_without_barricade_leaves_last_placement_none(tmp_path):
    world = make_world(tmp_path, ROOM)
    world.step(Action())
    assert world.last_placement is None


# --- Rejections ---


def test_wall_tile_rejected(tmp_path):
    world = make_world(tmp_path, ROOM)
    assert world.place_barricade((0, 0)) is PlacementResult.INVALID_TILE
    assert world.barricades == set()


def test_out_of_bounds_rejected(tmp_path):
    world = make_world(tmp_path, ROOM)
    for tile in [(-1, 2), (9, 2), (3, -1), (3, 5)]:
        assert world.place_barricade(tile) is PlacementResult.INVALID_TILE
    assert world.barricades == set()


def test_spawn_tile_rejected(tmp_path):
    world = make_world(tmp_path, ROOM)
    assert world.place_barricade((1, 1)) is PlacementResult.INVALID_TILE


def test_duplicate_rejected(tmp_path):
    world = make_world(tmp_path, ROOM)
    world.place_barricade((6, 1))
    assert world.place_barricade((6, 1)) is PlacementResult.OCCUPIED
    assert len(world.barricades) == 1


def test_player_tile_rejected(tmp_path):
    world = make_world(tmp_path, ROOM)
    assert world.place_barricade((4, 2)) is PlacementResult.OCCUPIED


def test_tile_partly_under_player_rejected(tmp_path):
    """Player straddling two tiles blocks both, so it can never be trapped inside a barricade."""
    world = make_world(tmp_path, ROOM)
    world.player.x = 5 * TS  # exactly on the (4,2)/(5,2) boundary
    assert world.place_barricade((5, 2)) is PlacementResult.OCCUPIED
    assert world.place_barricade((4, 2)) is PlacementResult.OCCUPIED


def test_tile_under_zombie_rejected(tmp_path):
    world = make_world(tmp_path, ROOM)
    world.spawn_zombie((6, 1), Genome.of(path_strategy=PathStrategy.DIRECT, swarm=0.0))
    assert world.place_barricade((6, 1)) is PlacementResult.OCCUPIED


def test_limit_reached(tmp_path):
    world = make_world(tmp_path, LEVEL1_PATH.read_text())
    candidates = [(5, 2), (7, 2), (9, 2), (11, 2), (13, 2), (15, 2), (17, 2)]
    results = [world.place_barricade(t) for t in candidates[: MAX_BARRICADES + 1]]
    assert results[:MAX_BARRICADES] == [PlacementResult.OK] * MAX_BARRICADES
    assert results[MAX_BARRICADES] is PlacementResult.LIMIT_REACHED
    assert len(world.barricades) == MAX_BARRICADES


def test_rejected_action_does_not_stop_rest_of_step(tmp_path):
    world = make_world(tmp_path, ROOM)
    x0 = world.player.x
    world.step(Action(move=(1.0, 0.0), barricade=(0, 0)))
    assert world.last_placement is PlacementResult.INVALID_TILE
    assert world.player.x > x0


# --- Sealing prevention (BFS) ---


def test_sealing_single_corridor_rejected(tmp_path):
    world = make_world(tmp_path, CORRIDOR)
    for tile in [(2, 1), (3, 1), (4, 1), (5, 1), (5, 2)]:
        assert world.place_barricade(tile) is PlacementResult.BLOCKS_PATH
    assert world.barricades == set()
    assert all_spawns_connected(world)


def test_dead_end_off_the_route_is_allowed(tmp_path):
    world = make_world(tmp_path, CORRIDOR)
    assert world.place_barricade((7, 1)) is PlacementResult.OK  # the spur on the right


def test_closing_one_of_two_routes_ok_but_not_both(tmp_path):
    world = make_world(tmp_path, TWO_ROUTES)
    assert world.place_barricade((1, 2)) is PlacementResult.OK  # left route
    assert world.place_barricade((7, 2)) is PlacementResult.BLOCKS_PATH  # right route, last one
    assert all_spawns_connected(world)


def test_cannot_ring_player_on_level1(tmp_path):
    """Try to box the player in on the real map; the last ring tile must be refused."""
    world = make_world(tmp_path, LEVEL1_PATH.read_text())
    doorways = [(14, 9), (18, 9)]  # the central room's only exits
    assert world.place_barricade(doorways[0]) is PlacementResult.OK
    assert world.place_barricade(doorways[1]) is PlacementResult.BLOCKS_PATH
    assert all_spawns_connected(world)


def test_sealing_checked_against_players_current_tile(tmp_path):
    """Connectivity target follows the player, not just the start tile."""
    world = make_world(tmp_path, CORRIDOR)
    world.player.x, world.player.y = 7.5 * TS, 1.5 * TS  # move player up into the spur
    assert world.place_barricade((7, 2)) is PlacementResult.BLOCKS_PATH
    assert world.place_barricade((4, 3)) is PlacementResult.OK  # now off the route


# --- Pathfinding treats barricades as walls ---


def test_world_neighbors_skip_barricades(tmp_path):
    world = make_world(tmp_path, ROOM)
    assert (5, 2) in world.neighbors((4, 2))
    world.barricades.add((5, 2))
    assert (5, 2) not in world.neighbors((4, 2))


def test_astar_and_greedy_route_around_barricade(tmp_path):
    world = make_world(tmp_path, ROOM)
    start, goal = (1, 2), (7, 2)
    assert len(astar(world, start, goal)) == 7  # straight line
    world.place_barricade((3, 2))
    for search in (astar, greedy_best_first):
        path = search(world, start, goal)
        assert path is not None
        assert (3, 2) not in path
        assert len(path) > 7


def test_barricade_blocks_player_movement(tmp_path):
    world = make_world(tmp_path, ROOM)
    world.place_barricade((5, 2))
    for _ in range(60):
        world.step(Action(move=(1.0, 0.0)))
    assert world.player.x <= 5 * TS - world.player.radius + 1e-6
