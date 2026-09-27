"""P1.5 — tile line of sight (supercover traversal between tile centres)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from game.engine import World
from game.map import TileMap

LEVEL1_PATH = Path(__file__).resolve().parent.parent / "maps" / "level1.txt"

# 8 x 5, border walls only.
OPEN = "\n".join(
    [
        "########",
        "#S.....#",
        "#......#",
        "#.....P#",
        "########",
    ]
)


def load(tmp_path, text: str) -> TileMap:
    path = tmp_path / "map.txt"
    path.write_text(text)
    return TileMap.load(path)


def with_walls(tm: TileMap, *extra) -> TileMap:
    """Same map with extra wall tiles, for small targeted cases."""
    return TileMap(tm.width, tm.height, tm.tile_size, tm.walls | set(extra), tm.spawn_points, tm.player_start)


@pytest.fixture
def room(tmp_path) -> TileMap:
    return load(tmp_path, OPEN)


# --- Basic cases ---


def test_same_tile(room):
    assert room.has_line_of_sight((2, 2), (2, 2))


def test_adjacent_and_straight_lines(room):
    assert room.has_line_of_sight((1, 1), (2, 1))
    assert room.has_line_of_sight((1, 2), (6, 2))
    assert room.has_line_of_sight((3, 1), (3, 3))


def test_diagonal_in_open_room(room):
    assert room.has_line_of_sight((1, 1), (3, 3))


def test_wall_between_blocks(room):
    tm = with_walls(room, (4, 2))
    assert not tm.has_line_of_sight((1, 2), (6, 2))
    assert tm.has_line_of_sight((1, 1), (6, 1))  # row above is clear


def test_blocked_endpoint_never_visible(room):
    assert not room.has_line_of_sight((1, 1), (0, 0))  # border wall
    assert not room.has_line_of_sight((0, 0), (1, 1))
    assert not room.has_line_of_sight((1, 1), (-3, 1))  # out of bounds


# --- Exact tile coverage ---


def test_shallow_line_checks_exact_tiles(room):
    """(1,1)->(5,2) touches (1,1),(2,1),(3,1),(3,2),(4,2),(5,2) and nothing else."""
    a, b = (1, 1), (5, 2)
    for wall in [(2, 1), (3, 1), (3, 2), (4, 2)]:
        assert not with_walls(room, wall).has_line_of_sight(a, b), wall
    for wall in [(4, 1), (2, 2), (1, 2), (5, 1)]:
        assert with_walls(room, wall).has_line_of_sight(a, b), wall


def test_cannot_see_through_diagonal_gap(room):
    tm = with_walls(room, (2, 1), (1, 2))
    assert not tm.has_line_of_sight((1, 1), (2, 2))


def test_cannot_graze_wall_corner(room):
    """Segment through a corner touching one wall is blocked (conservative)."""
    assert not with_walls(room, (2, 1)).has_line_of_sight((1, 1), (3, 3))
    assert not with_walls(room, (1, 2)).has_line_of_sight((1, 1), (3, 3))


# --- Properties on the real map ---


def all_floor(tm: TileMap):
    return [(x, y) for x in range(tm.width) for y in range(tm.height) if tm.is_walkable((x, y))]


def test_symmetric_on_level1():
    tm = TileMap.load(LEVEL1_PATH)
    floor = all_floor(tm)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        a = floor[rng.integers(len(floor))]
        b = floor[rng.integers(len(floor))]
        assert tm.has_line_of_sight(a, b) == tm.has_line_of_sight(b, a), (a, b)


def test_visible_implies_every_sampled_point_is_clear():
    """Cross-check against dense float sampling: if LOS says visible, no point
    along the segment may land in a wall."""
    tm = TileMap.load(LEVEL1_PATH)
    floor = all_floor(tm)
    rng = np.random.default_rng(1)
    visible = 0
    for _ in range(1000):
        a = floor[rng.integers(len(floor))]
        b = floor[rng.integers(len(floor))]
        if not tm.has_line_of_sight(a, b):
            continue
        visible += 1
        for t in np.linspace(0.0, 1.0, 400):
            x = a[0] + 0.5 + t * (b[0] - a[0])
            y = a[1] + 0.5 + t * (b[1] - a[1])
            assert tm.is_walkable((int(np.floor(x)), int(np.floor(y)))), (a, b, t)
    assert visible > 50  # the check actually exercised something


def test_level1_room_walls_hide_player():
    tm = TileMap.load(LEVEL1_PATH)
    start = tm.player_start
    assert tm.has_line_of_sight(start, (15, 9))  # inside the central room
    assert not tm.has_line_of_sight(start, (16, 13))  # just below the room's bottom wall
    for spawn in tm.spawn_points:
        assert not tm.has_line_of_sight(start, spawn)


# --- World: barricades block sight ---


def test_world_los_blocked_by_barricade(tmp_path):
    world = World(load(tmp_path, OPEN), np.random.default_rng(0))
    assert world.has_line_of_sight((1, 2), (6, 2))
    world.barricades.add((4, 2))
    assert not world.has_line_of_sight((1, 2), (6, 2))
    assert world.tilemap.has_line_of_sight((1, 2), (6, 2))  # plain map ignores barricades
