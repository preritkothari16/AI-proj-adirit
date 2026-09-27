"""P1.3 — TileMap loading, dimensions, walls, coordinate conversion.

Pathfinding is not exercised here (P1.4); has_line_of_sight is not
implemented here (P1.5).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from game.map import TileMap

LEVEL1_PATH = Path(__file__).resolve().parent.parent / "maps" / "level1.txt"


def write_map(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "test_map.txt"
    path.write_text(text)
    return path


# --- small fixture map (5x5) ---

FIXTURE_MAP = "\n".join(
    [
        "#####",
        "#S..#",
        "#.#.#",
        "#..P#",
        "#####",
    ]
)


def test_load_fixture_dimensions(tmp_path):
    path = write_map(tmp_path, FIXTURE_MAP)
    tm = TileMap.load(path, tile_size=10)
    assert tm.width == 5
    assert tm.height == 5
    assert tm.tile_size == 10


def test_load_fixture_walls_and_floor(tmp_path):
    path = write_map(tmp_path, FIXTURE_MAP)
    tm = TileMap.load(path)
    assert tm.is_walkable((0, 0)) is False  # corner wall
    assert tm.is_walkable((2, 2)) is False  # interior wall
    assert tm.is_walkable((1, 1)) is True  # spawn tile is floor
    assert tm.is_walkable((3, 3)) is True  # player tile is floor


def test_out_of_bounds_is_not_walkable(tmp_path):
    path = write_map(tmp_path, FIXTURE_MAP)
    tm = TileMap.load(path)
    assert tm.is_walkable((-1, 0)) is False
    assert tm.is_walkable((0, -1)) is False
    assert tm.is_walkable((5, 0)) is False
    assert tm.is_walkable((0, 5)) is False


def test_spawn_points_and_player_start(tmp_path):
    path = write_map(tmp_path, FIXTURE_MAP)
    tm = TileMap.load(path)
    assert tm.spawn_points == ((1, 1),)
    assert tm.player_start == (3, 3)


def test_neighbors_four_directional_no_diagonals(tmp_path):
    path = write_map(tmp_path, FIXTURE_MAP)
    tm = TileMap.load(path)
    # (1,1) is floor; only (2,1) and (1,2) among its 4-dir candidates are floor
    neighbors = set(tm.neighbors((1, 1)))
    assert neighbors == {(2, 1), (1, 2)}


def test_neighbors_excludes_walls_and_out_of_bounds(tmp_path):
    path = write_map(tmp_path, FIXTURE_MAP)
    tm = TileMap.load(path)
    assert tm.neighbors((0, 0)) == []  # wall tile itself


def test_world_to_tile_and_back(tmp_path):
    path = write_map(tmp_path, FIXTURE_MAP)
    tm = TileMap.load(path, tile_size=10)
    assert tm.world_to_tile((0.0, 0.0)) == (0, 0)
    assert tm.world_to_tile((15.0, 25.0)) == (1, 2)
    assert tm.world_to_tile((49.9, 0.0)) == (4, 0)
    # tile_to_world returns the tile centre
    assert tm.tile_to_world((0, 0)) == (5.0, 5.0)
    assert tm.tile_to_world((1, 2)) == (15.0, 25.0)


def test_world_to_tile_round_trip_stays_in_same_tile(tmp_path):
    path = write_map(tmp_path, FIXTURE_MAP)
    tm = TileMap.load(path, tile_size=10)
    for tile in [(0, 0), (2, 3), (4, 4)]:
        centre = tm.tile_to_world(tile)
        assert tm.world_to_tile(centre) == tile


def test_has_line_of_sight_not_yet_implemented(tmp_path):
    path = write_map(tmp_path, FIXTURE_MAP)
    tm = TileMap.load(path)
    with pytest.raises(NotImplementedError):
        tm.has_line_of_sight((1, 1), (3, 3))


# --- malformed maps ---


def test_ragged_rows_raise(tmp_path):
    path = write_map(tmp_path, "###\n#.\n###")
    with pytest.raises(ValueError):
        TileMap.load(path)


def test_unknown_character_raises(tmp_path):
    path = write_map(tmp_path, "###\n#X#\n###")
    with pytest.raises(ValueError):
        TileMap.load(path)


def test_missing_player_start_raises(tmp_path):
    path = write_map(tmp_path, "###\n#S#\n###")
    with pytest.raises(ValueError):
        TileMap.load(path)


def test_missing_spawn_raises(tmp_path):
    path = write_map(tmp_path, "###\n#P#\n###")
    with pytest.raises(ValueError):
        TileMap.load(path)


def test_multiple_player_starts_raises(tmp_path):
    path = write_map(tmp_path, "####\n#PP#\n#S.#\n####")
    with pytest.raises(ValueError):
        TileMap.load(path)


# --- the real level ---


def test_level1_dimensions():
    tm = TileMap.load(LEVEL1_PATH)
    assert tm.width == 32
    assert tm.height == 20


def test_level1_has_at_least_four_spawns_and_one_player_start():
    tm = TileMap.load(LEVEL1_PATH)
    assert len(tm.spawn_points) >= 4
    assert tm.player_start is not None
    assert tm.is_walkable(tm.player_start)
    for spawn in tm.spawn_points:
        assert tm.is_walkable(spawn)


def test_level1_border_is_solid_wall():
    tm = TileMap.load(LEVEL1_PATH)
    for x in range(tm.width):
        assert tm.is_walkable((x, 0)) is False
        assert tm.is_walkable((x, tm.height - 1)) is False
    for y in range(tm.height):
        assert tm.is_walkable((0, y)) is False
        assert tm.is_walkable((tm.width - 1, y)) is False


def test_level1_all_spawns_reach_player_start_via_bfs():
    """Sanity check for later phases (P1.4/P4.6): no spawn is sealed off.
    Uses TileMap.neighbors directly, not the pathfinding module (not built yet).
    """
    tm = TileMap.load(LEVEL1_PATH)
    from collections import deque

    seen = {tm.player_start}
    queue = deque([tm.player_start])
    while queue:
        current = queue.popleft()
        for n in tm.neighbors(current):
            if n not in seen:
                seen.add(n)
                queue.append(n)

    for spawn in tm.spawn_points:
        assert spawn in seen, f"spawn {spawn} is not reachable from player start"
