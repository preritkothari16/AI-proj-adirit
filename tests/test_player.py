"""P2.2 — Player movement, wall collision and sliding.

Most tests use a small ASCII fixture map through World, so collision is
exercised exactly as the game uses it (World.is_blocked -> Player.move).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from config import FIXED_DT, PLAYER_MAX_HP, PLAYER_RADIUS, PLAYER_SPEED
from game.engine import World
from game.map import TileMap
from game.player import Action, Player

TS = 32  # tile size used by the fixture map

# 7x7 room, player starts in the centre (3,3), open floor around it.
OPEN_ROOM = "\n".join(
    [
        "#######",
        "#S....#",
        "#.....#",
        "#..P..#",
        "#.....#",
        "#.....#",
        "#######",
    ]
)


def make_world(tmp_path, text=OPEN_ROOM) -> World:
    path = tmp_path / "room.txt"
    path.write_text(text)
    return World(TileMap.load(path, tile_size=TS), np.random.default_rng(0))


def run(world: World, move, ticks: int) -> None:
    for _ in range(ticks):
        world.step(Action(move=move))


# --- basics ---


def test_hp_starts_from_config(tmp_path):
    world = make_world(tmp_path)
    assert world.player.hp == PLAYER_MAX_HP
    assert world.player.max_hp == PLAYER_MAX_HP


def test_player_spawns_at_player_start_centre(tmp_path):
    world = make_world(tmp_path)
    assert world.player.pos == (3.5 * TS, 3.5 * TS)


def test_no_input_means_no_movement(tmp_path):
    world = make_world(tmp_path)
    start = world.player.pos
    run(world, (0.0, 0.0), 30)
    assert world.player.pos == start


def test_moves_at_configured_speed(tmp_path):
    world = make_world(tmp_path)
    x0 = world.player.x
    run(world, (1.0, 0.0), 10)  # 10 ticks, well short of the wall
    assert world.player.x - x0 == pytest.approx(PLAYER_SPEED * FIXED_DT * 10)
    assert world.player.y == 3.5 * TS


def test_diagonal_is_not_faster(tmp_path):
    world = make_world(tmp_path)
    x0, y0 = world.player.pos
    run(world, (1.0, 1.0), 10)
    dist = math.hypot(world.player.x - x0, world.player.y - y0)
    assert dist == pytest.approx(PLAYER_SPEED * FIXED_DT * 10)


def test_partial_input_moves_slower(tmp_path):
    """Magnitudes below 1 aren't scaled up (future analogue/bot input)."""
    world = make_world(tmp_path)
    x0 = world.player.x
    run(world, (0.5, 0.0), 10)
    assert world.player.x - x0 == pytest.approx(0.5 * PLAYER_SPEED * FIXED_DT * 10)


# --- walls ---


def test_cannot_pass_through_wall_right(tmp_path):
    world = make_world(tmp_path)
    run(world, (1.0, 0.0), 300)  # 5 s, far more than enough to cross the room
    # wall column 6 starts at x = 6*TS; player's right edge must stop exactly there
    assert world.player.x + PLAYER_RADIUS == pytest.approx(6 * TS)


def test_cannot_pass_through_wall_left_up_down(tmp_path):
    for move, check in [
        ((-1.0, 0.0), lambda p: p.x - PLAYER_RADIUS == pytest.approx(1 * TS)),
        ((0.0, -1.0), lambda p: p.y - PLAYER_RADIUS == pytest.approx(1 * TS)),
        ((0.0, 1.0), lambda p: p.y + PLAYER_RADIUS == pytest.approx(6 * TS)),
    ]:
        world = make_world(tmp_path)
        run(world, move, 300)
        assert check(world.player), f"failed for move {move}: pos={world.player.pos}"


def test_player_never_overlaps_a_wall(tmp_path):
    """Random walk for many ticks; the player box must never overlap a blocked tile."""
    world = make_world(tmp_path)
    rng = np.random.default_rng(123)
    for _ in range(2000):
        move = tuple(rng.uniform(-1, 1, size=2))
        world.step(Action(move=move))
        p = world.player
        assert not any(world.is_blocked(t) for t in p._overlapped_tiles(p.x, p.y))


def test_barricade_tile_blocks_movement(tmp_path):
    world = make_world(tmp_path)
    world.barricades.add((4, 3))  # tile directly right of the player
    run(world, (1.0, 0.0), 120)
    assert world.player.x + PLAYER_RADIUS == pytest.approx(4 * TS)


# --- sliding ---


def test_slides_along_wall_when_moving_diagonally(tmp_path):
    world = make_world(tmp_path)
    run(world, (1.0, 0.0), 300)  # press against the right wall first
    x_at_wall = world.player.x
    y0 = world.player.y
    run(world, (1.0, -1.0), 20)  # push diagonally into the wall and upwards
    assert world.player.x == pytest.approx(x_at_wall)  # x stays pinned
    assert world.player.y < y0  # but y keeps moving: sliding


def test_slides_into_corner_and_stops(tmp_path):
    world = make_world(tmp_path)
    run(world, (1.0, -1.0), 600)
    assert world.player.x + PLAYER_RADIUS == pytest.approx(6 * TS)
    assert world.player.y - PLAYER_RADIUS == pytest.approx(1 * TS)


def test_fits_through_one_tile_corridor(tmp_path):
    """A 1-tile gap (32 px) must be passable for a 20 px player."""
    corridor = "\n".join(
        [
            "#######",
            "#S....#",
            "###.###",
            "#..P..#",
            "#######",
        ]
    )
    world = make_world(tmp_path, corridor)
    run(world, (0.0, -1.0), 120)
    assert world.player.y < 2 * TS  # got past the gap row into row 1


# --- construction guards / independence ---


def test_rejects_speed_that_could_tunnel():
    with pytest.raises(ValueError):
        Player((0.0, 0.0), tile_size=TS, speed=TS / FIXED_DT)


def test_player_move_works_without_world():
    """Movement only needs an is_blocked callable, no World/pygame."""
    p = Player((50.0, 50.0), tile_size=TS)
    p.move((1.0, 0.0), FIXED_DT, is_blocked=lambda t: False)
    assert p.x == pytest.approx(50.0 + PLAYER_SPEED * FIXED_DT)
