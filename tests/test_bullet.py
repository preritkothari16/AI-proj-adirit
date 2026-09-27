"""P2.4 — bullet movement, wall collision, cleanup, firing and cooldown."""
from __future__ import annotations

import math

import numpy as np
import pytest

from config import BULLET_SPEED, FIRE_COOLDOWN, FIXED_DT
from game.bullet import Bullet
from game.engine import World
from game.map import TileMap
from game.player import Action

TS = 32
BOUNDS = (10 * TS, 10 * TS)


def never_blocked(tile):
    return False


# 9 wide x 5 tall, player at (4,2); walls on the border only.
ROOM = "\n".join(
    [
        "#########",
        "#S......#",
        "#...P...#",
        "#.......#",
        "#########",
    ]
)


def make_world(tmp_path) -> World:
    path = tmp_path / "room.txt"
    path.write_text(ROOM)
    return World(TileMap.load(path, tile_size=TS), np.random.default_rng(0))


# --- Bullet on its own ---


def test_bullet_moves_speed_times_dt_each_tick():
    b = Bullet((100.0, 100.0), (1.0, 0.0))
    b.update(FIXED_DT, never_blocked, TS, BOUNDS)
    assert b.x == pytest.approx(100.0 + BULLET_SPEED * FIXED_DT)
    assert b.y == pytest.approx(100.0)
    assert b.alive


def test_bullet_direction_is_normalised():
    b = Bullet((100.0, 100.0), (3.0, 4.0))  # length 5, should still move at BULLET_SPEED
    b.update(FIXED_DT, never_blocked, TS, BOUNDS)
    moved = math.hypot(b.x - 100.0, b.y - 100.0)
    assert moved == pytest.approx(BULLET_SPEED * FIXED_DT)
    assert (b.x - 100.0) / moved == pytest.approx(0.6)


def test_zero_direction_rejected():
    with pytest.raises(ValueError):
        Bullet((0.0, 0.0), (0.0, 0.0))


def test_bullet_dies_on_wall():
    wall_tile = (5, 3)  # bullet starts in tile (3,3) moving right
    b = Bullet((3.5 * TS, 3.5 * TS), (1.0, 0.0))
    for _ in range(60):
        b.update(FIXED_DT, lambda t: t == wall_tile, TS, BOUNDS)
        if not b.alive:
            break
    assert not b.alive
    assert 5 * TS <= b.x < 6 * TS  # stopped inside the wall tile, not past it


def test_fast_bullet_cannot_skip_a_wall():
    """Even at one full tile per tick, sub-stepping catches a 1-tile wall."""
    b = Bullet((1.5 * TS, 1.5 * TS), (1.0, 0.0), speed=2 * TS / FIXED_DT)
    for _ in range(10):
        b.update(FIXED_DT, lambda t: t == (3, 1), TS, BOUNDS)
    assert not b.alive
    assert 3 * TS <= b.x < 4 * TS


def test_bullet_dies_outside_map():
    b = Bullet((BOUNDS[0] - 2.0, 50.0), (1.0, 0.0))
    b.update(FIXED_DT, never_blocked, TS, BOUNDS)
    assert not b.alive


def test_bullet_dies_outside_map_negative_side():
    b = Bullet((2.0, 50.0), (-1.0, 0.0))
    b.update(FIXED_DT, never_blocked, TS, BOUNDS)
    assert not b.alive


def test_dead_bullet_does_not_move():
    b = Bullet((2.0, 50.0), (-1.0, 0.0))
    b.update(FIXED_DT, never_blocked, TS, BOUNDS)
    x = b.x
    b.update(FIXED_DT, never_blocked, TS, BOUNDS)
    assert b.x == x


# --- Firing through World ---


def test_shoot_spawns_bullet_toward_aim(tmp_path):
    world = make_world(tmp_path)
    px, py = world.player.pos
    world.step(Action(shoot=True, aim=(px + 100.0, py)))  # aim straight right
    assert len(world.bullets) == 1
    b = world.bullets[0]
    assert b.vx == pytest.approx(BULLET_SPEED)
    assert b.vy == pytest.approx(0.0)
    assert b.x > px  # already moved one tick


def test_no_shoot_no_bullet(tmp_path):
    world = make_world(tmp_path)
    world.step(Action(aim=(0.0, 0.0)))
    assert world.bullets == []


def test_aiming_at_own_centre_does_not_fire(tmp_path):
    world = make_world(tmp_path)
    world.step(Action(shoot=True, aim=world.player.pos))
    assert world.bullets == []


def test_fire_cooldown_limits_rate(tmp_path):
    world = make_world(tmp_path)
    px, py = world.player.pos
    fired = 0
    one_second = round(1.0 / FIXED_DT)
    for _ in range(one_second):
        world.step(Action(shoot=True, aim=(px, py - 100.0)))  # aim up
        # cooldown is reset to its full value only on the tick a shot is fired
        if world.player.cooldown_remaining == world.player.fire_cooldown_ticks:
            fired += 1
    assert fired == round(1.0 / FIRE_COOLDOWN)  # 4 shots per second at 0.25 s


def test_bullets_hit_walls_and_are_cleaned_up(tmp_path):
    world = make_world(tmp_path)
    px, py = world.player.pos
    for aim in [(px + 100, py), (px - 100, py), (px, py + 100), (px, py - 100)]:
        world.player.cooldown_remaining = 0
        world.step(Action(shoot=True, aim=aim))
    assert len(world.bullets) == 4
    for _ in range(60):  # plenty of time to reach any wall in a 9x5 room
        world.step(Action())
    assert world.bullets == []


def test_barricade_stops_bullets(tmp_path):
    world = make_world(tmp_path)
    world.barricades.add((6, 2))  # two tiles right of player
    px, py = world.player.pos
    world.step(Action(shoot=True, aim=(px + 100, py)))
    for _ in range(10):
        world.step(Action())
    assert world.bullets == []


def test_many_ticks_of_constant_fire_stay_bounded(tmp_path):
    """Holding fire for a long time must not accumulate bullets forever."""
    world = make_world(tmp_path)
    px, py = world.player.pos
    for i in range(3600):
        world.step(Action(shoot=True, aim=(px + math.cos(i), py + math.sin(i))))
    assert len(world.bullets) <= 5
