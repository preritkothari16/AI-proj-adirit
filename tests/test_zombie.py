"""P2.6 — dummy zombie: direct movement, bullet damage, death, contact attacks, player death."""
from __future__ import annotations

import numpy as np
import pytest

from config import (
    BULLET_DAMAGE,
    FIXED_DT,
    PLAYER_MAX_HP,
    ZOMBIE_ATTACK_COOLDOWN,
    ZOMBIE_ATTACK_DAMAGE,
    ZOMBIE_MAX_HP,
    ZOMBIE_SPEED,
)
from game.engine import World, _segment_circle_entry
from game.map import TileMap
from game.player import Action
from game.zombie import Zombie

TS = 32
ATTACK_TICKS = round(ZOMBIE_ATTACK_COOLDOWN / FIXED_DT)

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

# Wall column at x=3 between a zombie at (1,2) and the player at (5,2).
WALLED = "\n".join(
    [
        "#########",
        "#S.#....#",
        "#..#.P..#",
        "#..#....#",
        "#########",
    ]
)


def make_world(tmp_path, text: str = ROOM) -> World:
    path = tmp_path / "map.txt"
    path.write_text(text)
    return World(TileMap.load(path, tile_size=TS), np.random.default_rng(0))


def place_next_to_player(world: World) -> Zombie:
    """Zombie touching the player's right side (bodies exactly flush)."""
    p = world.player
    z = world.spawn_zombie(world.tilemap.world_to_tile(p.pos))
    z.x = p.x + p.radius + z.radius
    return z


def shoot_at(world: World, target) -> None:
    world.player.cooldown_remaining = 0
    world.step(Action(shoot=True, aim=target.pos))


# --- Movement ---


def test_spawn_zombie_at_tile_centre(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((1, 1))
    assert z.pos == world.tilemap.tile_to_world((1, 1))
    assert z.hp == ZOMBIE_MAX_HP
    assert world.zombies == [z]


def test_zombie_moves_straight_at_player(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((1, 2))  # same row, left of player
    x0 = z.x
    world.step(Action())
    assert z.x == pytest.approx(x0 + ZOMBIE_SPEED * FIXED_DT)
    assert z.y == pytest.approx(world.player.y)


def test_zombie_stops_on_contact(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((1, 2))
    for _ in range(300):
        world.step(Action())
    assert z.in_contact(world.player)
    x = z.x
    world.step(Action())
    assert z.x == x
    assert z.x < world.player.x - z.radius - world.player.radius + 1e-6  # never overlaps


def test_direct_zombie_is_stopped_by_wall(tmp_path):
    """No pathfinding yet: walking straight at the player through a wall just stalls."""
    world = make_world(tmp_path, WALLED)
    z = world.spawn_zombie((1, 2))
    for _ in range(300):
        world.step(Action())
    assert z.x <= 3 * TS - z.radius + 1e-6
    assert world.player.hp == PLAYER_MAX_HP


# --- Bullet damage and zombie death ---


def test_bullet_damages_zombie(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((7, 2))
    shoot_at(world, z)
    for _ in range(20):
        world.step(Action())
    assert z.hp == pytest.approx(ZOMBIE_MAX_HP - BULLET_DAMAGE)
    assert world.bullets == []  # bullet consumed by the hit


def test_zombie_dies_at_zero_hp_and_is_removed(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((7, 2))
    shots = int(np.ceil(ZOMBIE_MAX_HP / BULLET_DAMAGE))
    for _ in range(shots):
        shoot_at(world, z)
        for _ in range(20):
            world.step(Action())
    assert z.hp == 0.0
    assert not z.alive
    assert world.zombies == []


def test_hp_never_negative():
    z = Zombie((0.0, 0.0), TS)
    z.take_damage(ZOMBIE_MAX_HP * 10)
    assert z.hp == 0.0


def test_bullet_hits_only_the_first_zombie_in_line(tmp_path):
    world = make_world(tmp_path)
    near = world.spawn_zombie((6, 2))
    far = world.spawn_zombie((7, 2))
    shoot_at(world, far)
    for _ in range(20):
        world.step(Action())
    assert near.hp < ZOMBIE_MAX_HP
    assert far.hp == ZOMBIE_MAX_HP


def test_miss_does_no_damage(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((7, 3))
    world.step(Action(shoot=True, aim=(world.player.x, 0.0)))  # fire straight up, away from zombie
    for _ in range(30):
        world.step(Action())
    assert z.hp == ZOMBIE_MAX_HP


def test_segment_test_catches_fast_bullets():
    """A step far longer than the zombie is wide still registers the hit."""
    assert _segment_circle_entry((0.0, 0.0), (1000.0, 0.0), (500.0, 5.0), 13.0) is not None
    assert _segment_circle_entry((0.0, 0.0), (1000.0, 0.0), (500.0, 50.0), 13.0) is None
    assert _segment_circle_entry((0.0, 0.0), (100.0, 0.0), (500.0, 0.0), 13.0) is None  # short of it


# --- Attacks on the player ---


def test_contact_damages_player(tmp_path):
    world = make_world(tmp_path)
    z = place_next_to_player(world)
    world.step(Action())
    assert world.player.hp == pytest.approx(PLAYER_MAX_HP - ZOMBIE_ATTACK_DAMAGE)
    assert z.damage_dealt == pytest.approx(ZOMBIE_ATTACK_DAMAGE)


def test_attack_cooldown(tmp_path):
    world = make_world(tmp_path)
    place_next_to_player(world)
    for _ in range(2 * ATTACK_TICKS):
        world.step(Action())
    assert world.player.hp == pytest.approx(PLAYER_MAX_HP - 2 * ZOMBIE_ATTACK_DAMAGE)


def test_no_damage_without_contact(tmp_path):
    world = make_world(tmp_path)
    world.spawn_zombie((1, 1))
    world.step(Action())
    assert world.player.hp == PLAYER_MAX_HP


def test_player_dies_and_world_stops(tmp_path):
    world = make_world(tmp_path)
    place_next_to_player(world)
    hits_to_kill = int(np.ceil(PLAYER_MAX_HP / ZOMBIE_ATTACK_DAMAGE))
    for _ in range(hits_to_kill * ATTACK_TICKS + 5):
        world.step(Action())
    assert world.player.hp == 0.0
    assert not world.player.alive
    assert world.game_over
    tick, pos = world.tick, world.player.pos
    world.step(Action(move=(1.0, 0.0), shoot=True, aim=(0.0, 0.0)))
    assert world.tick == tick  # frozen once over
    assert world.player.pos == pos
    assert world.bullets == []


def test_killing_zombie_stops_its_attacks(tmp_path):
    world = make_world(tmp_path)
    z = place_next_to_player(world)
    world.step(Action())  # first hit lands
    hp = world.player.hp
    z.take_damage(ZOMBIE_MAX_HP)
    for _ in range(3 * ATTACK_TICKS):
        world.step(Action())
    assert world.player.hp == hp
    assert world.zombies == []


def test_player_can_shoot_approaching_zombies(tmp_path):
    """End-to-end: player holding fire at an approaching zombie kills it before it arrives."""
    world = make_world(tmp_path)
    z = world.spawn_zombie((1, 2))
    for _ in range(300):
        world.step(Action(shoot=True, aim=z.pos))
    assert not z.alive
    assert world.zombies == []
    assert world.player.hp == PLAYER_MAX_HP
