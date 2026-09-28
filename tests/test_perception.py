"""P3.3 — zombie perception (vision radius + line of sight) and FSM-driven behaviour."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ai.genome import Genome, PathStrategy
from config import FIXED_DT
from game.engine import World
from game.map import TileMap
from game.player import Action
from game.states import ZombieState

W, C, S, A = ZombieState.WANDER, ZombieState.CHASE, ZombieState.SEARCH, ZombieState.ATTACK
TS = 32


def genome(strategy=PathStrategy.DIRECT) -> Genome:
    """All-0.5 stats (vision 256 px, give-up 3 s), no swarm."""
    return Genome.of(path_strategy=strategy, swarm=0.0)


STATS = genome().decode()
GIVE_UP_TICKS = round(STATS.give_up_time / FIXED_DT)

# 14 x 7. Wall column at x=7 (rows 1-4) splits the room; row 5 is a gap underneath.
ARENA = "\n".join(
    [
        "##############",
        "#S.....#.....#",
        "#......#.....#",
        "#......#..P..#",
        "#......#.....#",
        "#............#",
        "##############",
    ]
)
HIDDEN = (3, 3)  # left of the wall: out of sight from anywhere right of it in rows 1-4


def make_world(tmp_path, seed: int = 0) -> World:
    path = tmp_path / "arena.txt"
    path.write_text(ARENA)
    return World(TileMap.load(path, tile_size=TS), np.random.default_rng(seed))


def put(world: World, body, tile) -> None:
    body.x, body.y = world.tilemap.tile_to_world(tile)


# --- Vision: radius and line of sight ---


def test_sees_player_in_open(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((8, 3), genome())
    assert z.sees_player(world)


def test_walls_block_vision(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((5, 3), genome())  # 5 tiles away, well within radius, wall in between
    assert math.dist(z.pos, world.player.pos) < z.vision_radius
    assert not z.sees_player(world)
    world.step(Action())
    assert z.state is W


def test_barricades_block_vision(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((8, 3), genome())
    world.barricades.add((9, 3))
    assert not z.sees_player(world)


def test_vision_radius_limits_sight(tmp_path):
    world = make_world(tmp_path)
    put(world, world.player, (12, 5))
    near = world.spawn_zombie((5, 5), genome())  # 7 tiles along the open bottom row
    far = world.spawn_zombie((1, 5), genome())  # 11 tiles: clear line, but too far
    assert near.sees_player(world)
    assert not far.sees_player(world)


def test_never_seen_has_no_memory(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((1, 1), genome())
    world.step(Action())
    assert z.last_known_pos is None
    assert z.time_since_seen == math.inf


# --- Losing the player -> SEARCH at the last known position ---


def chase_then_hide(tmp_path, strategy=PathStrategy.ASTAR):
    """Zombie sees and chases the player, then the player 'teleports' behind the wall."""
    world = make_world(tmp_path)
    z = world.spawn_zombie((8, 1), genome(strategy))
    world.step(Action())
    assert z.state is C
    seen_at = world.player.pos
    put(world, world.player, HIDDEN)
    world.step(Action())
    return world, z, seen_at


def test_losing_player_triggers_search(tmp_path):
    world, z, _ = chase_then_hide(tmp_path)
    assert not z.sees_player(world)
    assert z.state is S


def test_search_targets_last_known_position(tmp_path):
    world, z, seen_at = chase_then_hide(tmp_path)
    last_tile = world.tilemap.world_to_tile(seen_at)
    assert z.last_known_pos == seen_at
    assert z.path_goal == last_tile  # planning toward where the player was, not where it is
    assert z.path[-1] == last_tile
    for _ in range(GIVE_UP_TICKS - 10):
        world.step(Action())
        assert z.state is S
    assert math.dist(z.pos, seen_at) < 1.0  # got there and waits on the spot


@pytest.mark.parametrize("strategy", list(PathStrategy))
def test_every_strategy_searches_last_known_position(tmp_path, strategy):
    world, z, seen_at = chase_then_hide(tmp_path, strategy)
    for _ in range(GIVE_UP_TICKS - 10):
        world.step(Action())
    assert math.dist(z.pos, seen_at) < 1.0


def test_search_gives_up_and_wanders(tmp_path):
    world, z, _ = chase_then_hide(tmp_path)
    for _ in range(GIVE_UP_TICKS - 2):
        world.step(Action())
    assert z.state is S
    for _ in range(3):
        world.step(Action())
    assert z.state is W
    assert z.wander_target is not None


def test_search_resumes_chase_when_player_found(tmp_path):
    world, z, _ = chase_then_hide(tmp_path)
    put(world, world.player, (11, 1))  # back in view
    world.step(Action())
    assert z.state is C


# --- ATTACK behaviour ---


def test_attack_stands_still_and_hits(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((9, 3), genome())
    for _ in range(120):
        world.step(Action())
        if z.state is A:
            break
    assert z.state is A
    hp, pos = world.player.hp, z.pos
    for _ in range(90):
        world.step(Action())
        assert z.state is A
        assert z.pos == pos
    assert world.player.hp < hp


# --- WANDER behaviour ---


@pytest.mark.parametrize("strategy", list(PathStrategy))
def test_wander_moves_between_random_floor_tiles(tmp_path, strategy):
    world = make_world(tmp_path)
    put(world, world.player, (12, 1))  # far corner; keep out of the wanderer's sight
    world.player.hp = world.player.max_hp = 1e9
    z = world.spawn_zombie((2, 3), genome(strategy))
    z.sees_player = lambda _world: False  # isolate wandering from chasing
    targets, start = set(), z.pos
    for _ in range(30 * 60):
        world.step(Action())
        assert z.state is W
        assert not world.is_blocked(z.wander_target)
        assert not any(world.is_blocked(t) for t in z.occupied_tiles())
        targets.add(z.wander_target)
    assert len(targets) >= 3
    assert z.pos != start


def test_wander_is_deterministic_for_a_seed(tmp_path):
    def run(seed):
        world = make_world(tmp_path, seed)
        z = world.spawn_zombie((2, 3), genome(PathStrategy.ASTAR))
        for _ in range(600):
            world.step(Action())
        return z.pos, z.wander_target

    assert run(5) == run(5)
    assert run(5) != run(6)


# --- All four states reachable in one run ---


def test_all_four_states_reachable(tmp_path):
    world = make_world(tmp_path)
    z = world.spawn_zombie((1, 1), genome(PathStrategy.ASTAR))
    visited = []

    def step(n=1):
        for _ in range(n):
            world.step(Action())
            if not visited or visited[-1] is not z.state:
                visited.append(z.state)

    step()  # hidden behind the wall: WANDER
    put(world, z, (9, 3))  # walk up to the player
    step(120)  # CHASE -> ATTACK
    put(world, world.player, HIDDEN)  # player vanishes behind the wall
    step()  # ATTACK -> SEARCH
    step(GIVE_UP_TICKS + 5)  # give up -> WANDER
    assert set(visited) == {W, C, S, A}
    assert visited[:4] == [W, C, A, S]
    assert visited[4] is W
