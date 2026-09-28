"""P3.5 — zombie debug overlay (F1): paths, vision radius, state labels. SDL dummy driver, no window."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from ai.genome import Genome, PathStrategy
from game.engine import World
from game.map import TileMap
from game.player import Action
from game.renderer import STATE_COLOURS, Renderer
from game.states import ZombieState

ROOT = Path(__file__).resolve().parent.parent
LEVEL1_PATH = ROOT / "maps" / "level1.txt"


@pytest.fixture
def world():
    pygame.init()
    yield World(TileMap.load(LEVEL1_PATH), np.random.default_rng(0))
    pygame.quit()


def chaser(world: World, strategy=PathStrategy.ASTAR, vision: float = 0.5):
    """Zombie in the top-left corner, x-ray sight so it chases along a planned path."""
    z = world.spawn_zombie((2, 2), Genome.of(path_strategy=strategy, vision=vision, swarm=0.0))
    z.sees_player = lambda _w: True
    world.step(Action())
    assert z.state is ZombieState.CHASE
    return z


def render(world: World, debug: bool) -> pygame.Surface:
    renderer = Renderer(world, debug=debug)
    surface = pygame.Surface(renderer.screen_size)
    renderer.draw(surface, world)
    return surface


def colour_near(surface: pygame.Surface, point, colour, radius: int = 1) -> bool:
    x0, y0 = round(point[0]), round(point[1])
    w, h = surface.get_size()
    return any(
        surface.get_at((x, y))[:3] == colour
        for x in range(x0 - radius, x0 + radius + 1)
        for y in range(y0 - radius, y0 + radius + 1)
        if 0 <= x < w and 0 <= y < h
    )


def test_debug_is_off_by_default_and_toggles(world):
    renderer = Renderer(world)
    assert not renderer.debug
    renderer.toggle_debug()
    assert renderer.debug
    renderer.toggle_debug()
    assert not renderer.debug


def test_states_have_distinct_colours():
    assert set(STATE_COLOURS) == set(ZombieState)
    assert len(set(STATE_COLOURS.values())) == len(ZombieState)


def test_path_drawn_only_in_debug(world):
    z = chaser(world)
    waypoint = world.tilemap.tile_to_world(z.path[len(z.path) // 2])
    chase = STATE_COLOURS[ZombieState.CHASE]
    assert colour_near(render(world, True), waypoint, chase)
    assert not colour_near(render(world, False), waypoint, chase)


def test_direct_zombie_shows_straight_line_to_player(world):
    z = chaser(world, PathStrategy.DIRECT)
    assert z.path == []
    mid = ((z.x + world.player.x) / 2, (z.y + world.player.y) / 2)
    assert colour_near(render(world, True), mid, STATE_COLOURS[ZombieState.CHASE])


def test_vision_radius_circle(world):
    z = chaser(world, vision=0.0)  # 128 px
    surface = render(world, True)
    below = (z.x, z.y + z.vision_radius)  # straight down: away from the route to the player
    assert colour_near(surface, below, STATE_COLOURS[ZombieState.CHASE])
    assert not colour_near(render(world, False), below, STATE_COLOURS[ZombieState.CHASE])


def test_search_marks_last_known_position(world):
    z = chaser(world)
    z.state = ZombieState.SEARCH
    z.last_known_pos = world.tilemap.tile_to_world((8, 8))
    z.path = []
    assert colour_near(render(world, True), z.last_known_pos, STATE_COLOURS[ZombieState.SEARCH])


def test_state_label_drawn_under_zombie(world):
    z = chaser(world)
    surface = render(world, True)
    label_area = pygame.Rect(round(z.x) - 30, round(z.y + z.radius + 2), 60, 14)
    chase = STATE_COLOURS[ZombieState.CHASE]
    hits = sum(
        surface.get_at((x, y))[:3] == chase
        for x in range(label_area.left, label_area.right)
        for y in range(label_area.top, label_area.bottom)
    )
    assert hits > 10  # text pixels, not just a line crossing the area


@pytest.mark.parametrize("state", list(ZombieState))
def test_every_state_draws_without_error(world, state):
    z = chaser(world)
    z.state = state
    if state is ZombieState.WANDER:
        z.wander_target = (5, 5)
    render(world, True)


def test_overlay_does_not_change_world(world):
    z = chaser(world)
    before = (z.pos, list(z.path), z.state, z.repaths, world.tick, world.rng.bit_generator.state)
    render(world, True)
    assert (z.pos, list(z.path), z.state, z.repaths, world.tick, world.rng.bit_generator.state) == before


def test_main_runs_with_debug_flag():
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    result = subprocess.run(
        [sys.executable, "main.py", "--max-frames", "60", "--debug"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
