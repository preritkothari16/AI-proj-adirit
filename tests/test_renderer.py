"""P2.3 — renderer and input mapping, run without a real window (SDL dummy driver)."""
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

from game.engine import World
from game.map import TileMap
from game.player import Action, Controller
from game.renderer import BARRICADE_COLOUR, BULLET_COLOUR, ZOMBIE_COLOUR, PLAYER_COLOUR, WALL_COLOUR, Renderer
from main import HumanController, keys_to_move

ROOT = Path(__file__).resolve().parent.parent
LEVEL1_PATH = ROOT / "maps" / "level1.txt"


@pytest.fixture
def world():
    return World(TileMap.load(LEVEL1_PATH), np.random.default_rng(0))


def test_keys_to_move():
    assert keys_to_move(False, False, False, False) == (0.0, 0.0)
    assert keys_to_move(True, False, False, False) == (0.0, -1.0)  # W = up = -y
    assert keys_to_move(False, True, False, False) == (0.0, 1.0)
    assert keys_to_move(False, False, True, False) == (-1.0, 0.0)
    assert keys_to_move(False, False, False, True) == (1.0, 0.0)
    assert keys_to_move(True, False, False, True) == (1.0, -1.0)
    assert keys_to_move(True, True, True, True) == (0.0, 0.0)  # opposites cancel


def test_human_controller_satisfies_protocol():
    assert isinstance(HumanController(), Controller)


def test_screen_size_matches_map(world):
    assert Renderer(world).screen_size == (32 * 32, 20 * 32)


def test_draw_paints_walls_and_player(world):
    pygame.init()
    try:
        renderer = Renderer(world)
        surface = pygame.Surface(renderer.screen_size)
        renderer.draw(surface, world)
        assert surface.get_at((16, 16))[:3] == WALL_COLOUR  # tile (0,0) is border wall
        px, py = round(world.player.x), round(world.player.y)
        assert surface.get_at((px, py))[:3] == PLAYER_COLOUR
    finally:
        pygame.quit()


def test_main_loop_runs_for_a_few_frames():
    """Launch the real main() headlessly and let it exit on its own."""
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    result = subprocess.run(
        [sys.executable, "main.py", "--max-frames", "30"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_step_with_human_style_action_moves_player(world):
    x0 = world.player.x
    world.step(Action(move=keys_to_move(False, False, False, True)))
    assert world.player.x > x0


def test_draw_paints_bullets(world):
    pygame.init()
    try:
        renderer = Renderer(world)
        px, py = world.player.pos
        world.step(Action(shoot=True, aim=(px + 200.0, py)))
        surface = pygame.Surface(renderer.screen_size)
        renderer.draw(surface, world)
        b = world.bullets[0]
        assert surface.get_at((round(b.x), round(b.y)))[:3] == BULLET_COLOUR
    finally:
        pygame.quit()


def test_draw_paints_barricades(world):
    pygame.init()
    try:
        renderer = Renderer(world)  # background cached before the barricade exists
        world.place_barricade((5, 2))
        surface = pygame.Surface(renderer.screen_size)
        renderer.draw(surface, world)
        assert surface.get_at((5 * 32 + 16, 2 * 32 + 16))[:3] == BARRICADE_COLOUR
    finally:
        pygame.quit()


def test_draw_paints_zombies_and_game_over(world):
    pygame.init()
    try:
        renderer = Renderer(world)
        z = world.spawn_zombie((5, 2))
        surface = pygame.Surface(renderer.screen_size)
        renderer.draw(surface, world)
        assert surface.get_at((round(z.x), round(z.y)))[:3] == ZOMBIE_COLOUR
        world.player.take_damage(world.player.max_hp)
        renderer.draw(surface, world)  # game-over overlay must not crash
        assert surface.get_at((round(z.x), round(z.y)))[:3] != ZOMBIE_COLOUR  # dimmed
    finally:
        pygame.quit()
