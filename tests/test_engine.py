"""P2.1 — World construction, fixed-tick stepping, headless operation."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from config import FIXED_DT
from game.engine import World
from game.map import TileMap
from game.player import Action

ROOT = Path(__file__).resolve().parent.parent
LEVEL1_PATH = ROOT / "maps" / "level1.txt"


@pytest.fixture
def world():
    return World(TileMap.load(LEVEL1_PATH), np.random.default_rng(0))


def test_initial_state(world):
    assert world.tick == 0
    assert world.time == 0.0
    assert world.player.pos == world.tilemap.tile_to_world(world.tilemap.player_start)
    assert world.zombies == []
    assert world.bullets == []
    assert world.barricades == set()
    assert world.tilemap.width == 32


def test_step_advances_exactly_one_tick(world):
    world.step(Action())
    assert world.tick == 1
    assert world.time == pytest.approx(FIXED_DT)


def test_hundreds_of_steps_run_headlessly(world):
    for _ in range(600):
        world.step(Action(move=(1.0, 0.0), shoot=True))
    assert world.tick == 600
    assert world.time == pytest.approx(10.0)


def test_time_does_not_drift(world):
    for _ in range(3600):
        world.step(Action())
    assert world.time == 3600 * FIXED_DT


def test_step_records_last_action(world):
    action = Action(move=(0.0, -1.0), aim=(5.0, 5.0), shoot=True)
    world.step(action)
    assert world.last_action == action


def test_step_rejects_non_action(world):
    with pytest.raises(TypeError):
        world.step((1.0, 0.0))


def test_rng_must_be_generator():
    with pytest.raises(TypeError):
        World(TileMap.load(LEVEL1_PATH), rng=42)


def test_engine_does_not_load_pygame():
    """Import World in a fresh interpreter and check pygame never got loaded."""
    code = (
        "import sys; import game.engine; "
        "sys.exit(1 if 'pygame' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT)
    assert result.returncode == 0, "game.engine pulled in pygame"
