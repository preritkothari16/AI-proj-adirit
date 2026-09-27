"""World: the pygame-free simulation core (P2.1).

World owns all simulation state and advances it one fixed tick at a time via
step(action). Rendering, input and the app loop live in main.py and
game/renderer.py — this module must never import pygame, so headless runs,
the BotPlayer and tests work without a window.

Entity behaviour is filled in by later phases:
    bullets     P2.4
    barricades  P2.5
    zombies     P2.6 / P3.x
"""
from __future__ import annotations

from typing import Any, List, Set

import numpy as np

from config import FIXED_DT
from game.map import Tile, TileMap
from game.player import Action, Player


class World:
    def __init__(self, tilemap: TileMap, rng: np.random.Generator):
        if not isinstance(rng, np.random.Generator):
            raise TypeError("World needs a seeded np.random.Generator (np.random.default_rng(seed))")
        self.tilemap = tilemap
        self.rng = rng

        self.player = Player(tilemap.tile_to_world(tilemap.player_start), tilemap.tile_size)
        self.zombies: List[Any] = []
        self.bullets: List[Any] = []
        self.barricades: Set[Tile] = set()

        self.tick = 0
        self.last_action = Action()

    @property
    def time(self) -> float:
        """Simulation time in seconds. Derived from the tick count so it never drifts."""
        return self.tick * FIXED_DT

    def is_blocked(self, tile: Tile) -> bool:
        """True for walls, out-of-bounds tiles and barricades. Used for collision."""
        return not self.tilemap.is_walkable(tile) or tile in self.barricades

    def step(self, action: Action) -> None:
        """Advance the simulation by exactly one FIXED_DT tick."""
        if not isinstance(action, Action):
            raise TypeError(f"step() expects an Action, got {type(action).__name__}")
        self.last_action = action
        self.player.move(action.move, FIXED_DT, self.is_blocked)
        # Bullet and zombie updates are added here in P2.4+.
        self.tick += 1
