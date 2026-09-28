"""World: the pygame-free simulation core (P2.1).

World owns all simulation state and advances it one fixed tick at a time via
step(action). Rendering, input and the app loop live in main.py and
game/renderer.py — this module must never import pygame, so headless runs,
the BotPlayer and tests work without a window.

Each zombie's stats, movement strategy and swarm weight come from its genome (P3.4).
Once the player's HP hits 0 the world is over and step() stops advancing.
"""
from __future__ import annotations

import math
from typing import List, Optional, Set, Tuple

import numpy as np

from ai.genome import Genome
from config import FIXED_DT
from game.barricade import PlacementResult, check_placement
from game.bullet import Bullet
from game.map import Tile, TileMap
from game.player import Action, Player
from game.zombie import Zombie


class World:
    def __init__(self, tilemap: TileMap, rng: np.random.Generator):
        if not isinstance(rng, np.random.Generator):
            raise TypeError("World needs a seeded np.random.Generator (np.random.default_rng(seed))")
        self.tilemap = tilemap
        self.rng = rng

        self.player = Player(tilemap.tile_to_world(tilemap.player_start), tilemap.tile_size)
        self.zombies: List[Zombie] = []
        self._next_zid = 0
        self.bullets: List[Bullet] = []
        self.barricades: Set[Tile] = set()

        self.tick = 0
        self.last_action = Action()
        self.last_placement: PlacementResult | None = None  # result of the latest barricade attempt

    @property
    def game_over(self) -> bool:
        return not self.player.alive

    @property
    def time(self) -> float:
        """Simulation time in seconds. Derived from the tick count so it never drifts."""
        return self.tick * FIXED_DT

    def is_blocked(self, tile: Tile) -> bool:
        """True for walls, out-of-bounds tiles and barricades. Used for collision."""
        return not self.tilemap.is_walkable(tile) or tile in self.barricades

    def neighbors(self, tile: Tile) -> List[Tile]:
        """4-directional unblocked neighbours. Makes World a GraphLike, so
        pathfinding over the World automatically treats barricades as walls."""
        x, y = tile
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [t for t in candidates if not self.is_blocked(t)]

    def has_line_of_sight(self, a: Tile, b: Tile) -> bool:
        """Tile LOS where walls, out-of-bounds tiles and barricades all block sight."""
        return self.tilemap.has_line_of_sight(a, b, self.is_blocked)

    def place_barricade(self, tile: Tile) -> PlacementResult:
        result = check_placement(self, tile)
        if result is PlacementResult.OK:
            self.barricades.add(tile)
        self.last_placement = result
        return result

    def spawn_zombie(self, tile: Tile, genome: Genome) -> Zombie:
        """Add a zombie built from `genome` at the centre of `tile`. WaveManager (P4.6) calls this."""
        zombie = Zombie(self._next_zid, genome, self.tilemap.tile_to_world(tile), self.tilemap.tile_size)
        self._next_zid += 1
        self.zombies.append(zombie)
        return zombie

    def step(self, action: Action) -> None:
        """Advance the simulation by exactly one FIXED_DT tick. No-op once the game is over."""
        if not isinstance(action, Action):
            raise TypeError(f"step() expects an Action, got {type(action).__name__}")
        if self.game_over:
            return
        self.last_action = action
        if action.barricade is not None:
            self.place_barricade(action.barricade)
        self.player.move(action.move, FIXED_DT, self.is_blocked)
        self._handle_shooting(action)
        self._update_bullets()
        self._update_zombies()
        self.tick += 1

    def _handle_shooting(self, action: Action) -> None:
        p = self.player
        if p.cooldown_remaining > 0:
            p.cooldown_remaining -= 1
        if not action.shoot or p.cooldown_remaining > 0:
            return
        direction = (action.aim[0] - p.x, action.aim[1] - p.y)
        if direction == (0.0, 0.0):
            return  # aiming at your own centre: no direction to fire in
        self.bullets.append(Bullet(p.pos, direction))
        p.cooldown_remaining = p.fire_cooldown_ticks

    def _update_bullets(self) -> None:
        tm = self.tilemap
        bounds = (tm.width * tm.tile_size, tm.height * tm.tile_size)
        for bullet in self.bullets:
            start = bullet.pos
            bullet.update(FIXED_DT, self.is_blocked, tm.tile_size, bounds)
            target = self._first_zombie_hit(start, bullet.pos, bullet.radius)
            if target is not None:
                target.take_damage(bullet.damage)
                bullet.alive = False
        self.bullets = [b for b in self.bullets if b.alive]
        self.zombies = [z for z in self.zombies if z.alive]

    def _first_zombie_hit(
        self, a: Tuple[float, float], b: Tuple[float, float], bullet_radius: float
    ) -> Optional[Zombie]:
        """Zombie whose circle the segment a->b enters first, if any. Testing the
        whole segment (not just the end point) means a bullet can't skip a zombie."""
        best, best_t = None, math.inf
        for z in self.zombies:
            t = _segment_circle_entry(a, b, z.pos, z.radius + bullet_radius)
            if t is not None and t < best_t:
                best, best_t = z, t
        return best

    def _update_zombies(self) -> None:
        for zombie in self.zombies:
            zombie.update(FIXED_DT, self)


def _segment_circle_entry(
    a: Tuple[float, float], b: Tuple[float, float], centre: Tuple[float, float], r: float
) -> Optional[float]:
    """Smallest t in [0,1] where a + t(b-a) is within r of centre, or None."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    fx, fy = a[0] - centre[0], a[1] - centre[1]
    c = fx * fx + fy * fy - r * r
    if c <= 0.0:
        return 0.0  # segment starts inside the circle
    qa = dx * dx + dy * dy
    if qa == 0.0:
        return None
    qb = 2.0 * (fx * dx + fy * dy)
    disc = qb * qb - 4.0 * qa * c
    if disc < 0.0:
        return None
    t = (-qb - math.sqrt(disc)) / (2.0 * qa)
    return t if 0.0 <= t <= 1.0 else None
