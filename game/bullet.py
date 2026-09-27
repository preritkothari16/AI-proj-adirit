"""Bullets: physical projectiles that travel each tick (P2.4). Not hitscan.

A bullet is a point moving in a straight line. It dies when it enters a
blocked tile (wall or barricade) or leaves the map. Movement is sub-stepped
so a fast bullet can't skip over a tile or clip through a wall corner
between two samples. Zombie hits are added in P2.6.
"""
from __future__ import annotations

import math
from typing import Callable, Tuple

from config import BULLET_DAMAGE, BULLET_RADIUS, BULLET_SPEED

Tile = Tuple[int, int]


class Bullet:
    def __init__(
        self,
        pos: Tuple[float, float],
        direction: Tuple[float, float],
        speed: float = BULLET_SPEED,
        damage: float = BULLET_DAMAGE,
        radius: float = BULLET_RADIUS,
    ):
        dx, dy = direction
        length = math.hypot(dx, dy)
        if length == 0.0:
            raise ValueError("bullet direction must be non-zero")
        self.x, self.y = float(pos[0]), float(pos[1])
        self.vx = dx / length * speed
        self.vy = dy / length * speed
        self.damage = damage
        self.radius = radius
        self.alive = True

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def update(
        self,
        dt: float,
        is_blocked: Callable[[Tile], bool],
        tile_size: int,
        bounds: Tuple[float, float],
    ) -> None:
        """Move one tick. `bounds` is the map size in pixels (width, height)."""
        if not self.alive:
            return
        distance = math.hypot(self.vx, self.vy) * dt
        # sample at most every quarter tile so walls can't be skipped
        substeps = max(1, math.ceil(distance / (tile_size / 4)))
        sx, sy = self.vx * dt / substeps, self.vy * dt / substeps
        width, height = bounds
        for _ in range(substeps):
            self.x += sx
            self.y += sy
            if not (0.0 <= self.x < width and 0.0 <= self.y < height):
                self.alive = False
                return
            tile = (int(self.x // tile_size), int(self.y // tile_size))
            if is_blocked(tile):
                self.alive = False
                return
