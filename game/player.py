"""Player input contracts (P1.2) and the Player entity (P2.2).

Action is the single interface between any controller (human keyboard/mouse,
BotPlayer) and World.step(). Player (P2.2) is the controllable entity:
position, HP and wall collision, with no keyboard/pygame knowledge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple, runtime_checkable

from config import FIXED_DT, PLAYER_MAX_HP, PLAYER_RADIUS, PLAYER_SPEED

Tile = Tuple[int, int]
_EPS = 1e-6


@dataclass(frozen=True)
class Action:
    """One tick's worth of input, produced by any Controller."""

    move: Tuple[float, float] = (0.0, 0.0)
    aim: Tuple[float, float] = (0.0, 0.0)
    shoot: bool = False
    barricade: Optional[Tile] = None


@runtime_checkable
class Controller(Protocol):
    """Anything that can drive the player: HumanController (P2.3), BotPlayer (P5.1)."""

    def get_action(self, world) -> Action: ...


class Player:
    """The player entity: position in world pixels, HP, and wall-aware movement.

    Knows nothing about keyboards or pygame — it only consumes a move vector.
    Collision uses an axis-aligned square of half-width `radius`, resolved one
    axis at a time, which gives sliding along walls for free.
    """

    def __init__(
        self,
        pos: Tuple[float, float],
        tile_size: int,
        speed: float = PLAYER_SPEED,
        max_hp: float = PLAYER_MAX_HP,
        radius: float = PLAYER_RADIUS,
    ):
        if speed * FIXED_DT >= tile_size:
            raise ValueError("speed too high: player could tunnel through a tile in one tick")
        self.x, self.y = float(pos[0]), float(pos[1])
        self.tile_size = tile_size
        self.speed = speed
        self.max_hp = max_hp
        self.hp = max_hp
        self.radius = radius

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def move(self, direction: Tuple[float, float], dt: float, is_blocked: Callable[[Tile], bool]) -> None:
        """Move along `direction` for `dt` seconds. Diagonals are normalised so
        they aren't faster than straight movement."""
        dx, dy = direction
        length = math.hypot(dx, dy)
        if length == 0.0:
            return
        if length > 1.0:
            dx, dy = dx / length, dy / length

        step = self.speed * dt
        self._move_x(dx * step, is_blocked)
        self._move_y(dy * step, is_blocked)

    def _overlapped_tiles(self, x: float, y: float) -> list[Tile]:
        ts, r = self.tile_size, self.radius
        # subtract a hair from the max edge so touching a tile boundary doesn't count as overlap
        x0, x1 = int(math.floor((x - r) / ts)), int(math.floor((x + r - _EPS) / ts))
        y0, y1 = int(math.floor((y - r) / ts)), int(math.floor((y + r - _EPS) / ts))
        return [(tx, ty) for tx in range(x0, x1 + 1) for ty in range(y0, y1 + 1)]

    def _move_x(self, amount: float, is_blocked: Callable[[Tile], bool]) -> None:
        if amount == 0.0:
            return
        new_x = self.x + amount
        blocked = [t for t in self._overlapped_tiles(new_x, self.y) if is_blocked(t)]
        if blocked:
            ts = self.tile_size
            if amount > 0:
                new_x = min(tx for tx, _ in blocked) * ts - self.radius
            else:
                new_x = (max(tx for tx, _ in blocked) + 1) * ts + self.radius
        self.x = new_x

    def _move_y(self, amount: float, is_blocked: Callable[[Tile], bool]) -> None:
        if amount == 0.0:
            return
        new_y = self.y + amount
        blocked = [t for t in self._overlapped_tiles(self.x, new_y) if is_blocked(t)]
        if blocked:
            ts = self.tile_size
            if amount > 0:
                new_y = min(ty for _, ty in blocked) * ts - self.radius
            else:
                new_y = (max(ty for _, ty in blocked) + 1) * ts + self.radius
        self.y = new_y
