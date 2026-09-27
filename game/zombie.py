"""Zombie entity (P2.6: temporary fixed-stat dummy).

For now every zombie walks straight at the player (no pathfinding, so walls
can stall it), takes bullet damage, dies at 0 HP and hits the player on
contact. Genome-decoded stats, the FSM and path strategies replace the fixed
behaviour in P3.x.

Attack rule: a zombie is "in contact" when the gap between its body and the
player's is at most ZOMBIE_ATTACK_REACH px (circle distance). While in contact
it stops moving and deals ZOMBIE_ATTACK_DAMAGE once every
ZOMBIE_ATTACK_COOLDOWN seconds, counted in whole ticks like the fire cooldown.
"""
from __future__ import annotations

import math
from typing import Callable, Tuple

from config import (
    FIXED_DT,
    ZOMBIE_ATTACK_COOLDOWN,
    ZOMBIE_ATTACK_DAMAGE,
    ZOMBIE_ATTACK_REACH,
    ZOMBIE_MAX_HP,
    ZOMBIE_RADIUS,
    ZOMBIE_SPEED,
)
from game.player import Body, Player

Tile = Tuple[int, int]


class Zombie(Body):
    def __init__(
        self,
        pos: Tuple[float, float],
        tile_size: int,
        speed: float = ZOMBIE_SPEED,
        max_hp: float = ZOMBIE_MAX_HP,
        radius: float = ZOMBIE_RADIUS,
        attack_damage: float = ZOMBIE_ATTACK_DAMAGE,
    ):
        super().__init__(pos, tile_size, speed, radius)
        self.max_hp = max_hp
        self.hp = max_hp
        self.attack_damage = attack_damage
        self.attack_cooldown_ticks = max(1, round(ZOMBIE_ATTACK_COOLDOWN / FIXED_DT))
        self.attack_cooldown_remaining = 0
        self.damage_dealt = 0.0  # feeds FitnessStats later (P4.5)

    @property
    def alive(self) -> bool:
        return self.hp > 0.0

    def take_damage(self, amount: float) -> None:
        self.hp = max(0.0, self.hp - amount)

    def in_contact(self, player: Player) -> bool:
        gap = math.hypot(player.x - self.x, player.y - self.y) - self.radius - player.radius
        return gap <= ZOMBIE_ATTACK_REACH

    def update(self, dt: float, player: Player, is_blocked: Callable[[Tile], bool]) -> None:
        """One tick: tick down the attack cooldown, then either hit the player or walk at them."""
        if not self.alive:
            return
        if self.attack_cooldown_remaining > 0:
            self.attack_cooldown_remaining -= 1
        if self.in_contact(player):
            if self.attack_cooldown_remaining == 0 and player.alive:
                player.take_damage(self.attack_damage)
                self.damage_dealt += self.attack_damage
                self.attack_cooldown_remaining = self.attack_cooldown_ticks
            return
        self.move((player.x - self.x, player.y - self.y), dt, is_blocked)
