"""Frozen data contract for the zombie FSM (P1.2).

next_state() is a pure function — no World/Zombie access — so it can be
unit-tested exhaustively. Transitions (P3.1):

    WANDER -> CHASE   sees the player
    CHASE  -> ATTACK  within attack range
    CHASE  -> SEARCH  loses sight (and not in range)
    SEARCH -> CHASE   sees the player again
    SEARCH -> WANDER  time_since_seen >= give_up_time
    ATTACK -> CHASE   player steps out of range but is still visible
    ATTACK -> SEARCH  player out of range and out of sight

Anything else keeps the current state. "In range" means
dist_to_player <= attack_range; the caller picks the units (P3.x passes
the gap between bodies in px, matching Zombie.in_contact).

Note: this is the zombie behaviour FSM, not the game's screen states
(menu/playing/game-over), which live in main.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class ZombieState(Enum):
    WANDER = auto()
    CHASE = auto()
    SEARCH = auto()
    ATTACK = auto()


@dataclass(frozen=True)
class FSMInputs:
    """Everything next_state() needs to decide a transition."""

    sees_player: bool
    dist_to_player: float
    time_since_seen: float
    give_up_time: float
    attack_range: float


def next_state(state: ZombieState, inputs: FSMInputs) -> ZombieState:
    """Pure FSM transition function: current state + this tick's inputs -> next state."""
    in_range = inputs.dist_to_player <= inputs.attack_range

    if state is ZombieState.WANDER:
        return ZombieState.CHASE if inputs.sees_player else ZombieState.WANDER

    if state is ZombieState.CHASE:
        if in_range:
            return ZombieState.ATTACK
        return ZombieState.CHASE if inputs.sees_player else ZombieState.SEARCH

    if state is ZombieState.ATTACK:
        if in_range:
            return ZombieState.ATTACK
        return ZombieState.CHASE if inputs.sees_player else ZombieState.SEARCH

    if state is ZombieState.SEARCH:
        if inputs.sees_player:
            return ZombieState.CHASE
        if inputs.time_since_seen >= inputs.give_up_time:
            return ZombieState.WANDER
        return ZombieState.SEARCH

    raise ValueError(f"unknown zombie state: {state!r}")
