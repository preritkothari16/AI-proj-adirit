"""Frozen data contract for the zombie FSM (P1.2).

next_state() is a pure function — no World/Zombie access — so it can be
unit-tested exhaustively. Transition logic implemented in P3.1.

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
    """Pure FSM transition function. Implemented in P3.1."""
    raise NotImplementedError
