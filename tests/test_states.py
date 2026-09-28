"""P3.1 — zombie FSM transitions, table-driven.

Every row: (current state, inputs, expected next state). Inputs are built with
make() so each row only states what matters for it.
"""
from __future__ import annotations

import subprocess
import sys
from itertools import product

import pytest

from game.states import FSMInputs, ZombieState, next_state

W, C, S, A = ZombieState.WANDER, ZombieState.CHASE, ZombieState.SEARCH, ZombieState.ATTACK

RANGE = 4.0
GIVE_UP = 3.0
FAR = 100.0  # well outside attack range


def make(sees=False, dist=FAR, since=0.0, give_up=GIVE_UP, attack_range=RANGE) -> FSMInputs:
    return FSMInputs(
        sees_player=sees,
        dist_to_player=dist,
        time_since_seen=since,
        give_up_time=give_up,
        attack_range=attack_range,
    )


TRANSITIONS = [
    # --- required transitions ---
    pytest.param(W, make(sees=True), C, id="wander->chase on sight"),
    pytest.param(C, make(sees=True, dist=RANGE - 1), A, id="chase->attack in range"),
    pytest.param(C, make(sees=True, dist=RANGE), A, id="chase->attack at exact range"),
    pytest.param(C, make(sees=False, since=0.1), S, id="chase->search on sight lost"),
    pytest.param(S, make(sees=True, since=0.0), C, id="search->chase when found"),
    pytest.param(S, make(sees=False, since=GIVE_UP), W, id="search->wander at give-up time"),
    pytest.param(S, make(sees=False, since=GIVE_UP + 5), W, id="search->wander after give-up"),
    # --- leaving ATTACK ---
    pytest.param(A, make(sees=True, dist=RANGE + 1), C, id="attack->chase out of range, visible"),
    pytest.param(A, make(sees=False, dist=RANGE + 1), S, id="attack->search out of range, hidden"),
]

NON_TRANSITIONS = [
    pytest.param(W, make(sees=False), W, id="wander stays when player unseen"),
    pytest.param(W, make(sees=False, dist=0.0), W, id="wander ignores range without sight"),
    pytest.param(W, make(sees=False, since=GIVE_UP * 10), W, id="wander ignores give-up timer"),
    pytest.param(C, make(sees=True, dist=RANGE + 0.01), C, id="chase stays just out of range"),
    pytest.param(C, make(sees=True, dist=FAR), C, id="chase stays while visible"),
    pytest.param(S, make(sees=False, since=GIVE_UP - 0.01), S, id="search stays before give-up"),
    pytest.param(S, make(sees=False, since=0.0), S, id="search stays right after losing sight"),
    pytest.param(S, make(sees=False, dist=0.0, since=1.0), S, id="search does not attack blind"),
    pytest.param(A, make(sees=True, dist=RANGE), A, id="attack stays in range"),
    pytest.param(A, make(sees=False, dist=0.0), A, id="attack stays in range even unseen"),
]

# --- edge / priority cases ---
PRIORITY = [
    pytest.param(C, make(sees=False, dist=0.0), A, id="chase: range beats lost sight"),
    pytest.param(S, make(sees=True, since=GIVE_UP * 2), C, id="search: sight beats give-up"),
    pytest.param(W, make(sees=True, dist=0.0), C, id="wander goes via chase, never straight to attack"),
    pytest.param(S, make(sees=False, since=0.0, give_up=0.0), W, id="zero give-up time wanders immediately"),
]


@pytest.mark.parametrize("state, inputs, expected", TRANSITIONS)
def test_transitions(state, inputs, expected):
    assert next_state(state, inputs) is expected


@pytest.mark.parametrize("state, inputs, expected", NON_TRANSITIONS)
def test_non_transitions(state, inputs, expected):
    assert next_state(state, inputs) is expected


@pytest.mark.parametrize("state, inputs, expected", PRIORITY)
def test_priority_and_edges(state, inputs, expected):
    assert next_state(state, inputs) is expected


# --- properties over the whole input grid ---

GRID = [
    make(sees=sees, dist=dist, since=since)
    for sees, dist, since in product([False, True], [0.0, RANGE, RANGE + 1, FAR], [0.0, GIVE_UP - 1, GIVE_UP, GIVE_UP + 1])
]


@pytest.mark.parametrize("state", list(ZombieState))
def test_total_and_only_legal_edges(state):
    """Every state/input combination returns a state, and only along allowed edges."""
    allowed = {
        W: {W, C},
        C: {C, A, S},
        S: {S, C, W},
        A: {A, C, S},
    }
    for inputs in GRID:
        assert next_state(state, inputs) in allowed[state], inputs


def test_pure_and_deterministic():
    inputs = make(sees=True, dist=RANGE + 1)
    before = inputs
    assert all(next_state(C, inputs) is C for _ in range(10))
    assert inputs == before  # frozen dataclass, untouched


def test_unknown_state_rejected():
    with pytest.raises(ValueError):
        next_state("CHASE", make())  # type: ignore[arg-type]


def test_does_not_import_pygame():
    code = "import sys, game.states; sys.exit('pygame' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
