"""P1.2 — import/interface tests for the frozen data contracts.

These check shapes and types only. Behaviour is tested in each phase's own
test file; compute_fitness was implemented in P4.5.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ai.fitness import FitnessStats, compute_fitness
from ai.genome import GENE_NAMES, NUM_GENES, Genome, PathStrategy, ZombieStats
from game.player import Action, Controller
from game.states import FSMInputs, ZombieState


# --- ai.genome ---


def test_path_strategy_members():
    assert {m.value for m in PathStrategy} == {"direct", "greedy", "astar"}


def test_gene_names_length():
    assert len(GENE_NAMES) == NUM_GENES == 6


def test_zombie_stats_fields():
    field_names = {f.name for f in dataclasses.fields(ZombieStats)}
    assert field_names == {
        "speed",
        "max_hp",
        "vision_radius",
        "give_up_time",
        "path_strategy",
        "swarm_weight",
    }


def test_genome_accepts_correct_shape():
    g = Genome(np.zeros(NUM_GENES))
    assert g.genes.shape == (NUM_GENES,)
    assert isinstance(g.genes, np.ndarray)


def test_genome_rejects_wrong_shape():
    with pytest.raises(ValueError):
        Genome(np.zeros(NUM_GENES - 1))


def test_genome_methods_return_genomes():
    rng = np.random.default_rng(0)
    assert isinstance(Genome.random(rng), Genome)
    assert isinstance(Genome(np.zeros(NUM_GENES)).repair(), Genome)


# --- ai.fitness ---


def test_fitness_stats_defaults():
    stats = FitnessStats()
    assert stats.damage_dealt == 0.0
    assert stats.time_alive == 0.0
    assert stats.min_dist_to_player == float("inf")
    assert stats.reached_player is False


def test_compute_fitness_returns_float():
    assert isinstance(compute_fitness(FitnessStats()), float)


# --- game.states ---


def test_zombie_state_members():
    assert {s.name for s in ZombieState} == {"WANDER", "CHASE", "SEARCH", "ATTACK"}


def test_fsm_inputs_construction():
    inputs = FSMInputs(
        sees_player=True,
        dist_to_player=5.0,
        time_since_seen=0.0,
        give_up_time=3.0,
        attack_range=1.0,
    )
    assert inputs.sees_player is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        inputs.sees_player = False  # frozen


# --- game.player ---


def test_action_defaults_and_frozen():
    a = Action()
    assert a.move == (0.0, 0.0)
    assert a.aim == (0.0, 0.0)
    assert a.shoot is False
    assert a.barricade is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.shoot = True


def test_controller_protocol_is_structural():
    class DummyController:
        def get_action(self, world) -> Action:
            return Action()

    assert isinstance(DummyController(), Controller)

    class NotAController:
        pass

    assert not isinstance(NotAController(), Controller)
