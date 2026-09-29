"""GameSession (P4.7): the pygame-free game flow that ties waves and evolution together.

    MENU --start()--> WAVE --(wave over)--> REPORT --next_wave()--> WAVE ...
                        |
                        +-- player died ------------------------> GAME_OVER
                        +-- last wave survived -----------------> VICTORY
    GAME_OVER / VICTORY --start()--> WAVE 1 again (new run)

One wave = a WaveManager run on a fresh World: player at full HP at the start
tile, no barricades, the whole current population spawned. When it ends the
session scores every zombie with compute_fitness, files a WaveReport and (if
the player survived and waves remain) asks the optimizer for the next
population from those genomes and fitnesses. Wave 1 is the optimizer's
initial (random) population.

Player HP and barricades deliberately do not carry over between waves: every
generation then faces the same starting conditions, so fitness differences
come from the zombies, not from a weakened or fortified player.

Randomness: one SeedSequence(seed) is split in two, so the world's dice
(zombie wandering, spawn assignment) and the optimizer's dice (selection,
mutation) are independent streams; a run is reproducible from its seed and
the player's inputs.

The caller drives it: main.py (or a test) calls step(action) once per fixed
tick while phase is WAVE. Never imports pygame.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ai.fitness import compute_fitness
from ai.genetic import GeneticAlgorithm, Optimizer
from ai.genome import Genome, PathStrategy
from config import NUM_WAVES, WAVE_TIME_LIMIT
from game.engine import World
from game.map import TileMap
from game.player import Action
from game.wave_manager import WaveEnd, WaveManager

OptimizerFactory = Callable[[np.random.Generator], Optimizer]


class Phase(Enum):
    MENU = "menu"
    WAVE = "wave"
    REPORT = "report"
    VICTORY = "victory"
    GAME_OVER = "game_over"


@dataclass(frozen=True)
class PopulationSummary:
    """Average decoded traits of a population: what the evolution is producing."""

    size: int
    mean_speed: float  # px/s
    mean_hp: float
    mean_vision: float  # px
    mean_swarm: float  # swarm weight
    strategy_counts: Dict[PathStrategy, int]


@dataclass(frozen=True)
class WaveReport:
    """Result of one finished wave."""

    wave: int  # 1-based
    end_reason: WaveEnd
    time: float  # seconds the wave lasted
    zombies_total: int
    zombies_killed: int
    player_hp: float
    player_max_hp: float
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    mean_damage: float  # average damage a zombie dealt to the player
    fought: PopulationSummary  # the population that played this wave
    evolved: Optional[PopulationSummary]  # its offspring, i.e. next wave's population (None if the run ended)

    @property
    def survivors(self) -> int:
        return self.zombies_total - self.zombies_killed


def summarize(genomes: Sequence[Genome]) -> PopulationSummary:
    if len(genomes) == 0:
        raise ValueError("cannot summarize an empty population")
    stats = [g.decode() for g in genomes]
    counts = {s: 0 for s in PathStrategy}
    for st in stats:
        counts[st.path_strategy] += 1
    n = len(stats)
    return PopulationSummary(
        size=n,
        mean_speed=sum(s.speed for s in stats) / n,
        mean_hp=sum(s.max_hp for s in stats) / n,
        mean_vision=sum(s.vision_radius for s in stats) / n,
        mean_swarm=sum(s.swarm_weight for s in stats) / n,
        strategy_counts=counts,
    )


def _default_optimizer(rng: np.random.Generator) -> Optimizer:
    return GeneticAlgorithm(rng)


class GameSession:
    def __init__(
        self,
        tilemap: TileMap,
        seed: int = 0,
        num_waves: int = NUM_WAVES,
        wave_time_limit: float = WAVE_TIME_LIMIT,
        optimizer_factory: OptimizerFactory = _default_optimizer,
    ):
        if num_waves < 1:
            raise ValueError(f"num_waves must be >= 1, got {num_waves}")
        self.tilemap = tilemap
        self.num_waves = num_waves
        self.wave_time_limit = wave_time_limit
        self._optimizer_factory = optimizer_factory
        world_seed, optimizer_seed = np.random.SeedSequence(seed).spawn(2)
        self._world_rng = np.random.default_rng(world_seed)
        self._optimizer_rng = np.random.default_rng(optimizer_seed)

        self.phase = Phase.MENU
        self.wave = 0  # 1-based once started
        self.history: List[WaveReport] = []
        self.genomes: List[Genome] = []  # population for the current / upcoming wave
        self.optimizer: Optimizer = optimizer_factory(self._optimizer_rng)
        self.world = self._new_world()  # idle world so the menu has something to show
        self.wave_manager: Optional[WaveManager] = None

    # --- flow ---

    def start(self) -> None:
        """MENU / VICTORY / GAME_OVER -> wave 1 of a fresh run."""
        if self.phase not in (Phase.MENU, Phase.VICTORY, Phase.GAME_OVER):
            raise RuntimeError(f"cannot start from phase {self.phase.name}")
        self.history = []
        self.optimizer = self._optimizer_factory(self._optimizer_rng)
        self.genomes = list(self.optimizer.initial_population())
        self.wave = 0
        self._begin_wave()

    def next_wave(self) -> None:
        """REPORT -> the next wave, played by the evolved population."""
        if self.phase is not Phase.REPORT:
            raise RuntimeError(f"next_wave() needs phase REPORT, not {self.phase.name}")
        self._begin_wave()

    def step(self, action: Action) -> None:
        """Advance one fixed tick of the current wave; finishes the wave when it ends."""
        if self.phase is not Phase.WAVE:
            raise RuntimeError(f"step() needs phase WAVE, not {self.phase.name}")
        assert self.wave_manager is not None
        self.world.step(action)
        self.wave_manager.update(self.world)
        if self.wave_manager.is_over():
            self._finish_wave()

    # --- read-only info for HUD / screens ---

    @property
    def zombies_total(self) -> int:
        return len(self.genomes) if self.wave_manager is not None else 0

    @property
    def zombies_remaining(self) -> int:
        return self.wave_manager.alive_count if self.wave_manager is not None else 0

    @property
    def time_left(self) -> float:
        return self.wave_manager.time_left if self.wave_manager is not None else self.wave_time_limit

    @property
    def last_report(self) -> Optional[WaveReport]:
        return self.history[-1] if self.history else None

    @property
    def population_summary(self) -> Optional[PopulationSummary]:
        return summarize(self.genomes) if self.genomes else None

    # --- internals ---

    def _new_world(self) -> World:
        return World(self.tilemap, self._world_rng)

    def _begin_wave(self) -> None:
        self.wave += 1
        self.world = self._new_world()
        self.wave_manager = WaveManager(self.world, self.wave_time_limit)
        self.wave_manager.start_wave(self.genomes)
        self.phase = Phase.WAVE

    def _finish_wave(self) -> None:
        wm = self.wave_manager
        assert wm is not None and wm.end_reason is not None
        results = wm.results()
        genomes = [g for g, _ in results]
        fitnesses = [compute_fitness(s) for _, s in results]
        player_died = wm.end_reason is WaveEnd.PLAYER_DEAD
        last_wave = self.wave >= self.num_waves

        evolved: Optional[PopulationSummary] = None
        if not player_died and not last_wave:
            self.genomes = list(self.optimizer.next_generation(genomes, fitnesses))
            evolved = summarize(self.genomes)

        p = self.world.player
        self.history.append(
            WaveReport(
                wave=self.wave,
                end_reason=wm.end_reason,
                time=wm.time_elapsed,
                zombies_total=len(results),
                zombies_killed=sum(1 for z in wm.wave_zombies if not z.alive),
                player_hp=p.hp,
                player_max_hp=p.max_hp,
                best_fitness=max(fitnesses),
                mean_fitness=sum(fitnesses) / len(fitnesses),
                worst_fitness=min(fitnesses),
                mean_damage=sum(s.damage_dealt for _, s in results) / len(results),
                fought=summarize(genomes),
                evolved=evolved,
            )
        )
        if player_died:
            self.phase = Phase.GAME_OVER
        elif last_wave:
            self.phase = Phase.VICTORY
        else:
            self.phase = Phase.REPORT
