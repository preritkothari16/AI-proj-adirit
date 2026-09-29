"""WaveManager (P4.6): runs one wave of zombies on a World and collects their results.

    wm = WaveManager(world)
    wm.start_wave(genomes)          # spawn one zombie per genome
    while not wm.is_over():
        world.step(action)          # caller advances the world ...
        wm.update(world)            # ... then asks the manager whether the wave ended
    wm.results()                    # [(genome, FitnessStats)] in the order of `genomes`

Spawning: one zombie per genome, spread evenly over the map's spawn points
(each point gets floor(n / points) or one more). Which genome lands on which
point is a random permutation from `world.rng`, so a genome's fitness isn't
biased by its position in the list (GA elites are always first).

The wave ends (first that applies, checked in this order):
    PLAYER_DEAD  the player's HP hit 0
    ALL_DEAD     no zombie is alive
    TIME_UP      `time_limit` seconds of simulation ticks have passed
The reason is latched: once over, later update() calls change nothing.

World drops dead zombies from `world.zombies`, so the manager keeps its own
reference to every zombie it spawned. results() therefore covers the whole
population, dead or alive, and returns snapshots (copies) of the stats.
"""
from __future__ import annotations

import dataclasses
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

from ai.fitness import FitnessStats
from ai.genome import Genome
from config import FIXED_DT, WAVE_TIME_LIMIT

if TYPE_CHECKING:
    from game.engine import World
    from game.zombie import Zombie


class WaveEnd(Enum):
    PLAYER_DEAD = "player_dead"
    ALL_DEAD = "all_dead"
    TIME_UP = "time_up"


class WaveManager:
    def __init__(self, world: "World", time_limit: float = WAVE_TIME_LIMIT):
        if time_limit <= 0:
            raise ValueError(f"time_limit must be > 0, got {time_limit}")
        self.world = world
        self.time_limit = time_limit
        self.limit_ticks = max(1, round(time_limit / FIXED_DT))
        self.wave_zombies: List["Zombie"] = []  # every zombie of the current wave, in genome order
        self.start_tick = 0
        self.end_reason: Optional[WaveEnd] = None
        self._started = False

    # --- wave lifecycle ---

    def start_wave(self, genomes: Sequence[Genome]) -> List["Zombie"]:
        """Clear the previous wave's zombies and bullets and spawn one zombie per genome."""
        if len(genomes) == 0:
            raise ValueError("start_wave needs at least one genome")
        if not all(isinstance(g, Genome) for g in genomes):
            raise TypeError("start_wave expects Genome objects")
        world = self.world
        if world.game_over:
            raise ValueError("cannot start a wave: the player is dead")

        world.zombies = []
        world.bullets = []
        spawns = world.tilemap.spawn_points
        # Even spread: slot i -> spawn i % len(spawns); the permutation decides which genome gets which slot.
        slots = world.rng.permutation(len(genomes))
        self.wave_zombies = [world.spawn_zombie(spawns[int(slot) % len(spawns)], g) for g, slot in zip(genomes, slots)]
        self.start_tick = world.tick
        self.end_reason = None
        self._started = True
        return list(self.wave_zombies)

    def update(self, world: Optional["World"] = None) -> None:
        """Check the end conditions after a world tick. Call once per world.step()."""
        self._require_started()
        if world is not None and world is not self.world:
            raise ValueError("update() got a different World than this manager was built with")
        if self.end_reason is not None:
            return
        if not self.world.player.alive:
            self.end_reason = WaveEnd.PLAYER_DEAD
        elif not any(z.alive for z in self.wave_zombies):
            self.end_reason = WaveEnd.ALL_DEAD
        elif self.ticks_elapsed >= self.limit_ticks:
            self.end_reason = WaveEnd.TIME_UP

    def is_over(self) -> bool:
        self._require_started()
        return self.end_reason is not None

    # --- info ---

    @property
    def ticks_elapsed(self) -> int:
        """Whole world ticks since start_wave (stops advancing when the player dies)."""
        return self.world.tick - self.start_tick

    @property
    def time_elapsed(self) -> float:
        return self.ticks_elapsed * FIXED_DT

    @property
    def time_left(self) -> float:
        return max(0.0, (self.limit_ticks - self.ticks_elapsed) * FIXED_DT)

    @property
    def alive_count(self) -> int:
        return sum(1 for z in self.wave_zombies if z.alive)

    def results(self) -> List[Tuple[Genome, FitnessStats]]:
        """(genome, stats snapshot) for every zombie of the wave, in the order the genomes were given."""
        self._require_started()
        return [(z.genome, dataclasses.replace(z.fitness_stats)) for z in self.wave_zombies]

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("no wave started: call start_wave(genomes) first")
