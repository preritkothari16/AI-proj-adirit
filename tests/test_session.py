"""P4.7 — GameSession flow (headless) plus its HUD/screens and key handling."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from ai.fitness import compute_fitness
from ai.genetic import GeneticAlgorithm, Optimizer
from ai.genome import Genome, PathStrategy
from config import NUM_WAVES, POPULATION_SIZE, WAVE_TIME_LIMIT
from game.map import TileMap
from game.player import Action
from game.renderer import Renderer, evolution_line, screen_text, status_line
from game.session import GameSession, Phase, PopulationSummary, WaveReport, summarize
from game.wave_manager import WaveEnd
from main import handle_keydown

ROOT = Path(__file__).resolve().parent.parent
LEVEL1 = ROOT / "maps" / "level1.txt"
IDLE = Action()


def make_session(seed: int = 0, num_waves: int = 3, limit: float = 2.0, **kw) -> GameSession:
    return GameSession(TileMap.load(LEVEL1), seed=seed, num_waves=num_waves, wave_time_limit=limit, **kw)


def play_wave(session: GameSession, invulnerable: bool = True, max_ticks: int = 20_000) -> None:
    """Idle the player until the wave ends. Invulnerable keeps HP full so waves end by time / kills."""
    n = 0
    while session.phase is Phase.WAVE and n < max_ticks:
        if invulnerable:
            session.world.player.hp = session.world.player.max_hp
        session.step(IDLE)
        n += 1
    assert session.phase is not Phase.WAVE


def kill_player(session: GameSession) -> None:
    session.world.player.take_damage(1e9)
    session.step(IDLE)


class SpyOptimizer:
    """Wraps a GA and records what next_generation received."""

    def __init__(self, rng):
        self.inner = GeneticAlgorithm(rng)
        self.calls: list[tuple[list[Genome], list[float], list[Genome]]] = []

    def initial_population(self):
        return self.inner.initial_population()

    def next_generation(self, genomes, fitnesses):
        out = self.inner.next_generation(genomes, fitnesses)
        self.calls.append((list(genomes), list(fitnesses), list(out)))
        return out


# --- flow: menu / start ---


def test_starts_in_menu_with_idle_world():
    s = make_session()
    assert s.phase is Phase.MENU
    assert s.wave == 0 and s.history == []
    assert s.world.zombies == []
    assert s.zombies_remaining == 0


def test_step_and_next_wave_rejected_outside_their_phase():
    s = make_session()
    with pytest.raises(RuntimeError):
        s.step(IDLE)
    with pytest.raises(RuntimeError):
        s.next_wave()


def test_start_begins_wave_one_with_full_population():
    s = make_session()
    s.start()
    assert s.phase is Phase.WAVE and s.wave == 1
    assert len(s.world.zombies) == POPULATION_SIZE == s.zombies_total == s.zombies_remaining
    assert s.time_left == pytest.approx(2.0)
    assert [z.genome for z in s.world.zombies] == s.genomes  # wave 1 = optimizer's initial population


def test_start_rejected_mid_wave_and_report():
    s = make_session()
    s.start()
    with pytest.raises(RuntimeError):
        s.start()
    play_wave(s)
    assert s.phase is Phase.REPORT
    with pytest.raises(RuntimeError):
        s.start()


def test_bad_num_waves():
    with pytest.raises(ValueError):
        make_session(num_waves=0)


def test_default_config_values():
    s = GameSession(TileMap.load(LEVEL1))
    assert s.num_waves == NUM_WAVES == 10 and s.wave_time_limit == WAVE_TIME_LIMIT


# --- flow: wave -> report -> next wave ---


def test_wave_end_gives_report_phase_and_report():
    s = make_session()
    s.start()
    play_wave(s)
    assert s.phase is Phase.REPORT
    r = s.last_report
    assert isinstance(r, WaveReport) and len(s.history) == 1
    assert r.wave == 1 and r.zombies_total == POPULATION_SIZE
    assert r.end_reason in (WaveEnd.TIME_UP, WaveEnd.ALL_DEAD)
    assert 0.0 <= r.worst_fitness <= r.mean_fitness <= r.best_fitness <= 1.0
    assert 0 <= r.zombies_killed <= r.zombies_total and r.survivors == r.zombies_total - r.zombies_killed
    assert r.time <= 2.0 + 1e-9
    assert isinstance(r.fought, PopulationSummary) and isinstance(r.evolved, PopulationSummary)


def test_time_up_report_for_idle_wave_with_blind_zombies():
    s = make_session(limit=1.0)
    s.start()
    for z in s.world.zombies:
        z.sees_player = lambda w: False
    play_wave(s)
    r = s.last_report
    assert r.end_reason is WaveEnd.TIME_UP
    assert r.time == pytest.approx(1.0) and r.zombies_killed == 0


def test_all_dead_ends_wave_early_and_counts_kills():
    s = make_session()
    s.start()
    for z in s.world.zombies:
        z.take_damage(1e9)
    s.step(IDLE)
    assert s.phase is Phase.REPORT
    r = s.last_report
    assert r.end_reason is WaveEnd.ALL_DEAD and r.zombies_killed == r.zombies_total
    assert r.mean_fitness >= 0.0


def test_next_wave_starts_fresh_world_with_evolved_population():
    s = make_session()
    s.start()
    first_genomes = list(s.genomes)
    old_world = s.world
    s.world.place_barricade((10, 5))
    play_wave(s)
    evolved = list(s.genomes)
    s.next_wave()
    assert s.phase is Phase.WAVE and s.wave == 2
    assert s.world is not old_world
    assert s.world.tick == 0 and s.world.barricades == set()  # fresh conditions every wave
    assert s.world.player.hp == s.world.player.max_hp
    assert len(s.world.zombies) == POPULATION_SIZE
    assert [z.genome for z in s.world.zombies] == evolved
    assert any(not np.array_equal(a.genes, b.genes) for a, b in zip(first_genomes, evolved))


def test_wave_manager_is_per_wave_and_timer_restarts():
    s = make_session()
    s.start()
    wm1 = s.wave_manager
    play_wave(s)
    s.next_wave()
    assert s.wave_manager is not wm1
    assert s.time_left == pytest.approx(2.0)
    assert s.zombies_remaining == POPULATION_SIZE


# --- evolution wiring ---


def test_fitnesses_passed_to_optimizer_are_compute_fitness_of_results():
    spy_holder = {}

    def factory(rng):
        spy_holder["spy"] = SpyOptimizer(rng)
        return spy_holder["spy"]

    s = make_session(optimizer_factory=factory)
    s.start()
    play_wave(s)
    spy = spy_holder["spy"]
    assert len(spy.calls) == 1
    genomes_in, fitness_in, out = spy.calls[0]
    wm = s.wave_manager
    expected = [compute_fitness(st) for _, st in wm.results()]
    assert fitness_in == expected
    assert [g for g, _ in wm.results()] == genomes_in
    assert len(genomes_in) == len(out) == POPULATION_SIZE
    assert s.last_report.best_fitness == max(expected)
    assert s.last_report.mean_fitness == pytest.approx(sum(expected) / len(expected))


def test_best_genome_survives_into_next_wave_via_elitism():
    holder = {}

    def factory(rng):
        holder["spy"] = SpyOptimizer(rng)
        return holder["spy"]

    s = make_session(optimizer_factory=factory)
    s.start()
    play_wave(s)
    genomes_in, fitness_in, _ = holder["spy"].calls[0]
    best = genomes_in[int(np.argmax(fitness_in))]
    s.next_wave()
    assert any(np.array_equal(z.genome.genes, best.genes) for z in s.world.zombies)


def test_optimizer_called_once_per_finished_wave_but_not_after_last():
    holder = {}

    def factory(rng):
        holder["spy"] = SpyOptimizer(rng)
        return holder["spy"]

    s = make_session(num_waves=3, optimizer_factory=factory)
    s.start()
    for _ in range(2):
        play_wave(s)
        s.next_wave()
    assert len(holder["spy"].calls) == 2
    play_wave(s)
    assert s.phase is Phase.VICTORY
    assert len(holder["spy"].calls) == 2  # nothing left to evolve for
    assert s.last_report.evolved is None


def test_report_summaries_describe_fought_and_evolved_populations():
    s = make_session()
    s.start()
    fought_expected = summarize(s.genomes)
    play_wave(s)
    r = s.last_report
    assert r.fought == fought_expected
    assert r.evolved == summarize(s.genomes)
    assert sum(r.fought.strategy_counts.values()) == POPULATION_SIZE


def test_summarize_values_and_validation():
    g = Genome.of(path_strategy=PathStrategy.ASTAR)  # all-0.5: speed 90, hp 50, vision 256
    sm = summarize([g, g])
    assert sm.size == 2 and sm.mean_speed == pytest.approx(90.0)
    assert sm.mean_hp == pytest.approx(50.0) and sm.mean_vision == pytest.approx(256.0)
    assert sm.strategy_counts[PathStrategy.ASTAR] == 2 and sm.strategy_counts[PathStrategy.DIRECT] == 0
    with pytest.raises(ValueError):
        summarize([])


def test_evolution_improves_fitness_over_waves_against_idle_player():
    """Zombies that find an idle player earn more, so the population should trend upward over many waves.
    Averaged over seeds and split into early/late waves to smooth out noise."""
    early, late = [], []
    for seed in range(3):
        s = make_session(seed=seed, num_waves=8, limit=8.0)
        s.start()
        while s.phase is not Phase.VICTORY:
            play_wave(s)
            if s.phase is Phase.REPORT:
                s.next_wave()
        means = [r.mean_fitness for r in s.history]
        early.append(np.mean(means[:3]))
        late.append(np.mean(means[-3:]))
    assert np.mean(late) > np.mean(early)


# --- flow: game over / victory / restart ---


def test_player_death_is_game_over_with_report_and_no_evolution():
    holder = {}

    def factory(rng):
        holder["spy"] = SpyOptimizer(rng)
        return holder["spy"]

    s = make_session(optimizer_factory=factory)
    s.start()
    kill_player(s)
    assert s.phase is Phase.GAME_OVER
    r = s.last_report
    assert r.end_reason is WaveEnd.PLAYER_DEAD and r.wave == 1 and r.evolved is None
    assert holder["spy"].calls == []
    with pytest.raises(RuntimeError):
        s.next_wave()
    with pytest.raises(RuntimeError):
        s.step(IDLE)


def test_player_killed_by_zombies_ends_run():
    s = make_session(limit=60.0)
    s.start()
    n = 0
    while s.phase is Phase.WAVE and n < 3600:
        s.step(IDLE)  # idle, not invulnerable: 20 wandering zombies eventually find and kill the player
        n += 1
    assert s.phase in (Phase.GAME_OVER, Phase.REPORT)  # wave may also just time out on some seeds
    if s.phase is Phase.GAME_OVER:
        assert s.last_report.player_hp == 0.0


def test_victory_after_surviving_every_wave():
    s = make_session(num_waves=3)
    s.start()
    while s.phase is not Phase.VICTORY:
        play_wave(s)
        if s.phase is Phase.REPORT:
            s.next_wave()
    assert s.wave == 3 and len(s.history) == 3
    assert [r.wave for r in s.history] == [1, 2, 3]
    with pytest.raises(RuntimeError):
        s.next_wave()


def test_single_wave_run_is_victory_after_first_wave():
    s = make_session(num_waves=1)
    s.start()
    play_wave(s)
    assert s.phase is Phase.VICTORY


@pytest.mark.parametrize("end", [Phase.GAME_OVER, Phase.VICTORY])
def test_restart_from_end_screens_resets_the_run(end):
    s = make_session(num_waves=1)
    s.start()
    if end is Phase.GAME_OVER:
        kill_player(s)
    else:
        play_wave(s)
    assert s.phase is end
    s.start()
    assert s.phase is Phase.WAVE and s.wave == 1 and s.history == []
    assert s.world.player.alive and len(s.world.zombies) == POPULATION_SIZE


def test_restart_gives_a_new_random_population():
    s = make_session(num_waves=1)
    s.start()
    first = [g.genes.copy() for g in s.genomes]
    kill_player(s)
    s.start()
    assert any(not np.array_equal(a, g.genes) for a, g in zip(first, s.genomes))


# --- determinism ---


def run_history(seed: int):
    s = make_session(seed=seed, num_waves=3)
    s.start()
    while s.phase is not Phase.VICTORY:
        play_wave(s)
        if s.phase is Phase.REPORT:
            s.next_wave()
    return [(r.best_fitness, r.mean_fitness, r.zombies_killed, r.fought.mean_speed) for r in s.history]


def test_same_seed_same_run():
    assert run_history(4) == run_history(4)


def test_different_seed_different_run():
    assert run_history(4) != run_history(5)


def test_module_is_pygame_free():
    code = "import sys; import game.session; sys.exit(1 if 'pygame' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0


# --- HUD text ---


def test_status_line_shows_wave_time_hp_zombies():
    s = make_session(num_waves=5, limit=30.0)
    s.start()
    line = status_line(s)
    assert "Wave 1/5" in line and "Time 30s" in line and "HP 100/100" in line
    assert f"Zombies {POPULATION_SIZE}/{POPULATION_SIZE}" in line
    s.world.zombies[0].take_damage(1e9)
    s.world.player.take_damage(25)
    s.step(IDLE)
    line = status_line(s)
    assert f"Zombies {POPULATION_SIZE - 1}/{POPULATION_SIZE}" in line and "HP 75/100" in line


def test_evolution_line_first_wave_then_fitness_and_delta():
    s = make_session()
    s.start()
    assert "random first generation" in evolution_line(s)
    assert "spd" in evolution_line(s) and "A*/Greedy/Direct" in evolution_line(s)
    play_wave(s)
    s.next_wave()
    line = evolution_line(s)
    assert "last wave fitness best" in line and "(" not in line  # no delta with a single report
    play_wave(s)
    s.next_wave()
    assert "(" in evolution_line(s) and ("+" in evolution_line(s) or "-" in evolution_line(s).split("(")[1])


# --- screens ---


def test_screen_text_per_phase():
    s = make_session(num_waves=2)
    title, lines = screen_text(s)
    assert title == "ZOMBIE DARWIN" and any("ENTER" in l for l in lines)
    s.start()
    assert screen_text(s) is None  # no panel during a wave
    play_wave(s)
    title, lines = screen_text(s)
    assert title == "WAVE 1 COMPLETE"
    text = "\n".join(lines)
    assert "Fitness" in text and "Evolved to" in text and "wave 2" in text
    s.next_wave()
    play_wave(s)
    title, lines = screen_text(s)
    assert title == "VICTORY" and "Avg zombie fitness per wave" in "\n".join(lines)
    s.start()
    kill_player(s)
    title, lines = screen_text(s)
    assert title == "GAME OVER" and "wave 1" in "\n".join(lines) and "you were killed" in "\n".join(lines)


# --- rendering + key handling (SDL dummy driver) ---


@pytest.fixture
def pg():
    pygame.init()
    yield
    pygame.quit()


def draw_all_phases(s: GameSession, r: Renderer, surface):
    seen = []
    r.draw(surface, s.world, s)
    seen.append(s.phase)
    s.start()
    r.draw(surface, s.world, s)
    seen.append(s.phase)
    play_wave(s)
    r.draw(surface, s.world, s)
    seen.append(s.phase)
    s.next_wave()
    play_wave(s)
    r.draw(surface, s.world, s)
    seen.append(s.phase)
    s.start()
    kill_player(s)
    r.draw(surface, s.world, s)
    seen.append(s.phase)
    return seen


def test_renderer_draws_every_phase(pg):
    s = make_session(num_waves=2)
    r = Renderer(s.world)
    surface = pygame.Surface(r.screen_size)
    assert draw_all_phases(s, r, surface) == [Phase.MENU, Phase.WAVE, Phase.REPORT, Phase.VICTORY, Phase.GAME_OVER]


def test_panel_dims_the_scene_but_wave_does_not(pg):
    s = make_session()
    r = Renderer(s.world)
    surface = pygame.Surface(r.screen_size)
    s.start()
    z = s.world.zombies[0]
    from game.renderer import ZOMBIE_COLOUR

    r.draw(surface, s.world, s)
    assert surface.get_at((round(z.x), round(z.y)))[:3] == ZOMBIE_COLOUR  # HUD sits on the walls only
    play_wave(s)
    r.draw(surface, s.world, s)
    assert surface.get_at((16, 100))[:3] != (95, 95, 105)  # dimmed wall column


def test_hud_text_drawn_on_wall_rows(pg):
    s = make_session()
    r = Renderer(s.world)
    s.start()
    with_hud = pygame.Surface(r.screen_size)
    plain = pygame.Surface(r.screen_size)
    r.draw(with_hud, s.world, s)
    r.draw(plain, s.world)  # legacy HUD is on the bottom row only
    top = pygame.Rect(0, 0, r.screen_size[0], 32)
    assert pygame.image.tobytes(with_hud.subsurface(top), "RGB") != pygame.image.tobytes(plain.subsurface(top), "RGB")


def test_draw_without_session_still_works(pg):
    s = make_session()
    r = Renderer(s.world)
    r.draw(pygame.Surface(r.screen_size), s.world)


def test_enter_key_flow(pg):
    s = make_session(num_waves=2)
    r = Renderer(s.world)
    assert handle_keydown(s, r, pygame.K_RETURN) and s.phase is Phase.WAVE
    assert handle_keydown(s, r, pygame.K_RETURN) and s.phase is Phase.WAVE  # ignored mid-wave
    play_wave(s)
    assert s.phase is Phase.REPORT
    assert handle_keydown(s, r, pygame.K_SPACE) and s.phase is Phase.WAVE and s.wave == 2
    play_wave(s)
    assert s.phase is Phase.VICTORY
    assert handle_keydown(s, r, pygame.K_KP_ENTER) and s.phase is Phase.WAVE and s.wave == 1


def test_escape_quits_and_f1_toggles_debug(pg):
    s = make_session()
    r = Renderer(s.world)
    assert handle_keydown(s, r, pygame.K_ESCAPE) is False
    assert r.debug is False
    assert handle_keydown(s, r, pygame.K_F1) is True and r.debug is True
    assert handle_keydown(s, r, pygame.K_a) is True and s.phase is Phase.MENU


def test_main_smoke_from_menu():
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    result = subprocess.run(
        [sys.executable, "main.py", "--max-frames", "30"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


# --- full default-size MVP run ---


@pytest.mark.slow
def test_full_ten_wave_run_with_default_settings():
    s = GameSession(TileMap.load(LEVEL1), seed=1)
    s.start()
    while s.phase is not Phase.VICTORY:
        play_wave(s)
        if s.phase is Phase.REPORT:
            s.next_wave()
    assert len(s.history) == NUM_WAVES
    assert all(r.zombies_total == POPULATION_SIZE for r in s.history)
    assert all(r.time <= WAVE_TIME_LIMIT + 1e-9 for r in s.history)
