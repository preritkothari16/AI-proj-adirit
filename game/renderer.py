"""Pygame drawing (P2.3) + zombie debug overlay (P3.5). Reads World state, never changes it.

One of only two modules allowed to import pygame (the other is main.py).
World coordinates are pixels, and there is no camera: the whole 32x20 map
fits on screen, so world position == screen position.

Debug overlay (toggle with F1 in main.py), per zombie, coloured by FSM state:
  - vision radius circle
  - route it is steering along: its position -> remaining path tile centres ->
    the point it is heading for (player / last known position / wander tile)
  - last known player position (X) while searching
  - label: state and path strategy

With a GameSession (P4.7) draw() also shows the game-flow HUD (wave, timer, HP,
zombies remaining on the top wall row; evolution summary on the bottom one) and
the menu / between-wave report / victory / game-over panels. Without one it
falls back to the plain HP + zombie-count HUD.
"""
from __future__ import annotations

import pygame

from ai.genome import PathStrategy
from game.engine import World
from game.session import GameSession, Phase, WaveReport
from game.states import ZombieState
from game.wave_manager import WaveEnd

FLOOR_COLOUR = (40, 40, 46)
WALL_COLOUR = (95, 95, 105)
SPAWN_COLOUR = (70, 30, 30)
GRID_COLOUR = (48, 48, 55)
PLAYER_COLOUR = (80, 170, 255)
AIM_COLOUR = (200, 200, 200)
BULLET_COLOUR = (255, 220, 90)
BARRICADE_COLOUR = (150, 100, 50)
ZOMBIE_COLOUR = (90, 200, 90)
HP_BAR_BG = (120, 30, 30)
HP_BAR_FG = (80, 220, 80)
TEXT_COLOUR = (240, 240, 240)

STATE_COLOURS = {
    ZombieState.WANDER: (150, 150, 150),
    ZombieState.CHASE: (255, 70, 70),
    ZombieState.SEARCH: (255, 170, 40),
    ZombieState.ATTACK: (220, 80, 255),
}
STRATEGY_LABELS = {PathStrategy.DIRECT: "direct", PathStrategy.GREEDY: "greedy", PathStrategy.ASTAR: "A*"}
PANEL_COLOUR = (24, 24, 30)
PANEL_BORDER = (150, 150, 165)
DIM_TEXT_COLOUR = (170, 170, 180)
END_REASON_TEXT = {
    WaveEnd.ALL_DEAD: "all zombies destroyed",
    WaveEnd.TIME_UP: "time ran out",
    WaveEnd.PLAYER_DEAD: "you were killed",
}


# --- text for the HUD and screens (pure, so tests can check it without pixels) ---


def status_line(session: GameSession) -> str:
    """Top HUD: wave, timer, HP, zombies remaining."""
    p = session.world.player
    return (
        f"Wave {session.wave}/{session.num_waves}   Time {session.time_left:.0f}s   "
        f"HP {p.hp:.0f}/{p.max_hp:.0f}   Zombies {session.zombies_remaining}/{session.zombies_total}"
    )


def traits_text(pop) -> str:
    c = pop.strategy_counts
    return (
        f"spd {pop.mean_speed:.0f}  hp {pop.mean_hp:.0f}  vis {pop.mean_vision:.0f}  "
        f"A*/Greedy/Direct {c[PathStrategy.ASTAR]}/{c[PathStrategy.GREEDY]}/{c[PathStrategy.DIRECT]}"
    )


def evolution_line(session: GameSession) -> str:
    """Bottom HUD: how the last wave scored and what the current population looks like."""
    pop = session.population_summary
    if pop is None:
        return ""
    last = session.last_report
    if last is None:
        fitness = "random first generation"
    else:
        fitness = f"last wave fitness best {last.best_fitness:.2f} avg {last.mean_fitness:.2f}"
        if len(session.history) > 1:
            fitness += f" ({last.mean_fitness - session.history[-2].mean_fitness:+.2f})"
    return f"Evolution  {fitness}  |  {traits_text(pop)}"


def _fitness_trend(session: GameSession) -> str:
    return "Avg zombie fitness per wave: " + "  ".join(f"{r.mean_fitness:.2f}" for r in session.history)


def _report_lines(r: WaveReport, session: GameSession) -> list[str]:
    lines = [
        f"Ended: {END_REASON_TEXT[r.end_reason]}   Time {r.time:.0f}s",
        f"Zombies killed {r.zombies_killed}/{r.zombies_total}   Your HP {r.player_hp:.0f}/{r.player_max_hp:.0f}",
        f"Fitness  best {r.best_fitness:.2f}   avg {r.mean_fitness:.2f}   worst {r.worst_fitness:.2f}",
        f"Avg damage dealt per zombie {r.mean_damage:.1f}",
        "",
        "This wave:  " + traits_text(r.fought),
    ]
    if r.evolved is not None:
        lines.append("Evolved to: " + traits_text(r.evolved))
    if len(session.history) > 1:
        lines += ["", _fitness_trend(session)]
    return lines


def screen_text(session: GameSession) -> tuple[str, list[str]] | None:
    """(title, body lines) of the panel for the current phase; None during a wave."""
    phase = session.phase
    if phase is Phase.MENU:
        return "ZOMBIE DARWIN", [
            f"Survive {session.num_waves} waves. The zombies evolve between waves.",
            "",
            "WASD / arrows  move        Mouse  aim",
            "Left click  shoot        Right click  barricade",
            "F1  debug overlay        Esc  quit",
            "",
            "Press ENTER to start",
        ]
    report = session.last_report
    if report is None:
        return None
    if phase is Phase.REPORT:
        return f"WAVE {report.wave} COMPLETE", _report_lines(report, session) + [
            "",
            f"Press ENTER for wave {report.wave + 1}",
        ]
    if phase is Phase.VICTORY:
        head = [f"You survived all {session.num_waves} waves!", ""]
        return "VICTORY", head + _report_lines(report, session) + ["", "Press ENTER to play again"]
    if phase is Phase.GAME_OVER:
        head = [f"You fell on wave {report.wave} of {session.num_waves}.", ""]
        return "GAME OVER", head + _report_lines(report, session) + ["", "Press ENTER to play again"]
    return None


class Renderer:
    def __init__(self, world: World, debug: bool = False):
        self.debug = debug
        tm = world.tilemap
        self.screen_size = (tm.width * tm.tile_size, tm.height * tm.tile_size)
        # The map never changes during a run, so draw it once and blit it every frame.
        self._background = self._build_background(world)
        self._font = None  # created lazily so pygame.font is initialised by then
        self._small_font = None

    def toggle_debug(self) -> None:
        self.debug = not self.debug

    def _build_background(self, world: World) -> pygame.Surface:
        tm = world.tilemap
        ts = tm.tile_size
        surface = pygame.Surface(self.screen_size)
        surface.fill(FLOOR_COLOUR)
        spawns = set(tm.spawn_points)
        for ty in range(tm.height):
            for tx in range(tm.width):
                rect = pygame.Rect(tx * ts, ty * ts, ts, ts)
                if (tx, ty) in tm.walls:
                    pygame.draw.rect(surface, WALL_COLOUR, rect)
                elif (tx, ty) in spawns:
                    pygame.draw.rect(surface, SPAWN_COLOUR, rect)
                pygame.draw.rect(surface, GRID_COLOUR, rect, width=1)
        return surface

    def draw(self, surface: pygame.Surface, world: World, session: GameSession | None = None) -> None:
        surface.blit(self._background, (0, 0))
        ts = world.tilemap.tile_size
        for tx, ty in world.barricades:
            pygame.draw.rect(surface, BARRICADE_COLOUR, pygame.Rect(tx * ts + 2, ty * ts + 2, ts - 4, ts - 4))
        if self.debug:  # under the bodies so routes don't hide them
            for zombie in world.zombies:
                self._draw_zombie_debug(surface, zombie, world)
        for zombie in world.zombies:
            self._draw_zombie(surface, zombie)
        if self.debug:
            for zombie in world.zombies:
                self._draw_zombie_label(surface, zombie)
        self._draw_player(surface, world)
        # bullets last so the aim line never hides them
        for bullet in world.bullets:
            pygame.draw.circle(surface, BULLET_COLOUR, (round(bullet.x), round(bullet.y)), round(bullet.radius))
        if session is None:
            self._draw_hud(surface, world)
        else:
            self._draw_session(surface, session)

    def _draw_session(self, surface: pygame.Surface, session: GameSession) -> None:
        if session.phase is not Phase.MENU:
            self._draw_bar_text(surface, status_line(session), top=True)
            self._draw_bar_text(surface, evolution_line(session), top=False)
        screen = screen_text(session)
        if screen is not None:
            self._draw_panel(surface, *screen)

    def _draw_bar_text(self, surface: pygame.Surface, text: str, top: bool) -> None:
        if self._font is None:
            self._font = pygame.font.Font(None, 24)
        if self._small_font is None:
            self._small_font = pygame.font.Font(None, 16)
        # HUD lives on the solid border wall rows so it never covers playable floor.
        label = self._font.render(text, True, TEXT_COLOUR)
        y = 6 if top else self.screen_size[1] - label.get_height() - 8
        surface.blit(label, (8, y))
        if top and self.debug:
            hint = self._small_font.render("[F1] debug", True, DIM_TEXT_COLOUR)
            surface.blit(hint, (self.screen_size[0] - hint.get_width() - 8, 10))

    def _draw_panel(self, surface: pygame.Surface, title: str, lines: list[str]) -> None:
        shade = pygame.Surface(self.screen_size, pygame.SRCALPHA)
        shade.fill((0, 0, 0, 170))
        surface.blit(shade, (0, 0))
        title_font, body_font = pygame.font.Font(None, 64), pygame.font.Font(None, 26)
        title_img = title_font.render(title, True, TEXT_COLOUR)
        body = [body_font.render(line, True, TEXT_COLOUR if line else DIM_TEXT_COLOUR) for line in lines]
        line_h = body_font.get_linesize()
        width = max([title_img.get_width()] + [b.get_width() for b in body]) + 48
        height = title_img.get_height() + line_h * len(body) + 48
        panel = pygame.Rect(0, 0, width, height)
        panel.center = (self.screen_size[0] // 2, self.screen_size[1] // 2)
        pygame.draw.rect(surface, PANEL_COLOUR, panel)
        pygame.draw.rect(surface, PANEL_BORDER, panel, width=2)
        surface.blit(title_img, title_img.get_rect(midtop=(panel.centerx, panel.top + 20)))
        y = panel.top + 20 + title_img.get_height() + 8
        for img in body:
            surface.blit(img, img.get_rect(midtop=(panel.centerx, y)))
            y += line_h

    def _draw_zombie(self, surface: pygame.Surface, zombie) -> None:
        centre = (round(zombie.x), round(zombie.y))
        pygame.draw.circle(surface, ZOMBIE_COLOUR, centre, round(zombie.radius))
        self._draw_hp_bar(surface, zombie.x, zombie.y - zombie.radius - 6, zombie.hp / zombie.max_hp)

    def _draw_zombie_debug(self, surface: pygame.Surface, zombie, world: World) -> None:
        colour = STATE_COLOURS[zombie.state]
        centre = (round(zombie.x), round(zombie.y))
        pygame.draw.circle(surface, colour, centre, round(zombie.vision_radius), width=1)
        route = [centre] + [world.tilemap.tile_to_world(t) for t in zombie.path]
        target = self._debug_target(zombie, world)
        if target is not None:
            route.append(target)
        if len(route) >= 2:
            pygame.draw.lines(surface, colour, False, [(round(x), round(y)) for x, y in route], 2)
        if zombie.state is ZombieState.SEARCH and zombie.last_known_pos is not None:
            x, y = round(zombie.last_known_pos[0]), round(zombie.last_known_pos[1])
            pygame.draw.line(surface, colour, (x - 5, y - 5), (x + 5, y + 5), 2)
            pygame.draw.line(surface, colour, (x - 5, y + 5), (x + 5, y - 5), 2)

    @staticmethod
    def _debug_target(zombie, world: World):
        """Point the zombie is finally heading for in its current state (None when attacking)."""
        if zombie.state is ZombieState.CHASE:
            return world.player.pos
        if zombie.state is ZombieState.SEARCH:
            return zombie.last_known_pos
        if zombie.state is ZombieState.WANDER and zombie.wander_target is not None:
            return world.tilemap.tile_to_world(zombie.wander_target)
        return None

    def _draw_zombie_label(self, surface: pygame.Surface, zombie) -> None:
        if self._small_font is None:
            self._small_font = pygame.font.Font(None, 16)
        text = f"{zombie.state.name} {STRATEGY_LABELS[zombie.path_strategy]}"
        label = self._small_font.render(text, True, STATE_COLOURS[zombie.state])
        surface.blit(label, label.get_rect(midtop=(round(zombie.x), round(zombie.y + zombie.radius + 2))))

    def _draw_hp_bar(self, surface: pygame.Surface, cx: float, top: float, fraction: float) -> None:
        width = 24
        left = round(cx - width / 2)
        pygame.draw.rect(surface, HP_BAR_BG, pygame.Rect(left, round(top), width, 3))
        pygame.draw.rect(surface, HP_BAR_FG, pygame.Rect(left, round(top), round(width * fraction), 3))

    def _draw_hud(self, surface: pygame.Surface, world: World) -> None:
        if self._font is None:
            self._font = pygame.font.Font(None, 24)
        p = world.player
        text = f"HP {p.hp:.0f}/{p.max_hp:.0f}   Zombies {len(world.zombies)}"
        if self.debug:
            text += "   [F1] debug"
        label = self._font.render(text, True, TEXT_COLOUR)
        surface.blit(label, (8, self.screen_size[1] - label.get_height() - 6))  # on the bottom border wall
        if world.game_over:
            shade = pygame.Surface(self.screen_size, pygame.SRCALPHA)
            shade.fill((0, 0, 0, 160))
            surface.blit(shade, (0, 0))
            big = pygame.font.Font(None, 72).render("GAME OVER", True, TEXT_COLOUR)
            surface.blit(big, big.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] // 2)))

    def _draw_player(self, surface: pygame.Surface, world: World) -> None:
        p = world.player
        centre = (round(p.x), round(p.y))
        pygame.draw.circle(surface, PLAYER_COLOUR, centre, round(p.radius))
        aim = world.last_action.aim
        if aim != (0.0, 0.0):
            pygame.draw.line(surface, AIM_COLOUR, centre, (round(aim[0]), round(aim[1])), 1)
