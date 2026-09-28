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
"""
from __future__ import annotations

import pygame

from ai.genome import PathStrategy
from game.engine import World
from game.states import ZombieState

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

    def draw(self, surface: pygame.Surface, world: World) -> None:
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
        self._draw_hud(surface, world)

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
