"""Pygame drawing (P2.3). Reads World state, never changes it.

One of only two modules allowed to import pygame (the other is main.py).
World coordinates are pixels, and there is no camera: the whole 32x20 map
fits on screen, so world position == screen position.
"""
from __future__ import annotations

import pygame

from game.engine import World

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


class Renderer:
    def __init__(self, world: World):
        tm = world.tilemap
        self.screen_size = (tm.width * tm.tile_size, tm.height * tm.tile_size)
        # The map never changes during a run, so draw it once and blit it every frame.
        self._background = self._build_background(world)
        self._font = None  # created lazily so pygame.font is initialised by then

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
        for zombie in world.zombies:
            self._draw_zombie(surface, zombie)
        self._draw_player(surface, world)
        # bullets last so the aim line never hides them
        for bullet in world.bullets:
            pygame.draw.circle(surface, BULLET_COLOUR, (round(bullet.x), round(bullet.y)), round(bullet.radius))
        self._draw_hud(surface, world)

    def _draw_zombie(self, surface: pygame.Surface, zombie) -> None:
        centre = (round(zombie.x), round(zombie.y))
        pygame.draw.circle(surface, ZOMBIE_COLOUR, centre, round(zombie.radius))
        self._draw_hp_bar(surface, zombie.x, zombie.y - zombie.radius - 6, zombie.hp / zombie.max_hp)

    def _draw_hp_bar(self, surface: pygame.Surface, cx: float, top: float, fraction: float) -> None:
        width = 24
        left = round(cx - width / 2)
        pygame.draw.rect(surface, HP_BAR_BG, pygame.Rect(left, round(top), width, 3))
        pygame.draw.rect(surface, HP_BAR_FG, pygame.Rect(left, round(top), round(width * fraction), 3))

    def _draw_hud(self, surface: pygame.Surface, world: World) -> None:
        if self._font is None:
            self._font = pygame.font.Font(None, 24)
        p = world.player
        label = self._font.render(f"HP {p.hp:.0f}/{p.max_hp:.0f}   Zombies {len(world.zombies)}", True, TEXT_COLOUR)
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
