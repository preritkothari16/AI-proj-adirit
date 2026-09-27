"""Zombie Darwin — playable entry point (P2.3).

Owns the pygame window, input and frame timing only. All simulation lives in
World (game/engine.py); this file just turns keys into Actions and calls
world.step() at the fixed tick rate.

Run:  python main.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pygame

from config import FIXED_DT
from game.engine import World
from game.map import TileMap
from game.player import Action
from game.renderer import Renderer

LEVEL1_PATH = Path(__file__).resolve().parent / "maps" / "level1.txt"
TARGET_FPS = 60
DEMO_SPAWN_INTERVAL = 6.0  # seconds, temporary until WaveManager (P4.6)
MAX_STEPS_PER_FRAME = 5  # stops a long stall (e.g. dragging the window) turning into a burst of ticks


def keys_to_move(up: bool, down: bool, left: bool, right: bool) -> tuple[float, float]:
    """Pure helper: WASD booleans -> move vector. Opposite keys cancel out."""
    return (float(right) - float(left), float(down) - float(up))


class HumanController:
    """Reads keyboard/mouse state and produces an Action. Satisfies Controller."""

    def get_action(self, world) -> Action:
        keys = pygame.key.get_pressed()
        move = keys_to_move(
            up=keys[pygame.K_w] or keys[pygame.K_UP],
            down=keys[pygame.K_s] or keys[pygame.K_DOWN],
            left=keys[pygame.K_a] or keys[pygame.K_LEFT],
            right=keys[pygame.K_d] or keys[pygame.K_RIGHT],
        )
        mx, my = pygame.mouse.get_pos()
        buttons = pygame.mouse.get_pressed()
        # right click: try to barricade the tile under the cursor (World rejects invalid spots)
        barricade = world.tilemap.world_to_tile((mx, my)) if buttons[2] else None
        return Action(move=move, aim=(float(mx), float(my)), shoot=buttons[0], barricade=barricade)


def main(seed: int = 0, max_frames: int | None = None) -> None:
    pygame.init()
    world = World(TileMap.load(LEVEL1_PATH), np.random.default_rng(seed))
    renderer = Renderer(world)
    screen = pygame.display.set_mode(renderer.screen_size)
    pygame.display.set_caption("Zombie Darwin")
    clock = pygame.time.Clock()
    controller = HumanController()

    accumulator = 0.0
    next_spawn_tick = 0
    frames = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Fixed timestep: the simulation always advances in FIXED_DT ticks,
        # however long the frame actually took.
        accumulator += clock.tick(TARGET_FPS) / 1000.0
        steps = 0
        while accumulator >= FIXED_DT and steps < MAX_STEPS_PER_FRAME:
            # Temporary P2.6 spawner: one dummy per spawn point every few seconds.
            # WaveManager (P4.6) replaces this.
            if world.tick >= next_spawn_tick and not world.game_over:
                for tile in world.tilemap.spawn_points:
                    world.spawn_zombie(tile)
                next_spawn_tick = world.tick + round(DEMO_SPAWN_INTERVAL / FIXED_DT)
            world.step(controller.get_action(world))
            accumulator -= FIXED_DT
            steps += 1
        if steps == MAX_STEPS_PER_FRAME:
            accumulator = 0.0

        renderer.draw(screen, world)
        pygame.display.flip()

        frames += 1
        if max_frames is not None and frames >= max_frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Zombie Darwin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None, help="quit after N frames (smoke testing)")
    args = parser.parse_args()
    main(seed=args.seed, max_frames=args.max_frames)
