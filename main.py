"""Zombie Darwin — playable entry point (P2.3, game flow P4.7).

Owns the pygame window, input and frame timing only. All simulation and game
flow live in GameSession (game/session.py: menu -> wave -> report -> ... ->
victory / game over, with the zombies evolving between waves); this file turns
keys into Actions and calls session.step() at the fixed tick rate.

Keys: ENTER / SPACE start, next wave, play again.  F1 debug overlay.  Esc quit.

Run:  python main.py            (F1 toggles the zombie debug overlay)
      python main.py --debug    (start with the overlay on)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pygame

from config import FIXED_DT
from game.map import TileMap
from game.player import Action
from game.renderer import Renderer
from game.session import GameSession, Phase

LEVEL1_PATH = Path(__file__).resolve().parent / "maps" / "level1.txt"
TARGET_FPS = 60
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


def handle_keydown(session: GameSession, renderer: Renderer, key: int) -> bool:
    """Apply one key press. Returns False when the game should quit."""
    if key == pygame.K_ESCAPE:
        return False
    if key == pygame.K_F1:
        renderer.toggle_debug()
    elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
        if session.phase in (Phase.MENU, Phase.VICTORY, Phase.GAME_OVER):
            session.start()
        elif session.phase is Phase.REPORT:
            session.next_wave()
    return True


def main(seed: int = 0, max_frames: int | None = None, debug: bool = False) -> None:
    pygame.init()
    session = GameSession(TileMap.load(LEVEL1_PATH), seed=seed)
    renderer = Renderer(session.world, debug=debug)
    screen = pygame.display.set_mode(renderer.screen_size)
    pygame.display.set_caption("Zombie Darwin")
    clock = pygame.time.Clock()
    controller = HumanController()

    accumulator = 0.0
    frames = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                running = handle_keydown(session, renderer, event.key) and running

        # Fixed timestep: the simulation always advances in FIXED_DT ticks,
        # however long the frame actually took. Only waves tick; menu and reports wait.
        accumulator += clock.tick(TARGET_FPS) / 1000.0
        steps = 0
        while accumulator >= FIXED_DT and steps < MAX_STEPS_PER_FRAME and session.phase is Phase.WAVE:
            session.step(controller.get_action(session.world))
            accumulator -= FIXED_DT
            steps += 1
        if steps == MAX_STEPS_PER_FRAME or session.phase is not Phase.WAVE:
            accumulator = 0.0

        renderer.draw(screen, session.world, session)
        pygame.display.flip()

        frames += 1
        if max_frames is not None and frames >= max_frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Zombie Darwin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None, help="quit after N frames (smoke testing)")
    parser.add_argument("--debug", action="store_true", help="start with the zombie debug overlay on (F1 toggles)")
    args = parser.parse_args()
    main(seed=args.seed, max_frames=args.max_frames, debug=args.debug)
