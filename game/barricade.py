"""Barricade placement rules (P2.5).

Barricades are indestructible blocked tiles, capped at MAX_BARRICADES. A
placement is rejected if it would leave any spawn point without a path to
the player — checked with BFS (ai.pathfinding.is_reachable) against the map
as it *would* look with the new barricade added.

Pathfinding sees barricades as fully blocked (not high-cost), because World
itself is the graph zombies path over and World.neighbors() skips them.
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Iterable, List, Tuple

from ai.pathfinding import is_reachable
from config import MAX_BARRICADES

if TYPE_CHECKING:
    from game.engine import World

Tile = Tuple[int, int]


class PlacementResult(Enum):
    OK = "ok"
    INVALID_TILE = "invalid tile"  # out of bounds, a wall, or a spawn point
    OCCUPIED = "occupied"  # already barricaded, or the player / a zombie is on it
    LIMIT_REACHED = "limit reached"
    BLOCKS_PATH = "blocks path"  # a spawn would lose its route to the player


class _GraphWithExtraBlock:
    """World's walkable graph with one extra tile treated as blocked, for 'what if' BFS."""

    def __init__(self, world: "World", extra: Tile):
        self._world = world
        self._extra = extra

    def neighbors(self, tile: Tile) -> List[Tile]:
        return [t for t in self._world.neighbors(tile) if t != self._extra]


def check_placement(world: "World", tile: Tile, max_barricades: int = MAX_BARRICADES) -> PlacementResult:
    """Decide whether `tile` may be barricaded. Does not modify the world."""
    tm = world.tilemap
    if not tm.is_walkable(tile) or tile in tm.spawn_points:
        return PlacementResult.INVALID_TILE
    if tile in world.barricades or tile in world.player.occupied_tiles():
        return PlacementResult.OCCUPIED
    if any(tile in z.occupied_tiles() for z in world.zombies):
        return PlacementResult.OCCUPIED
    if len(world.barricades) >= max_barricades:
        return PlacementResult.LIMIT_REACHED

    player_tile = tm.world_to_tile(world.player.pos)
    graph = _GraphWithExtraBlock(world, tile)
    if not is_reachable(graph, tm.spawn_points, player_tile):
        return PlacementResult.BLOCKS_PATH
    return PlacementResult.OK


def all_spawns_connected(world: "World", spawns: Iterable[Tile] | None = None) -> bool:
    """True if every spawn can currently reach the player's tile. Handy invariant for tests."""
    tm = world.tilemap
    starts = tm.spawn_points if spawns is None else spawns
    return is_reachable(world, starts, tm.world_to_tile(world.player.pos))
