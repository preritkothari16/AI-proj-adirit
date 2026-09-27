"""A*, Greedy Best-First and BFS reachability (P1.4).

Operates on any object exposing `neighbors(tile) -> Iterable[Tile]` (see
GraphLike below) — deliberately does NOT import game/map.py or pygame, so
this module stays pure Python and usable standalone by ai/ code (see the
import-direction rule in PROJECT_CONTEXT.md). game.map.TileMap satisfies
this interface structurally; tests exercise both a minimal fake grid and
the real TileMap.

4-directional only, no diagonals — matches TileMap.neighbors().
"""
from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

Tile = Tuple[int, int]


class GraphLike(Protocol):
    def neighbors(self, tile: Tile) -> Iterable[Tile]: ...


def manhattan(a: Tile, b: Tile) -> int:
    """Admissible, consistent heuristic for 4-directional grid movement."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _reconstruct_path(came_from: Dict[Tile, Tile], start: Tile, goal: Tile) -> List[Tile]:
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def astar(
    graph: GraphLike,
    start: Tile,
    goal: Tile,
    stats: Optional[dict] = None,
) -> Optional[List[Tile]]:
    """Shortest tile path start->goal (inclusive), or None if unreachable.

    If `stats` is passed, sets stats['nodes_expanded'] — for experiments
    comparing A* vs Greedy vs Direct (P5.6). Return type is unchanged so
    existing callers don't need to pass it.
    """
    if start == goal:
        if stats is not None:
            stats["nodes_expanded"] = 0
        return [start]

    counter = 0  # heap tie-breaker; tiles aren't orderable against each other
    open_heap: List[tuple] = [(manhattan(start, goal), 0, counter, start)]
    open_set = {start}
    came_from: Dict[Tile, Tile] = {}
    g_score: Dict[Tile, int] = {start: 0}
    nodes_expanded = 0

    while open_heap:
        _, g, _, current = heapq.heappop(open_heap)
        if current not in open_set:
            continue  # stale entry, a better one was already processed
        open_set.discard(current)
        nodes_expanded += 1

        if current == goal:
            if stats is not None:
                stats["nodes_expanded"] = nodes_expanded
            return _reconstruct_path(came_from, start, goal)

        for neighbor in graph.neighbors(current):
            tentative_g = g + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + manhattan(neighbor, goal)
                counter += 1
                heapq.heappush(open_heap, (f, tentative_g, counter, neighbor))
                open_set.add(neighbor)

    if stats is not None:
        stats["nodes_expanded"] = nodes_expanded
    return None


def greedy_best_first(
    graph: GraphLike,
    start: Tile,
    goal: Tile,
    stats: Optional[dict] = None,
) -> Optional[List[Tile]]:
    """Tile path start->goal ordered purely by heuristic distance to goal,
    ignoring cost already spent. Faster to expand, not guaranteed shortest.
    Returns None if unreachable. See `astar` for the `stats` argument.
    """
    if start == goal:
        if stats is not None:
            stats["nodes_expanded"] = 0
        return [start]

    counter = 0
    open_heap: List[tuple] = [(manhattan(start, goal), counter, start)]
    came_from: Dict[Tile, Tile] = {}
    visited = {start}
    nodes_expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        nodes_expanded += 1

        if current == goal:
            if stats is not None:
                stats["nodes_expanded"] = nodes_expanded
            return _reconstruct_path(came_from, start, goal)

        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(open_heap, (manhattan(neighbor, goal), counter, neighbor))

    if stats is not None:
        stats["nodes_expanded"] = nodes_expanded
    return None


def _bfs_reaches(graph: GraphLike, start: Tile, goal: Tile) -> bool:
    if start == goal:
        return True
    visited = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in graph.neighbors(current):
            if neighbor == goal:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def is_reachable(graph: GraphLike, starts: Iterable[Tile], goal: Tile) -> bool:
    """True iff `goal` is reachable (via BFS) from every tile in `starts`.

    Used to validate barricade placement (P2.5): a barricade must never
    seal a spawn off from the player.
    """
    return all(_bfs_reaches(graph, start, goal) for start in starts)
