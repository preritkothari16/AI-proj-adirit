"""P1.4 — Manhattan heuristic, A*, Greedy Best-First, BFS reachability.

Uses a minimal local grid (mirrors TileMap's neighbors() interface) so
ai/pathfinding.py stays provably independent of game/map.py and pygame.
One integration test at the bottom runs against the real TileMap/level1.txt.
"""
from __future__ import annotations

from pathlib import Path

from ai.pathfinding import astar, greedy_best_first, is_reachable, manhattan

LEVEL1_PATH = Path(__file__).resolve().parent.parent / "maps" / "level1.txt"


class GridGraph:
    """Minimal stand-in for TileMap: same 4-directional neighbors() contract,
    built directly from ASCII rows so these tests don't depend on game/map.py.
    """

    def __init__(self, rows: list[str]):
        self.rows = rows
        self.height = len(rows)
        self.width = len(rows[0])
        assert all(len(r) == self.width for r in rows)

    def is_walkable(self, tile):
        x, y = tile
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return self.rows[y][x] != "#"

    def neighbors(self, tile):
        x, y = tile
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [t for t in candidates if self.is_walkable(t)]


def assert_valid_path(path, start, goal):
    """A path must start/end correctly and move one 4-dir step at a time."""
    assert path[0] == start
    assert path[-1] == goal
    for a, b in zip(path, path[1:]):
        assert manhattan(a, b) == 1, f"non-adjacent step {a} -> {b}"


# --- Manhattan heuristic ---


def test_manhattan_basic():
    assert manhattan((0, 0), (3, 4)) == 7


def test_manhattan_zero_for_same_tile():
    assert manhattan((5, 5), (5, 5)) == 0


def test_manhattan_symmetric():
    a, b = (2, 7), (9, 1)
    assert manhattan(a, b) == manhattan(b, a)


# --- A*: known shortest path (single unique corridor, hand-verified) ---

# #####
# #...#   <- row1: cols 1-3 open
# ###.#   <- row2: only col3 open
# #...#   <- row3: cols 1-3 open
# #####
FORCED_CORRIDOR = ["#####", "#...#", "###.#", "#...#", "#####"]


def test_astar_finds_known_shortest_path():
    graph = GridGraph(FORCED_CORRIDOR)
    start, goal = (1, 1), (1, 3)
    path = astar(graph, start, goal)
    assert path == [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3)]


def test_astar_start_equals_goal():
    graph = GridGraph(FORCED_CORRIDOR)
    assert astar(graph, (1, 1), (1, 1)) == [(1, 1)]


def test_astar_optimal_on_open_grid():
    """No obstacles: shortest path length must equal Manhattan distance + 1 tiles."""
    graph = GridGraph(["....."] * 5)
    start, goal = (0, 0), (4, 4)
    path = astar(graph, start, goal)
    assert_valid_path(path, start, goal)
    assert len(path) - 1 == manhattan(start, goal)


# --- Greedy Best-First ---


def test_greedy_finds_a_valid_path():
    graph = GridGraph(FORCED_CORRIDOR)
    start, goal = (1, 1), (1, 3)
    path = greedy_best_first(graph, start, goal)
    assert_valid_path(path, start, goal)
    # this map has only one possible route, so greedy must match A* here
    assert path == [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3)]


def test_greedy_start_equals_goal():
    graph = GridGraph(FORCED_CORRIDOR)
    assert greedy_best_first(graph, (1, 1), (1, 1)) == [(1, 1)]


def test_greedy_is_never_shorter_than_astar():
    """A* is optimal, so Greedy can only match or exceed its path length.
    Checked across several spawn -> player_start pairs on the real map.
    """
    from game.map import TileMap

    tm = TileMap.load(LEVEL1_PATH)
    for spawn in tm.spawn_points:
        astar_path = astar(tm, spawn, tm.player_start)
        greedy_path = greedy_best_first(tm, spawn, tm.player_start)
        assert astar_path is not None
        assert greedy_path is not None
        assert_valid_path(astar_path, spawn, tm.player_start)
        assert_valid_path(greedy_path, spawn, tm.player_start)
        assert len(greedy_path) >= len(astar_path)


# --- nodes_expanded stats (for P5.6 experiments) ---


def test_astar_reports_nodes_expanded():
    graph = GridGraph(FORCED_CORRIDOR)
    stats = {}
    astar(graph, (1, 1), (1, 3), stats=stats)
    assert stats["nodes_expanded"] > 0


def test_greedy_reports_nodes_expanded():
    graph = GridGraph(FORCED_CORRIDOR)
    stats = {}
    greedy_best_first(graph, (1, 1), (1, 3), stats=stats)
    assert stats["nodes_expanded"] > 0


def test_stats_argument_is_optional():
    graph = GridGraph(FORCED_CORRIDOR)
    # must not raise when stats isn't passed
    assert astar(graph, (1, 1), (1, 3)) is not None
    assert greedy_best_first(graph, (1, 1), (1, 3)) is not None


def test_nodes_expanded_zero_when_start_is_goal():
    graph = GridGraph(FORCED_CORRIDOR)
    stats = {}
    astar(graph, (1, 1), (1, 1), stats=stats)
    assert stats["nodes_expanded"] == 0


# --- blocked map returns None ---

# #######
# #.....#
# #.###.#
# #.#G#.#   <- goal fully walled in, no entrance
# #.###.#
# #.....#
# #######
SEALED_GOAL_MAP = [
    "#######",
    "#.....#",
    "#.###.#",
    "#.#G#.#",
    "#.###.#",
    "#.....#",
    "#######",
]


def test_astar_returns_none_when_goal_unreachable():
    graph = GridGraph(SEALED_GOAL_MAP)
    assert astar(graph, (1, 1), (3, 3)) is None


def test_greedy_returns_none_when_goal_unreachable():
    graph = GridGraph(SEALED_GOAL_MAP)
    assert greedy_best_first(graph, (1, 1), (3, 3)) is None


# --- BFS reachability (is_reachable) ---

# #####
# #.#.#   <- two parallel corridors, col1 and col3, separated by a wall
# #.#.#
# #.#.#
# #####
SPLIT_CORRIDORS = ["#####", "#.#.#", "#.#.#", "#.#.#", "#####"]


def test_is_reachable_true_within_same_corridor():
    graph = GridGraph(SPLIT_CORRIDORS)
    assert is_reachable(graph, [(1, 1)], (1, 3)) is True


def test_is_reachable_detects_sealed_path():
    graph = GridGraph(SPLIT_CORRIDORS)
    assert is_reachable(graph, [(1, 1)], (3, 1)) is False


def test_is_reachable_requires_all_starts_to_reach_goal():
    graph = GridGraph(SPLIT_CORRIDORS)
    # (3,1) cannot reach (1,3): mixed starts must fail overall
    assert is_reachable(graph, [(1, 1), (3, 1)], (1, 3)) is False


def test_is_reachable_trivially_true_when_start_is_goal():
    graph = GridGraph(SPLIT_CORRIDORS)
    assert is_reachable(graph, [(1, 1)], (1, 1)) is True


def test_is_reachable_true_on_real_level1_from_every_spawn():
    from game.map import TileMap

    tm = TileMap.load(LEVEL1_PATH)
    assert is_reachable(tm, tm.spawn_points, tm.player_start) is True


# --- module independence ---


def test_pathfinding_module_does_not_import_pygame():
    source = Path("ai/pathfinding.py").read_text()
    assert "import pygame" not in source
