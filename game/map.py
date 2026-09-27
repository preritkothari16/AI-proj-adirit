"""TileMap: loads a fixed ASCII map and answers walkability/coordinate queries (P1.3).

Format (one character per tile, one line per row):
    #  wall
    .  floor
    S  floor + spawn point
    P  floor + player start

Pathfinding lives in ai/pathfinding.py (P1.4). Line of sight (P1.5) is a grid
traversal between tile centres, see has_line_of_sight().
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

Tile = Tuple[int, int]  # (col, row)

_WALL = "#"
_FLOOR = "."
_SPAWN = "S"
_PLAYER_START = "P"
_VALID_CHARS = {_WALL, _FLOOR, _SPAWN, _PLAYER_START}


@dataclass(frozen=True)
class TileMap:
    width: int
    height: int
    tile_size: int
    walls: frozenset[Tile]
    spawn_points: Tuple[Tile, ...]
    player_start: Tile

    @classmethod
    def load(cls, path: str | Path, tile_size: int = 32) -> "TileMap":
        """Parse an ASCII map file. Raises ValueError on any malformed input."""
        text = Path(path).read_text().splitlines()
        rows = [line for line in text if line != ""]
        if not rows:
            raise ValueError(f"{path}: empty map")

        height = len(rows)
        width = len(rows[0])
        for row_index, row in enumerate(rows):
            if len(row) != width:
                raise ValueError(
                    f"{path}: row {row_index} has length {len(row)}, expected {width}"
                )
            for col_index, ch in enumerate(row):
                if ch not in _VALID_CHARS:
                    raise ValueError(
                        f"{path}: unrecognized character {ch!r} at ({col_index}, {row_index})"
                    )

        walls: set[Tile] = set()
        spawn_points: list[Tile] = []
        player_start: Tile | None = None

        for row_index, row in enumerate(rows):
            for col_index, ch in enumerate(row):
                tile = (col_index, row_index)
                if ch == _WALL:
                    walls.add(tile)
                elif ch == _SPAWN:
                    spawn_points.append(tile)
                elif ch == _PLAYER_START:
                    if player_start is not None:
                        raise ValueError(
                            f"{path}: multiple player-start tiles "
                            f"({player_start} and {tile})"
                        )
                    player_start = tile

        if not spawn_points:
            raise ValueError(f"{path}: no spawn points ('S') found")
        if player_start is None:
            raise ValueError(f"{path}: no player start ('P') found")

        return cls(
            width=width,
            height=height,
            tile_size=tile_size,
            walls=frozenset(walls),
            spawn_points=tuple(spawn_points),
            player_start=player_start,
        )

    def in_bounds(self, tile: Tile) -> bool:
        x, y = tile
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, tile: Tile) -> bool:
        return self.in_bounds(tile) and tile not in self.walls

    def neighbors(self, tile: Tile) -> list[Tile]:
        """4-directional (N/S/E/W) walkable neighbours, no diagonals."""
        x, y = tile
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [t for t in candidates if self.is_walkable(t)]

    def world_to_tile(self, pos: Tuple[float, float]) -> Tile:
        x, y = pos
        return (int(x // self.tile_size), int(y // self.tile_size))

    def tile_to_world(self, tile: Tile) -> Tuple[float, float]:
        """Centre of the tile, in world (pixel) coordinates."""
        x, y = tile
        return ((x + 0.5) * self.tile_size, (y + 0.5) * self.tile_size)

    def has_line_of_sight(
        self, a: Tile, b: Tile, is_blocked: Optional[Callable[[Tile], bool]] = None
    ) -> bool:
        """True if the straight segment between the centres of `a` and `b`
        crosses no blocked tile. Endpoints count too, so a blocked `a` or `b`
        is never visible.

        Supercover traversal in exact integer arithmetic: every tile the
        segment touches is checked. When the segment passes exactly through a
        tile corner, both tiles beside the corner are checked, so you can't see
        through a diagonal gap or graze a wall's corner. Symmetric: LOS(a, b)
        == LOS(b, a).

        `is_blocked` defaults to walls + out of bounds; World passes its own
        so barricades block sight too.
        """
        blocked = is_blocked if is_blocked is not None else (lambda t: not self.is_walkable(t))
        x, y = a
        dx, dy = b[0] - x, b[1] - y
        nx, ny = abs(dx), abs(dy)
        sx, sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)

        if blocked((x, y)):
            return False
        ix = iy = 0  # tile boundaries crossed so far on each axis
        while ix < nx or iy < ny:
            # Compare where the next x-boundary and next y-boundary are hit along
            # the segment: (0.5 + ix) / nx vs (0.5 + iy) / ny, cross-multiplied.
            decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
            if decision == 0:
                # exactly through a corner: both side tiles must be clear
                if blocked((x + sx, y)) or blocked((x, y + sy)):
                    return False
                x, y = x + sx, y + sy
                ix, iy = ix + 1, iy + 1
            elif decision < 0:
                x, ix = x + sx, ix + 1
            else:
                y, iy = y + sy, iy + 1
            if blocked((x, y)):
                return False
        return True
