"""Zombie entity: movement (P3.2), perception + FSM (P3.3), genome-driven stats + swarm (P3.4).

Zombie(zid, genome, pos) decodes its genome once (Genome.decode) into speed,
max_hp, vision_radius, give_up_time (aggression), path_strategy and
swarm_weight. Radius, attack damage/cooldown and reach are shared constants.

Each tick a zombie:
  1. perceives: sees the player if within `vision_radius` px (centre to centre)
     AND World line of sight between their tiles is clear (walls and barricades
     block). Seeing updates `last_known_pos` and resets the time since seen.
  2. runs the pure FSM (game.states.next_state) on those inputs.
  3. acts on the new state:
       WANDER  walk to a random floor tile (world.rng); pick a new one on
               arrival, when unreachable, or after ZOMBIE_WANDER_RETARGET s.
       CHASE   navigate to the player.
       SEARCH  navigate to the last known player position, then wait there
               until the FSM gives up (give_up_time) and returns to WANDER.
       ATTACK  stand still, hit once per ZOMBIE_ATTACK_COOLDOWN.

Navigation uses `path_strategy`:
    DIRECT  walk straight at the target; walls can stall it (no planning).
    GREEDY  follow a Greedy Best-First tile path (fast, not always shortest).
    ASTAR   follow an A* tile path (shortest).
Greedy/A* paths are lists of tile centres over the World graph (barricades
count as walls). A path is re-planned every ZOMBIE_REPATH_INTERVAL seconds and
immediately when the target tile changes. On the target tile it closes the
last pixels directly.

Swarm: while moving (WANDER/CHASE/SEARCH) the navigation direction gets
swarm_weight * (unit vector toward the centroid of living allies within
ZOMBIE_SWARM_RADIUS px) added to it. swarm_weight 0 -> no swarm force at all.
Not applied while attacking. A waypoint counts as reached within the swarm
pull's length of one step, so the sideways pull can't stall path following.

Attack rule: "in contact" = gap between the zombie's and the player's circles
is at most ZOMBIE_ATTACK_REACH px. That gap is the FSM's dist_to_player and
ZOMBIE_ATTACK_REACH its attack_range, so ATTACK == in contact.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

from ai.genome import Genome, PathStrategy
from ai.pathfinding import astar, greedy_best_first
from config import (
    FIXED_DT,
    TILE_SIZE,
    ZOMBIE_ATTACK_COOLDOWN,
    ZOMBIE_ATTACK_DAMAGE,
    ZOMBIE_ATTACK_REACH,
    ZOMBIE_RADIUS,
    ZOMBIE_REPATH_INTERVAL,
    ZOMBIE_SWARM_RADIUS,
    ZOMBIE_WANDER_RETARGET,
)
from game.player import Body, Player
from game.states import FSMInputs, ZombieState, next_state

if TYPE_CHECKING:
    from game.engine import World

Tile = Tuple[int, int]
Vec = Tuple[float, float]

_ARRIVE_EPS = 0.5  # px; closer than this to a waypoint centre counts as reached

_SEARCHES = {PathStrategy.GREEDY: greedy_best_first, PathStrategy.ASTAR: astar}


class Zombie(Body):
    def __init__(self, zid: int, genome: Genome, pos: Vec, tile_size: int = TILE_SIZE):
        stats = genome.decode()
        super().__init__(pos, tile_size, stats.speed, ZOMBIE_RADIUS)
        self.zid = zid
        self.genome = genome
        self.stats = stats
        self.max_hp = stats.max_hp
        self.hp = stats.max_hp
        self.attack_damage = ZOMBIE_ATTACK_DAMAGE
        self.swarm_weight = stats.swarm_weight
        self.attack_cooldown_ticks = max(1, round(ZOMBIE_ATTACK_COOLDOWN / FIXED_DT))
        self.attack_cooldown_remaining = 0
        self.damage_dealt = 0.0  # feeds FitnessStats later (P4.5)

        # perception + FSM
        self.vision_radius = stats.vision_radius
        self.give_up_time = stats.give_up_time
        self.state = ZombieState.WANDER
        self.last_known_pos: Optional[Vec] = None  # player position at last sighting
        self.ticks_since_seen: Optional[int] = None  # None = never seen
        self.wander_target: Optional[Tile] = None
        self.wander_ticks_left = 0
        self.wander_retarget_ticks = max(1, round(ZOMBIE_WANDER_RETARGET / FIXED_DT))

        # navigation
        self.path_strategy = stats.path_strategy
        self.path: List[Tile] = []  # remaining waypoints (tiles), next one first
        self.path_goal: Optional[Tile] = None  # tile the current path was planned to
        self.repath_ticks = max(1, round(ZOMBIE_REPATH_INTERVAL / FIXED_DT))
        self.repath_remaining = 0  # 0 = plan on the next update
        self.repaths = 0  # how many searches ran (for P5.6 experiments)
        self.nodes_expanded = 0  # total search effort (for P5.6 experiments)
        self._swarm: Vec = (0.0, 0.0)  # this tick's swarm force, set in update()

    @property
    def alive(self) -> bool:
        return self.hp > 0.0

    def take_damage(self, amount: float) -> None:
        self.hp = max(0.0, self.hp - amount)

    def gap_to(self, player: Player) -> float:
        return math.hypot(player.x - self.x, player.y - self.y) - self.radius - player.radius

    def in_contact(self, player: Player) -> bool:
        return self.gap_to(player) <= ZOMBIE_ATTACK_REACH

    @property
    def time_since_seen(self) -> float:
        return math.inf if self.ticks_since_seen is None else self.ticks_since_seen * FIXED_DT

    # --- perception ---

    def sees_player(self, world: "World") -> bool:
        p = world.player
        if math.hypot(p.x - self.x, p.y - self.y) > self.vision_radius:
            return False
        tm = world.tilemap
        return world.has_line_of_sight(tm.world_to_tile(self.pos), tm.world_to_tile(p.pos))

    def _perceive(self, world: "World") -> FSMInputs:
        sees = self.sees_player(world)
        if sees:
            self.last_known_pos = world.player.pos
            self.ticks_since_seen = 0
        elif self.ticks_since_seen is not None:
            self.ticks_since_seen += 1
        return FSMInputs(
            sees_player=sees,
            dist_to_player=self.gap_to(world.player),
            time_since_seen=self.time_since_seen,
            give_up_time=self.give_up_time,
            attack_range=ZOMBIE_ATTACK_REACH,
        )

    # --- swarm ---

    def swarm_force(self, world: "World") -> Vec:
        """swarm_weight * unit vector toward nearby living allies' centroid; (0, 0) if none."""
        if self.swarm_weight <= 0.0:
            return (0.0, 0.0)
        near = [
            z
            for z in world.zombies
            if z is not self and z.alive and math.hypot(z.x - self.x, z.y - self.y) <= ZOMBIE_SWARM_RADIUS
        ]
        if not near:
            return (0.0, 0.0)
        cx = sum(z.x for z in near) / len(near) - self.x
        cy = sum(z.y for z in near) / len(near) - self.y
        length = math.hypot(cx, cy)
        if length == 0.0:
            return (0.0, 0.0)
        return (self.swarm_weight * cx / length, self.swarm_weight * cy / length)

    # --- per-tick update ---

    def update(self, dt: float, world: "World") -> None:
        if not self.alive:
            return
        self._swarm = self.swarm_force(world)
        if self.attack_cooldown_remaining > 0:
            self.attack_cooldown_remaining -= 1

        previous = self.state
        self.state = next_state(self.state, self._perceive(world))
        if self.state is ZombieState.WANDER and previous is not ZombieState.WANDER:
            self.wander_target = None  # fresh destination after giving up

        if self.state is ZombieState.ATTACK:
            self._attack(world.player)
        elif self.state is ZombieState.CHASE:
            p = world.player
            self._navigate(dt, world, world.tilemap.world_to_tile(p.pos), p.pos)
        elif self.state is ZombieState.SEARCH:
            self._search(dt, world)
        else:
            self._wander(dt, world)

    def _attack(self, player: Player) -> None:
        if self.attack_cooldown_remaining == 0 and player.alive:
            player.take_damage(self.attack_damage)
            self.damage_dealt += self.attack_damage
            self.attack_cooldown_remaining = self.attack_cooldown_ticks

    def _search(self, dt: float, world: "World") -> None:
        if self.last_known_pos is None:
            return  # can't happen via the FSM (SEARCH needs a prior CHASE), but stay put if forced
        goal = world.tilemap.world_to_tile(self.last_known_pos)
        self._navigate(dt, world, goal, self.last_known_pos)

    def _wander(self, dt: float, world: "World") -> None:
        tm = world.tilemap
        here = tm.world_to_tile(self.pos)
        if self.wander_ticks_left > 0:
            self.wander_ticks_left -= 1
        if self.wander_target is None or here == self.wander_target or self.wander_ticks_left == 0:
            self.wander_target = self._pick_wander_target(world)
            self.wander_ticks_left = self.wander_retarget_ticks
        if not self._navigate(dt, world, self.wander_target, tm.tile_to_world(self.wander_target)):
            self.wander_target = None  # unreachable: pick another next tick

    def _pick_wander_target(self, world: "World") -> Tile:
        tm = world.tilemap
        while True:  # the map always has floor (player start), so this terminates
            tile = (int(world.rng.integers(tm.width)), int(world.rng.integers(tm.height)))
            if not world.is_blocked(tile):
                return tile

    # --- navigation ---

    def _navigate(self, dt: float, world: "World", goal: Tile, final_pos: Vec) -> bool:
        """Move one tick toward `goal` (a tile); on that tile, head for `final_pos`.
        Returns False only if a planner found no route."""
        if self.path_strategy is PathStrategy.DIRECT:
            self._step_toward(final_pos, dt, world)
            return True

        if self.repath_remaining > 0:
            self.repath_remaining -= 1
        if self.repath_remaining == 0 or goal != self.path_goal:
            if not self._plan(world, goal):
                return False

        if not self.path:
            self._step_toward(final_pos, dt, world)  # already on the goal tile
            return True

        tx, ty = world.tilemap.tile_to_world(self.path[0])
        self._step_toward((tx, ty), dt, world)
        arrive = _ARRIVE_EPS + math.hypot(*self._swarm) * self.speed * dt
        if math.hypot(tx - self.x, ty - self.y) <= arrive:
            self.path.pop(0)
        return True

    def _step_toward(self, target: Vec, dt: float, world: "World") -> None:
        step = self.speed * dt
        # Scale so a target closer than one step is reached exactly instead of overshot
        # (Body.move only normalises vectors longer than 1).
        dx, dy = (target[0] - self.x) / step, (target[1] - self.y) / step
        length = math.hypot(dx, dy)
        if length > 1.0:
            dx, dy = dx / length, dy / length
        sx, sy = self._swarm  # (0, 0) unless the genome swarms and allies are near
        self.move((dx + sx, dy + sy), dt, world.is_blocked)

    def _plan(self, world: "World", goal: Tile) -> bool:
        start = world.tilemap.world_to_tile(self.pos)
        stats: dict = {}
        path = _SEARCHES[self.path_strategy](world, start, goal, stats)
        self.repaths += 1
        self.nodes_expanded += stats.get("nodes_expanded", 0)
        self.path_goal = goal
        self.repath_remaining = self.repath_ticks
        # path[0] is the tile we're already in; head straight for the next centre
        # (wall sliding absorbs any off-centre start).
        self.path = path[1:] if path else []
        return path is not None
