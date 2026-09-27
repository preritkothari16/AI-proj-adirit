# Zombie Darwin — Project Context

## Project Goal
University AI project (2 devs, 4 weeks). Top-down wave-survival game on a 32×20 grid: 10 waves × 20 zombies, max 60 s per wave. Player moves, shoots, places barricades. Zombies evolve between waves via GA; Hill Climbing + Random Search are baselines. Goal: demonstrate AI syllabus concepts inside a playable game.

## Tech Stack
Python 3.10+, Pygame, NumPy, Matplotlib, pytest, Git/GitHub.

**Repo:** https://github.com/preritkothari16/AI-proj-adirit (branch `main`). Local `D:\PROJECTS\AI_Game` tracks `origin/main`.

## Architecture
Target below. Built so far: `config.py`, `requirements.txt`, `pyproject.toml` (pytest config), `.gitignore`, package skeleton (`game/`, `ai/`, `analytics/`, `tests/` with `__init__.py`), and the P1.2 contract files marked ✅. Rest not created yet.

```
zombie_darwin/
  main.py            # pygame app loop, screen states (menu/playing/between waves/game over), HumanController
  simulate.py        # headless CLI runner for experiments (addition to original target)
  config.py          # all constants
  game/
    engine.py        # World: pygame-free simulation, step(action)
    states.py        # ✅ zombie FSM: ZombieState enum + FSMInputs + next_state() stub
    map.py           # TileMap, ASCII loader, line of sight
    player.py        # ✅ Action, Controller protocol (Player entity itself: P2.2)
    zombie.py        # Zombie: FSM + steering + stats tracking
    bullet.py        # simple projectile
    barricade.py     # placement rules (BFS check)
    wave_manager.py  # spawn/timer/end/results
    renderer.py      # pygame drawing + debug overlay
  ai/
    pathfinding.py   # astar, greedy_best_first, is_reachable (BFS)
    genome.py        # ✅ Genome, ZombieStats, PathStrategy (random/repair/decode stubbed: P4.1)
    fitness.py       # ✅ FitnessStats (compute_fitness stubbed: P4.5)
    genetic.py       # Optimizer protocol, GeneticAlgorithm, RandomSearch
    hill_climbing.py # HillClimber
    kmeans.py        # stretch
    bot_player.py    # BotPlayer (Controller)
  analytics/
    logger.py        # RunLogger (CSV per generation)
    plots.py
  maps/level1.txt
  data/runs/
  tests/
```

## Roadmap
Milestones are themes; B's evolution track (M4 P4.1–4.4) runs in parallel with A's M1–M2.

- **M1 Foundation:** P1.1 skeleton+config · P1.2 data contracts · P1.3 TileMap · P1.4 pathfinding · P1.5 line of sight
- **M2 Playable:** P2.1 World · P2.2 Player · P2.3 renderer+main · P2.4 bullets · P2.5 barricades · P2.6 dummy zombie
- **M3 Zombie AI:** P3.1 FSM · P3.2 movement strategies · P3.3 vision/wander/search · P3.4 genome→zombie+swarm · P3.5 debug overlay
- **M4 Evolution:** P4.1 Genome · P4.2 GA operators · P4.3 toy convergence+Random · P4.4 HillClimber · P4.5 fitness · P4.6 WaveManager · P4.7 10-wave loop (**MVP**)
- **M5 Experiments:** P5.1 BotPlayer · P5.2 logger · P5.3 headless runner · P5.4 plots · P5.5 GA vs Random/HC · P5.6 pathfinding exp · P5.7 budget exp
- **M6 Submission:** P6.1 tuning · P6.2 visual polish · P6.3 README/report figures · P6.4 final sweep · P6.5 K-means (stretch)

Owners: **A** = game/engine/render, **B** = ai/evolution/analytics.
Checkpoints: IC1 Map↔Pathfinding · IC2 Barricade↔BFS · IC3 Genome↔Zombie · IC4 WaveManager↔GA (MVP) · IC5 Bot↔World headless · IC6 Runner↔Logger/Plots.

## Current Progress
### Completed
- Architecture audit; roadmap with phases agreed.
- P1.1 (lite) — skeleton, `config.py`, pytest wired up.
- P1.2 — frozen data contracts implemented as stubs, 13 interface tests passing.
- P1.3 — TileMap loader + `maps/level1.txt`, 31 tests passing (13 + 18 new).
- P1.4 — A*, Greedy Best-First, BFS `is_reachable`.
- P1.5 — Line of sight (supercover grid traversal); barricades block sight via `World.has_line_of_sight`.
- P2.1 — `World` (pygame-free, fixed tick).
- P2.2 — Player movement, wall collision, sliding.
- P2.3 — Pygame renderer + `main.py` loop + HumanController. Run: `python main.py`.
- P2.4 — Physical bullets with fire cooldown.
- P2.5 — Barricades: placement rules + BFS sealing check, right-click to place.
- P2.6 — Dummy zombie (fixed stats, walks straight at player), bullet damage, zombie death, contact attacks, player death / game over. **M2 playable.**
### In Progress
- None.
### Not Started
- P3.1 onward (M3), P4.x (B).

## Current Phase
M2 done → M3 Zombie AI. Next: **P3.1** (FSM `next_state`) for A; **P4.1** (Genome logic) for B.

## Important Interfaces
(Frozen in P1.2 — implemented, not stubs, unless noted. Change only by agreement.)

- `ai/genome.py`: `Genome(genes: np.ndarray)` shape (6,), floats [0,1], order = `GENE_NAMES` = speed, health, vision, aggression, path_strategy, swarm. `__post_init__` casts to `np.float64` and raises `ValueError` on wrong shape. `Genome.random(rng)`, `repair()`, `decode() -> ZombieStats` all raise `NotImplementedError` (bodies land in P4.1). `PathStrategy(Enum){DIRECT, GREEDY, ASTAR}`. `ZombieStats` frozen dataclass: `speed, max_hp, vision_radius, give_up_time, path_strategy, swarm_weight`.
- `ai/fitness.py`: `FitnessStats(damage_dealt=0.0, time_alive=0.0, min_dist_to_player=inf, reached_player=False)`. `compute_fitness(stats) -> float` raises `NotImplementedError` (body in P4.5).
- `game/states.py`: `ZombieState(Enum){WANDER, CHASE, SEARCH, ATTACK}`, `FSMInputs` frozen dataclass: `sees_player, dist_to_player, time_since_seen, give_up_time, attack_range`. `next_state(state, inputs) -> ZombieState` raises `NotImplementedError` (body in P3.1).
- `game/player.py`: `Body(pos, tile_size, speed, radius)` — shared base for Player and Zombie: `x, y, pos, speed, radius, move(direction, dt, is_blocked)`, `occupied_tiles()`; tunnelling guard raises `ValueError` if `speed*FIXED_DT >= tile_size`. `Action` frozen dataclass: `move=(0,0), aim=(0,0), shoot=False, barricade=None`. `Controller` is a `@runtime_checkable Protocol` with `get_action(self, world) -> Action`. `Player(Body)(pos, tile_size, speed=PLAYER_SPEED, max_hp=PLAYER_MAX_HP, radius=PLAYER_RADIUS)`: adds `hp, max_hp, alive, take_damage(amount)` (HP clamped at 0), fire cooldown; `move(direction, dt, is_blocked: Callable[[Tile], bool])` — normalises vectors longer than 1 (diagonals not faster), smaller magnitudes kept as-is.
- `game/map.py`: `TileMap` frozen dataclass — `TileMap.load(path, tile_size=32)`, `width, height, tile_size`, `walls: frozenset[Tile]`, `spawn_points: tuple[Tile, ...]`, `player_start: Tile`. `is_walkable(t)` (bounds + not a wall), `in_bounds(t)`, `neighbors(t)` (4-dir N/S/E/W, walkable only), `world_to_tile(pos)` (floor-div by `tile_size`), `tile_to_world(t)` (tile *centre*, not corner). `has_line_of_sight(a, b, is_blocked=None) -> bool` — tile-centre to tile-centre, every touched tile must be clear (endpoints included); `is_blocked` defaults to walls + OOB. `load()` raises `ValueError` on: ragged rows, unknown chars, 0 spawns, 0 or 2+ player starts. Tile = `tuple[int, int]` (col, row).
- `ai/pathfinding.py`: works on any `GraphLike` (structural `Protocol`, just needs `neighbors(tile) -> Iterable[Tile]`) — does **not** import `game.map`, satisfying the import-direction rule; `TileMap` satisfies it duck-typed. `manhattan(a, b) -> int`. `astar(graph, start, goal, stats=None) -> list[Tile] | None`, `greedy_best_first(graph, start, goal, stats=None) -> list[Tile] | None` — pass a `dict` as `stats` to get `stats['nodes_expanded']` after the call (added for P5.6 experiments; return type unchanged, so existing calls don't need it). `is_reachable(graph, starts: Iterable[Tile], goal) -> bool` — BFS, True only if **every** tile in `starts` reaches `goal` (used for barricade validation, P2.5).
- `game/barricade.py`: `check_placement(world, tile, max_barricades=MAX_BARRICADES) -> PlacementResult` (pure, no mutation). `PlacementResult(Enum){OK, INVALID_TILE (OOB/wall/spawn), OCCUPIED (already barricaded / under player's or any zombie's hitbox), LIMIT_REACHED, BLOCKS_PATH}` — checks run in that order. `all_spawns_connected(world)` helper.
- `game/zombie.py`: `Zombie(Body)(pos, tile_size, speed, max_hp, radius, attack_damage)` — defaults from `ZOMBIE_*` config (temporary fixed stats; genome-decoded in P3.4). `hp, max_hp, alive, take_damage()`, `damage_dealt` (for fitness), `in_contact(player)`, `update(dt, player, is_blocked)`: if in contact → hit when attack cooldown is 0, else move straight at player.
- `game/engine.py`: `World(tilemap, rng)` — `rng` must be `np.random.Generator`. Attributes: `tilemap`, `rng`, `player: Player` (spawned at centre of `player_start`), `zombies: list[Zombie]`, `bullets: list[Bullet]`, `barricades: set[Tile]`, `tick`, `last_action`, `last_placement: PlacementResult | None`. `time` = `tick * FIXED_DT`. `is_blocked(tile)` = wall, out of bounds, or barricade. `neighbors(tile)` = 4-dir unblocked tiles, so **World is a `GraphLike`** — zombies path over `world`, not `tilemap`. `place_barricade(tile) -> PlacementResult`. `spawn_zombie(tile) -> Zombie`. `has_line_of_sight(a, b)` = tilemap LOS with `is_blocked` (barricades block sight). `game_over` = player HP 0. `step(action)`: no-op once `game_over`; else place barricade if `action.barricade` → move player → fire if `action.shoot` and cooldown ready (toward `action.aim`, a world-pixel point; no shot if aim == player centre) → update bullets (segment-vs-circle hit test on each bullet's per-tick path; first zombie hit takes `bullet.damage`, bullet dies) → drop dead bullets and zombies → zombies update (move/attack) → tick += 1.
- `game/wave_manager.py`: `start_wave(genomes)`, `update(world)`, `is_over()`, `results() -> list[(Genome, FitnessStats)]`.
- `ai/genetic.py`: `Optimizer.next_generation(genomes, fitnesses) -> list[Genome]`; `GeneticAlgorithm`, `RandomSearch`. `ai/hill_climbing.py`: `HillClimber`. All take `rng`.
- `analytics/logger.py`: `RunLogger(path).log_generation(gen, genomes, fitnesses)`.

## Architectural Decisions
- **Only `main.py` and `game/renderer.py` import pygame.** Why: headless sim + tests need no window.
- **Import direction: `ai/` pure modules (pathfinding, genome, fitness, genetic, hill_climbing) never import `game/`.** Only `ai/bot_player.py` may. Why: avoids circular imports, lets B test in isolation.
- **Fixed timestep (FIXED_DT = 1/60) + seeded `np.random.Generator` passed in.** Why: reproducible, headless == rendered.
- **Human and bot share `Controller -> Action`.** Why: BotPlayer is drop-in.
- **Genes in [0,1]; speed+health+vision share a fixed stat budget (`repair()`).** Why: forces trade-offs; otherwise GA maxes everything.
- **Common `Optimizer` interface; HC = 20 parallel hill climbers; RandomSearch baseline.** Why: fair comparison, same evaluation budget.
- **`states.py` = zombie FSM (pure function), not screen states.** Screen states live in `main.py`. Why: FSM unit-testable without World.
- **`engine.py` holds `World` (logic), not the pygame loop.** Why: keeps pygame out of simulation.
- **Bullets = simple fast projectiles (`bullet.py`), not hitscan.** Why: matches target structure; circle collision is simple and visible.
- **Barricades indestructible, max `MAX_BARRICADES`=5; rejected if BFS (with the candidate added) finds any spawn cut off from the player's *current* tile. Pathfinding treats them as fully blocked (not high-cost) via `World.neighbors`; they also block bullets. Spawn tiles and tiles under the player can't be barricaded.** Why: simplest valid rule; blocked-not-costly keeps A*/Greedy unchanged, and the BFS check guarantees a path always exists.
- **Zombies repath on ~0.5 s timer or target tile change.** Why: performance in headless runs.
- **Swarm gene = one steering weight toward nearby allies' centroid.** Why: meaningful without full boids.
- **"Budget experiments" = sweep stat-budget size.** Why: tests how trade-off pressure changes evolved strategies.
- **`simulate.py` added at root** as headless CLI. Why: target had no home for the runner.
- **`ai/pathfinding.py` takes a structural `GraphLike` Protocol, not `TileMap` directly.** Why: enforces the "ai/ never imports game/" rule without losing type hints; `TileMap` satisfies it because it already has `neighbors()`.
- **Bullets are points moving each tick, sub-stepped every ≤ tile/4 px; die on blocked tile (walls + barricades) or leaving the map. Fire cooldown counted in whole ticks (`Player.cooldown_remaining`, 15 ticks = 0.25 s).** Why: no hitscan per plan; sub-stepping stops tunnelling; tick counters avoid float timing drift.
- **`main.py` uses a fixed-timestep accumulator (max 5 ticks/frame) around `world.step()`.** Why: simulation always advances in FIXED_DT regardless of real frame time, so play matches headless runs. `Renderer` pre-draws the static map once and only reads World. No camera: world px == screen px (1024×640).
- **Player collision = axis-aligned square (half-width `PLAYER_RADIUS`), resolved X then Y and clamped flush to the tile edge.** Why: simplest correct tile collision; per-axis resolution gives wall sliding for free. `Player.move` takes an `is_blocked` callable so it never needs World or pygame, and barricades block via `World.is_blocked`.
- **Attack rule (P2.6): contact = gap between player and zombie circles ≤ `ZOMBIE_ATTACK_REACH` (4 px). In contact the zombie stops and deals `ZOMBIE_ATTACK_DAMAGE`=10 every `ZOMBIE_ATTACK_COOLDOWN`=1 s (tick-counted). Zombies don't collide with each other or push the player.** Why: simplest rule that makes the player killable; later FSM ATTACK state reuses it. Dummy stats: speed 90, HP 50 (2 bullets), radius 10.
- **Line of sight = exact-integer supercover traversal between tile centres; at an exact corner both side tiles must be clear (no seeing through diagonal gaps or grazing wall corners). Barricades block sight.** Why: exact (no float edge cases), symmetric, conservative; consistent with barricades blocking bullets. Vision (P3.3) converts pixel positions with `world_to_tile` first.
- **Bullet↔zombie hits use the swept segment of each tick, not the end point.** Why: bullets can't skip zombies regardless of speed. One zombie per bullet (no piercing).
- **Game over freezes `World.step`.** Why: headless runs and the renderer can just check `world.game_over`. `main.py` has a temporary demo spawner (one zombie per spawn every 6 s) until WaveManager (P4.6).
- **`stats: dict | None` out-param on `astar`/`greedy_best_first` for `nodes_expanded`**, instead of changing the return type. Why: keeps the frozen `list[Tile] | None` signature; P5.6 experiments opt in by passing a dict.

## Known Bugs / Issues
- None. Open: fitness weights TBD — tune in P6.1. `STAT_BUDGET` in `config.py` is a placeholder value.

## Tests / Verification
- `tests/test_contracts.py` — 13 tests. Import/shape/type checks only; `NotImplementedError` stubs asserted, not implemented — real behavior tests come with P3.1/P4.1/P4.5.
- `tests/test_map.py` — 17 tests: 5x5 fixture map (load, walls, bounds, spawns/player, 4-dir neighbors, world↔tile conversion), malformed-map rejection, and `maps/level1.txt` (32×20 dims, border solid, ≥4 spawns + 1 player start, all spawns BFS-reachable from player start).
- `tests/test_pathfinding.py` — 21 tests: Manhattan heuristic, A* on a hand-verified forced-corridor map (unique path, exact tile sequence asserted), A* optimality on an open grid, Greedy completeness + "never shorter than A*" invariant (checked from every real spawn to player_start), `nodes_expanded` stats plumbing, blocked/sealed-goal → `None`, `is_reachable` (same-corridor True, split-corridor False, mixed starts False, real map True from every spawn), and a check that the module never does `import pygame`.
- `tests/test_engine.py` — 8 tests: initial state, one step = one tick, 600 headless steps, no time drift over 3600 steps, last_action recorded, type checks, and a subprocess check that importing `game.engine` never loads pygame.
- `tests/test_player.py` — 15 tests: HP from config, spawn position, configured speed, diagonal not faster, walls in all 4 directions (exact flush stop), barricade blocks, random 2000-tick walk never overlaps a blocked tile, diagonal sliding, corner stop, fits 1-tile corridor, tunnelling guard, movement works without World.
- `tests/test_renderer.py` — 9 tests (SDL dummy driver, no window): WASD→move mapping, HumanController satisfies Controller, screen size, walls/player pixels drawn, `main.py --max-frames 30` exits cleanly, key-derived Action moves player, bullets, barricades, zombies + game-over overlay drawn.
- `tests/test_bullet.py` — 15 tests: speed per tick, normalised direction, wall kill (stops inside wall tile), fast bullet can't skip a wall, out-of-map kill (both sides), fire toward aim, no fire without shoot / at own centre, 4 shots/s cooldown, cleanup after hitting walls, barricades stop bullets, 60 s of constant fire stays bounded.
- `tests/test_barricade.py` — 21 tests: valid placement via `place_barricade` and `Action`, check doesn't mutate; rejections (wall, OOB, spawn, duplicate, player tile incl. straddled tile, zombie tile, limit); sealing a 1-tile corridor rejected, dead-end spur allowed, 2 routes → only one closable, level1 room doorways can't both be closed, target follows player; `World.neighbors` skips barricades, A*/Greedy route around, barricade blocks movement.
- `tests/test_zombie.py` — 15 tests: spawn, direct movement at speed, stops on contact without overlapping, stalled by wall (no pathfinding), bullet damage, death + removal at 0 HP, HP never negative, bullet hits only first zombie in line, miss = no damage, swept hit test catches fast bullets, contact damage + `damage_dealt`, attack cooldown, no damage at range, player death freezes world, dead zombie stops attacking, end-to-end shoot approaching zombie.
- `tests/test_line_of_sight.py` — 12 tests: same/adjacent/straight/diagonal, wall between, blocked/OOB endpoints, exact tile set for a shallow line, diagonal-gap and corner-graze blocked, symmetry (2000 random level1 pairs), visible ⇒ dense float samples all clear, level1 room hides player from spawns, World LOS blocked by barricade.
- **147/147 passing** (`python -m pytest -q`).

## Next Tasks
1. A → P3.1 FSM `next_state`, then P3.2 movement strategies (path over `world`, repath ~0.5 s).
2. B → P4.1 Genome logic (`random`/`repair`/`decode`).
3. A → P3.3 vision uses `World.has_line_of_sight` + `vision_radius`.

## Last Session Summary
Committed + pushed P2.3–P2.6 (e63dc78). Implemented P1.5: `TileMap.has_line_of_sight(a, b, is_blocked=None)` (integer supercover traversal, corner-conservative, symmetric) and `World.has_line_of_sight` (barricades block). Removed the NotImplementedError stub test, added 12 LOS tests; 147/147 passing. P1.5 committed + pushed.
