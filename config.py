"""Central constants for Zombie Darwin. No other module should hard-code these values."""

# --- Map / grid ---
GRID_WIDTH = 32
GRID_HEIGHT = 20
TILE_SIZE = 32  # pixels per tile, rendering only

# --- Simulation ---
FIXED_DT = 1.0 / 60.0  # seconds per tick, used for both headless and rendered runs

# --- Player ---
PLAYER_SPEED = 150.0  # pixels per second
PLAYER_MAX_HP = 100.0
PLAYER_RADIUS = 10.0  # pixels; collision uses a square of side 2*radius

# --- Waves ---
NUM_WAVES = 10
ZOMBIES_PER_WAVE = 20
WAVE_TIME_LIMIT = 60.0  # seconds

# --- Genetic Algorithm ---
POPULATION_SIZE = 20
TOURNAMENT_SIZE = 3
ELITISM_COUNT = 2
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.10  # per-gene probability

# --- Genome stat budget (placeholder, tune in P6.1) ---
# speed + health + vision genes are rescaled to sum to this after mutation/crossover,
# forcing trade-offs instead of every zombie maxing all stats.
STAT_BUDGET = 1.5
