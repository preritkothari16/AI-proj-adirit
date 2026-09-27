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

# --- Bullets ---
BULLET_SPEED = 600.0  # pixels per second (10 px per tick)
BULLET_RADIUS = 3.0  # pixels, used for drawing and (P2.6) zombie hits
BULLET_DAMAGE = 25.0  # applied to zombies in P2.6
FIRE_COOLDOWN = 0.25  # seconds between shots

# --- Dummy zombie (P2.6; fixed stats until genome decoding in P3.4) ---
ZOMBIE_SPEED = 90.0  # pixels per second, slower than the player
ZOMBIE_MAX_HP = 50.0  # two bullets
ZOMBIE_RADIUS = 10.0  # pixels; same square collision as the player
ZOMBIE_ATTACK_DAMAGE = 10.0  # per hit on the player
ZOMBIE_ATTACK_COOLDOWN = 1.0  # seconds between hits from one zombie
ZOMBIE_ATTACK_REACH = 4.0  # pixels of gap allowed between bodies that still counts as contact

# --- Barricades ---
MAX_BARRICADES = 5  # indestructible; placement rejected if it would seal a spawn off

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
