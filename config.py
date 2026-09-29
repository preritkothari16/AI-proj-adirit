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

# --- Zombies ---
# Per-zombie stats come from its genome (Genome.decode, P3.4). Each gene in [0, 1]
# maps linearly onto its range; an all-0.5 genome gives the old P2.6 dummy
# (speed 90, 50 HP, 256 px vision, 3 s give-up).
ZOMBIE_SPEED_RANGE = (40.0, 140.0)  # px/s; player is 150
ZOMBIE_HP_RANGE = (10.0, 90.0)  # 1 to 4 bullets
ZOMBIE_VISION_RANGE = (128.0, 384.0)  # px, 4 to 12 tiles
ZOMBIE_GIVE_UP_RANGE = (1.0, 5.0)  # s; decoded from the aggression gene
ZOMBIE_SWARM_WEIGHT_MAX = 1.0  # swarm gene 1 -> pull toward allies as strong as toward the target
ZOMBIE_SWARM_RADIUS = 96.0  # px (3 tiles); allies closer than this count for swarming

# Shared by all zombies (not evolved)
ZOMBIE_RADIUS = 10.0  # pixels; same square collision as the player
ZOMBIE_ATTACK_DAMAGE = 10.0  # per hit on the player
ZOMBIE_ATTACK_COOLDOWN = 1.0  # seconds between hits from one zombie
ZOMBIE_REPATH_INTERVAL = 0.5  # seconds between Greedy/A* re-plans (also re-plans when the target tile changes)
ZOMBIE_ATTACK_REACH = 4.0  # pixels of gap allowed between bodies that still counts as contact
ZOMBIE_WANDER_RETARGET = 6.0  # seconds before a wanderer picks a new destination even if not there yet

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
MUTATION_SIGMA = 0.10  # std-dev of the Gaussian added to a mutated gene (genes live in [0, 1])

# --- Fitness (P4.5; weights are placeholders, tune in P6.1) ---
# fitness = sum(weight * term), every term in [0, 1] -> fitness in [0, sum of weights] (= 1 with these).
# damage    = damage_dealt / FITNESS_DAMAGE_REF            (capped at 1)
# proximity = 1 - min_gap_to_player / FITNESS_DISTANCE_REF (capped to [0, 1]; never approached = 0)
# survival  = time_alive / WAVE_TIME_LIMIT                 (capped at 1)
# reached   = 1 if the zombie ever touched the player else 0
FITNESS_W_DAMAGE = 0.5
FITNESS_W_PROXIMITY = 0.25
FITNESS_W_SURVIVAL = 0.1
FITNESS_W_REACHED = 0.15
FITNESS_DAMAGE_REF = PLAYER_MAX_HP  # HP; total damage that kills the player
FITNESS_DISTANCE_REF = GRID_WIDTH * TILE_SIZE  # px: map width; a gap this big or more scores 0 proximity

# --- Genome stat budget (placeholder, tune in P6.1) ---
# speed + health + vision genes are rescaled to sum to this by Genome.repair() (random, mutation, crossover),
# forcing trade-offs instead of every zombie maxing all stats.
STAT_BUDGET = 1.5
