"""Physics and environment constants for Carrom."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoardConfig:
    size: float = 1.0
    wall_thickness: float = 0.02
    pocket_radius: float = 0.06         # visual pocket size; also the wall gap at each corner
    # Pocket *detection* radius — larger than pocket_radius because walls with rounded endcaps
    # prevent a coin's centre from reaching within pocket_radius of the corner.  0.09 = coin
    # edge just touching the pocket rim, which is a realistic capture threshold.
    pocket_capture_radius: float = 0.09
    coin_radius: float = 0.03
    striker_radius: float = 0.035
    striker_offset: float = 0.08


# Physics constants
DT = 1.0 / 120.0
MAX_SIM_STEPS = 1800
SETTLE_VELOCITY = 0.02
MAX_FORCE = 6.0
STRIKER_MASS = 1.2
COIN_MASS = 0.9
FRICTION = 0.15        # contact friction at collision points (piece-piece, piece-wall)
ELASTICITY = 0.92      # coefficient of restitution for rubber cushions / polished wood
# Coulomb (kinetic) friction between piece and board surface (boric-acid-powdered wood, μ≈0.04).
# Expressed as deceleration in simulation units/s² so it is independent of piece mass.
BOARD_DECEL = 2.5

# Game constants
NUM_BLACK = 9
NUM_WHITE = 9
NUM_QUEEN = 1
MAX_COINS = NUM_BLACK + NUM_WHITE + NUM_QUEEN

AGENT_ID = "agent"
OPPONENT_ID = "opponent"
