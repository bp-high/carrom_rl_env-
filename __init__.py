"""OpenEnv Carrom — a physics-based RL environment for the board game Carrom.

Features real Newtonian physics (friction, elasticity, collisions),
LLM-friendly text actions, and Green Agent efficiency tracking.
"""

from carrom_env.models import Action, Observation
from client import CarromEnv

__all__ = ["Action", "Observation", "CarromEnv"]
