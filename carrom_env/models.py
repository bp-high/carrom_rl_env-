"""Typed models for OpenEnv Carrom."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action model — supports both raw numeric and LLM-friendly text actions
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """Carrom shot action.

    For LLM agents, set ``action_type="text"`` and provide a natural-language
    ``text`` description such as ``"aim at coin black_inner_0 with medium force
    from center"``.  The environment will parse it into numeric parameters.

    For numeric agents, set ``action_type="numeric"`` (default) and provide
    ``placement_x``, ``angle``, and ``force`` directly.
    """

    action_type: str = Field(
        "numeric",
        description='Either "numeric" (default) or "text".  '
        'When "text", the environment parses the ``text`` field into shot parameters.',
    )
    # --- numeric fields (used when action_type == "numeric") ---
    placement_x: float = Field(0.0, description="Striker x placement along baseline [-0.4, 0.4].")
    angle: float = Field(0.0, description="Shot angle in radians. 0 points toward +y.")
    force: float = Field(0.5, description="Normalized shot force in [0, 1].")
    spin: Optional[float] = Field(0.0, description="Optional spin value; currently unused.")
    shot_id: Optional[int] = Field(None, description="Client-provided shot identifier.")

    # --- text field (used when action_type == "text") ---
    text: Optional[str] = Field(
        None,
        description='Natural-language shot description, e.g. '
        '"aim at coin black_inner_0 with strong force from the left side".',
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class CoinInfo(BaseModel):
    """Per-coin info exposed in the text observation for LLM readability."""

    coin_id: str
    color: str
    x: float
    y: float
    pocketed: bool
    nearest_pocket: str = Field(description="Which pocket this coin is closest to.")
    pocket_distance: float = Field(description="Distance to the nearest pocket.")


class Observation(BaseModel):
    positions: List[List[float]] = Field(..., description="[N,2] positions in board coordinates.")
    velocities: List[List[float]] = Field(..., description="[N,2] velocities.")
    pocketed: List[bool] = Field(..., description="[N] pocketed flags.")
    agent_score: int
    opponent_score: int
    current_player: str
    remaining_coins: int
    turn_number: int = 0
    max_turns: int = 200
    coins: List[CoinInfo] = Field(default_factory=list, description="Per-coin detail for LLM agents.")
    text_summary: str
    reward: float = Field(0.0, description="Reward for this step.")
    done: bool = Field(False, description="Whether the episode is finished.")


class PieceState(BaseModel):
    piece_id: str
    kind: str
    color: str
    position: List[float]
    velocity: List[float]
    pocketed: bool


class State(BaseModel):
    rng_seed: Optional[int]
    step_count: int
    current_player: str
    board_size: float
    pieces: List[PieceState]
    agent_score: int
    opponent_score: int
