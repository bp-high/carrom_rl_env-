"""GRPO training utilities for the Carrom environment (ICF rules).

Provides reward functions, prompt formatting, and rollout collection
compatible with TRL's GRPOTrainer.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional

from carrom_env.env import CarromEnv
from carrom_env.models import Action, Observation


# ---------------------------------------------------------------------------
# System prompt — ICF-aware
# ---------------------------------------------------------------------------

CARROM_SYSTEM_PROMPT = """\
You are a Carrom agent playing under ICF (International Carrom Federation) rules.

Board: 1.0 × 1.0 square centred at (0,0).  Pockets at the four corners (±0.5, ±0.5).
Your striker is on the BOTTOM baseline (y ≈ -0.42).

Colour assignment
-----------------
YOU play WHITE coins (+1 pt each).  Opponent plays BLACK coins.
Queen (red, centre) = +3 pts — must be "covered" by pocketing a white coin on the
same or next shot.

ICF rules to follow
-------------------
- Pocket a WHITE coin  → score +1, take another turn
- Pocket a BLACK coin  → DUE: coin returns to board centre, turn ENDS (no score)
- Pocket STRIKER       → FOUL: one pocketed coin returns to board, turn ends
- Miss (nothing own)   → turn passes to opponent

Respond with a JSON object and nothing else:
{"placement_x": <-0.4 to 0.4>, "angle": <radians, 0=straight>, "force": <0.0 to 1.0>}

Think step by step: identify reachable WHITE coins near pockets, then choose angle/force."""


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(obs: Observation) -> str:
    return f"<|system|>\n{CARROM_SYSTEM_PROMPT}\n<|user|>\n{obs.text_summary}\n<|assistant|>"


def format_chat_prompt(obs: Observation) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": CARROM_SYSTEM_PROMPT},
        {"role": "user",   "content": obs.text_summary},
    ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(response: str) -> Optional[Action]:
    text = response.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$",          "", text).strip()
    match = re.search(r"\{[^}]+\}", text)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        return Action(
            placement_x=float(data.get("placement_x", 0.0)),
            angle=float(data.get("angle", 0.0)),
            force=max(0.0, min(1.0, float(data.get("force", 0.5)))),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Reward functions for GRPO
# ---------------------------------------------------------------------------

def carrom_reward_fn(
    prompts: List[str],
    completions: List[str],
    env_rewards: List[float],
    **kwargs,
) -> List[float]:
    """Combined reward for GRPO: format + range + environment outcome."""
    rewards = []
    for prompt, completion, env_reward in zip(prompts, completions, env_rewards):
        reward = float(env_reward)
        action = parse_response(completion)

        if action is not None:
            reward += 0.2
            if -0.4 <= action.placement_x <= 0.4:      reward += 0.05
            if 0.1  <= action.force       <= 0.95:     reward += 0.05
            if -math.pi / 2 <= action.angle <= math.pi / 2: reward += 0.05
        else:
            reward -= 0.5

        rewards.append(reward)
    return rewards


def carrom_reward_for_trl(completions: List[Any], **kwargs) -> List[float]:
    """Reward function matching TRL GRPOTrainer's expected signature.

    Evaluates each completion by:
    1. Parsing the JSON action (+0.3 if valid, -0.5 if not)
    2. Checking parameter ranges (+0.1 each for placement, force, angle)
    3. Executing in a fresh env instance for the actual game reward
       — includes due-coin penalty from ICF rules

    Completions can be plain strings or chat message lists (TRL passes both).
    """
    rewards = []
    for completion in completions:
        # TRL may pass chat lists (vLLM path) or plain strings
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)

        reward = 0.0
        action = parse_response(text)

        if action is not None:
            reward += 0.3
            if -0.4 <= action.placement_x <= 0.4:           reward += 0.1
            if 0.15 <= action.force       <= 0.9:           reward += 0.1
            if -math.pi / 2 <= action.angle <= math.pi / 2: reward += 0.1

            try:
                env = CarromEnv(seed=hash(text) % 100_000)
                env.reset()
                _, env_reward, _, _, info = env.step(action)
                reward += env_reward
                if info.get("coin_potted", 0) > 0:
                    reward += 0.5
                # Additional signal for ICF due violations
                reward -= 0.3 * info.get("due_coins", 0)
            except Exception:
                pass
        else:
            reward -= 0.5

        rewards.append(reward)
    return rewards


# ---------------------------------------------------------------------------
# Helpers for offline rollout collection
# ---------------------------------------------------------------------------

def compute_env_reward(
    env: CarromEnv,
    response: str,
) -> tuple[float, bool, Observation]:
    action = parse_response(response)
    if action is None:
        obs = Observation(
            positions=[], velocities=[], pocketed=[],
            agent_score=env.agent_score,
            opponent_score=env.opponent_score,
            current_player="agent",
            remaining_coins=0,
            text_summary="Parse error.",
        )
        return -0.5, False, obs
    obs, reward, terminated, truncated, info = env.step(action)
    return reward, terminated or truncated, obs


def collect_rollouts(
    generate_fn,
    num_rollouts: int = 8,
    max_turns_per_episode: int = 15,
    seed: int = 0,
) -> Dict[str, List]:
    """Collect rollouts for offline GRPO training.

    Args:
        generate_fn: Callable(prompt: str) -> str.
        num_rollouts: Number of episodes (group size G in GRPO).
        max_turns_per_episode: Max steps per episode.
        seed: Base seed.
    """
    all_prompts, all_completions, all_rewards = [], [], []
    for rollout_idx in range(num_rollouts):
        env = CarromEnv(seed=seed + rollout_idx)
        obs = env.reset()
        for _ in range(max_turns_per_episode):
            prompt     = format_prompt(obs)
            completion = generate_fn(prompt)
            reward, done, obs = compute_env_reward(env, completion)
            all_prompts.append(prompt)
            all_completions.append(completion)
            all_rewards.append(reward)
            if done:
                break
    return {"prompts": all_prompts, "completions": all_completions, "rewards": all_rewards}
