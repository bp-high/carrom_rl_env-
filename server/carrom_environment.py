"""Server-side CarromEnvironment implementing the OpenEnv Environment interface."""

from __future__ import annotations

from typing import Optional

from openenv.core.env_server.interfaces import Environment

from carrom_env.env import CarromEnv
from carrom_env.models import Action, Observation


class CarromEnvironment(Environment):
    """OpenEnv-compatible server environment for Carrom.

    Wraps CarromEnv and returns Observation objects with reward/done
    fields, matching the openenv ``create_app`` expectations.
    """

    def __init__(self) -> None:
        self._env = CarromEnv()

    def reset(self, seed: Optional[int] = None, **kwargs) -> Observation:
        obs = self._env.reset(seed=seed)
        obs.reward = 0.0
        obs.done = False
        return obs

    def step(self, action: Action, **kwargs) -> Observation:
        obs, reward, terminated, truncated, info = self._env.step(action)
        obs.reward = reward
        obs.done = terminated or truncated
        return obs

    # ------------------------------------------------------------------
    # Split-turn API (used by the visual Gradio UI)
    # ------------------------------------------------------------------

    def step_agent(self, action: Action) -> Observation:
        """Agent turn only — does not auto-play opponent."""
        obs, reward, terminated, truncated, info = self._env.step_agent(action)
        obs.reward = reward
        obs.done = terminated or truncated
        return obs

    def step_agent_animated(self, action: Action):
        """Agent turn with animation snapshots. Returns (obs, snapshots)."""
        obs, reward, terminated, truncated, info = self._env.step_agent_animated(action)
        obs.reward = reward
        obs.done = terminated or truncated
        return obs, info.get("snapshots", [])

    def step_opponent(self):
        """Opponent auto-reply. No-op if not opponent's turn.

        Returns (obs, opp_action) where opp_action is the Action the
        opponent chose, or None if no turn was played.
        """
        obs, reward, terminated, truncated, info = self._env.step_opponent()
        obs.reward = reward
        obs.done = terminated or truncated
        return obs, info.get("opp_action")

    def step_opponent_animated(self):
        """Opponent turn with animation snapshots.
        Returns (obs, opp_action, snapshots)."""
        obs, reward, terminated, truncated, info = self._env.step_opponent_animated()
        obs.reward = reward
        obs.done = terminated or truncated
        return obs, info.get("opp_action"), info.get("snapshots", [])

    def get_opponent_action(self):
        """Peek at what action the opponent *would* play (without executing)."""
        return self._env._opponent_action()

    @property
    def needs_opponent_turn(self) -> bool:
        return self._env.needs_opponent_turn

    @property
    def state(self):
        return self._env.state()

    def close(self) -> None:
        pass
