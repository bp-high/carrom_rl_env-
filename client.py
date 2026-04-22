"""Carrom environment client."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult, StateT

from carrom_env.models import Action, Observation


class CarromEnv(EnvClient["Action", "Observation", StateT]):
    """Client for connecting to a Carrom Environment server.

    Example (async)::

        async with CarromEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            print(result.observation.text_summary)
            result = await env.step(Action(placement_x=0.0, angle=0.1, force=0.6))
            print(result.reward)

    Example (sync)::

        with CarromEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset()
            result = env.step(Action(placement_x=0.0, angle=0.1, force=0.6))
    """

    def _step_payload(self, action: Action) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Observation]:
        obs = Observation(**payload)
        return StepResult(
            observation=obs,
            reward=obs.reward if isinstance(obs.reward, (int, float)) else 0.0,
            done=obs.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> Any:
        from openenv.core.env_server.types import State

        return State(**payload)
