"""
WebSocket client for the Clinical Triage Gym.

Provides a typed client that connects to the TriageEnvironment server
and handles serialization/deserialization of actions and observations.
"""

from typing import Any, Dict

from openenv.core.client_types import StateT, StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from .models import TriageAction, TriageObservation


class TriageEnv(EnvClient[TriageAction, TriageObservation, State]):
    """Typed WebSocket client for the Clinical Triage Gym.

    Example (async):
        >>> async with TriageEnv(base_url="ws://localhost:8000") as env:
        ...     result = await env.reset(task_id="easy")
        ...     result = await env.step(TriageAction(action_type="assess"))
        ...     result = await env.step(TriageAction(
        ...         action_type="triage", urgency=4, specialty="orthopedics"
        ...     ))

    Example (sync):
        >>> env = TriageEnv(base_url="ws://localhost:8000").sync()
        >>> with env:
        ...     result = env.reset(task_id="easy")
        ...     result = env.step(TriageAction(action_type="triage", urgency=4, specialty="orthopedics"))
    """

    def _step_payload(self, action: TriageAction) -> Dict[str, Any]:
        """Convert a TriageAction to JSON for the server."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TriageObservation]:
        """Parse server response into a typed StepResult."""
        obs = TriageObservation(**payload)
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse state response."""
        return State(**payload)
