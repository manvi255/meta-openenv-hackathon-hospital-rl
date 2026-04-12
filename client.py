# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kernel Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import HospitalAction, HospitalObservation
except (ImportError, ModuleNotFoundError):
    from models import HospitalAction, HospitalObservation

class HospitalEnv(
    EnvClient[HospitalAction, HospitalObservation, State]
):
    """
    Client for the Hospital RL Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with HospitalEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.available_beds)
        ...
        ...     result = client.step(HospitalAction(action_type="ADMIT", patient_id="p1"))
        ...     print(result.reward)

    Example with Docker:
        >>> client = HospitalEnv.from_docker_image("hospital-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(HospitalAction(action_type="REJECT", patient_id="p2"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: HospitalAction) -> Dict:
        """
        Convert HospitalAction to JSON payload for step message.

        Args:
            action: HospitalAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[HospitalObservation]:
        """
        Parse server response into StepResult[HospitalObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with HospitalObservation
        """
        obs_data = payload.get("observation", {})
        observation = HospitalObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
