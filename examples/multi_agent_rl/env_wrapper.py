"""
SKRL MAPPO bridge for :class:`~environment.Go2MasqLocomotionEnv`.

Reads per-agent :attr:`~environment.Go2MasqLocomotionEnv.observation_spaces` /
:attr:`~environment.Go2MasqLocomotionEnv.action_spaces` from the env (after
:meth:`~genesis_forge.ManagedEnvironment.build`) and builds decentralized observation
tensors from ``extras["observations"]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from environment import OBS_SHARED_KEY, Go2MasqLocomotionEnv
from gymnasium import spaces
from skrl.envs.wrappers.torch.base import MultiAgentEnvWrapper
from skrl.utils.spaces.torch import (
    flatten_tensorized_space,
    tensorize_space,
)

from genesis_forge.managed_env import ManagedEnvironment


class SkrlMasqWrapper(MultiAgentEnvWrapper):
    """SKRL MAPPO wrapper for :class:`~environment.Go2MasqLocomotionEnv`."""

    def __init__(self, env: ManagedEnvironment):
        super().__init__(env)
        self._unwrapped: Go2MasqLocomotionEnv = env.unwrapped
        self._has_reset = False
        self._info: dict[str, Any] = {}
        self._last_combined_obs: torch.Tensor | None = None
        self._last_agent_obs: dict[str, torch.Tensor] = {}

    @property
    def possible_agents(self) -> list[str]:
        return self._unwrapped.agents

    @property
    def observation_spaces(self) -> dict[str, spaces.Box]:
        return self._unwrapped.observation_spaces

    @property
    def action_spaces(self) -> dict[str, spaces.Box]:
        return self._unwrapped.action_spaces

    @property
    def state_spaces(self) -> dict[str, spaces.Box]:
        return self._unwrapped.state_spaces

    def state(self) -> dict[str, torch.Tensor | None]:
        if self._last_combined_obs is None:
            raise RuntimeError("state() called before reset/step")
        flat = flatten_tensorized_space(
            tensorize_space(
                self.state_spaces[self.possible_agents[0]],
                self._last_combined_obs,
                device=self.device,
            )
        )
        return {uid: flat for uid in self.possible_agents}

    def reset(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if not self._has_reset:
            observations, self._info = self._env.reset()
            assert observations.ndim == 2
            self._handle_observations(observations)
            self._has_reset = True
        return self._last_agent_obs, self._info

    def step(
        self,
        actions: Mapping[str, torch.Tensor],
    ) -> tuple[
        Mapping[str, torch.Tensor],
        Mapping[str, torch.Tensor],
        Mapping[str, torch.Tensor],
        Mapping[str, torch.Tensor],
        Mapping[str, Any],
    ]:
        combined_actions = torch.cat([actions[k] for k in self.possible_agents], dim=-1)
        observations, rew, term, trunc, extras = self._env.step(combined_actions)
        
        self._handle_observations(observations)

        # Convert rewards, terminations, and truncations to dicts keyed by agent
        rew = rew.unsqueeze(-1)
        rewards = {k: rew.to(dtype=torch.float32) for k in self.possible_agents}
        terminated = {
            k: term.view(-1, 1).to(dtype=torch.bool) for k in self.possible_agents
        }
        truncated = {
            k: trunc.view(-1, 1).to(dtype=torch.bool) for k in self.possible_agents
        }

        return self._last_agent_obs, rewards, terminated, truncated, extras

    def _handle_observations(self, combined_obs: torch.Tensor) -> None:
        self._last_combined_obs = combined_obs
        obs_dict = self._unwrapped.extras["observations"]

        # Combine the shared observations with the agent-specific observations
        shared = obs_dict[OBS_SHARED_KEY]
        agent_obs: dict[str, torch.Tensor] = {}
        for agent in self.possible_agents:
            agent_obs[agent] = torch.cat([shared, obs_dict[agent]], dim=-1)
        self._last_agent_obs = agent_obs

    def render(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        self._env.close()
