import torch
from typing import Any
import genesis as gs
from importlib import metadata

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.wrappers.wrapper import Wrapper


class RslRlWrapper(Wrapper):
    """
    A wrapper for the rsl_rl framework.

    IMPORTANT: This should be the last wrapper, as the change in the step and get_observations methods might break other wrappers.

    What it does:
     - Combines the terminated and truncated tensors into a single tensor (i.e. `terminated | truncated`).
     - Add the truncated tensor to the extras dictionary as "time_outs".
     - Returns observations and extras from the `get_observations` method.
    """

    can_be_wrapped = False

    def __init__(self, env: GenesisEnv):
        super().__init__(env)

        self.obs_tensor_dict = False
        try:
            major_version = int(metadata.version("rsl-rl-lib").split(".")[0])
            if major_version >= 3:
                self.obs_tensor_dict = True
        except:
            pass

    @property
    def device(self) -> str:
        return gs.device

    def build(self):
        """
        Call an initial reset after building the environment.
        """
        super().build()
        self.env.reset()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Returns a single "dones" tensor, instead of the terminated and truncated tensors (via `terminated | truncated`).
        Add the truncated tensor to the extras dictionary as "time_outs".
        """
        (
            obs,
            rewards,
            terminated,
            truncated,
            extras,
        ) = super().step(actions)

        # Combine terminated and truncated
        dones = terminated | truncated

        # Add observations and timeouts to extras
        if extras is None:
            extras = {}
        extras["time_outs"] = truncated
        extras = self._add_observations_to_extras(obs, extras)

        # Convert logging items from tensors to floats
        if "episode" in extras:
            for key, value in extras["episode"].items():
                if isinstance(value, torch.Tensor):
                    extras["episode"][key] = value.float().mean().item()

        obs = self._format_obs_group(obs)
        return obs, rewards, dones, extras

    def get_observations(self):
        """
        Returns observations as well as an extras dictionary with the observations added to the `extras["observations"]["critic"]` key.
        """
        obs = self.env.get_observations()
        extras = self._add_observations_to_extras(obs, self.env.extras)
        obs = self._format_obs_group(obs)
        return obs, extras

    def _add_observations_to_extras(self, obs: torch.Tensor, extras: dict):
        """
        Add the observations to the extras dictionary.
        """
        if "observations" not in extras:
            extras["observations"] = {}
        extras["observations"]["critic"] = obs
        return extras

    def _format_obs_group(self, obs: torch.Tensor):
        """
        If we're using rsl_rl 3.0+, put the observations into a dictionary under the "policy" key
        """
        if not self.obs_tensor_dict:
            return obs
        return {"policy": obs}
