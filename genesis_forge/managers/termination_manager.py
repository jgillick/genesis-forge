import torch
from typing import TypedDict, Callable, Any

import genesis as gs
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager


class TerminationConfig(TypedDict):
    """Defines a termination condition."""

    fn: Callable[[GenesisEnv, ...], torch.Tensor]
    """Function that will be called to calculate a termination signal for the environment."""

    params: dict[str, Any]
    """Additional parameters to pass to the function."""

    time_out: bool
    """Set to True if a positive result is a time out and not a termination."""


class TerminationManager(BaseManager):
    """
    Handles calculating and logging the "dones" (termination or truncation) for the environments.

    This works with a dictionary configuration of termination conditions. For each dictionary item,
    a function will be called to calculate a termination signal for the environment.

    Args:
        env: The environment to manage the termination for.
        term_cfg: A dictionary of termination conditions.
        logging_enabled: Whether to log the termination signals to tensorboard.
        logging_tag: The section tag used to log the termination signals to tensorboard.

    Example with ManagedEnvironment::

        class MyEnv(ManagedEnvironment):
            def config(self):
                self.termination_manager = TerminationManager(
                    self,
                    term_cfg={
                        "Min Height": {
                            "fn": mdp.terminations.min_height,
                            "params": {"min_height": 0.05},
                        },
                    },
                )

    Example using the termination manager directly::

        class MyEnv(GenesisEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.termination_manager = TerminationManager(
                    self,
                    term_cfg={
                        "Min Height": {
                            "fn": mdp.terminations.min_height,
                            "params": {"min_height": 0.5},
                        },
                        "Rolled over": {
                            "fn": mdp.terminations.max_angle,
                            "params": { "quat_threshold": 0.35 },
                        },
                    },
                )

            def build(self):
                super().build()
                self.termination_manager.build()

            def step(self, actions: torch.Tensor):
                super().step(actions)
                # ...handle actions...

                # Calculate dones (terminated or truncated)
                terminated, truncated = self.termination_manager.step()
                dones = terminated | truncated
                reset_env_idx = dones.nonzero(as_tuple=False).reshape((-1,))

                # Reset environments
                if reset_env_idx.numel() > 0:
                    self.reset(reset_env_idx)

                return obs, rewards, terminated, truncated, info

            def reset(self, envs_idx: Sequence[int] = None):
                super().reset(envs_idx)
                # ...do reset logic here...x

                self.termination_manager.reset(envs_idx)
                return obs, info

    """

    def __init__(
        self,
        env: GenesisEnv,
        term_cfg: dict[str, TerminationConfig],
        logging_enabled: bool = True,
        logging_tag: str = "Terminations",
    ):
        super().__init__(env, type="termination")

        self.term_cfg = term_cfg
        self.logging_enabled = logging_enabled
        self.logging_tag = logging_tag
        self._terminated_buf = torch.zeros(
            env.num_envs, device=gs.device, dtype=torch.bool
        )
        self._truncated_buf = torch.zeros_like(self._terminated_buf)

    @property
    def dones(self) -> torch.Tensor:
        """The termination signals for the environments. Shape is (num_envs,)."""
        return self._terminated_buf | self._truncated_buf

    @property
    def terminated(self) -> torch.Tensor:
        """The termination signals for the environments. Shape is (num_envs,)."""
        return self._terminated_buf

    @property
    def truncated(self) -> torch.Tensor:
        """The truncation signals for the environments. Shape is (num_envs,)."""
        return self._truncated_buf

    def step(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the termination/truncation signals for this step

        Returns:
            terminated - The termination signals for the environments. Shape is (num_envs,).
            truncated - The truncation signals for the environments. Shape is (num_envs,).
        """
        if not self.enabled:
            return self._terminated_buf, self._truncated_buf

        self._terminated_buf[:] = False
        self._truncated_buf[:] = False
        logging_dict = self.env.extras[self.env.extras_logging_key]
        for name, cfg in self.term_cfg.items():
            try:
                fn = cfg["fn"]
                params = cfg.get("params", dict())
                trunc = cfg.get("time_out", False)

                # Get termination value
                value = fn(self.env, **params).detach()

                # Add to the correct buffer using in-place operations
                if trunc:
                    self._truncated_buf |= value
                else:
                    self._terminated_buf |= value

                # Logging, if there are some terminations and timeouts
                dones = value.nonzero(as_tuple=True)[0]
                if self.logging_enabled and dones.numel() > 0:
                    logging_dict[f"{self.logging_tag} / {name}"] = (
                        value.float().mean().detach()
                    )

            except Exception as e:
                print(f"Error calculating termination for '{name}'")
                raise e

        self.env.extras["terminations"] = self._terminated_buf
        self.env.extras["truncations"] = self._truncated_buf
        return self._terminated_buf, self._truncated_buf
