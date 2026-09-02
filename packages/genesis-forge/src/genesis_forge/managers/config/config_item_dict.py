from typing import Any, NotRequired, Protocol, TypedDict

import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.config.mdp_fn import MdpFn


class ConfigCallbackFn(Protocol):
    """
    A simple callback function used to provide input to a manager.
    The first argument will always be the environment, followed by the config dict values.

    Example::
        
        ##
        # This is the config callback reward function
        #
        def reward_target_height(
            env: GenesisEnv,
            target_height: Union[float, torch.Tensor] = None
        ) -> torch.Tensor:
            base_pos = env.robot.get_pos()
            return torch.square(base_pos[:, 2] - target_height)
        
        ...

        ##
        # Here's how it is used
        RewardManager(
            self,
            cfg={
                "height": {
                    "weight": -50.0,
                    "fn": reward_target_height,
                    "params": {
                        "target_height": 0.3,
                    },
                },
            }
        }

    Args:
        env: the environemnt
        **params: Other args that are provided by the dict "params" value

    Return:
        result: torch.Tensor, shape (n_envs, 1)
    """

    def __call__(self, env: GenesisEnv, *params: Any, **kwargs: Any) -> torch.Tensor: ...


class ConfigItemDict(TypedDict):
    """Defines a manager config item used to """

    fn: ConfigCallbackFn | MdpFn
    """
    Function that will be called to calculate results (reward, observations, etc), or execute
    an action (e.g. entity reset) for each environment."""

    params: NotRequired[dict[str, Any]]
    """Additional parameters to pass to the function."""
 
