from typing import TypedDict, Callable, Any

from genesis_forge.genesis_env import GenesisEnv
from genesis.engine.entities import RigidEntity

ResetConfigFn = Callable[[GenesisEnv, RigidEntity, list[int], ...], None]


class ResetConfigFnClass:
    """
    The shape of the class that can be used as a reset function
    """

    def __init__(self, env: GenesisEnv, entity: RigidEntity, envs_idx: list[int]):
        pass

    def build(self):
        pass

    def __call__(
        self,
        env: GenesisEnv,
        entity: RigidEntity,
        envs_idx: list[int],
    ):
        pass


class EntityResetConfig(TypedDict):
    """Defines an entity reset item."""

    fn: ResetConfigFn | ResetConfigFnClass
    """
    Function, or class function, that will be called on reset.
    The args passed to the function are:
        - env: The environment instance.
        - entity: The entity instance.
        - envs_idx: The environment ids for which the entity is to be reset.
        - **params: Additional parameters to pass to the function from the params dictionary.
    """

    params: dict[str, Any]
    """Additional parameters to pass to the function."""

    weight: float
    """The weight of the reward item."""
