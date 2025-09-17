import inspect
from typing import TypedDict, Callable, Any

from genesis_forge.genesis_env import GenesisEnv
from genesis.engine.entities import RigidEntity

ResetConfigFn = Callable[[GenesisEnv, RigidEntity, list[int], ...], None]


class ResetConfigFnClass:
    """
    The shape of the class that can be used as a reset function
    """

    def __init__(self, env: GenesisEnv, entity: RigidEntity):
        self.env = env
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

    Args:
        env: The environment instance.
        entity: The entity instance.
        envs_idx: The environment ids for which the entity is to be reset.
        **params: Additional parameters to pass to the function from the params dictionary.
    """

    params: dict[str, Any]
    """Additional parameters to pass to the function."""

    weight: float
    """The weight of the reward item."""


from typing import Callable


class ParamsDict(dict):
    """
    The params dictionary that will call a function when a key is set or changed.
    We use this to rebuild a reset class, if a parameter is changed.
    """

    def __init__(self, params: dict, on_change: Callable[[], None]):
        super().__init__(params)
        self._on_change = on_change

    def register_action(self, key, action_function):
        """Registers an action function to be called when 'key' is changed."""
        self._actions[key] = action_function

    def __setitem__(self, key, value):
        """Call the on change function when a value is changed."""
        super().__setitem__(key, value)
        self._on_change()

    def __delitem__(self, key):
        """Call the on change function when a value is deleted."""
        super().__delitem__(key)
        self._on_change()


class ConfigItem:
    """
    The on-reset config dict get's wrapped in this class to provide a clean interface and to rebuild function classes when params are changed.
    """

    def __init__(self, cfg: EntityResetConfig, env: GenesisEnv):
        self._env = env
        self._entity = None

        self._cfg = cfg
        self._fn = cfg["fn"]
        params = cfg.get("params", {}) or {}
        self._params = ParamsDict(params, self._rebuild)

        self._initialized = True
        self._is_class = inspect.isclass(cfg["fn"])
        if self._is_class:
            self._initialized = False

    @property
    def fn(self):
        return self._fn

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params: dict):
        """Overwrite the params dictionary"""
        if not isinstance(params, ParamsDict):
            params = ParamsDict(params, self._rebuild)
        self._params = params

        if self._is_class:
            self._rebuild()

    def build(self, entity: RigidEntity):
        """Build the function class"""
        self._entity = entity
        if not self._is_class:
            return
        self._init_fn_class()

    def execute(self, envs_idx: list[int]):
        """
        Call the function for the given environment ids.

        Args:
            envs_idx: The environment ids to call the function for.
        """
        self._fn(self._env, self._entity, envs_idx, **self._params)

    def _init_fn_class(self):
        """Initialize the function class"""
        params = self._cfg.get("params", {}) or {}
        if self._initialized:
            return

        instance: ResetConfigFnClass = self._fn(self._env, self._entity, **params)
        instance.build()

        self._fn = instance
        self._initialized = False

    def _rebuild(self):
        """Rebuild the function class"""
        if not self._is_class:
            return
        self._init_fn_class()
