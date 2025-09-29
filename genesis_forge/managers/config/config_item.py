import inspect

from genesis_forge.genesis_env import GenesisEnv

from genesis_forge.managers.config.params_dict import ParamsDict
from genesis_forge.managers.config.mdp_fn_class import MdpFnClass


class ConfigItem:
    """
    A config item for a manager config.
    The manager config dict values get wrapped in this class to manage building function classes, and rebuilding them
    when the config item parameters are changed.
    """

    def __init__(self, cfg: dict, env: GenesisEnv):
        self._env = env
        self._entity = None
        self._kwargs = {}

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
        self._params = ParamsDict(params.copy(), self._rebuild)
        if self._is_class:
            self._rebuild()

    def build(self, **kwargs):
        """
        Build the function class

        Args:
            **kwargs: Additional parameters to pass to the build and call methods of the function class.
        """
        self._kwargs = kwargs
        if not self._is_class:
            return
        self._init_fn_class()

    def execute(self, envs_idx: list[int]):
        """
        Call the function for the given environment ids.

        Args:
            envs_idx: The environment ids to call the function for.
        """
        self._fn(self._env, **self._kwargs, envs_idx=envs_idx, **self._params)

    def _init_fn_class(self):
        """Initialize the function class"""
        params = self._cfg.get("params", {}) or {}
        if self._initialized:
            return

        instance: MdpFnClass = self._fn(self._env, **self._kwargs, **params)
        instance.build()

        self._fn = instance
        self._initialized = False

    def _rebuild(self):
        """Rebuild the function class"""
        if not self._is_class:
            return
        self._init_fn_class()


class TerminationConfigItem(ConfigItem):
    """
    A config item for a termination condition.
    """

    def __init__(self, cfg: dict, env: GenesisEnv):
        super().__init__(cfg, env)
        self.time_out = cfg.get("time_out", False)


class RewardConfigItem(ConfigItem):
    """
    A config item for a reward condition.
    """

    def __init__(self, cfg: dict, env: GenesisEnv):
        super().__init__(cfg, env)
        self.weight = cfg.get("weight", 0.0)


class ObservationConfigItem(ConfigItem):
    """
    A config item for an observation condition.
    """

    def __init__(self, cfg: dict, env: GenesisEnv):
        super().__init__(cfg, env)
        self.scale = cfg.get("scale", 1.0)
        self.noise = cfg.get("noise", None)
