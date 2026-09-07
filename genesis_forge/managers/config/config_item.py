import inspect
from types import MappingProxyType

import torch

from genesis_forge.genesis_env import GenesisEnv

from .config_item_dict import ConfigItemDict
from .mdp_fn import MdpFn

_NO_PARAMS = MappingProxyType({})
"""Read-only stand-in for the params of an MdpFn, whose params live on the instance."""


class ConfigItem:
    """
    A config item for a manager config.

    The manager config dict values get wrapped in this class, which drives the lifecycle
    of the executing function, params, and any build steps that might be necessary.
    """

    def __init__(self, cfg: ConfigItemDict, env: GenesisEnv):
        self._env = env
        self._entity = None
        self._kwargs = {}
        self._built = False

        self._cfg = cfg
        self._fn = cfg["fn"]
        params = cfg.get("params", {}) or {}

        self._is_mdp_class_fn = isinstance(self._fn, MdpFn)

        # A class constructor, not an instance, was passed as the function.
        if inspect.isclass(self._fn):
            # This is an MdpFn subclass
            if issubclass(self._fn, MdpFn):
                param_string = ", ".join(f"{k}=..." for k in params) or "..."
                raise TypeError(
                    f"{self._fn.__name__} must be constructed, not passed as a class. "
                    f"Use {self._fn.__name__}({param_string})"
                )
            # Any other class
            raise TypeError(
                f"{self._fn.__name__} is a class constructor, not a callable instance. "
                "If this is a subclass of MdpFnClass or ResetMdpFnClass, see the 1.0.0 UPGRADE.md notes"
            )

        # Cannot pass params to a MdpFn instance
        if self._is_mdp_class_fn:
            if params:
                raise ValueError(
                    f"{type(self._fn).__name__} declares its params as fields, so the "
                    f"config 'params' dict must be empty. Pass {sorted(params)!r} to "
                    f"the constructor instead: {type(self._fn).__name__}(...)"
                )
            self._params = _NO_PARAMS
        else:
            self._params = dict(params)

    """
    Properties
    """

    @property
    def fn(self):
        """The callable for this config item."""
        return self._fn

    @property
    def is_mdp_class_fn(self) -> bool:
        """Whether ``fn`` is a constructed :class:`MdpFn` instance."""
        return self._is_mdp_class_fn

    @property
    def params(self):
        """
        The params provided for the function.

        Empty for an :class:`MdpFn`, whose params are typed fields on the instance --
        read and write them there (``cfg.fn.target_height``) to get type checking.
        """
        return self._params

    @params.setter
    def params(self, params: dict):
        """Overwrite all params at once, rebuilding once."""
        if self._is_mdp_class_fn:
            self._fn.update(**params)
            return
        self._params = dict(params)

    """
    Lifecycle Operations
    """

    def build(self, **kwargs):
        """
        Prepare the function or MdpFn class for training.

        Args:
            **kwargs: Additional context injected by the manager to any MdpFn class.
                      For example, the entity manager passes ``entity`` here.
        """
        self._kwargs = kwargs
        if self._is_mdp_class_fn:
            self._fn.context(self._env, **kwargs)
            self._fn.safe_build()
        self._built = True

    def reset(self, envs_idx: torch.Tensor):
        """
        Reset the function for the given environments.
        No-op if the function is a plain function, or has not been built yet.
        """
        if self._built and self._is_mdp_class_fn:
            self._fn.reset(envs_idx)

    def execute(self, **kwargs):
        """
        Call the function, passing along the build-time context and params.

        Args:
            **kwargs: Additional per-call arguments. For example, the entity manager
                      passes ``envs_idx`` here for a reset function.

        Returns:
            Whatever the function returns.
        """
        return self._fn(self._env, **self._kwargs, **kwargs, **self._params)


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

    def increment_weight(self, increment: float, limit: float | None = None):
        """
        Increment the weight value by the given amount.

        Args:
            increment: The amount to increment the weight by (+/-).
            limit: Do not set the value beyond this limit.
        """
        self.weight = directional_clamp(self.weight + increment, increment, limit)
        return self.weight

    def increment_param(self, param: str, increment: float, limit: float | None = None):
        """
        Increment a float parameter value by the given amount.

        Args:
            param: The parameter to increment.
            increment: The amount to increment the parameter by (+/-).
            limit: Do not set the value beyond this limit.
        """
        # Get value
        if self._is_mdp_class_fn:
            if param not in self._fn._param_names():
                raise AttributeError(
                    f"{type(self._fn).__name__} has no param '{param}'. "
                    f"Declared params: {sorted(self._fn._param_names())!r}"
                )
            value = getattr(self._fn, param)
        else:
            value = self.params[param]

        # Increment
        value = directional_clamp(value + increment, increment, limit)

        # Update value
        if self._is_mdp_class_fn:
            setattr(self._fn, param, value)
        else:
            self.params[param] = value

        return value


class ObservationConfigItem(ConfigItem):
    """
    A config item for an observation condition.
    """

    def __init__(self, cfg: dict, env: GenesisEnv):
        super().__init__(cfg, env)
        self.scale = cfg.get("scale", 1.0)
        self.noise = cfg.get("noise", None)


def directional_clamp(value: float, increment: float, limit: float | None = None) -> float:
    """Clamp an incremented value to `limit`, in whichever direction it moved."""
    if limit is None:
        return value
    return min(value, limit) if increment > 0 else max(value, limit)
