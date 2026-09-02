from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import torch
    from genesis.engine.entities import RigidEntity

    from genesis_forge.genesis_env import GenesisEnv


@dataclass(kw_only=True, eq=False)
class MdpFn:
    """
    Base class for an MDP function whose parameters are typed dataclass fields.

    Example - A reward function that targets a height::

        @dataclass(kw_only=True, eq=False)
        class base_height(MdpFn):
            target_height: float = 0.3
            entity_manager: EntityManager | None = None

            def __call__(self, env: GenesisEnv) -> torch.Tensor:
                pos = self.entity_manager.entity.get_pos()
                return torch.square(pos[:, 2] - self.target_height)

    Used in a manager config, and adjusted mid-training by a curriculum::

        self.reward_manager = RewardManager(self, cfg={
            "height": {
                "weight": -50.0,
                "fn": base_height(target_height=0.3),
            },
        })

    Change the reward param at runtime::

        self.reward_manager["height"].fn.target_height = 0.35
    """

    # ClassVars, not fields
    _env: ClassVar[GenesisEnv | None] = None
    _can_build: ClassVar[bool] = False
    _building: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs) -> None:
        """ Enforces the @dataclass decorator on subclasses. """
        super().__init_subclass__(**kwargs)
        dataclasses.dataclass(kw_only=True, eq=False)(cls)

    """
    Properties
    """

    @property
    def env(self) -> GenesisEnv:
        """
        The environment this function is bound to.
        """
        if self._env is None:
            raise RuntimeError(
                f"{type(self).__name__} is not bound to an environment yet. "
                "It needs to be set via the context method."
            )
        return self._env

    """
    Public implementation API
    """

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        """
        The function execution entry point, which calculates a value for every environment.

        Args:
            env: The Genesis Forge environment.

        Returns:
            torch.Tensor: Shape ``(num_envs,)`` for rewards and terminations,
                                ``(num_envs, N)`` for observations.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement __call__(self, env)"
        )

    def build(self) -> None:
        """
        Build/rebuild any state or buffers necessary for the function call.

        Called once by the manager during the environment build phase, and again after
        any param is changed. Must be idempotent -- it is safe to call any number of
        times, and re-deriving from scratch is the expected implementation.
        """

    def reset(self, envs_idx: list[int]) -> None:
        """
        Called every time one or more environments reset.

        Args:
            envs_idx: The environment ids being reset.
        """

    """
    Runtime param changes
    """

    def update(self, **params: Any) -> None:
        """
        Change several params at once, rebuilding only once.

        Assigning params one at a time rebuilds once per assignment and exposes
        :meth:`build` to a partially updated param set. Use this when a curriculum
        changes more than one param together.

        Args:
            **params: Declared param names and their new values.

        Raises:
            AttributeError: If a name is not a declared param.
        """
        if not params:
            return

        # Check if we're trying to set unknown params
        unknown = sorted(set(params) - self._param_names())
        if unknown:
            raise AttributeError(
                f"{type(self).__name__} has no param(s) {unknown!r}. "
            )

        # Set the param values, bypassing the rebuild hook for each value, and then rebuild once at the end
        for name, value in params.items():
            object.__setattr__(self, name, value)

        if self._can_build:
            self.safe_build()

    """
    Lifecycle, driven by the manager
    """

    def context(self, env: GenesisEnv) -> None:
        """
        Store the environment this function is bound to.

        Called by the manager during the environment build phase.

        Override to store additional values -- see :meth:`ResetMdpFn.context`, which
        also stores ``entity``.
        """
        object.__setattr__(self, "_env", env)

    def safe_build(self) -> None:
        """
        Run :meth:`build` with re-entrancy guarded, so a build that assigns one of its
        own declared fields does not create an infinite build loop.
        """
        object.__setattr__(self, "_building", True)
        try:
            self.build()
        finally:
            object.__setattr__(self, "_building", False)
        object.__setattr__(self, "_can_build", True)

    """
    Internal methods
    """

    @classmethod
    def _param_names(cls) -> frozenset[str]:
        """The declared param names, excluding ClassVars and other non-field attributes."""
        cached = cls.__dict__.get("_param_names_cache")
        if cached is None:
            cached = frozenset(f.name for f in dataclasses.fields(cls))
            cls._param_names_cache = cached
        return cached

    def __setattr__(self, name: str, value: Any) -> None:
        """Rebuild when a param changes."""
        super().__setattr__(name, value)
        if self._can_build and not self._building and name in self._param_names():
            self.safe_build()


@dataclass(kw_only=True, eq=False)
class ResetMdpFn(MdpFn):
    """
    Base class for an entity reset function, used by the :class:`EntityManager`.
    """

    _entity: ClassVar[RigidEntity | None] = None

    @property
    def entity(self) -> RigidEntity:
        """
        The entity this function resets.

        Raises:
            RuntimeError: If accessed before the manager has bound the function.
        """
        if self._entity is None:
            raise RuntimeError(
                f"{type(self).__name__} is not bound to an entity yet. "
                "It needs to be set via the context method."
            )
        return self._entity

    def __call__(
        self,
        env: GenesisEnv,
        entity: RigidEntity,
        envs_idx: list[int],
    ) -> None:
        """
        Apply this reset to the given environments.

        Args:
            env: The Genesis Forge environment.
            entity: The entity being reset.
            envs_idx: The environment ids being reset.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement __call__(self, env, entity, envs_idx)"
        )

    def context(self, env: GenesisEnv, entity: RigidEntity = None) -> None:
        """Store the environment and the entity this function resets."""
        super().context(env)
        object.__setattr__(self, "_entity", entity)
