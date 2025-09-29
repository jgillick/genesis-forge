from genesis_forge.genesis_env import GenesisEnv
from genesis.engine.entities import RigidEntity


class MdpFnClass:
    """
    A callable function class that can be used in place of an MDP function.
    The build method will be called automatically during the build phase of the environment, and
    if any of the mdp params are changed.
    """

    def __init__(self, env: GenesisEnv):
        self.env = env
        pass

    def build(self):
        """Called during the environment build phase and when MDP params are changed."""
        pass

    def __call__(
        self,
        env: GenesisEnv,
        envs_idx: list[int],
    ):
        """The callable MDP function."""
        pass


class ResetMdpFnClass(MdpFnClass):
    """
    An MDP function class for an EntityManager on-reset functions
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
