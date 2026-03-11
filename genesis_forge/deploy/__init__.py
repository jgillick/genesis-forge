"""
genesis_forge.deploy
====================

Utilities for deploying trained policies to a real robot without any
Genesis or simulation dependency.

**Training-time** (export)::

    from genesis_forge.deploy import export

    env = Go2SimpleEnv(num_envs=1)
    env = RslRlWrapper(env)
    env.build()

    config = export(env, path="./deploy/go2_config.json")

**Deployment-time** (robot, no Genesis required)::

    from genesis_forge.deploy import DeploymentConfig, ObservationBuilder, ActionDecoder

    config = DeploymentConfig.from_json("./deploy/go2_config.json")
    obs_builder = ObservationBuilder(config)
    action_decoder = ActionDecoder(config)

    policy = torch.jit.load("policy.pt")

    obs_builder.reset()
    while True:
        obs = obs_builder.build_obs({
            "angle_velocity":    imu.gyro,
            "linear_velocity":   imu.linear_vel,
            "projected_gravity": gravity_vec,
            "dof_position":      joint_positions,
            "dof_velocity":      joint_velocities,
            "actions":           last_actions,
        })
        with torch.no_grad():
            raw_actions = policy(obs.unsqueeze(0)).squeeze(0)
        joint_commands = action_decoder.decode(raw_actions)
        # {"FL_hip_joint": 0.12, "FL_thigh_joint": 0.85, ...}

**Custom action manager types**::

    # Training side (uses Genesis)
    class VelocityActionManager(BaseActionManager):
        deploy_type = "velocity"

        def export(self) -> dict:
            config = super().export()
            config.update({"max_velocity": self._max_velocity.tolist()})
            return config

    # Deployment side (no Genesis)
    from genesis_forge.deploy import register_action_decoder

    @register_action_decoder("velocity")
    def decode_velocity(actions, params):
        max_vel = torch.tensor(params["max_velocity"])
        return torch.clamp(actions, -max_vel, max_vel)
"""

from .config import DeploymentConfig
from .export import export
from .observation_builder import ObservationBuilder
from .action_decoder import ActionDecoder, register_action_decoder
from genesis_forge.utils.rolling_buffer import RollingBuffer
from genesis_forge.utils.transforms import (
    position_decode,
    position_within_limits_decode,
)

__all__ = [
    "DeploymentConfig",
    "export",
    "ObservationBuilder",
    "ActionDecoder",
    "register_action_decoder",
    "RollingBuffer",
    "position_decode",
    "position_within_limits_decode",
]
