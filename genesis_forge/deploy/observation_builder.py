from __future__ import annotations

import torch

from genesis_forge.deploy.config import DeploymentConfig
from genesis_forge.rolling_buffer import RollingBuffer


class ObservationBuilder:
    """
    Standalone observation assembler for deployment on a real robot.

    Mirrors the behavior of
    :class:`~genesis_forge.managers.ObservationManager` but without any
    Genesis or simulation dependency.  Provide raw sensor values (keyed by the
    same observation names used during training) and receive back the flat
    observation tensor ready to feed into the policy.

    Supports observation history: if the training environment used
    ``history_len > 1``, this class maintains the same rolling buffer and
    concatenates frames in the same order, using the same
    :class:`~genesis_forge.utils.transforms.RollingBuffer` as the training
    manager to guarantee identical behavior.

    Call :meth:`reset` whenever the robot is reset to clear the history buffer.

    Example::

        from genesis_forge.deploy import DeploymentConfig, ObservationBuilder

        config = DeploymentConfig.from_json("deploy_config.json")
        obs_builder = ObservationBuilder(config)

        # At the start of an episode
        obs_builder.reset()

        while running:
            obs = obs_builder.build_obs({
                "angle_velocity":    imu.gyro,         # list or tensor, shape (3,)
                "linear_velocity":   imu.linear_vel,   # shape (3,)
                "projected_gravity": gravity_vec,       # shape (3,)
                "dof_position":      joint_positions,  # shape (n_joints,)
                "dof_velocity":      joint_velocities, # shape (n_joints,)
                "actions":           last_actions,     # shape (n_actions,)
            })
            # obs.shape == (num_observations,)
            raw_actions = policy(obs.unsqueeze(0)).squeeze(0)
            ...
    """

    def __init__(self, config: DeploymentConfig):
        self._config = config
        self._single_obs_size = sum(o.dim for o in config.observations)

        init_frame = torch.zeros(self._single_obs_size)
        self._history = RollingBuffer(config.observation_history_len, init_frame)

        # Pre-allocated output buffer of shape (num_observations,)
        self._output = torch.zeros(config.num_observations)

    def reset(self) -> None:
        """
        Clear the observation history.

        Call this at the start of each episode (i.e. whenever the robot is
        reset) so that stale frames from the previous episode do not leak into
        the new one.
        """
        self._history.reset()

    def build_obs(self, values: dict[str, torch.Tensor | list | float]) -> torch.Tensor:
        """
        Assemble the observation tensor from raw sensor values.

        The values are scaled using the same factors configured during training.
        No noise is applied (matching the behaviour of
        :meth:`~genesis_forge.managers.ObservationManager.get_observations`
        when called with explicit override values).

        Args:
            values: A dict mapping each observation name to a raw sensor value.
                    Keys must match the observation names defined in the training
                    environment's ``ObservationManager`` config.  Values may be
                    a :class:`torch.Tensor`, a Python ``list``, or a scalar
                    ``float``.

        Returns:
            A 1-D float tensor of shape ``(num_observations,)``, ready to pass
            directly to the policy (use ``.unsqueeze(0)`` to add a batch dim).

        Raises:
            KeyError: If a required observation key is missing from ``values``.
        """
        buffer = self._history.rotate()

        offset = 0
        for obs_item in self._config.observations:
            name = obs_item.name
            if name not in values:
                raise KeyError(
                    f"Observation '{name}' not found in provided values. "
                    f"Expected keys: {[o.name for o in self._config.observations]}"
                )

            val = values[name]
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, dtype=torch.float32)
            val = val.float().flatten()

            # Apply scale
            scale = obs_item.scale
            if scale is not None and scale != 1.0:
                val = val * scale

            buffer[offset : offset + obs_item.dim] = val
            offset += obs_item.dim

        self._history.push(buffer)
        self._history.output(self._output)
        return self._output.clone()
