from collections.abc import Callable
from typing import TypedDict, cast

import genesis as gs
import torch

from genesis_forge.gamepads import Gamepad
from genesis_forge.genesis_env import GenesisEnv

from .command_manager import CommandManager, CommandRangeValue


class AttitudeCommandRange(TypedDict):
    pitch: CommandRangeValue
    roll: CommandRangeValue


class AttitudeCommandManager(CommandManager):
    """
    Generates a pitch and roll command for the robot's base from a uniform distribution,
    for robots that lean on purpose, like a quadruped bowing or leaning into a turn.

    The angles are in radians, and follow the right-hand rule about the base's y and x
    axes, as in Genesis's `quat_to_xyz(quat, rpy=True)`. With x forward and y left, a
    positive pitch tips the nose down, and a positive roll tips the right side down.

    Use it with the `rewards.command_tracking_attitude` reward. While the robot leans,
    the velocity rewards should measure in the heading frame (`frame="heading"`), so the
    lean isn't counted as motion along the wrong axis.

    Args:
        env: The environment to control
        range: The ranges of pitch and roll, in radians
        resample_time_sec: The time interval between changing the command,
                           or None to only change it on reset
        level_probability: The probability of an environment being commanded to stay
                           level (0.0 = never, 1.0 = always)

    Example::

        class MyEnv(ManagedEnvironment):
            def config(self):
                self.attitude_command = AttitudeCommandManager(
                    self,
                    range={
                        "pitch": (-0.3, 0.3),
                        "roll": (-0.2, 0.2),
                    },
                    level_probability=0.2,
                )

                RewardManager(
                    self,
                    cfg={
                        "tracking_attitude": {
                            "weight": 1.0,
                            "fn": rewards.command_tracking_attitude(
                                attitude_command=self.attitude_command,
                                entity_manager=self.robot_manager,
                            ),
                        },
                        # ... other rewards ...
                    },
                )

                ObservationManager(
                    self,
                    cfg={
                        "attitude_cmd": {"fn": self.attitude_command.observation},
                        # ... other observations ...
                    },
                )
    """

    def __init__(
        self,
        env: GenesisEnv,
        range: AttitudeCommandRange,
        resample_time_sec: float | None = 5.0,
        level_probability: float = 0.0,
    ):
        super().__init__(env, range=range, resample_time_sec=resample_time_sec)
        self.level_probability = level_probability

    """
    Properties
    """

    @property
    def range(self) -> AttitudeCommandRange:
        """The pitch and roll range dict, in radians."""
        return cast(AttitudeCommandRange, self._range)

    @range.setter
    def range(self, range: AttitudeCommandRange):
        """Update the pitch and roll ranges."""
        CommandManager.range.fset(self, range)

    @property
    def pitch(self) -> torch.Tensor:
        """The commanded pitch, in radians. Shape is (num_envs,)."""
        return self.command[:, self._range_idx["pitch"]]

    @property
    def roll(self) -> torch.Tensor:
        """The commanded roll, in radians. Shape is (num_envs,)."""
        return self.command[:, self._range_idx["roll"]]

    def level_envs(self, threshold: float = 0.01) -> torch.Tensor:
        """
        The environments commanded to stay level: both pitch and roll are within
        `threshold` radians of zero.

        Args:
            threshold: The largest angle, in radians, still counted as level.

        Returns:
            Boolean tensor, shape (num_envs,)
        """
        return torch.all(torch.abs(self.command) <= threshold, dim=1)

    def when_level(
        self, threshold: float = 0.01
    ) -> Callable[[GenesisEnv], torch.Tensor]:
        """
        Returns a `when` function for reward and termination items. It is True wherever
        the command asks the robot to stay level.

        Example::

            "flat_orientation": {
                "fn": rewards.flat_orientation_l2(entity_manager=self.robot_manager),
                "weight": -5.0,
                "when": self.attitude_command.when_level(),
            },

        Args:
            threshold: The largest angle, in radians, still counted as level.
        """
        return lambda env: self.level_envs(threshold)

    def when_tilted(
        self, threshold: float = 0.01
    ) -> Callable[[GenesisEnv], torch.Tensor]:
        """
        Returns a `when` function for reward and termination items. It is True wherever
        the command asks the robot to pitch or roll.

        Args:
            threshold: The largest angle, in radians, still counted as level.
        """
        return lambda env: ~self.level_envs(threshold)

    """
    Operations
    """

    def resample_command(self, env_ids: torch.Tensor):
        """Resample the command, then level the environments picked by `level_probability`."""
        super().resample_command(env_ids)
        if not self.enabled:
            return

        # torch.rand samples [0, 1), so `<` makes 0.0 never level an environment and 1.0
        # always level it
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=gs.device)
        level = torch.rand(len(env_ids), device=gs.device) < self.level_probability
        self._command[env_ids[level], :] = 0.0

    def use_gamepad(
        self,
        gamepad: Gamepad,
        pitch_axis: int | None = None,
        roll_axis: int | None = None,
    ):
        """
        Use a connected gamepad to control the command.

        Pushing the pitch stick forward tips the nose down, and pushing the roll stick
        right tips the right side down. Set an axis to `None` if that angle should not
        be controlled by the gamepad. It's then held at zero, or the closest value to
        zero within its range.

        Args:
            gamepad: The gamepad to use.
            pitch_axis: Map this gamepad axis index to the pitch.
            roll_axis: Map this gamepad axis index to the roll.

        Example::

            # Pitch on the right stick's vertical axis, and roll on its horizontal axis
            env.attitude_command.use_gamepad(gamepad, pitch_axis=3, roll_axis=2)
        """
        super().use_gamepad(
            gamepad,
            range_axis={"pitch": pitch_axis, "roll": roll_axis},
            # A stick reads -1 when pushed forward, which should be the most nose-down pitch
            invert_axis={"pitch": True},
        )
