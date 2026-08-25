"""Export a trained Genesis Forge environment for deployment on a real robot.

Training-machine half of the deployment story. It reads the observation and action
pipelines out of a *built* environment, proves the simulation-free runtime
reproduces them exactly, and writes a bundle the robot can load::

    from genesis_forge.deployment import export

    env = MyEnv(num_envs=1)
    env.build()
    export(env, "./my_policy")

The robot-side half is the separate ``genesis-forge-deploy`` package, which depends
only on numpy so it installs on a Raspberry Pi or Jetson.
"""

from .capture import Capture, ExportError, capture_environment
from .exporter import export
from .parity import (
    ParityError,
    ParityReport,
    check_parity,
    check_policy_parity,
)

__all__ = [
    "Capture",
    "ExportError",
    "ParityError",
    "ParityReport",
    "capture_environment",
    "check_parity",
    "check_policy_parity",
    "export",
]
