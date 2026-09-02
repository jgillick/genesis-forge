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

The modules behind it:

* :mod:`~genesis_forge.deployment.errors` -- every exception raised here
* :mod:`~genesis_forge.deployment.provenance` -- where a bundle came from
* :mod:`~genesis_forge.deployment.capture` -- reading the contract off managers
* :mod:`~genesis_forge.deployment.comparison` -- tolerances and float comparison
* :mod:`~genesis_forge.deployment.sampling` -- the inputs parity compares on
* :mod:`~genesis_forge.deployment.parity` -- the gate itself
* :mod:`~genesis_forge.deployment.exporter` -- ``export()``
"""

from .capture import Capture, capture_environment
from .comparison import PIPELINE_ATOL, PIPELINE_RTOL
from .errors import ExportError, ParityError
from .exporter import export
from .parity import ParityReport, check_parity

__all__ = [
    "PIPELINE_ATOL",
    "PIPELINE_RTOL",
    "Capture",
    "ExportError",
    "ParityError",
    "ParityReport",
    "capture_environment",
    "check_parity",
    "export",
]
