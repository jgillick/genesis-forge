"""Simulation-free runtime for deploying Genesis Forge policies to real robots.

This package reproduces the observation-assembly and action-processing pipelines
from training, reading them out of a bundle exported by ``genesis_forge``. It
depends on **numpy only** -- importing anything here must never pull in torch or
the Genesis simulator, so it installs on a Raspberry Pi or Jetson.

Typical robot-side use::

    from genesis_forge_runtime import load_bundle

    bundle = load_bundle("./go2_walk")
    print(bundle.describe())          # what to wire up

    observation_assembler = bundle.create_observation_assembler()
    action_processor = bundle.create_action_processor()

    while True:
        obs = observation_assembler.assemble({
            "robot_ang_vel": gyro,
            "actions": action_processor.last_raw_actions_by_manager["action_manager"],
        })
        targets = action_processor.process(policy(obs))
        send_to_motors(targets.by_joint)

The modules behind it, roughly in dependency order:

* :mod:`~genesis_forge_runtime.constants` -- the manifest's vocabulary
* :mod:`~genesis_forge_runtime.errors` -- every exception raised here
* :mod:`~genesis_forge_runtime.serialization` -- JSON to numpy and back
* :mod:`~genesis_forge_runtime.observation_schema` /
  :mod:`~genesis_forge_runtime.action_schema` -- the two halves of the contract
* :mod:`~genesis_forge_runtime.manifest` -- the contract as a whole
* :mod:`~genesis_forge_runtime.bundle` -- reading and writing a bundle directory
* :mod:`~genesis_forge_runtime.observations` /
  :mod:`~genesis_forge_runtime.processors` / :mod:`~genesis_forge_runtime.actions`
  -- the runtime itself

**Trust model:** a bundle is trusted input, equivalent to executable code --
loading one may import processor classes it names. Only load bundles you produced.
"""

from .action_schema import ActionManagerSpec, ActuatorSpec
from .actions import ActionProcessor, ProcessedActions
from .bundle import Bundle, load_bundle, load_manifest, save_bundle
from .constants import (
    ARCHIVE_SUFFIX,
    EXTRACT_MARKER,
    GOLDEN_FILENAME,
    MANIFEST_FILENAME,
    MIN_SUPPORTED_SCHEMA_VERSION,
    POLICY_DIRNAME,
    SCHEMA_VERSION,
)
from .errors import (
    ActionError,
    BundleError,
    MalformedBundleError,
    ObservationError,
    SchemaVersionError,
)
from .manifest import Manifest, Provenance
from .observation_schema import ObservationEntry, ObservationLayout
from .observations import ObservationAssembler
from .processors import ActionManagerProcessor, AffineProcessor

__version__ = "1.0.0"

__all__ = [
    "ARCHIVE_SUFFIX",
    "ActionError",
    "ActionManagerProcessor",
    "ActionManagerSpec",
    "ActionProcessor",
    "ActuatorSpec",
    "AffineProcessor",
    "Bundle",
    "BundleError",
    "EXTRACT_MARKER",
    "GOLDEN_FILENAME",
    "MANIFEST_FILENAME",
    "MIN_SUPPORTED_SCHEMA_VERSION",
    "MalformedBundleError",
    "Manifest",
    "ObservationAssembler",
    "ObservationEntry",
    "ObservationError",
    "ObservationLayout",
    "POLICY_DIRNAME",
    "ProcessedActions",
    "Provenance",
    "SCHEMA_VERSION",
    "SchemaVersionError",
    "__version__",
    "load_bundle",
    "load_manifest",
    "save_bundle",
]
