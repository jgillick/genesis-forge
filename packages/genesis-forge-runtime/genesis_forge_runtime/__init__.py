"""Simulation-free runtime for deploying Genesis Forge policies to real robots.

This package reproduces the observation-assembly and action-decoding pipelines
from training, reading them out of a bundle exported by ``genesis_forge``. It
depends on **numpy only** -- importing anything here must never pull in torch or
the Genesis simulator, so it installs on a Raspberry Pi or Jetson.

Typical robot-side use::

    from genesis_forge_runtime import load_bundle

    bundle = load_bundle("./my_policy")
    print(bundle.describe())          # what to wire up

    observation_assembler = bundle.create_observation_assembler()
    action_decoder = bundle.create_action_decoder()

    while True:
        obs = observation_assembler.assemble({
            "robot_ang_vel": gyro,
            "actions": action_decoder.last_target_actions_by_manager["action_manager"],
        })
        targets = action_decoder.decode(policy(obs))
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
  :mod:`~genesis_forge_runtime.decoders` / :mod:`~genesis_forge_runtime.actions`
  -- the runtime itself

**Trust model:** a bundle is trusted input, equivalent to executable code --
loading one may import decoder classes it names. Only load bundles you produced.
"""

from .action_schema import ActionManagerSpec, ActuatorSpec
from .actions import ActionDecoder, DecodedActions
from .bundle import Bundle, load_bundle, load_manifest, save_bundle
from .constants import (
    ARCHIVE_SUFFIX,
    EXTRACT_MARKER,
    GOLDEN_FILENAME,
    HISTORY_NEWEST_FIRST,
    MANIFEST_FILENAME,
    MIN_SUPPORTED_SCHEMA_VERSION,
    POLICY_FORMAT_ONNX,
    POLICY_FORMAT_TORCHSCRIPT,
    POLICY_STEM,
    SCHEMA_VERSION,
)
from .decoders import AffineDecoder, ManagerDecoder
from .errors import (
    BundleError,
    DecoderError,
    MalformedBundleError,
    ObservationError,
    SchemaVersionError,
)
from .manifest import Manifest, PolicySpec, Provenance
from .observation_schema import ObservationEntry, ObservationLayout
from .observations import ObservationAssembler

__version__ = "1.0.0"

__all__ = [
    "ARCHIVE_SUFFIX",
    "EXTRACT_MARKER",
    "GOLDEN_FILENAME",
    "HISTORY_NEWEST_FIRST",
    "MANIFEST_FILENAME",
    "MIN_SUPPORTED_SCHEMA_VERSION",
    "POLICY_FORMAT_ONNX",
    "POLICY_FORMAT_TORCHSCRIPT",
    "POLICY_STEM",
    "SCHEMA_VERSION",
    "ActionDecoder",
    "ActionManagerSpec",
    "ActuatorSpec",
    "AffineDecoder",
    "Bundle",
    "BundleError",
    "DecodedActions",
    "DecoderError",
    "MalformedBundleError",
    "ManagerDecoder",
    "Manifest",
    "ObservationAssembler",
    "ObservationEntry",
    "ObservationError",
    "ObservationLayout",
    "PolicySpec",
    "Provenance",
    "SchemaVersionError",
    "__version__",
    "load_bundle",
    "load_manifest",
    "save_bundle",
]
