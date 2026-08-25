"""Simulation-free runtime for deploying Genesis Forge policies to real robots.

This package reproduces the observation-assembly and action-decoding pipelines
from training, reading them out of a bundle exported by ``genesis_forge``. It
depends on **numpy only** -- importing anything here must never pull in torch or
the Genesis simulator, so it installs on a Raspberry Pi or Jetson.

Typical robot-side use::

    from genesis_forge_deploy import load_bundle

    bundle = load_bundle("./my_policy")
    print(bundle.describe())          # what to wire up

    assembler = bundle.observation_assembler()
    decoder = bundle.action_decoder()

    while True:
        obs = assembler.assemble({"robot_ang_vel": gyro, ...})
        actions = policy(obs)          # onnxruntime, or anything else
        targets = decoder.decode(actions)
        send_to_motors(targets)

**Trust model:** a bundle is trusted input, equivalent to executable code --
loading one may import decoder classes it names. Only load bundles you produced.
"""

from .actions import (
    ActionDecoder,
    AffineDecoder,
    DecodedActions,
    DecoderError,
    ManagerDecoder,
)
from .bundle import (
    GOLDEN_FILENAME,
    HISTORY_NEWEST_FIRST,
    MANIFEST_FILENAME,
    MIN_SUPPORTED_SCHEMA_VERSION,
    POLICY_FILENAME,
    SCHEMA_VERSION,
    SOURCE_PIPELINE_STATE,
    SOURCE_SENSOR,
    STAGE_PROCESSED_ACTIONS,
    STAGE_RAW_ACTIONS,
    ActionManagerSpec,
    ActuatorSpec,
    Bundle,
    BundleError,
    MalformedBundleError,
    Manifest,
    ObservationEntry,
    ObservationLayout,
    PolicySpec,
    Provenance,
    SchemaVersionError,
    load_bundle,
    load_manifest,
    save_bundle,
)
from .observations import ObservationAssembler, ObservationError

__version__ = "1.0.0"

__all__ = [
    "GOLDEN_FILENAME",
    "HISTORY_NEWEST_FIRST",
    "MANIFEST_FILENAME",
    "MIN_SUPPORTED_SCHEMA_VERSION",
    "POLICY_FILENAME",
    "SCHEMA_VERSION",
    "SOURCE_PIPELINE_STATE",
    "SOURCE_SENSOR",
    "STAGE_PROCESSED_ACTIONS",
    "STAGE_RAW_ACTIONS",
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
