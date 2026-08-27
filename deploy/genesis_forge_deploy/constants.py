"""The deployment bundle's vocabulary.

These strings and numbers are the manifest's contract. They are deliberately
separate from the code that uses them: a bundle may be read by tooling that never
imports the rest of this package, and the values must not drift.
"""

#: Schema version this runtime writes and understands.
SCHEMA_VERSION = 1

#: Oldest bundle schema this runtime can still read.
MIN_SUPPORTED_SCHEMA_VERSION = 1

MANIFEST_FILENAME = "manifest.json"
GOLDEN_FILENAME = "golden.npz"

#: The policy is stored under this stem, keeping its own extension, so a bundle
#: never misrepresents what it holds.
POLICY_STEM = "policy"

#: Policy formats the bundle understands. The bundle records what the file *is*;
#: it does not require any particular one. The runtime itself is format-agnostic --
#: it hands you an observation vector and takes actions back, whatever ran between.
POLICY_FORMAT_ONNX = "onnx"
POLICY_FORMAT_TORCHSCRIPT = "torchscript"

#: An observation entry the user reads off real hardware each tick.
SOURCE_SENSOR = "sensor"
#: An observation entry whose value comes from the pipeline's own previous output.
SOURCE_PIPELINE_STATE = "pipeline_state"

#: Pipeline-state entry echoing the raw policy output.
STAGE_RAW_ACTIONS = "raw_actions"
#: Pipeline-state entry echoing an action manager's decoded joint targets.
STAGE_TARGET_ACTIONS = "target_actions"

#: History is concatenated newest-first, matching ObservationManager.
HISTORY_NEWEST_FIRST = "newest_first"
