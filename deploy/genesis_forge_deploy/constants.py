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

#: A bundle written as a single file rather than a directory: a zip archive with
#: the same contents. The extension names the bundle, not the container -- as with
#: .jar and .whl -- so it stays true if the container ever changes. Reading goes by
#: content rather than by name, so a bundle renamed to .zip still loads.
ARCHIVE_SUFFIX = ".gfb"

#: Written inside an extracted archive, holding a fingerprint of the archive it
#: came from. A later load re-extracts unless the fingerprint still matches, so
#: replacing the archive can never leave a robot running the previous bundle.
EXTRACT_MARKER = ".source"

#: The policy is stored under this stem, keeping its own extension, so a bundle
#: never misrepresents what it holds.
POLICY_STEM = "policy"

#: Policy formats the bundle understands. The bundle records what the file *is*;
#: it does not require any particular one. The runtime itself is format-agnostic --
#: it hands you an observation vector and takes actions back, whatever ran between.
POLICY_FORMAT_ONNX = "onnx"
POLICY_FORMAT_TORCHSCRIPT = "torchscript"



#: History is concatenated newest-first, matching ObservationManager.
HISTORY_NEWEST_FIRST = "newest_first"
