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

#: Suffix for a bundle written as one file: a zip holding what the directory
#: would. Bundles are recognised by content, so a renamed one still loads.
ARCHIVE_SUFFIX = ".gfb"

#: Written inside an extracted archive, holding a fingerprint of the archive it
#: came from. A later load re-extracts unless the fingerprint still matches, so
#: replacing the archive can never leave a robot running the previous bundle.
EXTRACT_MARKER = ".source"

#: Directory inside the bundle holding the exported policy, under the filenames
#: the exporter was given. The runtime never opens them.
POLICY_DIRNAME = "policy"
