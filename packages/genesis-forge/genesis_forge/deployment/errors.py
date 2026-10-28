"""Every error the export side raises.

Mirrors ``genesis_forge_runtime.errors`` on the runtime side, so both halves of the
deployment story expose one place to catch from.
"""


class ExportError(Exception):
    """The environment cannot be exported as it is currently configured."""


class ParityError(Exception):
    """The deployment pipeline disagreed with the training pipeline."""
