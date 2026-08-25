"""Every error this package raises.

Collected in one module so a robot-side caller can see the whole hierarchy at a
glance and catch at whatever granularity it wants::

    try:
        bundle = load_bundle(path)
    except SchemaVersionError:
        ...   # re-export from a matching Genesis Forge
    except BundleError:
        ...   # anything else wrong with the bundle
"""


class BundleError(Exception):
    """Base class for every error raised while reading a bundle."""


class SchemaVersionError(BundleError):
    """The bundle was written by an incompatible version of Genesis Forge."""


class MalformedBundleError(BundleError):
    """The bundle is missing a required section or holds an unusable value."""


class ObservationError(Exception):
    """A value handed to the assembler was missing, mis-sized, or unexpected."""


class DecoderError(Exception):
    """A decoder could not be resolved, or was handed unusable policy output."""
