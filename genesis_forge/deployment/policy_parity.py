"""Check an exported policy file against the trained policy it came from.

Closes the seam the pipeline gate cannot see: the graph or module in the bundle is
compared against the policy it was exported from, on the same observations. That
catches a conversion that changed the maths -- and, more often in practice, a
stale or mismatched file, which is silent and expensive to discover on hardware.

The bundle does not require any particular format. ONNX is the documented path, so
it has the most support here, but a TorchScript module is verified the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .comparison import POLICY_ATOL, POLICY_RTOL
from .errors import ParityError

#: File extensions we recognise, and what format they mean.
_EXTENSION_FORMATS = {
    ".onnx": "onnx",
    ".pt": "torchscript",
    ".pth": "torchscript",
    ".jit": "torchscript",
    ".ts": "torchscript",
}

#: torch's archives are ZIPs; ONNX files are raw protobuf.
_ZIP_MAGIC = b"PK\x03\x04"


def infer_policy_format(policy_path: str | Path) -> str:
    """Work out what kind of policy file this is.

    Uses the extension, then checks the file's own bytes for a contradiction --
    a ``.onnx`` file that is really a torch archive is a mislabelling worth
    catching at export time rather than on the robot.

    Args:
        policy_path: The exported policy file.

    Returns:
        The format name, or ``"unknown"`` for an extension we do not recognise
        (which is fine: the bundle records it and the runtime never loads it).

    Raises:
        ParityError: The extension and the file's contents disagree.
    """
    path = Path(policy_path)
    declared = _EXTENSION_FORMATS.get(path.suffix.lower(), "unknown")

    with path.open("rb") as handle:
        looks_like_archive = handle.read(4) == _ZIP_MAGIC

    if declared == "onnx" and looks_like_archive:
        raise ParityError(
            f"'{path.name}' has an .onnx extension but its contents are a torch "
            f"archive, not an ONNX graph. Export it with "
            f"runner.export_policy_to_onnx(...), or give the file its real "
            f"extension so the bundle records what it actually holds."
        )
    if declared == "torchscript" and not looks_like_archive:
        raise ParityError(
            f"'{path.name}' looks like a TorchScript file by extension, but its "
            f"contents are not a torch archive. Check that the export completed."
        )
    return declared


def validate_policy(
    policy_path: str | Path,
    reference_policy: Any,
    observations: np.ndarray,
    *,
    policy_format: str | None = None,
    input_name: str = "obs",
    rtol: float = POLICY_RTOL,
    atol: float = POLICY_ATOL,
) -> float:
    """Verify an exported policy against the one it came from, whatever its format.

    Args:
        policy_path: The exported policy file.
        reference_policy: What to check the file against -- the trained policy
            itself, as a torch callable taking a float32 tensor and returning
            actions.
        observations: Observation vectors to compare on, shaped ``(ticks, size)``.
        policy_format: Override the format instead of inferring it from the file.
        input_name: The ONNX graph's input name. Ignored for other formats.
        rtol: Relative tolerance, applied against the reference action. This is
            the one that matters -- the drift between runtimes scales with how
            large the actions are.
        atol: Absolute tolerance, which only takes over near zero.

    Returns:
        The largest absolute difference observed.

    Raises:
        ParityError: The two disagreed, or the format has no validator.
    """
    resolved = policy_format or infer_policy_format(policy_path)

    if resolved == "onnx":
        return validate_onnx_policy(
            policy_path,
            reference_policy,
            observations,
            input_name=input_name,
            rtol=rtol,
            atol=atol,
        )
    if resolved == "torchscript":
        return validate_torchscript_policy(
            policy_path, reference_policy, observations, rtol=rtol, atol=atol
        )
    raise ParityError(
        f"No validator for policy format '{resolved}'. The file is still packaged "
        f"into the bundle -- omit the trained policy to skip this check, or pass "
        f"policy_format= if the format was mis-detected."
    )


def validate_onnx_policy(
    policy_path: str | Path,
    reference_policy: Any,
    observations: np.ndarray,
    *,
    input_name: str = "obs",
    rtol: float = POLICY_RTOL,
    atol: float = POLICY_ATOL,
) -> float:
    """Compare an ONNX graph under onnxruntime against the trained torch policy."""
    try:
        import onnxruntime
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ParityError(
            "onnxruntime is required to verify an exported ONNX policy. Install it "
            "with `pip install onnxruntime`, or export without a policy file."
        ) from error

    session = onnxruntime.InferenceSession(
        str(policy_path), providers=["CPUExecutionProvider"]
    )

    def run(batched: np.ndarray) -> np.ndarray:
        return np.asarray(session.run(None, {input_name: batched})[0]).ravel()

    return _compare(
        run, reference_policy, observations, rtol=rtol, atol=atol, kind="ONNX graph"
    )


def validate_torchscript_policy(
    policy_path: str | Path,
    reference_policy: Any,
    observations: np.ndarray,
    *,
    rtol: float = POLICY_RTOL,
    atol: float = POLICY_ATOL,
) -> float:
    """Compare a TorchScript module against the trained torch policy."""
    try:
        module = torch.jit.load(str(policy_path), map_location="cpu")
    except Exception as error:
        raise ParityError(
            f"Could not load '{Path(policy_path).name}' as TorchScript: {error}"
        ) from error
    module.eval()

    def run(batched: np.ndarray) -> np.ndarray:
        # The module is loaded onto the CPU, so pin its input there too -- training
        # may well have left torch's default device pointing at the GPU.
        with torch.no_grad():
            output = module(
                torch.as_tensor(batched, dtype=torch.float32, device="cpu")
            )
        return output.detach().cpu().numpy().ravel()

    return _compare(
        run,
        reference_policy,
        observations,
        rtol=rtol,
        atol=atol,
        kind="TorchScript module",
    )


def _policy_device(reference_policy: Any) -> torch.device:
    """Where to put the reference policy's input.

    The exported file is always loaded onto the CPU, but the policy we compare it
    against may still be on the training device. Feeding it a CPU tensor would fail
    with a bare device-mismatch error, so follow the policy instead.
    """
    try:
        return next(reference_policy.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def _compare(
    run_exported: Any,
    reference_policy: Any,
    observations: np.ndarray,
    *,
    rtol: float,
    atol: float,
    kind: str,
) -> float:
    """Run both policies over the same observations and report the worst gap.

    Compared relative to the reference action, since the drift between runtimes
    grows with activation magnitude: the same graph emitting velocity targets
    around 10 drifts far further than one emitting joint angles around 1.
    """
    device = _policy_device(reference_policy)
    worst = 0.0
    for observation in np.atleast_2d(observations).astype(np.float32):
        batched = observation[None, :]
        exported_actions = run_exported(batched)
        with torch.no_grad():
            reference = (
                reference_policy(
                    torch.as_tensor(batched, dtype=torch.float32, device=device)
                )
                .detach()
                .cpu()
                .numpy()
                .ravel()
            )

        difference = np.abs(exported_actions - reference)
        worst = max(worst, float(difference.max()))

        allowed = atol + rtol * np.abs(reference)
        if np.any(difference > allowed):
            index = int(np.argmax(difference - allowed))
            raise ParityError(
                f"The exported {kind} disagrees with the trained policy. Action "
                f"{index} differs by {difference[index]:.3e}: the file produced "
                f"{exported_actions[index]:.6g}, the trained policy produced "
                f"{reference[index]:.6g} (allowed {allowed[index]:.3e}, from "
                f"rtol={rtol:g}, atol={atol:g}).\n"
                f"A difference this size is not rounding -- the usual causes are a "
                f"stale file from an earlier run, or an observation normalizer that "
                f"did not make it into the exported graph. The bundle was not "
                f"written."
            )
    return worst
