"""Check an exported ONNX policy against the live torch policy.

Closes the seam the pipeline gate cannot see. A policy graph that silently dropped
its observation normalizer passes every other check and then misbehaves on
hardware -- the classic sim-to-real failure, and invisible without this.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .comparison import POLICY_ATOL
from .errors import ParityError


def check_policy_parity(
    policy_path: str,
    torch_policy: Any,
    observations: np.ndarray,
    *,
    input_name: str = "obs",
    atol: float = POLICY_ATOL,
) -> float:
    """Compare an exported ONNX policy against the live torch policy.

    Closes the seam the pipeline check cannot see: a policy graph that silently
    dropped its observation normalizer passes every other check and then misbehaves
    on hardware.

    Args:
        policy_path: The exported ``.onnx`` file.
        torch_policy: A callable taking a float32 tensor and returning actions.
        observations: Observation vectors to compare on, shaped ``(ticks, size)``.
        input_name: The ONNX graph's input name.
        atol: Absolute tolerance.

    Returns:
        The largest absolute difference observed.

    Raises:
        ParityError: The two disagreed, or onnxruntime is not installed.
    """
    try:
        import onnxruntime
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ParityError(
            "onnxruntime is required to verify an exported policy. Install it with "
            "`pip install onnxruntime`, or export without a policy file."
        ) from error

    session = onnxruntime.InferenceSession(
        str(policy_path), providers=["CPUExecutionProvider"]
    )
    worst = 0.0
    for observation in np.atleast_2d(observations).astype(np.float32):
        batched = observation[None, :]
        onnx_actions = np.asarray(session.run(None, {input_name: batched})[0]).ravel()
        with torch.no_grad():
            torch_actions = (
                torch_policy(torch.as_tensor(batched, dtype=torch.float32))
                .detach()
                .cpu()
                .numpy()
                .ravel()
            )
        worst = max(worst, float(np.max(np.abs(onnx_actions - torch_actions))))
        if worst > atol:
            raise ParityError(
                f"The exported ONNX policy disagrees with the trained torch policy "
                f"(largest difference {worst:.3e}, tolerance {atol:.1e}). The usual "
                f"cause is an observation normalizer that did not make it into the "
                f"exported graph."
            )
    return worst
