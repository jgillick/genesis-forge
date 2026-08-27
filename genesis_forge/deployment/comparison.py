"""Deciding whether two pipelines agree.

Tolerances are tiered by what can legitimately differ:

* numpy vs torch pipeline math -- the same operations in two libraries, so
  near-bit-exact. Ordering and scale bugs produce large errors, which means a
  tight bound costs nothing and catches everything.
* onnxruntime vs torch -- graph rewrites change accumulation order, so looser,
  and *relative*: that drift scales with activation magnitude, and action spaces
  are not all unit-scale. A wheeled robot emitting velocity targets around 10
  drifts ~30x further than one emitting joint positions around 1, for exactly the
  same graph.
"""

from __future__ import annotations

import numpy as np
import torch

from .errors import ParityError

PIPELINE_RTOL = 1.3e-6
PIPELINE_ATOL = 1e-5

#: Measured on a trained wheeled-robot policy (actions ~10, obs width 18): matched
#: weights drift at most 2.7e-05 over N(0,1) observations, while the *closest*
#: wrong checkpoint diverges by at least 2.3e-01. Anything between those is not a
#: case that occurs, so rtol sits ~30x above the noise and ~2000x below a real bug.
POLICY_RTOL = 1e-4
POLICY_ATOL = 1e-5


def max_abs_error(numpy_values: np.ndarray, torch_values: torch.Tensor) -> float:
    expected = torch_values.detach().cpu().numpy().ravel()
    actual = np.asarray(numpy_values).ravel()
    if expected.shape != actual.shape:
        return float("inf")
    if expected.size == 0:
        return 0.0
    return float(np.max(np.abs(actual - expected)))


def require_close(
    numpy_values: np.ndarray,
    torch_values: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    component: str,
    detail: str,
) -> None:
    expected = torch_values.detach().cpu().numpy().ravel()
    actual = np.asarray(numpy_values).ravel()

    if expected.shape != actual.shape:
        raise ParityError(
            f"Parity failed in {component}. {detail}: the deployment pipeline "
            f"produced {actual.shape[0]} value(s) where training produced "
            f"{expected.shape[0]}."
        )

    if np.allclose(actual, expected, rtol=rtol, atol=atol):
        return

    difference = np.abs(actual - expected)
    worst = int(np.argmax(difference))
    raise ParityError(
        f"Parity failed in {component}. {detail}. Largest difference "
        f"{difference[worst]:.3e} at index {worst}: deployment produced "
        f"{actual[worst]:.6g}, training produced {expected[worst]:.6g} "
        f"(tolerance rtol={rtol:g}, atol={atol:g}). The bundle was not written."
    )
