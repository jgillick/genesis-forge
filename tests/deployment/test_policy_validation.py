"""Verifying an exported policy file against the policy it came from.

The bundle does not require a particular format -- it records what the file is.
ONNX is the documented path; TorchScript is verified the same way. Anything else
is packaged unverified rather than refused.
"""

import numpy as np
import pytest
import torch

from genesis_forge.deployment import (
    ParityError,
    infer_policy_format,
    validate_policy,
    validate_torchscript_policy,
)

OBS, ACT = 4, 2


def a_policy(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Linear(OBS, ACT).eval()


def as_torchscript(module: torch.nn.Module, path) -> str:
    torch.jit.save(torch.jit.trace(module, torch.zeros(1, OBS)), str(path))
    return str(path)


def observations(count: int = 4) -> np.ndarray:
    return np.random.default_rng(0).uniform(-2, 2, (count, OBS)).astype(np.float32)


"""
Recognising what a policy file is
"""


def test_the_format_is_inferred_from_the_extension(tmp_path):
    policy = as_torchscript(a_policy(), tmp_path / "policy.pt")

    assert infer_policy_format(policy) == "torchscript"


def test_an_unrecognised_extension_is_reported_as_unknown(tmp_path):
    blob = tmp_path / "policy.weights"
    blob.write_bytes(b"some custom format")

    assert infer_policy_format(blob) == "unknown"


def test_an_onnx_extension_over_a_torch_archive_is_caught(tmp_path):
    """A bundle must not claim to hold an ONNX graph when it holds a torch archive."""
    mislabelled = tmp_path / "policy.onnx"
    as_torchscript(a_policy(), mislabelled)

    with pytest.raises(ParityError) as error:
        infer_policy_format(mislabelled)

    message = str(error.value)
    assert "policy.onnx" in message
    assert "torch archive" in message


def test_a_torchscript_extension_over_a_non_archive_is_caught(tmp_path):
    truncated = tmp_path / "policy.pt"
    truncated.write_bytes(b"not a torch archive at all")

    with pytest.raises(ParityError) as error:
        infer_policy_format(truncated)

    assert "TorchScript" in str(error.value)


"""
TorchScript validation
"""


def test_a_matching_torchscript_module_passes(tmp_path):
    policy = a_policy()
    exported = as_torchscript(policy, tmp_path / "policy.pt")

    worst = validate_torchscript_policy(exported, policy, observations())

    assert worst < 1e-6


def test_a_torchscript_module_from_a_different_policy_is_caught(tmp_path):
    """The realistic failure: a stale file from an earlier run."""
    stale = as_torchscript(a_policy(seed=1), tmp_path / "policy.pt")

    with pytest.raises(ParityError) as error:
        validate_torchscript_policy(stale, a_policy(seed=2), observations())

    message = str(error.value)
    assert "TorchScript module" in message
    assert "stale file" in message  # the likely cause is named, not just the number


def test_a_file_that_is_not_torchscript_is_reported_clearly(tmp_path):
    broken = tmp_path / "policy.ts"
    broken.write_bytes(b"PK\x03\x04 but not really a module")

    with pytest.raises(ParityError) as error:
        validate_torchscript_policy(broken, a_policy(), observations())

    assert "TorchScript" in str(error.value)


"""
Dispatching on format
"""


def test_validate_policy_routes_torchscript_to_its_validator(tmp_path):
    policy = a_policy()
    exported = as_torchscript(policy, tmp_path / "policy.pt")

    assert validate_policy(exported, policy, observations()) < 1e-6


def test_an_unknown_format_has_no_validator(tmp_path):
    blob = tmp_path / "policy.weights"
    blob.write_bytes(b"hand-rolled weights for a C++ runtime")

    with pytest.raises(ParityError) as error:
        validate_policy(blob, a_policy(), observations())

    message = str(error.value)
    assert "No validator" in message
    # It is packaged regardless -- the bundle does not require a format we know.
    assert "still packaged" in message


def test_the_format_can_be_declared_instead_of_inferred(tmp_path):
    """An unusual extension does not stop you naming the format yourself."""
    policy = a_policy()
    exported = as_torchscript(policy, tmp_path / "policy.weights")

    worst = validate_policy(
        exported, policy, observations(), policy_format="torchscript"
    )

    assert worst < 1e-6


"""
ONNX validation
"""


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_matching_onnx_graph_passes(tmp_path):
    pytest.importorskip("onnxruntime")
    from genesis_forge.deployment import validate_onnx_policy

    policy = a_policy()
    exported = tmp_path / "policy.onnx"
    torch.onnx.export(
        policy, (torch.zeros(1, OBS),), str(exported),
        input_names=["obs"], output_names=["actions"], opset_version=18, dynamo=False,
    )

    assert validate_onnx_policy(exported, policy, observations()) < 1e-5


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_an_onnx_graph_from_a_different_policy_is_caught(tmp_path):
    pytest.importorskip("onnxruntime")
    from genesis_forge.deployment import validate_onnx_policy

    exported = tmp_path / "policy.onnx"
    torch.onnx.export(
        a_policy(seed=1), (torch.zeros(1, OBS),), str(exported),
        input_names=["obs"], output_names=["actions"], opset_version=18, dynamo=False,
    )

    with pytest.raises(ParityError) as error:
        validate_onnx_policy(exported, a_policy(seed=2), observations())

    assert "ONNX graph" in str(error.value)


"""
Devices

Training leaves torch's default device pointing at the GPU, so validation has to be
explicit about where its tensors go rather than inheriting that.
"""


def an_accelerator() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.fixture
def default_device_is_the_gpu():
    device = an_accelerator()
    if device is None:
        pytest.skip("no accelerator available to set as the default device")
    torch.set_default_device(device)
    try:
        yield device
    finally:
        torch.set_default_device("cpu")


def test_validation_works_with_the_default_device_on_the_gpu(
    default_device_is_the_gpu, tmp_path
):
    """The realistic post-training state: torch's default device is the GPU.

    Both sides have to be explicit -- the exported module is loaded onto the CPU
    and needs a CPU input, while the reference policy is still on the accelerator
    and needs one there.
    """
    policy = a_policy()
    assert next(policy.parameters()).device.type == default_device_is_the_gpu
    exported = as_torchscript(policy, tmp_path / "policy.pt")

    assert validate_torchscript_policy(exported, policy, observations()) < 1e-5


"""
Tolerance shape

The drift between runtimes scales with activation magnitude, so the bound is
relative. An absolute-only bound fails any policy whose actions are not unit-scale
-- a wheeled robot emitting velocity targets around 10 is the case that found this.
"""

BIG_ACTION_SCALE = 10.0


class BigActionPolicy(torch.nn.Module):
    """A policy with actions around 10, like velocity targets."""

    def __init__(self, drift: float = 0.0):
        super().__init__()
        torch.manual_seed(0)
        self.linear = torch.nn.Linear(OBS, ACT)
        self.drift = drift

    def forward(self, observations):
        return self.linear(observations) * BIG_ACTION_SCALE * (1.0 + self.drift)


def test_float32_drift_passes_on_a_policy_with_large_actions(tmp_path):
    """Relative drift of 3e-5 on actions of ~10 is rounding, not a broken export.

    The absolute difference here is ~3e-4 -- thirty times the old absolute bound,
    which is why a real trained wheeled-robot policy could not be exported.
    """
    exported = as_torchscript(BigActionPolicy(drift=3e-5), tmp_path / "policy.pt")

    worst = validate_torchscript_policy(exported, BigActionPolicy(), observations())

    assert worst > 1e-5, "the difference must exceed the old absolute bound"
    assert worst < 1e-2


def test_a_real_mismatch_is_still_caught_on_a_policy_with_large_actions(tmp_path):
    """Relative drift of 1% is a broken export at any action scale."""
    exported = as_torchscript(BigActionPolicy(drift=1e-2), tmp_path / "policy.pt")

    with pytest.raises(ParityError) as error:
        validate_torchscript_policy(exported, BigActionPolicy(), observations())

    assert "disagrees with the trained policy" in str(error.value)


def test_near_zero_actions_still_get_an_absolute_floor(tmp_path):
    """Relative tolerance alone would be vanishingly tight around zero."""
    zeros = torch.nn.Linear(OBS, ACT)
    torch.nn.init.zeros_(zeros.weight)
    torch.nn.init.zeros_(zeros.bias)

    exported = as_torchscript(zeros, tmp_path / "policy.pt")

    assert validate_torchscript_policy(exported, zeros, observations()) == 0.0
