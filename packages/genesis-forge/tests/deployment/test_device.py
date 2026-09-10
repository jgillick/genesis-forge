"""The parity gate must work on the device the environment was built on.

Training happens on a GPU. The managers place their buffers on ``gs.device`` at
build time, so the tensors the gate feeds them have to live there too -- and the
rest of the suite cannot see this, because it pins ``gs.device`` to the CPU.
"""

import genesis as gs
import pytest
import torch

from genesis_forge.deployment import capture_environment, check_parity


def an_accelerator():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


@pytest.fixture
def built_on_an_accelerator(deployable_env, monkeypatch):
    """Move the environment's buffers off the CPU, as a real build would."""
    device = an_accelerator()
    if device is None:
        pytest.skip("no accelerator available to stand in for a GPU trainer")
    monkeypatch.setattr(gs, "device", device, raising=False)

    for manager in deployable_env.managers["action"]:
        for name, value in list(vars(manager).items()):
            if isinstance(value, torch.Tensor):
                setattr(manager, name, value.to(device))
    return deployable_env


def test_parity_runs_against_managers_that_are_not_on_the_cpu(built_on_an_accelerator):
    report = check_parity(capture_environment(built_on_an_accelerator))

    assert report.max_observation_error < 1e-5
    assert all(error < 1e-5 for error in report.max_action_error.values())
