"""Prove the deployment runtime reproduces the training pipeline, before export.

Manager-owned deployment counterparts are written by hand, so nothing structural
guarantees the numpy decode matches the torch ``process_actions`` it mirrors. This
gate closes that: it runs the *actual* ``genesis_forge_deploy`` classes -- the same
code the robot imports -- against the live torch pipeline on identical inputs, and
refuses the export if they disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from .comparison import PIPELINE_ATOL, PIPELINE_RTOL, max_abs_error, require_close
from .sampling import (
    preserved_observation_history,
    sample_actions,
    sample_observation_values,
    torch_observations,
)

if TYPE_CHECKING:  # pragma: no cover
    from genesis_forge_deploy import Manifest

    from .capture import Capture


@dataclass
class ParityReport:
    """What the parity run compared, and the worst disagreement it saw."""

    ticks: int
    max_observation_error: float = 0.0
    max_action_error: dict[str, float] = field(default_factory=dict)
    golden: dict[str, np.ndarray] = field(default_factory=dict)

    def summary(self) -> str:
        actions = ", ".join(
            f"{name}: {error:.2e}" for name, error in sorted(self.max_action_error.items())
        )
        return (
            f"parity over {self.ticks} tick(s) -- observations within "
            f"{self.max_observation_error:.2e}"
            + (f"; actions {actions}" if actions else "")
        )


def check_parity(
    capture: Capture,
    *,
    ticks: int = 6,
    seed: int = 0,
    rtol: float = PIPELINE_RTOL,
    atol: float = PIPELINE_ATOL,
) -> ParityReport:
    """Compare the numpy deployment pipeline against the live torch pipeline.

    Runs a multi-tick sequence so history stacking and any per-step decoder state
    are exercised, not just a single snapshot. Inputs mix seeded random values with
    clip-boundary values, since clipping is where the two implementations are most
    likely to diverge.

    Args:
        capture: What the exporter read out of the built environment.
        ticks: How many sequential ticks to compare.
        seed: Seed for the sampled inputs, so a failure is reproducible.
        rtol: Relative tolerance for the numpy-vs-torch comparison.
        atol: Absolute tolerance for the numpy-vs-torch comparison.

    Returns:
        A :class:`ParityReport`, including golden samples to ship in the bundle.

    Raises:
        ParityError: The two pipelines disagreed. The message names the component.
    """
    from genesis_forge_deploy import ActionDecoder, ObservationAssembler

    manifest: Manifest = capture.manifest
    assembler = ObservationAssembler(manifest.observations)
    decoder = ActionDecoder(manifest.actions)

    rng = np.random.default_rng(seed)
    report = ParityReport(ticks=ticks)

    golden_observations: list[np.ndarray] = []
    golden_policy_inputs: list[np.ndarray] = []
    golden_actions: list[np.ndarray] = []
    golden_targets: list[np.ndarray] = []

    # Training mutates its own history buffer as a side effect of observing, so
    # snapshot it and put it back -- exporting must not disturb the environment.
    with preserved_observation_history(capture.observation_manager):
        assembler.reset()
        decoder.reset()

        for tick in range(ticks):
            # Every entry is caller-supplied now -- including the ones that echo the
            # pipeline's own output -- so both sides get the same dict.
            observation_values = sample_observation_values(manifest, rng)

            numpy_obs = assembler.assemble(observation_values)
            torch_obs = torch_observations(capture, observation_values)

            error = max_abs_error(numpy_obs, torch_obs)
            report.max_observation_error = max(report.max_observation_error, error)
            require_close(
                numpy_obs,
                torch_obs,
                rtol=rtol,
                atol=atol,
                component="observation assembly",
                detail=(
                    f"tick {tick}: the deployment assembler and the training "
                    f"ObservationManager produced different vectors"
                ),
            )

            raw_actions = sample_actions(manifest, rng, tick=tick)
            decoded = decoder.decode(raw_actions)

            for spec in manifest.actions:
                manager = capture.action_managers[spec.name]
                chunk = raw_actions[spec.slice_start : spec.slice_end]
                torch_chunk = torch.as_tensor(
                    np.tile(chunk, (capture.num_envs, 1)), dtype=torch.float32
                )
                torch_targets = manager.process_actions(torch_chunk).detach()[0]
                numpy_targets = decoded.by_manager[spec.name]

                error = max_abs_error(numpy_targets, torch_targets)
                report.max_action_error[spec.name] = max(
                    report.max_action_error.get(spec.name, 0.0), error
                )
                require_close(
                    numpy_targets,
                    torch_targets,
                    rtol=rtol,
                    atol=atol,
                    component=f"action manager '{spec.name}' ({spec.deploy_type})",
                    detail=(
                        f"tick {tick}: the deployment decoder and the manager's "
                        f"process_actions produced different joint targets"
                    ),
                )

            golden_observations.append(numpy_obs)
            golden_policy_inputs.append(
                np.concatenate(
                    [observation_values[name] for name in sorted(observation_values)]
                )
                if observation_values
                else np.zeros(0, dtype=np.float32)
            )
            golden_actions.append(raw_actions)
            golden_targets.append(decoded.targets)

    report.golden = {
        "observations": np.asarray(golden_observations, dtype=np.float32),
        "raw_actions": np.asarray(golden_actions, dtype=np.float32),
        "joint_targets": np.asarray(golden_targets, dtype=np.float32),
    }
    if golden_policy_inputs and golden_policy_inputs[0].size:
        report.golden["sensor_inputs"] = np.asarray(
            golden_policy_inputs, dtype=np.float32
        )
    return report
