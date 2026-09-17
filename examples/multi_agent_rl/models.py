"""Gaussian legs + deterministic centralized critics for SKRL 2.x MAPPO."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from skrl.models.torch import Model
from skrl.models.torch.deterministic import DeterministicMixin
from skrl.models.torch.gaussian import GaussianMixin
from torch import nn


class MasqGaussianPolicy(GaussianMixin, Model):
    """Gaussian actor conditioned on decentralized observations."""

    def __init__(
        self,
        observation_space,
        action_space,
        *,
        hidden_dims=(256, 128, 64),
        device=None,
        clip_actions=False,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=None,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=True,
            reduction="sum",
            role="policy",
        )
        layers: list[nn.Module] = []
        prev = self.num_observations
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ELU()])
            prev = h
        layers.append(nn.Linear(prev, self.num_actions))
        self.net = nn.Sequential(*layers)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        x = inputs["observations"]
        mean = self.net(x)
        log_std = self.log_std_parameter.expand_as(mean)
        return mean, {"log_std": log_std}


class MasqValue(DeterministicMixin, Model):
    """Scalar critic on centralized states only."""

    def __init__(
        self,
        state_space,
        *,
        hidden_dims=(512, 256, 128),
        device=None,
    ):
        dummy_action = spaces.Box(
            low=-np.ones(1, dtype=np.float32),
            high=np.ones(1, dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )
        Model.__init__(
            self,
            observation_space=None,
            state_space=state_space,
            action_space=dummy_action,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False, role="value")
        layers: list[nn.Module] = []
        prev = self.num_states
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ELU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        x = inputs["states"]
        return self.net(x), {}
