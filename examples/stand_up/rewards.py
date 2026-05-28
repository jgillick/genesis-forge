import torch
import math
from genesis_forge.genesis_env import GenesisEnv


def stand_and_balance_reward(
    env: GenesisEnv,
    entity_manager,
    target_height: float = 0.28,
    max_tilt_deg: float = 20.0,
) -> torch.Tensor:
    """
    Positive reward for being upright and standing.

    This is intentionally a *dense* reward (paid every step while the condition holds)
    so the policy learns to both reach the posture and maintain it until timeout.
    """
    base_pos = entity_manager.base_pos
    projected_gravity = entity_manager.get_projected_gravity()
    tilt = torch.norm(projected_gravity[:, :2], dim=1)
    upright = tilt < math.sin(math.radians(max_tilt_deg))
    tall_enough = base_pos[:, 2] > target_height
    return (upright & tall_enough).float()