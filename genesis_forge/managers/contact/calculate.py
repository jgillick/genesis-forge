"""
PyTorch JIT scripts used to process and calculate the contact forces.
"""

import torch


@torch.jit.script
def _inv_transform_by_quat(
    vec: torch.Tensor,
    quat: torch.Tensor,
) -> None:
    """
    JIT-compiled inverse quaternion rotation (world to local frame).

    Args:
        vec: Vectors to rotate and output (..., 3) - modified in place
        quat: Unit quaternions (..., 4) in [w, x, y, z] format
    """
    # Extract quaternion components [w, x, y, z]
    qw = quat[..., 0]
    qx = quat[..., 1]
    qy = quat[..., 2]
    qz = quat[..., 3]

    # Extract vector components
    vx = vec[..., 0]
    vy = vec[..., 1]
    vz = vec[..., 2]

    # First cross product: q_xyz × v
    cx = qy * vz - qz * vy
    cy = qz * vx - qx * vz
    cz = qx * vy - qy * vx

    # Second cross product: q_xyz × (q_xyz × v)
    ccx = qy * cz - qz * cy
    ccy = qz * cx - qx * cz
    ccz = qx * cy - qy * cx

    # Write result back in place
    vec[..., 0] = vx - 2.0 * qw * cx + 2.0 * ccx
    vec[..., 1] = vy - 2.0 * qw * cy + 2.0 * ccy
    vec[..., 2] = vz - 2.0 * qw * cz + 2.0 * ccz


@torch.jit.script
def calculate_contact_forces(
    contact_forces: torch.Tensor,
    contact_positions: torch.Tensor,
    link_a: torch.Tensor,
    link_b: torch.Tensor,
    links_quat: torch.Tensor,
    target_link_ids: torch.Tensor,
    has_with_filter: bool,
    with_link_ids: torch.Tensor,
    output_forces: torch.Tensor,
    output_positions: torch.Tensor,
) -> None:
    """
    Process the data from the get_contacts method, and filter/accumulate the data by target link IDs.
    Optionally, if `with_link_ids` is defined, only contacts made between `target_link_ids` and `with_link_ids` will be considered.

    Args:
        contact_forces: Contact force data, shape (n_envs, n_contacts, 3)
        contact_positions: Contact position data, shape (n_envs, n_contacts, 3)
        link_a: First link in each contact, shape (n_envs, n_contacts)
        link_b: Second link in each contact, shape (n_envs, n_contacts)
        links_quat: Link quaternions, shape (n_envs, n_links, 4)
        target_link_ids: Target link IDs to collect contact forces for. Tensor shape: (n_target_links)
        has_with_filter: If the with_link_filter should be applied.
        with_link_ids: If defined, only contacts made with these link IDs AND target_link_ids will be considered. Tensor shape: (n_with_links)
        output_forces: Output force tensor, shape (n_envs, n_target_links, 3)
        output_positions: Output position tensor, shape (n_envs, n_target_links, 3)
    """
    n_contacts = contact_forces.shape[1]

    # Zero outputs
    output_forces.zero_()
    output_positions.zero_()

    # Early exit if no contacts
    if n_contacts == 0:
        return

    # Expand for broadcasting: compare all contacts against all targets at once
    link_a_exp = link_a.unsqueeze(-1)  # (n_envs, n_contacts, 1)
    link_b_exp = link_b.unsqueeze(-1)  # (n_envs, n_contacts, 1)
    targets = target_link_ids.view(1, 1, -1)  # (1, 1, n_targets)

    # Compute masks for all targets at once: (n_envs, n_contacts, n_targets)
    is_target_a = link_a_exp == targets
    is_target_b = link_b_exp == targets

    # Apply with_link filter if enabled
    if has_with_filter:
        with_links = with_link_ids.view(1, 1, -1)  # (1, 1, n_with)
        link_a_with = (link_a_exp == with_links).any(
            dim=-1, keepdim=True
        )  # (n_envs, n_contacts, 1)
        link_b_with = (link_b_exp == with_links).any(
            dim=-1, keepdim=True
        )  # (n_envs, n_contacts, 1)
        is_target_a = is_target_a & link_b_with
        is_target_b = is_target_b & link_a_with

    # Valid mask: (n_envs, n_contacts, n_targets)
    valid_mask = is_target_a | is_target_b

    # =========================================================================
    # Calculate positions
    # =========================================================================
    valid_mask_pos = valid_mask.unsqueeze(2)  # (n_envs, n_contacts, 1, n_targets)
    positions_exp = contact_positions.unsqueeze(-1)  # (n_envs, n_contacts, 3, 1)

    # Multiply and sum over contacts
    pos_sum = (positions_exp * valid_mask_pos.float()).sum(
        dim=1
    )  # (n_envs, 3, n_targets)

    # Count per target
    contact_count = (
        valid_mask.float().sum(dim=1).clamp(min=1.0).unsqueeze(1)
    )  # (n_envs, n_targets)

    # Average and transpose to (n_envs, n_targets, 3)
    avg_pos = pos_sum / contact_count  # (n_envs, 3, n_targets)
    output_positions.copy_(avg_pos.permute(0, 2, 1))

    # =========================================================================
    # Calculate forces
    # =========================================================================

    # Sign: +1 for is_target_b, -1 for is_target_a_only
    force_sign = (
        is_target_b.float() - (is_target_a & ~is_target_b).float()
    )  # (n_envs, n_contacts, n_targets)
    force_sign = force_sign.unsqueeze(2)  # (n_envs, n_contacts, 1, n_targets)

    # Multiply and sum over contacts
    forces_exp = contact_forces.unsqueeze(-1)  # (n_envs, n_contacts, 3, 1)
    world_forces = (forces_exp * force_sign).sum(dim=1)  # (n_envs, 3, n_targets)

    # Transpose to (n_envs, n_targets, 3)
    output_forces.copy_(world_forces.permute(0, 2, 1))

    # =========================================================================
    # Convert forces to local frame
    # =========================================================================
    target_quats = links_quat[:, target_link_ids, :]
    _inv_transform_by_quat(output_forces, target_quats)
