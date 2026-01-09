"""
PyTorch JIT scripts used to process and calculate the contact forces.
"""

import torch


@torch.jit.script
def _inv_transform_by_quat_jit(vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled inverse quaternion rotation (world to local frame).

    Rotates vectors by the inverse of the given quaternions.
    Uses the formula: v' = q* v q (where q* is quaternion conjugate)

    Args:
        vec: Vectors to rotate (..., 3)
        quat: Unit quaternions (..., 4) in [w, x, y, z] format (Genesis convention)

    Returns:
        Rotated vectors (..., 3)
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

    # For inverse rotation, use conjugate: q* = [w, -x, -y, -z]
    # Efficient formula: v' = v + 2*w*(q_xyz × v) + 2*(q_xyz × (q_xyz × v))
    # With conjugate (negated xyz): v' = v - 2*w*(q_xyz × v) + 2*(q_xyz × (q_xyz × v))

    # First cross product: q_xyz × v
    cx = qy * vz - qz * vy
    cy = qz * vx - qx * vz
    cz = qx * vy - qy * vx

    # Second cross product: q_xyz × (q_xyz × v)
    ccx = qy * cz - qz * cy
    ccy = qz * cx - qx * cz
    ccz = qx * cy - qy * cx

    # Final result: v - 2*w*cross1 + 2*cross2
    rx = vx - 2.0 * qw * cx + 2.0 * ccx
    ry = vy - 2.0 * qw * cy + 2.0 * ccy
    rz = vz - 2.0 * qw * cz + 2.0 * ccz

    return torch.stack([rx, ry, rz], dim=-1)


@torch.jit.script
def calculate_contact_forces(
    contact_forces: torch.Tensor,
    contact_positions: torch.Tensor,
    link_a: torch.Tensor,
    link_b: torch.Tensor,
    links_quat: torch.Tensor,
    target_link_ids: torch.Tensor,
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
        with_link_ids: If defined, only contacts made with these link IDs AND target_link_ids will be considered. Tensor shape: (n_with_links)
        output_forces: Output force tensor, shape (n_envs, n_target_links, 3)
        output_positions: Output position tensor, shape (n_envs, n_target_links, 3)
    """
    n_envs = contact_forces.shape[0]
    n_contacts = contact_forces.shape[1]
    n_targets = target_link_ids.shape[0]
    has_with_filter = with_link_ids.numel() > 0

    # Zero outputs
    output_forces.zero_()
    output_positions.zero_()

    # Early exit if no contacts
    if n_contacts == 0:
        return

    # Pre-compute quaternions for all contacts (shared across all targets)
    batch_idx = (
        torch.arange(n_envs, device=link_a.device).view(-1, 1).expand(-1, n_contacts)
    )
    quat_a = links_quat[batch_idx, link_a]  # (n_envs, n_contacts, 4)
    quat_b = links_quat[batch_idx, link_b]  # (n_envs, n_contacts, 4)

    # Transform forces to local frame
    force_local_a = _inv_transform_by_quat_jit(-contact_forces, quat_a)
    force_local_b = _inv_transform_by_quat_jit(contact_forces, quat_b)

    # Compute the filter mask for with_link_ids
    if has_with_filter:
        with_links = with_link_ids.view(1, 1, -1)
        link_a_exp = link_a.unsqueeze(-1)
        link_b_exp = link_b.unsqueeze(-1)
        link_a_in_with = (link_a_exp == with_links).any(dim=-1)
        link_b_in_with = (link_b_exp == with_links).any(dim=-1)
    else:
        # Dummy tensors (won't be used)
        link_a_in_with = torch.zeros_like(link_a, dtype=torch.bool)
        link_b_in_with = torch.zeros_like(link_b, dtype=torch.bool)

    # Process each target link
    for t_idx in range(n_targets):
        target_link = target_link_ids[t_idx]

        # Find contacts where link_a or link_b matches this target
        is_target_a = link_a == target_link
        is_target_b = link_b == target_link

        # Apply with_link filter
        # target_a valid if link_b is in with_links
        # target_b valid if link_a is in with_links
        if has_with_filter:
            is_target_a = is_target_a & link_b_in_with
            is_target_b = is_target_b & link_a_in_with

        # Combined mask for any valid contact
        valid_mask = is_target_a | is_target_b
        valid_mask_float = valid_mask.float()

        # Count valid contacts for position averaging
        contact_count = valid_mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)

        # Accumulate positions: (n_envs, n_contacts, 3) -> (n_envs, 3)
        masked_positions = contact_positions * valid_mask_float.unsqueeze(-1)
        avg_position = masked_positions.sum(dim=1) / contact_count
        output_positions[:, t_idx, :] = avg_position

        # Select force based on which link is target (is_target_b takes priority)
        is_b_float = is_target_b.float().unsqueeze(-1)  # (n_envs, n_contacts, 1)
        is_a_only_float = (is_target_a & ~is_target_b).float().unsqueeze(-1)

        selected_force = force_local_a * is_a_only_float + force_local_b * is_b_float

        # Apply validity mask and sum
        masked_force = selected_force * valid_mask_float.unsqueeze(-1)
        output_forces[:, t_idx, :] = masked_force.sum(dim=1)
