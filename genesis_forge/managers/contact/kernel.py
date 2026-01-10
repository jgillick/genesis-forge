import gstaichi as ti
from genesis.utils.geom import ti_inv_transform_by_quat


@ti.kernel
def kernel_get_contact_forces(
    contact_forces: ti.types.ndarray(),
    contact_positions: ti.types.ndarray(),
    link_a: ti.types.ndarray(),
    link_b: ti.types.ndarray(),
    links_quat: ti.types.ndarray(),
    target_link_ids: ti.types.ndarray(),
    has_with_filter: ti.i32,
    with_link_ids: ti.types.ndarray(),
    output_forces: ti.types.ndarray(),
    output_positions: ti.types.ndarray(),
    position_counts: ti.types.ndarray(),
):
    """
    Accumulates contact forces and positions for target links, optionally filtering by with_link_ids.

    Forces are accumulated in world frame, then transformed to local frame at the end.
    This is more efficient than transforming each contact force individually.

    Args:
        contact_forces: Contact force data (n_envs, n_contacts, 3)
        contact_positions: Contact position data (n_envs, n_contacts, 3)
        link_a: First link in each contact (n_envs, n_contacts)
        link_b: Second link in each contact (n_envs, n_contacts)
        links_quat: Link quaternions (n_envs, n_links, 4)
        target_link_ids: Target link IDs to track (n_target_links)
        has_with_filter: Whether to apply with_link filter (0 or 1)
        with_link_ids: Filter links (n_with_links) - only used if has_with_filter=True
        output_forces: Output force tensor (n_envs, n_target_links, 3)
        output_positions: Output position tensor (n_envs, n_target_links, 3)
        position_counts: Position count tensor (n_envs, n_target_links) - internal use only
    """
    # Pass 1: Accumulate forces (world frame) and positions
    for i_b, i_c, i_t in ti.ndrange(
        output_forces.shape[0], link_a.shape[-1], target_link_ids.shape[-1]
    ):
        contact_link_a = link_a[i_b, i_c]
        contact_link_b = link_b[i_b, i_c]
        target_link = target_link_ids[i_t]

        # Check if this contact involves our target link
        is_target_a = contact_link_a == target_link
        is_target_b = contact_link_b == target_link

        if is_target_a or is_target_b:
            # Apply with_link filter if specified
            should_include = True
            if has_with_filter:
                should_include = False
                for i_w in range(with_link_ids.shape[-1]):
                    with_link = with_link_ids[i_w]
                    if (is_target_a and contact_link_b == with_link) or (
                        is_target_b and contact_link_a == with_link
                    ):
                        should_include = True
                        break

            if should_include:
                position_counts[i_b, i_t] += 1
                # Accumulate positions and forces in world frame
                # Force sign: +1 if target is link_b, -1 if target is link_a
                sign = 1.0 if is_target_b else -1.0
                for j in ti.static(range(3)):
                    output_positions[i_b, i_t, j] += contact_positions[i_b, i_c, j]
                    output_forces[i_b, i_t, j] += sign * contact_forces[i_b, i_c, j]

    # Pass 2: Transform forces to local frame and average positions
    for i_b, i_t in ti.ndrange(output_forces.shape[0], output_forces.shape[1]):
        target_link = target_link_ids[i_t]

        # Get quaternion for this target link
        quat = ti.Vector.zero(ti.f32, 4)
        for j in ti.static(range(4)):
            quat[j] = links_quat[i_b, target_link, j]

        # Transform accumulated force from world to local frame
        force_vec = ti.Vector.zero(ti.f32, 3)
        for j in ti.static(range(3)):
            force_vec[j] = output_forces[i_b, i_t, j]

        force_local = ti_inv_transform_by_quat(force_vec, quat)

        for j in ti.static(range(3)):
            output_forces[i_b, i_t, j] = force_local[j]

        # Average positions
        if position_counts[i_b, i_t] > 0:
            for j in ti.static(range(3)):
                output_positions[i_b, i_t, j] = (
                    output_positions[i_b, i_t, j] / position_counts[i_b, i_t]
                )
