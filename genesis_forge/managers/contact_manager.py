import re
import torch
import genesis as gs
from typing import TypedDict, Tuple
import gstaichi as ti
from genesis.utils.geom import ti_inv_transform_by_quat
from genesis.engine.entities import RigidEntity

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager


@ti.kernel
def _kernel_get_contact_forces(
    contact_forces: ti.types.ndarray(),
    contact_positions: ti.types.ndarray(),
    link_a: ti.types.ndarray(),
    link_b: ti.types.ndarray(),
    links_quat: ti.types.ndarray(),
    target_link_ids: ti.types.ndarray(),
    with_link_ids: ti.types.ndarray(),
    output_forces: ti.types.ndarray(),
    output_positions: ti.types.ndarray(),
    position_counts: ti.types.ndarray(),
    has_with_filter: ti.i32,
    use_quaternion_transform: ti.i32,
):
    """
    Unified Taichi kernel for calculating contact forces and positions.

    This kernel accumulates contact forces and positions for target links, optionally filtering
    by with_link_ids. It's memory-efficient and avoids large intermediate tensors.

    Args:
        contact_forces: Contact force data (n_envs, n_contacts, 3)
        contact_positions: Contact position data (n_envs, n_contacts, 3)
        link_a: First link in each contact (n_envs, n_contacts)
        link_b: Second link in each contact (n_envs, n_contacts)
        links_quat: Link quaternions (n_envs, n_links, 4) - only used if use_quaternion_transform=True
        target_link_ids: Target link IDs to track (n_target_links)
        with_link_ids: Filter links (n_with_links) - only used if has_with_filter=True
        output_forces: Output force tensor (n_envs, n_target_links, 3)
        output_positions: Output position tensor (n_envs, n_target_links, 3)
        position_counts: Position count tensor (n_envs, n_target_links) - internal use only
        has_with_filter: Whether to apply with_link filter (0 or 1)
        use_quaternion_transform: Whether to apply quaternion transforms (0 or 1)
    """
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
                # Get contact force
                force_vec = ti.Vector.zero(ti.f32, 3)
                for j in ti.static(range(3)):
                    force_vec[j] = contact_forces[i_b, i_c, j]

                # Get contact position
                pos_vec = ti.Vector.zero(ti.f32, 3)
                for j in ti.static(range(3)):
                    pos_vec[j] = contact_positions[i_b, i_c, j]

                # Prepare force and position for accumulation
                if use_quaternion_transform:
                    # Get quaternions for both links
                    quat_a = ti.Vector.zero(ti.f32, 4)
                    quat_b = ti.Vector.zero(ti.f32, 4)

                    for j in ti.static(range(4)):
                        quat_a[j] = links_quat[i_b, contact_link_a, j]
                        quat_b[j] = links_quat[i_b, contact_link_b, j]

                    # Transform force and position to local frame of target link
                    if is_target_a:
                        # Force is applied to link_a (our target), transform to its local frame
                        force_vec = ti_inv_transform_by_quat(-force_vec, quat_a)
                        pos_vec = ti_inv_transform_by_quat(pos_vec, quat_a)
                    else:
                        # Force is applied to link_b, but we want it in target link's frame
                        force_vec = ti_inv_transform_by_quat(force_vec, quat_b)
                        pos_vec = ti_inv_transform_by_quat(pos_vec, quat_b)
                else:
                    # No quaternion transform - use forces and positions directly
                    force_multiplier = -1.0 if is_target_a else 1.0
                    force_vec = force_vec * force_multiplier
                    pos_vec = pos_vec

                # Accumulate force and position
                for j in ti.static(range(3)):
                    output_forces[i_b, i_t, j] += force_vec[j]
                    output_positions[i_b, i_t, j] += pos_vec[j]
                position_counts[i_b, i_t] += 1

    # Final pass: compute average positions for all links
    for i_b, i_t in ti.ndrange(output_forces.shape[0], output_forces.shape[1]):
        if position_counts[i_b, i_t] > 0:
            for j in ti.static(range(3)):
                output_positions[i_b, i_t, j] = (
                    output_positions[i_b, i_t, j] / position_counts[i_b, i_t]
                )


class ContactDebugVisualizerConfig(TypedDict):
    """Defines the configuration for the contact debug visualizer."""

    envs_idx: list[int]
    """The indices of the environments to visualize. If None, all environments will be visualized."""

    color: Tuple[float, float, float, float]
    """The color of the contact ball"""

    radius: float
    """The radius of the visualization sphere"""


DEFAULT_VISUALIZER_CONFIG: ContactDebugVisualizerConfig = {
    "envs_idx": None,
    "size": 0.02,
    "color": (0.5, 0.0, 0.0, 1.0),
}


class ContactManager(BaseManager):
    """
    Tracks the contact forces between entity links in the environment.

    Example with ManagedEnvironment::
        class MyEnv(ManagedEnvironment):

            # ... Construct scene and other env setup ...

            def config(self):
                # Define contact manager
                self.foot_contact_manager = ContactManager(
                    self,
                    link_names=[".*_Foot"],
                )

                # Use contact manager in rewards
                self.reward_manager = RewardManager(
                    self,
                    term_cfg={
                        "Foot contact": {
                            "weight": 5.0,
                            "fn": rewards.has_contact,
                            "params": {
                                "contact_manager": self.foot_contact_manager,
                                "min_contacts": 4,
                            },
                        },
                    },
                )

                # ... other managers here ...

    Example using the contact manager directly::

        class MyEnv(GenesisEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.contact_manager = ContactManager(
                    self,
                    link_names=[".*_Foot"],
                )

            def build(self):
                super().build()
                self.contact_manager.build()

            def step(self, actions: torch.Tensor):
                super().step(actions)
                self.contact_manager.step()
                return obs, rewards, terminations, timeouts, info

            def reset(self, envs_idx: list[int] | None = None):
                super().reset(envs_idx)
                self.contact_manager.reset(envs_idx)
                return obs, info

            def calculate_rewards():
                # Reward for each foot in contact with something with at least 1.0N force
                CONTACT_THRESHOLD = 1.0
                CONTACT_WEIGHT = 0.005
                has_contact = self.contact_manager.contacts[:,:].norm(dim=-1) > CONTACT_THRESHOLD
                contact_reward = has_contact.sum(dim=1).float() * CONTACT_WEIGHT

                # Access contact positions for debugging or additional analysis
                contact_positions = self.contact_manager.contact_positions
                # contact_positions shape: (n_envs, n_target_links, 3)
                # Positions are automatically averaged when multiple contacts occur

                # ...additional reward calculations here...

    Filtering::

        class MyEnv(ManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.scene = gs.Scene(
                    # ... scene options ...
                )

                # Add terrain
                self.terrain = self.scene.add_entity(gs.morphs.Plane())

                # add robot
                self.robot = self.scene.add_entity(
                    gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf"),
                )

            def config(self):
                # Track all contacts between the robot's feet and the terrain
                self.contact_manager = ContactManager(
                    self,
                    entity_attr="robot",
                    link_names=[".*_foot"],
                    with_entity_attr="terrain",
                )

                # ...other managers here...

            # ...other operations here...
    """

    def __init__(
        self,
        env: GenesisEnv,
        link_names: list[str],
        entity_attr: RigidEntity = "robot",
        with_entity_attr: RigidEntity = None,
        with_links_names: list[int] = None,
        track_air_time: bool = False,
        air_time_contact_threshold: float = 1.0,
        debug_visualizer: bool = False,
        debug_visualizer_cfg: ContactDebugVisualizerConfig = DEFAULT_VISUALIZER_CONFIG,
        use_taichi_kernel: bool = True,
        use_quaternion_transform: bool = False,
    ):
        """
        Args:
            env: The environment to track the contact forces for.
            link_names: The names, or name regex patterns, of the entity links to track the contact forces for.
            entity_attr: The environment attribute which contains the entity with the links we're tracking. Defaults to `robot`.
            with_entity_attr: Filter the contact forces to only include contacts with the entity assigned to this environment attribute.
            with_links_names: Filter the contact forces to only include contacts with these links.
            track_air_time: Whether to track the air time of the entity link contacts.
            air_time_contact_threshold: When track_air_time is True, this is the threshold for the contact forces to be considered.
            debug_visualizer: Whether to visualize the contact points.
            debug_visualizer_cfg: The configuration for the contact debug visualizer.
            use_taichi_kernel: Whether to use the optimized Taichi kernel for contact force calculation.
                             Defaults to True for better memory performance.
            use_quaternion_transform: Whether to apply quaternion transformations to forces.
                                     Only used when use_taichi_kernel=True. Defaults to False for better performance.
        """
        super().__init__(env, "contact")

        self._link_names = link_names
        self._air_time_contact_threshold = air_time_contact_threshold
        self._track_air_time = track_air_time
        self._entity_attr = entity_attr
        self._with_entity_attr = with_entity_attr
        self._with_links_names = with_links_names
        self._with_link_ids = None
        self._target_link_ids = None

        self.debug_visualizer = debug_visualizer
        self.visualizer_cfg = {**DEFAULT_VISUALIZER_CONFIG, **debug_visualizer_cfg}
        self._debug_nodes = []

        # Performance optimization options
        self._use_taichi_kernel = use_taichi_kernel
        self._use_quaternion_transform = use_quaternion_transform

        self.contacts: torch.Tensor | None = None
        """Contact forces experienced by the entity links."""

        self.contact_positions: torch.Tensor | None = None
        """Contact positions for each target link."""

        self.contact_position_counts: torch.Tensor | None = None
        """Number of contacts contributing to each position (for averaging). Internal use only."""

        self.last_air_time: torch.Tensor | None = None
        """Time spent (in s) in the air before the last contact."""

        self.current_air_time: torch.Tensor | None = None
        """Time spent (in s) in the air since the last detach."""

        self.last_contact_time: torch.Tensor | None = None
        """Time spent (in s) in contact before the last detach."""

        self.current_contact_time: torch.Tensor | None = None
        """Time spent (in s) in contact since the last contact."""

    """
    Helper Methods
    """

    def has_made_contact(self, dt: float, time_margin: float = 1.0e-8) -> torch.Tensor:
        """
        Checks if links that have established contact within the last :attr:`dt` seconds.

        This function checks if the links have established contact within the last :attr:`dt` seconds
        by comparing the current contact time with the given time period. If the contact time is less
        than the given time period, then the links are considered to be in contact.

        Args:
            dt: The time period since the contact was established.
            time_margin: Adds a little error margin to the dt time period.

        Returns:
            A boolean tensor indicating the links that have established contact within the last
            :attr:`dt` seconds. Shape is (n_envs, n_target_links)

        Raises:
            RuntimeError: If the manager is not configured to track air time.
        """
        # check if the sensor is configured to track contact time
        if not self._track_air_time:
            raise RuntimeError(
                "The contact sensor is not configured to track air time."
                "Please enable the 'track_air_time' in the manager configuration."
            )
        # check if the bodies are in contact
        currently_in_contact = self.current_contact_time > 0.0
        less_than_dt_in_contact = self.current_contact_time < (dt + time_margin)
        return currently_in_contact * less_than_dt_in_contact

    def has_broken_contact(
        self, dt: float, time_margin: float = 1.0e-8
    ) -> torch.Tensor:
        """Checks links that have broken contact within the last :attr:`dt` seconds.

        This function checks if the links have broken contact within the last :attr:`dt` seconds
        by comparing the current air time with the given time period. If the air time is less
        than the given time period, then the links are considered to not be in contact.

        Args:
            dt: The time period since the contact was broken.
            time_margin: Adds a little error margin to the dt time period.

        Returns:
            A boolean tensor indicating the links that have broken contact within the last
            :attr:`dt` seconds. Shape is (n_envs, n_target_links)

        Raises:
            RuntimeError: If the manager is not configured to track air time.
        """
        # check if the sensor is configured to track contact time
        if not self._track_air_time:
            raise RuntimeError(
                "The contact manager is not configured to track air time."
                "Please enable the 'track_air_time' in the manager configuration."
            )
        currently_detached = self.current_air_time > 0.0
        less_than_dt_detached = self.current_air_time < (dt + time_margin)
        return currently_detached * less_than_dt_detached

    """
    Operations
    """

    def build(self):
        """Initialize link indices and buffers."""
        super().build()

        # Get the link indices
        entity = self.env.__getattribute__(self._entity_attr)
        self._target_link_ids = self._get_links_idx(entity, self._link_names)
        if self._with_entity_attr or self._with_links_names:
            with_entity_attr = (
                self._with_entity_attr
                if self._with_entity_attr is not None
                else "robot"
            )
            with_entity = self.env.__getattribute__(with_entity_attr)
            self._with_link_ids = self._get_links_idx(
                with_entity, self._with_links_names
            )

        print(f"Target link ids: {self._target_link_ids.shape}")
        print(f"With link ids: {self._with_link_ids.shape}")

        # Initialize buffers
        link_count = self._target_link_ids.shape[0]
        self.contacts = torch.zeros(
            (self.env.num_envs, link_count, 3), device=gs.device
        )
        self.contact_positions = torch.zeros(
            (self.env.num_envs, link_count, 3), device=gs.device
        )
        self.contact_position_counts = torch.zeros(
            (self.env.num_envs, link_count), device=gs.device
        )
        if self._track_air_time:
            self.last_air_time = torch.zeros(
                (self.env.num_envs, link_count), device=gs.device
            )
            self.current_air_time = torch.zeros_like(self.last_air_time)
            self.last_contact_time = torch.zeros_like(self.last_air_time)
            self.current_contact_time = torch.zeros_like(self.last_air_time)

    def reset(self, envs_idx: list[int] | None = None):
        super().reset(envs_idx)
        if envs_idx is None:
            envs_idx = torch.arange(self.env.num_envs, device=gs.device)

        if not self.enabled:
            return

        # reset the current air time
        if self._track_air_time:
            self.current_air_time[envs_idx] = 0.0
            self.current_contact_time[envs_idx] = 0.0
            self.last_air_time[envs_idx] = 0.0
            self.last_contact_time[envs_idx] = 0.0

    def step(self):
        super().step()
        if not self.enabled:
            return
        self._calculate_contact_forces()
        self._calculate_air_time()

    """
    Implementation
    """

    def _get_links_idx(
        self, entity: RigidEntity, names: list[str] = None
    ) -> torch.Tensor:
        """
        Find the global link indices for the given link names or regular expressions.

        Args:
            entity: The entity to find the links in.
            names: The names, or name regex patterns, of the links to find.

        Returns:
            List of global link indices.
        """
        # If link names are not defined, assume all links
        if names is None:
            return torch.tensor([link.idx for link in entity.links], device=gs.device)

        ids = []
        for pattern in names:
            for link in entity.links:
                if pattern == link.name or re.match(f"^{pattern}$", link.name):
                    ids.append(link.idx)

        return torch.tensor(ids, device=gs.device)

    def _calculate_contact_forces(self):
        """
        Calculate contact forces using either optimized Taichi kernel or memory-efficient PyTorch.

        Returns:
            Tensor of shape (n_envs, n_target_links, 3)
        """
        if self._use_taichi_kernel:
            self._calculate_contact_forces_taichi()
        else:
            self._calculate_contact_forces_pytorch_optimized()

    def _calculate_contact_forces_taichi(self):
        """
        Calculate contact forces using optimized Taichi kernel.

        This method is memory-efficient and avoids large intermediate tensors.
        """
        contacts = self.env.scene.rigid_solver.collider.get_contacts(
            as_tensor=True, to_torch=True
        )
        force = contacts["force"]
        link_a = contacts["link_a"]
        link_b = contacts["link_b"]
        position = contacts["position"]

        # Clear output tensors
        self.contacts.fill_(0.0)
        self.contact_positions.fill_(0.0)
        self.contact_position_counts.fill_(0.0)

        # Prepare tensors for Taichi kernel
        target_links = self._target_link_ids.to(gs.device)
        with_links = (
            self._with_link_ids.to(gs.device)
            if self._with_link_ids is not None
            else torch.empty(0, device=gs.device)
        )

        # Get link quaternions (only needed if using quaternion transform)
        links_quat = None
        if self._use_quaternion_transform:
            links_quat = self.env.scene.rigid_solver.get_links_quat()
        else:
            # Create dummy quaternion tensor (won't be used)
            links_quat = torch.zeros(1, 1, 4, device=gs.device)

        # Call unified kernel
        _kernel_get_contact_forces(
            force.contiguous(),
            position.contiguous(),
            link_a.contiguous(),
            link_b.contiguous(),
            links_quat.contiguous(),
            target_links.contiguous(),
            with_links.contiguous(),
            self.contacts.contiguous(),
            self.contact_positions.contiguous(),
            self.contact_position_counts.contiguous(),
            1 if self._with_link_ids is not None else 0,
            1 if self._use_quaternion_transform else 0,
        )

        # Handle debug visualization
        if self.debug_visualizer:
            valid_counts = self.contact_position_counts > 0
            self.contact_positions[valid_counts] = self.contact_positions[
                valid_counts
            ] / self.contact_position_counts[valid_counts].unsqueeze(-1)
            self._render_debug_visualizer_taichi(self.contact_positions)

    def _calculate_contact_forces_pytorch_optimized(self):
        """
        Memory-optimized PyTorch implementation using scatter operations.

        This avoids large intermediate tensors by using scatter_add operations.
        """
        contacts = self.env.scene.rigid_solver.collider.get_contacts(
            as_tensor=True, to_torch=True
        )
        force = contacts["force"]
        link_a = contacts["link_a"]
        link_b = contacts["link_b"]
        position = contacts["position"]

        # Clear output tensors
        self.contacts.fill_(0.0)
        self.contact_positions.fill_(0.0)
        self.contact_position_counts.fill_(0.0)

        # Get target and with link IDs
        target_links = self._target_link_ids.to(gs.device)
        n_target_links = target_links.shape[0]

        # Create mapping from link ID to target index
        target_link_map = torch.zeros(
            target_links.max().item() + 1, dtype=torch.long, device=gs.device
        )
        target_link_map[target_links] = torch.arange(n_target_links, device=gs.device)

        # Process contacts in batches to avoid memory issues
        batch_size = min(1000, link_a.shape[1])  # Process up to 1000 contacts at a time

        for start_idx in range(0, link_a.shape[1], batch_size):
            end_idx = min(start_idx + batch_size, link_a.shape[1])

            # Get batch of contacts
            batch_link_a = link_a[:, start_idx:end_idx]
            batch_link_b = link_b[:, start_idx:end_idx]
            batch_force = force[:, start_idx:end_idx]
            batch_position = position[:, start_idx:end_idx]

            # Create masks for target links
            mask_a = (
                batch_link_a.unsqueeze(-1) == target_links.unsqueeze(0).unsqueeze(0)
            ).any(dim=-1)
            mask_b = (
                batch_link_b.unsqueeze(-1) == target_links.unsqueeze(0).unsqueeze(0)
            ).any(dim=-1)

            # Apply with_link filter if specified
            if self._with_link_ids is not None:
                with_links = self._with_link_ids.to(gs.device)
                mask_with_a = (
                    batch_link_a.unsqueeze(-1) == with_links.unsqueeze(0).unsqueeze(0)
                ).any(dim=-1)
                mask_with_b = (
                    batch_link_b.unsqueeze(-1) == with_links.unsqueeze(0).unsqueeze(0)
                ).any(dim=-1)

                # Only include contacts where one link is target and other is with_link
                valid_contacts = (mask_a & mask_with_b) | (mask_b & mask_with_a)
            else:
                valid_contacts = mask_a | mask_b

            # Accumulate forces using scatter operations
            if valid_contacts.any():
                # Get indices for scatter operations
                env_indices = (
                    torch.arange(self.env.num_envs, device=gs.device)
                    .unsqueeze(-1)
                    .expand(-1, batch_link_a.shape[1])
                )
                batch_indices = (
                    torch.arange(batch_link_a.shape[1], device=gs.device)
                    .unsqueeze(0)
                    .expand(self.env.num_envs, -1)
                )

                # Process link_a contacts
                link_a_valid = valid_contacts & mask_a
                if link_a_valid.any():
                    link_a_target_indices = target_link_map[batch_link_a[link_a_valid]]
                    env_indices_a = env_indices[link_a_valid]
                    forces_a = -batch_force[
                        link_a_valid
                    ]  # Negative for Newton's 3rd law
                    positions_a = batch_position[link_a_valid]

                    self.contacts[env_indices_a, link_a_target_indices] += forces_a
                    self.contact_positions[
                        env_indices_a, link_a_target_indices
                    ] += positions_a
                    self.contact_position_counts[
                        env_indices_a, link_a_target_indices
                    ] += 1

                # Process link_b contacts
                link_b_valid = valid_contacts & mask_b
                if link_b_valid.any():
                    link_b_target_indices = target_link_map[batch_link_b[link_b_valid]]
                    env_indices_b = env_indices[link_b_valid]
                    forces_b = batch_force[link_b_valid]
                    positions_b = batch_position[link_b_valid]

                    self.contacts[env_indices_b, link_b_target_indices] += forces_b
                    self.contact_positions[
                        env_indices_b, link_b_target_indices
                    ] += positions_b
                    self.contact_position_counts[
                        env_indices_b, link_b_target_indices
                    ] += 1

        # Compute average positions for PyTorch implementation
        valid_counts = self.contact_position_counts > 0
        self.contact_positions[valid_counts] = self.contact_positions[
            valid_counts
        ] / self.contact_position_counts[valid_counts].unsqueeze(-1)

        # Handle debug visualization
        if self.debug_visualizer:
            self._render_debug_visualizer_pytorch(position, force, link_a, link_b)

    def _calculate_air_time(self):
        """
        Track air time values for the links
        """
        if not self._track_air_time:
            return

        dt = self.env.scene.dt

        # Check contact state of bodies
        is_contact = (
            torch.norm(self.contacts[:, :, :], dim=-1)
            > self._air_time_contact_threshold
        )
        is_new_contact = (self.current_air_time > 0) * is_contact
        is_new_detached = (self.current_contact_time > 0) * ~is_contact

        # Update the last contact time if body has just become in contact
        self.last_air_time = torch.where(
            is_new_contact,
            self.current_air_time + dt,
            self.last_air_time,
        )

        # Increment time for bodies that are not in contact
        self.current_air_time = torch.where(
            ~is_contact,
            self.current_air_time + dt,
            0.0,
        )

        # Update the last contact time if body has just detached
        self.last_contact_time = torch.where(
            is_new_detached,
            self.current_contact_time + dt,
            self.last_contact_time,
        )

        # Increment time for bodies that are in contact
        self.current_contact_time = torch.where(
            is_contact,
            self.current_contact_time + dt,
            0.0,
        )

    def _render_debug_visualizer_taichi(
        self,
        contact_pos: torch.Tensor,
    ):
        """
        Visualize contact points for Taichi implementation.
        """
        # Clear existing debug objects
        for node in self._debug_nodes:
            self.env.scene.clear_debug_object(node)
        self._debug_nodes = []

        if not self.debug_visualizer:
            return

        self._render_debug_spheres(
            contact_pos,
        )

    def _render_debug_visualizer_pytorch(
        self,
        contact_pos: torch.Tensor,
        force: torch.Tensor,
        link_a: torch.Tensor,
        link_b: torch.Tensor,
    ):
        """
        Visualize contact points for PyTorch implementation.
        """
        # Clear existing debug objects
        for node in self._debug_nodes:
            self.env.scene.clear_debug_object(node)
        self._debug_nodes = []

        if not self.debug_visualizer:
            return

        # Create mask for target links
        target_links = self._target_link_ids.to(gs.device)
        mask_a = (link_a.unsqueeze(-1) == target_links.unsqueeze(0).unsqueeze(0)).any(
            dim=-1
        )
        mask_b = (link_b.unsqueeze(-1) == target_links.unsqueeze(0).unsqueeze(0)).any(
            dim=-1
        )

        # Apply with_link filter if specified
        if self._with_link_ids is not None:
            with_links = self._with_link_ids.to(gs.device)
            mask_with_a = (
                link_a.unsqueeze(-1) == with_links.unsqueeze(0).unsqueeze(0)
            ).any(dim=-1)
            mask_with_b = (
                link_b.unsqueeze(-1) == with_links.unsqueeze(0).unsqueeze(0)
            ).any(dim=-1)
            target_mask = (mask_a & mask_with_b) | (mask_b & mask_with_a)
        else:
            target_mask = mask_a | mask_b

        self._render_debug_spheres(contact_pos, target_mask)

    def _render_debug_spheres(
        self, contact_pos: torch.Tensor, link_mask: torch.Tensor | None = None
    ):
        """
        Render debug spheres for contact points.
        """
        # Filter to only the environments we want to visualize
        cfg = self.visualizer_cfg
        if cfg["envs_idx"] is not None:
            contact_pos = contact_pos[cfg["envs_idx"]]
            if link_mask is not None:
                link_mask = link_mask[cfg["envs_idx"]]

        if link_mask is not None:
            contact_pos = contact_pos[link_mask]

        # Draw debug spheres
        if contact_pos.shape[0] > 0:
            node = self.env.scene.draw_debug_spheres(
                poss=contact_pos,
                radius=cfg["size"],
                color=cfg["color"],
            )
            self._debug_nodes.append(node)

    def __repr__(self):
        attrs = [f"link_names={self._link_names}"]
        if self._entity_attr:
            attrs.append(f"entity_attr={self._entity_attr}")
        if self._with_entity_attr:
            attrs.append(f"with_entity_attr={self._with_entity_attr}")
        if self._with_links_names:
            attrs.append(f"with_links_names={self._with_links_names}")
        if self._track_air_time:
            attrs.append(f"track_air_time={self._track_air_time}")
            if self._air_time_contact_threshold:
                attrs.append(
                    f"air_time_contact_threshold={self._air_time_contact_threshold}"
                )
        attrs_str = ", ".join(attrs)
        return f"{self.__class__.__name__}({attrs_str})"
