import re
import torch
import genesis as gs
from typing import TypedDict, Tuple
from genesis.engine.entities import RigidEntity

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager


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

        self.contacts: torch.Tensor | None = None
        """Contact forces experienced by the entity links."""

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

        # Initialize buffers
        link_count = self._target_link_ids.shape[0]
        self.contacts = torch.zeros(
            (self.env.num_envs, link_count, 3), device=gs.device
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
            try:
                # Find link by name
                link = entity.get_link(pattern)
                if link is not None:
                    ids.append(link.idx)
            except:
                # Find link by regex
                for link in entity.links:
                    if re.match(pattern, link.name):
                        ids.append(link.idx)

        return torch.tensor(ids, device=gs.device)

    def _calculate_contact_forces(self):
        """
        Reduce all contacts down to a single force vector for each target link.

        Args:
            collision_data: Dict with 'force', 'link_a', 'link_b' tensors
            target_link_ids: List of link IDs to accumulate forces for
            with_link_ids: Optional list of link IDs to filter contacts by.
                        Only contacts involving these links will be considered.

        Returns:
            Tensor of shape (n_envs, n_target_links, 3)
        """
        contacts = self.env.scene.rigid_solver.collider.get_contacts(
            as_tensor=True, to_torch=True
        )
        force = contacts["force"]
        link_a = contacts["link_a"]
        link_b = contacts["link_b"]
        position = contacts["position"]

        # Convert target_link_ids to tensor for broadcasting
        target_links = self._target_link_ids.to(gs.device)
        n_target_links = target_links.shape[0]
        target_links = target_links.view(1, 1, n_target_links)

        # For mask filtering
        link_a_expanded = link_a.unsqueeze(-1)
        link_b_expanded = link_b.unsqueeze(-1)
        mask_target_a = (link_a_expanded == target_links).any(dim=-1)
        mask_target_b = (link_b_expanded == target_links).any(dim=-1)

        # Filter contacts by with_link_ids if specified
        with_link_mask = None
        if self._with_link_ids is not None:
            with_links = self._with_link_ids.to(gs.device)
            n_with_links = with_links.shape[0]
            with_links = with_links.view(1, 1, n_with_links)

            mask_with_a = (link_a_expanded == with_links).any(dim=-1)
            mask_with_b = (link_b_expanded == with_links).any(dim=-1)
            with_link_mask = (mask_with_a & mask_target_b) | (
                mask_with_b & mask_target_a
            )

            # Apply filters to tensors
            force = force * with_link_mask.unsqueeze(-1)

        # Concatenate links and forces - each force applies to both links in the pair
        all_links = torch.cat([link_a, link_b], dim=1)
        all_forces = torch.cat([force, force], dim=1)

        # Create mask for where each target link appears
        all_links = all_links.unsqueeze(-1)
        mask = all_links == target_links

        # Apply mask and sum contact forces
        force_expanded = all_forces.unsqueeze(-2)
        masked_forces = force_expanded * mask.unsqueeze(-1)
        self.contacts = masked_forces.sum(dim=1)

        # Get the position of the contact for visualization
        if self.debug_visualizer:
            # Filter out the positions that aren't with target links
            target_mask = mask_target_a | mask_target_b
            if with_link_mask is not None:
                target_mask = with_link_mask
            self._render_debug_visualizer(position, target_mask)

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

    def _render_debug_visualizer(
        self, contact_pos: torch.Tensor, link_mask: torch.Tensor
    ):
        """
        Visualize the contact points.
        """
        # Clear existing debug objects
        for node in self._debug_nodes:
            self.env.scene.clear_debug_object(node)
        self._debug_nodes = []

        # End here if the debug visualizer is not enabled
        if not self.debug_visualizer:
            return

        # Filter to only the environments we want to visualize
        cfg = self.visualizer_cfg
        if cfg["envs_idx"] is not None:
            contact_pos = contact_pos[cfg["envs_idx"]]
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
