import re
import torch
import genesis as gs
from genesis.engine.entities import RigidEntity

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager
from genesis_forge.managers.contact.config import (
    ContactDebugVisualizerConfig,
    DEFAULT_VISUALIZER_CONFIG,
)
from genesis_forge.managers.contact.kernel import kernel_get_contact_forces


class ContactManager(BaseManager):
    """
    Tracks the contact forces between entity links in the environment.

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
        use_quaternion_transform: Whether to apply quaternion transformations to force vectors.
                                    Defaults to False for better performance.

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
        use_quaternion_transform: bool = False,
    ):
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
        self._contact_position_counts = None

        # Performance optimization options
        self._use_quaternion_transform = use_quaternion_transform

        self.contacts: torch.Tensor | None = None
        """Contact forces experienced by the entity links."""

        self.contact_positions: torch.Tensor | None = None
        """Contact positions for each target link."""

        self.last_air_time: torch.Tensor | None = None
        """Time spent (in s) in the air before the last contact."""

        self.current_air_time: torch.Tensor | None = None
        """Time spent (in s) in the air since the last detach."""

        self.last_contact_time: torch.Tensor | None = None
        """Time spent (in s) in contact before the last detach."""

        self.current_contact_time: torch.Tensor | None = None
        """Time spent (in s) in contact since the last contact."""

    """
    Properties
    """

    @property
    def target_link_ids(self) -> torch.Tensor:
        """The global link indices for the target links."""
        return self._target_link_ids

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
        self._target_link_ids = self._get_links_idx(self._entity_attr, self._link_names)
        if self._with_entity_attr or self._with_links_names:
            with_entity_attr = (
                self._with_entity_attr
                if self._with_entity_attr is not None
                else "robot"
            )
            self._with_link_ids = self._get_links_idx(
                with_entity_attr, self._with_links_names
            )

        # Initialize buffers
        link_count = self._target_link_ids.shape[0]
        self.contacts = torch.zeros(
            (self.env.num_envs, link_count, 3), device=gs.device
        )
        self.contact_positions = torch.zeros(
            (self.env.num_envs, link_count, 3), device=gs.device
        )
        self._contact_position_counts = torch.zeros(
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

    def _get_links_idx(self, entity_attr: str, names: list[str] = None) -> torch.Tensor:
        """
        Find the global link indices for the given link names or regular expressions.

        Args:
            entity: The entity to find the links in.
            names: The names, or name regex patterns, of the links to find.

        Returns:
            List of global link indices.
        """
        entity = self.env.__getattribute__(entity_attr)

        # If link names are not defined, assume all links
        if names is None:
            return torch.tensor([link.idx for link in entity.links], device=gs.device)

        ids = []
        for pattern in names:
            found = False
            for link in entity.links:
                if pattern == link.name or re.match(f"^{pattern}$", link.name):
                    ids.append(link.idx)
                    found = True
            if not found:
                names = [link.name for link in entity.links]
                print(
                    f"Warning: Link {pattern} not found in entity {self._entity_attr}"
                )
                print(f"Available links: {names}")

        return torch.tensor(ids, device=gs.device)

    def _calculate_contact_forces(self):
        """
        Calculate contact forces using on the target links.

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

        # Validate physics engine outputs to prevent NaN/inf propagation
        # Replace invalid values with zeros
        if torch.isnan(force).any() or torch.isinf(force).any():
            force = torch.nan_to_num(force, nan=0.0, posinf=0.0, neginf=0.0)
            print("Warning: Invalid contact forces detected (NaN/inf) and sanitized")

        # Clear output tensors
        self.contacts.fill_(0.0)
        self.contact_positions.fill_(0.0)
        self._contact_position_counts.fill_(0.0)

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
        kernel_get_contact_forces(
            force.contiguous(),
            position.contiguous(),
            link_a.contiguous(),
            link_b.contiguous(),
            links_quat.contiguous(),
            target_links.contiguous(),
            with_links.contiguous(),
            self.contacts.contiguous(),
            self.contact_positions.contiguous(),
            self._contact_position_counts.contiguous(),
            1 if self._with_link_ids is not None else 0,
            1 if self._use_quaternion_transform else 0,
        )

        # Handle debug visualization
        if self.debug_visualizer:
            self._render_debug_visualizer()

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

    def _render_debug_visualizer(self):
        """
        Visualize contact points
        """
        # Clear existing debug objects
        for node in self._debug_nodes:
            self.env.scene.clear_debug_object(node)
        self._debug_nodes = []

        if not self.debug_visualizer:
            return

        cfg = self.visualizer_cfg

        # Filter to only the environments we want to visualize
        contacts = self.contacts
        contact_pos = self.contact_positions
        if cfg["envs_idx"] is not None:
            contacts = contacts[cfg["envs_idx"]]
            contact_pos = contact_pos[cfg["envs_idx"]]

        # Filter out contacts below the force threshold
        if "force_threshold" in cfg and cfg["force_threshold"] != 0.0:
            force_mask = torch.norm(contacts, dim=-1) > cfg["force_threshold"]
            contact_pos = contact_pos[force_mask]

        # Draw debug spheres
        if contact_pos.numel() > 0:
            node = self.env.scene.draw_debug_spheres(
                poss=contact_pos,
                radius=cfg["size"],
                color=cfg["color"],
            )
            if node is not None:
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
