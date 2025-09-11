import torch
import torch.nn.functional as F
import genesis as gs
from genesis.engine.entities import RigidEntity

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers import BaseManager


class TerrainManager(BaseManager):
    """
    Provides utility functions for the terrain.
    The manager maps out the sizes and heights of the terrain and subterrain.
    This allows your environment to calculate the robot's height above rough terrain.
    You can also generate random positions on the terrain or subterrain to place your robots on reset.

    Example::
        class MyEnv(ManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.scene = gs.Scene(
                    # ... scene options ...
                )

                # Add terrain
                self.terrain = self.scene.add_entity(
                    morph=gs.morphs.Terrain(
                        n_subterrains=(2, 2),
                        subterrain_size=(25, 25),
                        subterrain_types=[
                            ["flat_terrain", "random_uniform_terrain"],
                            ["discrete_obstacles_terrain", "pyramid_stairs_terrain"],
                        ],
                    ),
                )

            def config(self):
                self.terrain_manager = TerrainManager(
                    self,
                    terrain_attr="terrain",
                )

             def reset(self, envs_idx: list[int] = None) -> tuple[torch.Tensor, dict[str, Any]]:
                # Randomize positions on the terrain
                pos = self.terrain_manager.generate_random_env_pos(
                    envs_idx=envs_idx,
                    subterrain="flat_terrain",
                    height_offset=0.15,
                )
                self.robot.set_pos(pos, envs_idx=envs_idx)

    Args:
        env: The environment instance.
        terrain_attr: The attribute name of the terrain in the environment.

    Returns:
        The terrain manager instance.
    """

    def __init__(
        self,
        env: GenesisEnv,
        terrain_attr: str = "terrain",
    ):
        super().__init__(env, type="terrain")

        self._origin = (0, 0, 0)
        self._bounds = (0, 0, 0, 0)  # x_min, x_max, y_min, y_max
        self._size = (0, 0)
        self._terrain: RigidEntity = None
        self._terrain_attr = terrain_attr
        self._subterrain_bounds = {}
        self._height_field: torch.Tensor | None = None
        self._env_pos_buffer = torch.zeros(
            (self.env.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )

    def build(self):
        """Cache the terrain height field"""
        self._terrain = self.env.__getattribute__(self._terrain_attr)
        self._map_terrain()

    def get_terrain_height(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Get interpolated terrain height at world coordinates (x, y).

        Args:
            x: Tensor of shape (n_envs,)
            y: Tensor of shape (n_envs,)

        Returns:
            Heights in the torch.Tensor of shape (n_envs,)
        """
        # No height field, so we can assume the height is consistent
        if self._height_field is None:
            base_height = torch.zeros_like(x)
            base_height[:] = self._origin[2]
            return base_height

        # Normalize coordinates to [-1, 1] range expected by grid_sample
        (x_min, x_max, y_min, y_max) = self._bounds
        norm_x = 2 * (x - x_min) / (x_max - x_min) - 1
        norm_y = 2 * (y - y_min) / (y_max - y_min) - 1
        coords = torch.stack([norm_x, norm_y], dim=1)

        # Reshape grid to (n_coords, 1, 1, 2) for grid_sample
        grid = coords.unsqueeze(1).unsqueeze(1)

        # Expand the height field to match the number of coordinates
        height_field = self._height_field.expand(
            coords.shape[0], 1, -1, -1
        )  # (n_coords, 1, height, width)

        # Border padding mode isn't supported on Mac GPU (mps)
        # https://github.com/pytorch/pytorch/issues/125098
        if gs.device.type == "mps":
            padding_mode = "zeros"
            grid = grid.clamp(-1, 1)
        else:
            padding_mode = "border"

        interpolated = F.grid_sample(
            height_field,  # (n_coords, 1, height, width)
            grid,  # (n_coords, 1, 1, 2)
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=True,
        )

        # Extract the height values at the specific coordinates
        heights = interpolated.squeeze(1).squeeze(1).squeeze(1)  # (n_coords,)

        return heights

    def generate_random_positions(
        self,
        num: int | None = None,
        usable_ratio: float = 0.5,
        subterrain: str | None = None,
        height_offset: float = 0.1e-3,
        output: torch.Tensor | None = None,
        out_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Distribute X/Y/Z positions across the terrain or subterrain.
        The X & Y positions will be random points and the Z position will be at the approximate terrain height at that point.

        Args:
            num: The number of positions to generate. Not necessary if output is provided
            output: The position tensor to update in-place.
            out_idx: The indices of the output position tensor to update.
            usable_ratio: How much of the terrain/subterrain area should be used for random positions.
                          For example, 0.25 will only generate positions within the center 25% of the area of the terrain/subterrain.
                          This helps avoid placing things right on th edge of the terrain/subterrain.
            subterrain: The subterrain to generate positions for.
                        If None, positions will be generated for the entire terrain.
            height_offset: The offset to add to the terrain height.
                           Since the height is approximate, this can prevent items being placed below the terrain.

        Returns:
            The positions tensor of shape (num, 3)
        """
        # Prep output buffer
        assert (
            output is not None or num is not None
        ), "Either output or num must be provided"
        if output is None:
            output = torch.zeros(num, 3, device=gs.device)
        if out_idx is None:
            out_idx = torch.arange(output.shape[0], device=gs.device)
        if num is None:
            num = out_idx.shape[0]

        # Get total bounds
        bounds = self._bounds
        size = self._size
        if subterrain is not None and subterrain in self._subterrain_bounds:
            size = self._subterrain_size
            bounds = self._subterrain_bounds[subterrain]

        (x_origin, x_max, y_origin, y_max) = bounds
        (x_size, y_size) = size

        # Adjust size based on buffer ratio
        usable_x_size = x_size * usable_ratio
        usable_y_size = y_size * usable_ratio
        buffer_x_size = (x_size - usable_x_size) / 2
        buffer_y_size = (y_size - usable_y_size) / 2

        # Calculate the bounds of the usable area within the section
        x_min = x_origin + buffer_x_size
        x_max = x_origin + x_size - buffer_x_size
        y_min = y_origin + buffer_y_size
        y_max = y_origin + y_size - buffer_y_size

        # Output
        output[out_idx, 0] = torch.rand(num, device=gs.device) * (x_max - x_min) + x_min
        output[out_idx, 1] = torch.rand(num, device=gs.device) * (y_max - y_min) + y_min
        terrain_heights = self.get_terrain_height(
            output[out_idx, 0], output[out_idx, 1]
        )
        output[out_idx, 2] = terrain_heights + height_offset
        return output

    def generate_random_env_pos(
        self,
        envs_idx: list[int] | None = None,
        usable_ratio: float = 0.5,
        subterrain: str | None = None,
        height_offset: float = 0.1e-3,
    ) -> torch.Tensor:
        """
        Generate one X/Y/Z position on the terrain for each environment.
        The X & Y positions will be random points and the Z position will be at the approximate terrain height at that point.

        Args:
            envs_idx: The indices of the environments to generate positions for.
                      If None, positions will be generated for all environments.
            usable_ratio: How much of the terrain/subterrain area should be used for random positions.
                          For example, 0.25 will only generate positions within the center 25% of the area of the terrain/subterrain.
                          This helps avoid placing things right on th edge of the terrain/subterrain.
            subterrain: The subterrain to generate positions for.
                        If None, positions will be generated for the entire terrain.
            height_offset: The offset to add to the terrain height.

        Returns:
            The position tensor of shape (1, 3)
        """
        if envs_idx is None:
            envs_idx = torch.arange(self.env.num_envs, device=gs.device)

        # Update the position buffer in-place
        self.generate_random_positions(
            output=self._env_pos_buffer,
            out_idx=envs_idx,
            usable_ratio=usable_ratio,
            subterrain=subterrain,
            height_offset=height_offset,
        )
        return self._env_pos_buffer[envs_idx]

    """
    Implementation
    """

    def _map_terrain(self):
        """Map out terrain and subterrain sizes and bounds."""
        (terrain_geom,) = self._terrain.geoms
        morph = self._terrain.morph
        aabb = terrain_geom.get_AABB()
        pos = terrain_geom.get_pos()

        # If there are parallel environments, take values for the first environment
        if aabb.ndim == 3:
            aabb = aabb[0]
        if pos.ndim == 2:
            pos = pos[0]

        # Get the total bounds of the terrain
        (x_min, y_min, _) = aabb[0]
        (x_max, y_max, _) = aabb[1]
        self._origin = pos
        self._size = (x_max - x_min, y_max - y_min)
        self._bounds = (x_min, x_max, y_min, y_max)

        # Get subterrain bounds
        if hasattr(morph, "n_subterrains") and morph.n_subterrains is not None:
            self._subterrain_size = morph.subterrain_size
            self._subterrain_bounds = {}
            i = 0
            for x in range(morph.n_subterrains[0]):
                for y in range(morph.n_subterrains[1]):
                    name = morph.subterrain_types[x][y]
                    x_min = self._origin[0] + x * self._subterrain_size[0]
                    y_min = self._origin[1] + y * self._subterrain_size[1]
                    x_max = x_min + self._subterrain_size[0]
                    y_max = y_min + self._subterrain_size[1]

                    self._subterrain_bounds[name] = (x_min, x_max, y_min, y_max)
                    i += 1

        # Height field
        if "height_field" in terrain_geom.metadata:
            height_field = terrain_geom.metadata["height_field"]
            vertical_scale = morph.vertical_scale
            self._height_field = torch.as_tensor(
                height_field, device=gs.device, dtype=gs.tc_float
            )

            # Adjust for the vertical scale
            self._height_field *= vertical_scale

            # Reshape from (width, height) to (height, width) for grid_sample calculation
            # We only need one copy since all environments share the same terrain
            self._height_field = self._height_field.T
