# Terrain Manager

The Terrain Manager loads a terrain and provides helpful function, like calculating your robot's hight above a varied terrain, or placing it at random places around a specific subterrain.

You can see a full example using the reward manager in [examples/rough_terrain](https://github.com/jgillick/genesis-forge/tree/main/examples/rough_terrain).

<video autoplay="" muted="" loop="" playsinline="" controls="" src="../../_static/terrain.webm"></video>

## Basic Usage

```python
from genesis_forge.managers import TerrainManager

class MyEnv(ManagedEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scene = gs.Scene()

        # Define your terrain, like usual
        self.terrain = scene.add_entity(
            morph=gs.morphs.Terrain(
                n_subterrains=(1, 2),
                subterrain_size=(15, 15),
                subterrain_types=[
                    ["flat_terrain", "fractal_terrain"],
                ],
            ),
        )

    def config(self):
        # Load your terrain into the manager
        self.terrain_manager = TerrainManager(
            self,
            terrain_attr="terrain",
        )

        # Place the robot in random places around the flat_terrain subterrain
        self.robot_manager = EntityManager(
            self,
            entity_attr="robot",
            on_reset={
                "position": {
                    "fn": reset.randomize_terrain_position,
                    "params": {
                        "terrain_manager": self.terrain_manager,
                        "subterrain": "flat_terrain", # Select this subterrain for placement
                        "height_offset": 0.3, # place the robot this high above the terrain
                    },
                },
            },
        )

        # Reward height above the terrain
        self.reward_manager = RewardManager(
            self,
            logging_enabled=True,
            cfg={
                "base_height_target": {
                    "weight": -50.0,
                    "fn": rewards.base_height,
                    "params": {
                        "target_height": 0.3,
                        "terrain_manager": self.terrain_manager,
                    },
                },
            }
        )
```

## Terrain Utilities

### Get height at X/Y position

Get terrain height at specific positions:

```python
# Get height at robot position
robot_pos = self.robot.get_pos()
terrain_height = self.terrain_manager.get_terrain_height(robot_pos[:, 0], robot_pos[:, 1])
```

### Generate random locations around a subterrain

If you want to get 10 random X/Y coordinates around a specific subterrain:

```python
pos = self.terrain_manager.generate_random_positions(num=10, subterrain="flat_terrain")
```

To integrate this into a reset function to place your robots:

```python
def reset(envs_idx: list[int]):
    pos = terrain_manager.generate_random_env_pos(
        envs_idx=envs_idx,
        subterrain="fractal_terrain",
        height_offset=0.4,
    )
    self.robot.set_pos(pos, envs_idx=envs_idx)
```
