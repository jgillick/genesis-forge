# Go2 - Rough Terrain

**NOTE:** This example requires [Genesis Simulator](https://github.com/Genesis-Embodied-AI/Genesis) version 0.3.4+, in order to get this [bug fix](https://github.com/Genesis-Embodied-AI/Genesis/issues/1727), which affects rough terrain contacts.

Teaches the Go2 robot to walk on rough terrain. This environment uses a combination of the TerrainManager and EntityManager to place each robot randomly at a different place of terrain at each reset.

```python
def __init__(self):
    # ...other scene initialization...
    self.terrain = self.scene.add_entity(
        morph=gs.morphs.Terrain(
            n_subterrains=(1, 1),
            subterrain_size=(24, 24),
            subterrain_types="fractal_terrain",
        ),
    )

def config(self):
    # Terrain manager helps the EntityManager safetly place the robot above the terrain on reset
    self.terrain_manager = TerrainManager(self, terrain_attr="terrain")

    # Robot manager
    # Randomize the robot's position on the terrain after reset
    self.robot_manager = EntityManager(
        self,
        entity_attr="robot",
        on_reset={
            "position": {
                "fn": reset.randomize_terrain_position,
                "params": {
                    "terrain_manager": self.terrain_manager,
                    "height_offset": HEIGHT_OFFSET,
                },
            },
        },
    )

    # The terrain manager is used to automatically calculate the base height above the terrain
    RewardManager(
        self,
        logging_enabled=True,
        cfg={
            "base_height_target": {
                "weight": -50.0,
                "fn": rewards.base_height,
                "params": {
                    "target_height": 0.3,
                    "terrain_manager": self.terrain_manager, # <- this line
                },
            },
            # ... other rewards ...
        },
    )

    # ... other managers ...

```

## Training

We will be training the robot with the [rsl_rl](https://github.com/leggedrobotics/rsl_rl) training library. So first, we need to install that and tensorboard:

```bash
pip install tensorboard rsl-rl-lib>=2.2.4
```

Now you can run the training with:

```bash
python ./train.py
```

You can view the training progress with:

```bash
tensorboard --logdir ./go2-terrain/
```

The Genesis Forge training environment will also save videos while training that can be viewed in `./logs/go2-terrain/videos`.

## Evaluation

Now you can view the trained policy:

```bash
python ./eval.py ./logs/go2-terrain/
```
