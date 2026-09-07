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
    self.terrain_manager = TerrainManager(self, terrain=self.terrain)

    # Robot manager
    # Randomize the robot's position on the terrain after reset
    self.robot_manager = EntityManager(
        self,
        entity=self.robot,
        on_reset={
            "position": {
                "fn": reset.randomize_terrain_position(
                    terrain_manager=self.terrain_manager,
                    height_offset=HEIGHT_OFFSET,
                ),
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
                "fn": rewards.base_height(
                    target_height=0.3,
                    terrain_manager=self.terrain_manager, # <- this line
                ),
            },
            # ... other rewards ...
        },
    )

    # ... other managers ...

```


## Training

### With [uv](https://docs.astral.sh/uv/) (recommended)

Training:

```shell
uv run ./train.py
```

Evaluation:

```shell
uv run ./eval.py
```

### Without uv:

Install dependencies

```shell
pip install -e ../../ "rsl-rl-lib~=5.0" tensorboard
```

Train:

```shell
python ./train.py
```

Evaluation:

```shell
python ./eval.py
```

## Monitor training status

You can view the training progress with:

```shell
tensorboard --logdir ./logs/
```

## Training videos

The Genesis Forge training environment will also save videos while training that can be viewed in `./logs/go2-terrain/videos`.
