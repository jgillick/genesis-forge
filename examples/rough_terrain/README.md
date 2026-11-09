# Go2 - Rough Terrain

Teaches the Go2 robot to walk on uneven ground. The terrain is a `TerrainLayout` of named subterrains, and every robot starts each episode at a random spot near the middle, where they meet.

```python
def __init__(self):
    # ...other scene initialization...

    # Create 4 different kind of terrains
    self.terrain_layout = terrain.TerrainLayout(
        self.scene,
        subterrain_size=(12, 12),
        vertical_scale=0.001,  # the Go2 robot is small
        grid=[
            [
                terrain.RandomUniform(
                    name="rough",
                    min_height=0.0,
                    max_height=0.1,
                    step=0.05,
                    downsampled_scale=0.25,
                ),
                terrain.PyramidStairs(name="stairs_up", step_height=0.1),
            ],
            [
                terrain.PyramidStairs(name="stairs_down", step_height=-0.1),
                terrain.RandomGrid(name="rough_grid", grid_height=0.05),
            ],
        ],
    )


    # ... other managers ...
```

Want the robots to work their way up to harder ground as they get better? See the [terrain curriculum example](../terrain_curriculum).

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
pip install -e ../../packages/genesis-forge "rsl-rl-lib~=5.0" tensorboard
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
