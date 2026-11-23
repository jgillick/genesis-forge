# Go2 - Terrain Curriculum

Teaches the Go2 robot to walk on rough ground by starting it on flat ground and moving it to rougher terrain as it gets better. Each robot has its own level, so the robots that learn quickly move ahead without waiting for the rest.

## The terrain

The terrain has 5 levels, from flat ground (`level_0`) to 20cm bumps (`level_4`). Each level is one 12×12m subterrain, so a robot's level is also the index of the subterrain it's placed on:

```python
grid = []
for level in range(TERRAIN_LEVELS):
    rough = terrain.RandomUniform(
        name=f"level_{level}",
        min_height=0.0,
        max_height=0.05 * level,  # <~ flat at level 0, up to 20cm bumps at level 4
        step=0.01,
        downsampled_scale=0.25,
    )
    grid.append([rough])

self.terrain_layout = terrain.TerrainLayout(
    scene,
    subterrain_size=(12.0, 12.0),
    vertical_scale=0.001,  # the Go2 robot is small
    grid=grid,
)
```

## Moving between levels

When a robot's episode ends, its level changes based on how well it followed its velocity commands. That's its average `command_linear_velocity` reward for the episode, from 0 to 1:

- **Up a level** if it scored 0.8 or higher and lasted the whole episode.
- **Down a level** if it scored below 0.5, or fell before the episode ended. Walking off the edge of its level also ends the episode, but doesn't count as a fall, since following its commands well can take it there.
- Otherwise, it stays where it is.
- Robots that make it past the top level start over at a random level, so the policy doesn't forget the easier ground.

Each robot's level goes into a tensor, which the reset function reads to place it:

```python
"position": {
    "fn": reset.randomize_terrain_position(
        terrain_manager=self.terrain_manager,
        subterrain=self.env_subterrain_idx,  # <~ one subterrain index for each robot
        height_offset=HEIGHT_OFFSET,
    ),
},
```

All of this happens in `reset()` and `update_terrain_levels()` in [environment.py](./environment.py).

## Training

### With [uv](https://docs.astral.sh/uv/) (recommended)

Training:

```shell
uv run ./train.py
```

Evaluation, which starts the robot on the roughest level:

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

`Curriculum / terrain_level_avg` shows the average level of all the robots, so you can watch them work their way up, and `Curriculum / terrain_level_max` the highest level any of them has reached.

## Training videos

The Genesis Forge training environment will also save videos while training that can be viewed in `./logs/go2-terrain-curriculum/videos`.
