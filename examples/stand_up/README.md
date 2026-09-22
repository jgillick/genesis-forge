# Go2 Stand-Up Example

Train the Unitree Go2 to stand up from random collapsed poses on flat ground. The rewards have been shaped to encourage moving the joints deliberately for a controlled ascent. 

There are 100 random ground positions, stored in [`ground_positions.py`](ground_positions.py), and each time the robot resets, one of them is chosen as a starting point. To generate new random positions, run the [`generate_random_ground_pos.py`](./generate_random_ground_pos.py) script


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

The Genesis Forge training environment will also save videos while training that can be viewed in `./logs/go2-stand-up/videos`.
