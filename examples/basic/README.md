# Go2 Basic Locomotion Example

A simple program that teaches the Go2 robot to walk forward.

This example uses the Genesis Forge managed environment class, which provides a modular setup to build and configure an RL environment.

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

```bash
tensorboard --logdir ./logs/
```

## Training videos

The Genesis Forge training environment will also save videos while training that can be viewed in `./logs/go2-basic/videos`.

For example:

https://github.com/user-attachments/assets/be46df1b-35e5-4b5b-9bbc-f543210dd463
