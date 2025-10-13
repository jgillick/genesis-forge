# Berkeley Humanoid Robot

A simple locomotion training environment for the [Berkeley Humanoid Robot](https://berkeley-humanoid.com/) using the model from the [Mujoco Menagerie repository](https://github.com/google-deepmind/mujoco_menagerie/tree/main/berkeley_humanoid)

![Berkeley Humanoid Robot Image](./berkeley_humanoid.png)

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
tensorboard --logdir ./logs/
```

The Genesis Forge training environment will also save videos while training that can be viewed in `./logs/berkeley-humanoid/videos`.

## Evaluation

Now you can view the trained policy:

```bash
python ./eval.py ./logs/berkeley-humanoid/
```
