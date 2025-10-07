# Go2 Locomotion Gait Learning

This example teaches the Go2 quadruped robot to move with 4 different gaits. This implements the method described by ["Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition"](https://arxiv.org/pdf/2011.01387) (Siekmann et al., 2020), and adapted from the [Legged Gym walk this way environment](https://github.com/lupinjia/genesis_lr/blob/2b44e231007ae89c7f2189f6858240ffae3e28dc/legged_gym/envs/go2/go2_wtw/go2_wtw.py).

This is an advanced example, with multiple interesting parts:

- A custom command manager: [GaitCommandManager](./gate_command_manager.py)
- Curriculum learning
- Privileged critic observations and prior observation history
- Action manager delay, to emulate natural latency in the system

## How it works

The gait command manager is able to teach a variety of gaits by breaking down any gait into sequences of periodic phases (swing and stance), where:

- During swing phases, foot forces are penalized while velocities are allowed, encouraging the robot to lift its foot
- During stance phases, the opposite occurs

This leads to natural gait emergence without requiring reference trajectories or motion capture data.

### Supported Gaits

- **Walk**: A slow, stable gait where each foot lifts in sequence (FL → RR → FR → RL)
- **Trot**: Diagonal pairs of feet move together (FL/RR → FR/LR)
- **Pronk**: All four feet synchronized, with clear aerial phase (FL/FR/RL/RR)
- **Pace**: A running gait where the left and right legs move together (FL/RL → FR/RR)
- **Bound**: A hopping gait where the front and back legs move together (FL/FR → RL/RR)

### Curriculum learning

For a complex training program like this, it's best to setup a curriculum learning process. This way the robot has the ability to learn one behavior before being expected to learn the rest. In this case, we start by only teaching the trot gait. As the gait reward value hits `0.8`, we add an additional gait (pronk). This repeats until we've included all configured gaits.

Similarly, we increment the minimum foot clearance and gait period target ranges as the robot's rewards hit target values.

You can see this in the `update_curriculum` function in `environment.py`.

### Privileged Observations

In `environment.py`, you'll notice two `ObservationManager` blocks, with different names:

- `policy` - Observations for the "actor" policy.
- `critic` - Observations for the "critic" policy.

This creates an asymmetrical observation system, where the critic receives more information than would be available to the trained model, which helps the value estimates become more accurate. This provides a better training signal for the actor, leading to faster and more stable policy learning.

In this case, we concatenate the policy observations with the critic observations in PPO config, in `train.py` with the following grouping config:

```python
{
    #...
    "obs_groups": {
        "policy": ["policy"],
        "critic": ["critic", "policy"],
    },
}
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
tensorboard --logdir ./logs/
```

The Genesis Forge training environment will also save videos while training that can be viewed in `./logs/go2-gait/videos`.

## Evaluation

Now you can view the trained policy:

```bash
python ./eval.py
```
