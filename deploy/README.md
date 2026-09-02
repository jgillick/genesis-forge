# genesis-forge-deploy

Simulation-free runtime for deploying [Genesis Forge](https://github.com/jgillick/genesis-forge)
policies to real robots.

This package deliberately depends on **numpy only** — no torch, no Genesis simulator —
so it installs cleanly on a Raspberry Pi or Jetson.

After training, export a bundle from your built environment:

```python
from genesis_forge.deployment import export

bundle = export(env, "./my_policy")   # writes ./my_policy.gfb
print(bundle.describe())
```

Then, on the robot:

```python
from genesis_forge_deploy import load_bundle

bundle = load_bundle("./my_policy.gfb")   # a directory works too
print(bundle.describe())                 # what to wire up

observation_assembler = bundle.create_observation_assembler()
action_decoder = bundle.create_action_decoder()

while True:
    observation = observation_assembler.assemble({
        "robot_ang_vel": imu.gyro,
        "dof_pos": joints.positions,
        "actions": action_decoder.last_raw_actions,   # zeros before the first tick
    })
    targets = action_decoder.decode(policy(observation))
    send_to_motors(targets.by_joint)
```

See the Genesis Forge deployment guide for the full control-loop walkthrough.

## Trust model

A bundle is **trusted input, equivalent to executable code**: loading one may import
decoder classes named inside it. Only load bundles you produced yourself. Treat a
bundle from a third party the same way you would treat an unpickled checkpoint.
