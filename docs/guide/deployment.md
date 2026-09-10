# Deploy to your robot

Once a policy is trained, getting it onto a real robot means reproducing two
pipelines exactly as training built them:

- **Observation assembly** — where each value sits in the vector, the scale applied to
  it, and how many previous ticks are stacked alongside the current one.
- **Action decoding** — the scale, offset, and clipping applied to the policy's
  output, and which joint each number belongs to.

Recreating those by hand is tedious and easy to get subtly wrong, and a subtle
mistake shows up as a robot misbehaving for no obvious reason. Genesis Forge
captures both pipelines from your built environment and gives you a simulation-free
runtime that applies them on the robot, between your sensors and your motors.

## The two pieces

| Package                                                                                                        | Where it runs         | Depends on                               |
| -------------------------------------------------------------------------------------------------------------- | --------------------- | ---------------------------------------- |
| `genesis-forge`                                                                                                | Your training machine | Genesis, torch, the RL framework         |
| [`genesis-forge-runtime`](https://github.com/jgillick/genesis-forge/tree/main/packages/genesis-forge-runtime/) | The robot             | numpy (plus `onnxruntime` if you use it) |

The robot never installs the simulator. That is the point of the split: a Raspberry
Pi has no business downloading a physics engine.

## The whole flow

On the training machine, once your environment is built:

```python
from genesis_forge.deployment import export

# Build your environment
env = MyEnv(num_envs=1)
env.build()

# Export trained policy to a portable format (onnx, TensorScript, etc)
# ...

# Export
bundle = export(env, "./go2_walk.gfb", policy_path="policy.onnx")
print(bundle.describe())   # bundle summary
```

Copy the single file it wrote to your robot:

```bash
scp go2_walk.gfb robot:~/
```

Then, on the robot:

```python
from genesis_forge_runtime import load_bundle

bundle = load_bundle("~/go2_walk.gfb")
observation_assembler = bundle.create_observation_assembler()
action_decoder = bundle.create_action_decoder()

while True:
    observation = observation_assembler.assemble({...})   # your sensor readings
    actions = policy(observation) # get actions from your policy
    targets = action_decoder.decode() # decode the actions into DOF values
    send_to_motors(targets.by_joint)
```

[`examples/wheeled_robot`](https://github.com/jgillick/genesis-forge/tree/main/examples/wheeled_robot)
is that whole path working on a real four-wheeled car: `train.py`, then `deploy.py`,
and finally `on_robot/run.py` on a Raspberry Pi. It is worth reading once end to end before
you build your own loop.

## Exporting

By default `export()` writes a `.gfb` file, which is a **bundle** archive, holding:

```
manifest.json         # the deployment contract, human readable
golden.npz            # recorded input/output pairs, for an on-robot smoke test
policy/               # your exported policy, if you passed one
  policy.onnx
  policy.onnx.data    # its weights, when the export put them in a separate file
```

A policy is not always one file: ONNX keeps tensors above a size threshold in a
companion `.onnx.data`, and the graph is useless without it. Give all policy file paths
to the export function:

```python
export(env, "./go2_walk",
  policy_path=[
    "./policy.onnx",
    "./policy.onnx.data",
  ]
)
```

Then, on the robot, `bundle.policy_files` gives the policy file names, in the order you listed them, and
`bundle.policy_dir` is the folder they sit in, relative to the bundle's root.

To look at `manifest.json` while you are working, pass `archive=False` for a plain
directory instead of the single file:

```python
export(env, "./go2_walk", policy_path="policy.onnx", archive=False)
```

`load_bundle` can read either form (`.gfb` file or plain directory), so nothing in your control loop changes. A `.gfb` is
just a zip, so `unzip go2_walk.gfb` works too.

### The parity gate

Before anything reaches disk, export runs the deployment classes — the same code the
robot imports — against your live training pipeline, and refuses to write if they
disagree:

```
ParityError: Parity failed in action manager 'action_manager' (position). tick 2: the
deployment decoder and the manager's process_actions produced different joint targets.
Largest difference 4.500e-01 at index 1: deployment produced 1.35, training produced
0.9 (tolerance rtol=1.3e-06, atol=1e-05). The bundle was not written.
```

A bundle that exists is a bundle that passed.

The two halves are not equally strong:

- **Action decoding is verified end to end.** The numpy decoder is compared against
  the manager's own `process_actions`, so scale, offset, clipping and joint order
  cannot drift apart unnoticed.
- **Observation assembly is verified only as far as the layout.** Feeding both sides
  the same values proves that ordering, scaling and history stacking agree. It also
  means your observation functions never run, so nothing checks that the `dof_pos`
  your robot reports uses the same units, sign and zero point as training did — see
  [Where each observation comes from](#where-each-observation-comes-from).

### Recording where the bundle came from

When a robot misbehaves, the first question is which export produced the bundle it is
running. Every bundle answers half of that itself: the exporter stamps the time and
the Genesis Forge and torch versions. It cannot know which checkpoint you trained, so
pass that yourself:

```python
export(
    env,
    "./go2_walk",
    policy_path="policy.onnx",
    additional_provenance={
        "checkpoint": "logs/my_run/model_500.pt",
        "framework": "rsl_rl",
        "framework_version": "5.4.2",
    },
)
```

And those values land under `additional`:

```json
"provenance": {
  "exported_at": "2026-08-27T18:02:27+00:00",
  "genesis_forge_version": "1.0.0",
  "torch_version": "2.13.0",
  "additional": { "checkpoint": "logs/my_run/model_500.pt", ... }
}
```

Anything JSON-serializable works — a git commit, a robot serial number, etc.

## Running on the robot

Install just the runtime:

```bash
pip install genesis-forge-runtime
```

Copy the bundle over, and load it:

```python
from genesis_forge_runtime import load_bundle

bundle = load_bundle("./go2_walk.gfb")
print(bundle.describe())
```

```
Bundle: go2_walk
  control rate: 50.0 Hz (dt=0.02)
  observation vector: 45 values (15 per tick x 3 history)
  values you supply each tick:
    - robot_ang_vel (3 values), in rad/s -- Body-frame angular velocity
    - dof_pos (12 values), in rad -- Joint positions relative to the default pose
    - actions (12 values) -- Previous policy output
  joint targets produced (12):
    - [position] FL_hip, FL_thigh, FL_calf, ...
```

Then the control loop:

```python
import numpy as np
from genesis_forge_runtime import load_bundle

bundle = load_bundle("./go2_walk")
observation_assembler = bundle.create_observation_assembler()
action_decoder = bundle.create_action_decoder()
policy = ...  # see "Running the policy" below

while True:
    observation = observation_assembler.assemble({
        "robot_ang_vel": my_imu.read(),
        "dof_pos": my_actuators.positions(),
        "actions": action_decoder.last_raw_actions,
    })

    actions = policy(observation)
    targets = action_decoder.decode(actions)
    for joint_name, target in targets.by_joint.items():
        my_actuators.set_position(joint_name, target)
```

You address everything by name, never by index, and nothing is filled in for you: a
missing, mis-sized or unrecognized entry raises, as does a policy output containing
`NaN` or infinity.

### Where each observation comes from

`bundle.describe()` lists what to supply each tick, but not where to get it. Every
entry falls into one of three groups, and the third is where deployments usually go
wrong.

**1. Read straight off a sensor.** Joint positions and velocities from your encoders,
angular velocity from an IMU. Pass the raw reading — the assembler applies whatever
scaling training used, so scaling it yourself applies it twice.

**2. Previous actions.** An entry that echoes the policy's own previous output —
common in locomotion policies. You read it off the decoder rather than off hardware.
Which property depends on how you wrote the observation in training:

| In training                                        | On the robot                                           |
| -------------------------------------------------- | ------------------------------------------------------ |
| `observations.current_actions()`                   | `action_decoder.last_raw_actions`                      |
| `observations.current_actions(action_manager=mgr)` | `action_decoder.last_raw_actions_by_manager["<name>"]` |
| `action_manager.get_actions()`                     | `action_decoder.last_target_actions`                   |

The first two are raw policy output, the third decoded joint targets. Getting this
wrong is quiet: with one action manager the two are often the same width, so the
assembler accepts either.

**3. Derived, with no sensor behind it.** `projected_gravity`, `base_lin_vel` and
similar read simulator state no sensor reports, so you compute them — from an
IMU's orientation, a state estimator, or similar. If an entry has no plausible source
on your robot at all (e.g. your robot does not have an IMU), you might be able to effectively move those observations to a privileged observer in training, to keep them off the deployed observation list (see the [wheeled_robot example](https://github.com/jgillick/genesis-forge/tree/main/examples/wheeled_robot))

### Match the control rate and gains

The bundle records the rate the policy was trained at (`bundle.manifest.control_hz`)
and the actuator gains from training:

```python
for actuator in bundle.manifest.actuators:
    print(
      actuator.joint_names,
      actuator.values["kp"],
      actuator.values["kv"],
    )
```

The policy has no clock — it maps observations to actions. The rate matters because
each action is _held_ until the next tick, so a slower loop applies every command for
longer; and because stacked history spans a wall-clock window, so the same slots cover
a different span of time. Balancing and locomotion policies are sensitive to both; a
velocity-command robot much less so.

Match the gains for the same reason: the policy learned what a given target does to a
joint driven at those gains. Match both before blaming the policy.

### Reset when you restart control

Call `observation_assembler.reset()` and `action_decoder.reset()` whenever you
(re)start the loop. Observation history (when applicable) starts zero-filled, which is where every training episode
began, so the robot resumes from a state the policy knows.

### The first run

A policy that behaved in simulation can still do something violent on a real floor.
Put the robot on a stand or a harness, keep a hand on the power, and leave the
decoder's `check_finite` on so a `NaN` stops the loop instead of reaching a motor.

Before any of that, `golden.npz` is a free bench check: it holds the observations and
joint targets the parity gate ran on, so you can feed the recorded observations
through your policy and decoder with the motors disconnected and confirm you get the
recorded targets back.

### Troubleshooting

If the robot misbehaves, here's what to look for, in rough order of likelihood:

- **A wrong observation** — units, frame or sign disagreeing with training. Much the
  most common, and the export cannot catch it. See
  [Where each observation comes from](#where-each-observation-comes-from).
- **The wrong action feedback sent to observations** — `last_raw_actions` and `last_target_actions` are
  different vectors, and often the same width, so the wrong one passes silently.
- **Control rate** — a loop at 30 Hz because inference and I/O were not counted is not
  the 50 Hz the policy trained at. Pace off `bundle.manifest.dt`; see
  [Match the control rate and gains](#match-the-control-rate-and-gains).
- **Actuator gains** — match `bundle.manifest.actuators`.
- **No reset** — stale history from a previous run or a long pause.
- **The exported graph** — a normalizer left out of the ONNX graph looks fine at every
  step above. See [Verifying the exported policy](#verifying-the-exported-policy).

## Running the policy

The runtime doesn't care how you run inference. In the example above, we just used
a fake function `policy()` to represent sending observations to the policy and receiving
actions. Run your exported graph with whatever engine you like — ONNX is the usual choice on a Pi or Jetson, because `onnxruntime` installs without pulling in torch.

For example, with onnx:

```bash
pip install genesis-forge-runtime onnxruntime
```

```python
import onnxruntime

session = onnxruntime.InferenceSession(
    str(bundle.path / bundle.policy_path),
    providers=["CPUExecutionProvider"]
)

actions = session.run(
    None,
    {"obs": observation[None, :].astype("float32")}
)[0].ravel()
```

TorchScript works too:

```python
import torch

module = torch.jit.load(
    bundle.path / bundle.policy_path
).eval()

with torch.no_grad():
    actions = module(
        torch.from_numpy(observation)[None, :]
    ).numpy().ravel()
```

### Exporting the policy

**rsl_rl** ships an ONNX exporter:

```python
runner = OnPolicyRunner(env, cfg)
runner.load(checkpoint)

runner.export_policy_to_onnx(
    path="./exported",
    filename="policy.onnx",
)
```

It fuses the observation normalizer into the graph and names the input `obs` and the
output `actions`.

**skrl** has no built-in exporter, so wrap the deterministic policy together with its
state preprocessor and export that with `torch.onnx.export`. Two details matter:
`RunningStandardScaler` keeps its statistics in float64 (cast them to float32 first),
and it clamps normalized observations to ±5.0 — miss that and the graph will agree
with training on ordinary inputs and diverge on extreme ones.

For **TorchScript**, trace or script the same wrapped policy and save it with
`torch.jit.save`. The normalizer has to be inside what you trace, exactly as it does
for ONNX.

### Verifying the exported policy

Genesis Forge packages the policy files but never opens them, so checking that the
exported graph still computes what the trained policy computes is yours — and worth
doing: a normalizer that silently failed to make it into the graph is the classic
sim-to-real failure, and nothing else would catch it.

`bundle.golden["observations"]` holds the vectors the parity gate ran on, which
are exactly the right inputs to compare against:

To see an example of verifying an onnx policy, look at the [`verify_onnx_policy` function](https://github.com/jgillick/genesis-forge/tree/main/examples/wheeled_robot/deploy.py)

## Action managers and what their targets mean

Every built-in action manager is deployable out of the box. The bundle records which
one produced each target, because that determines what you do with the number:

| Manager                             | `deploy_type`            | Targets are                                      |
| ----------------------------------- | ------------------------ | ------------------------------------------------ |
| `PositionActionManager`             | `position`               | Joint positions                                  |
| `PositionWithinLimitsActionManager` | `position_within_limits` | Joint positions, mapped into each joint's limits |
| `VelocityActionManager`             | `velocity`               | Joint/wheel velocities                           |

The arithmetic behind all three is identical, which is why the bundle names the type
rather than leaving you to infer it. `targets.by_joint` is keyed by joint name either
way, so check the type before wiring it to a motor call:

```python
for spec in bundle.manifest.actions:
    print(spec.deploy_type, spec.joint_names)
```

### Grouped joints

`action_groups` lets one policy output drive several joints — a robot's wheels on
one side, say. The bundle records which action drives each joint, and the runtime
fans them out the same way before decoding, so `targets.by_joint` still has an entry
per joint:

```
  actions: 2 policy output(s) -> 4 joint target(s)
  joint targets produced (4, from 2 policy outputs):
    - [velocity] TT_Motor-1_axel, TT_Motor-2_axel, TT_Motor-3_axel, TT_Motor-4_axel
```

Grouping shares the _action_, not the decode: every scale, offset and clip bound
stays per joint. That is what lets mirrored wheels take one command and turn opposite
ways — the wheeled-robot example exports `scale: [-20, 20, -20, 20]` behind two
actions.

## Custom action managers

A `BaseActionManager` subclass participates in deployment by describing its decode as
plain data and shipping a decoder that reproduces it. On the training side:

```python
from genesis_forge.managers.action.base import BaseActionManager, DeploymentActionConfig

class CartesianImpedanceActionManager(BaseActionManager):
    deploy_type = "cartesian_impedance"

    def get_deployment_config(self):
        return DeploymentActionConfig(
            deploy_type=self.deploy_type,
            config={"stiffness": self._stiffness.tolist()},
            decoder_import_path="my_robot.decoders:CartesianImpedanceDecoder",
        )
```

And on the robot, in a module that imports without torch:

```python
import numpy as np
from genesis_forge_runtime import ManagerDecoder

class CartesianImpedanceDecoder(ManagerDecoder):
    def reset(self):
        # Config arrives as plain JSON data, so convert once here rather than
        # on every tick. `reset` runs at construction and at episode start.
        self._stiffness = np.asarray(self.spec.config["stiffness"], dtype=np.float32)

    def decode(self, actions):
        return np.asarray(actions, dtype=np.float32) * self._stiffness
```

If your manager is an affine one (scale, offset, clip), subclass
`AffineDofActionManager` instead: set `deploy_type` and the built-in decoder handles
it with no extra code. Either way the parity gate checks your decoder against your
`process_actions` like any other.

!!! note "This contract is still settling"

    The deployment contract has been through one real hardware deployment so far,
    and may still change. The bundle carries a `schema_version` so an out-of-date
    bundle fails loudly rather than misbehaving.

## Trust model

**Only load bundles you produced.** A bundle naming a custom decoder records its
import path, and `create_action_decoder()` imports that module — running whatever is
at its top level. Since a bundle from elsewhere can name any module on the robot's
path, treat one the way you would treat a checkpoint from a stranger.
