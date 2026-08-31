# Deployment

Once a policy is trained, getting it onto a real robot means reproducing two
pipelines exactly as training built them:

- **Observation assembly** — the order the values go in, what each is scaled by, and
  how many past ticks are stacked.
- **Action decoding** — the scale, offset, and clipping applied to the policy's
  output, and which joint each number belongs to.

Recreating those by hand is tedious and easy to get subtly wrong, and a subtle
mistake shows up as a robot that misbehaves for no obvious reason. Genesis Forge
captures both pipelines from your built environment and gives you a simulation-free
runtime that replays them.

## The two pieces

| Package | Where it runs | Depends on |
|---------|---------------|------------|
| `genesis-forge` | Your training machine | Genesis, torch, the RL framework |
| `genesis-forge-deploy` | The robot | numpy (plus `onnxruntime` if you use it) |

The robot never installs the simulator. That is the point of the split: a Raspberry
Pi has no business downloading a physics engine.

## Exporting

After your environment is built, export it:

```python
from genesis_forge.deployment import export

env = MyEnv(num_envs=1)
env.build()

export(env, "./my_policy", checkpoint="logs/my_run/model_500.pt")
```

This writes a **bundle** — a directory holding:

```
my_policy/
  manifest.json   # the deployment contract, human readable
  golden.npz      # recorded input/output pairs, for an on-robot smoke test
  policy.onnx     # your exported policy, if you passed one
```

### The parity gate

Export does not simply write out what it found. Before anything reaches disk, it
runs the actual deployment classes — the same code the robot imports — against your
live training pipeline on identical inputs, over several ticks, including values
that sit right on the clipping boundaries. If the two disagree, the export fails and
names the component that diverged:

```
ParityError: Parity failed in action manager 'action_manager' (position). tick 2: the
deployment decoder and the manager's process_actions produced different joint targets.
Largest difference 4.500e-01 at index 1: deployment produced 1.35, training produced
0.9 (tolerance rtol=1.3e-06, atol=1e-05). The bundle was not written.
```

A bundle that exists is a bundle that passed.

## Running on the robot

Install just the runtime:

```bash
pip install genesis-forge-deploy
```

Copy the bundle over, and ask it what to wire up:

```python
from genesis_forge_deploy import load_bundle

bundle = load_bundle("./my_policy")
print(bundle.describe())
```

```
Bundle: my_policy
  control rate: 50.0 Hz (dt=0.02)
  observation vector: 45 values (15 per tick x 3 history)
  sensor values you supply each tick:
    - robot_ang_vel (3 values) in rad/s, scaled by 0.25 -- Body-frame angular velocity
    - dof_pos (12 values) in rad -- Joint positions relative to the default pose
  values you feed back from the decoder:
    - actions (12 values), from decoder.last_target_actions_by_manager["action_manager"]
  joint targets produced (12):
    - [position] FL_hip, FL_thigh, FL_calf, ...
```

Then the control loop:

```python
import numpy as np
from genesis_forge_deploy import load_bundle

bundle = load_bundle("./my_policy")
assembler = bundle.observation_assembler()
decoder = bundle.action_decoder()
policy = ...  # see "Running the policy" below

while True:
    observation = assembler.assemble({
        "robot_ang_vel": imu.gyro,        # rad/s, body frame
        "dof_pos": joints.positions,      # rad, relative to default pose
        # bundle.describe() prints this line verbatim -- copy it.
        "actions": decoder.last_target_actions_by_manager["action_manager"],
    })

    targets = decoder.decode(policy(observation))

    for joint_name, target in targets.by_joint.items():
        motors[joint_name].set_position(target)   # a position manager -> position

    sleep_until_next_tick(bundle.manifest.dt)
```

That `"actions"` entry is the policy's own previous output fed back in — a common
input in locomotion policies. You read it off the decoder rather than off a sensor.
`bundle.describe()` tells you which entries work that way and which decoder property
to use, and the assembler raises if you leave one out.

You do not annotate any of this. Export determines which observations echo the
pipeline's own output by running each one against known values and seeing what comes
back, so an ordinary `lambda env: self.action_manager.get_actions()` is recognised as
it stands. An entry that *transforms* the actions before returning them is treated as
a sensor input, since export will not guess; if you need to override the
classification, set `"pipeline_state"` on the observation config.

It also tells apart the two feedback shapes, which is worth knowing because they hold
different numbers:

| In training | Feeds back | On the robot |
|---|---|---|
| `lambda env: mgr.get_actions()` | decoded joint targets | `decoder.last_target_actions_by_manager[...]` |
| `current_actions(action_manager=mgr)` | that manager's raw policy output | `decoder.last_raw_actions_by_manager[...]` |
| `current_actions()` | the whole raw policy vector | `decoder.last_raw_actions` |

The per-manager forms are used whenever an entry belongs to a specific manager, since
the flat properties hold the entire policy vector — which is the same thing only when
you have exactly one action manager.

A few things the runtime does for you:

- **Names, not indices.** You supply values by name and get targets back by joint
  name, so there is no index arithmetic to get wrong.
- **It refuses bad input.** A missing entry, a wrong-length value, or an unknown name
  raises immediately and says which one. Silence would mean a misaligned vector.
- **It refuses bad output.** If the policy emits `NaN` or infinity, the decoder raises
  rather than passing it to your motors.
- **Feedback values are explicit, not magic.** If your observation config feeds
  actions back in, you pass `decoder.last_target_actions` (or `last_raw_actions`)
  yourself. Nothing is filled in behind your back, so a feedback wire you forget
  raises immediately instead of quietly feeding zeros forever.

### Match the control rate and gains

The bundle records the rate the policy was trained at (`bundle.manifest.control_hz`)
and the actuator gains from training:

```python
for actuator in bundle.manifest.actuators:
    print(actuator.joint_names, actuator.values["kp"], actuator.values["kv"])
```

A policy trained at 50 Hz behaves differently at 200 Hz, and one trained against
particular PD gains behaves differently against others. Both are worth matching
before blaming the policy.

### Reset when you restart control

History starts zero-filled, which is exactly what training does at the start of every
episode -- so it is a state the policy knows well. Call `assembler.reset()` and
`decoder.reset()` whenever you (re)start the control loop, so the robot begins from
the same state an episode began from in training.

The first `history_length` ticks still carry less information than a full buffer, so
it is worth holding the robot in a safe posture until it fills.

## Running the policy

The runtime does not care how you run inference — hand `assemble()`'s output to
anything that takes a float32 vector. ONNX with `onnxruntime` is the usual choice on
a Pi or Jetson:

```bash
pip install genesis-forge-deploy[onnx]
```

```python
import onnxruntime

session = onnxruntime.InferenceSession("my_policy/policy.onnx",
                                       providers=["CPUExecutionProvider"])

def policy(observation):
    return session.run(None, {"obs": observation[None, :].astype("float32")})[0].ravel()
```

### Exporting the policy to ONNX

**rsl_rl** ships this already:

```python
runner.export_policy_to_onnx(path="./my_policy", filename="policy.onnx")
```

It fuses the observation normalizer into the graph and names the input `obs` and the
output `actions`.

**skrl** has no built-in exporter, so wrap the deterministic policy together with its
state preprocessor and export that with `torch.onnx.export`. Two details matter:
`RunningStandardScaler` keeps its statistics in float64 (cast them to float32 first),
and it clamps normalized observations to ±5.0 — miss that and the graph will agree
with training on ordinary inputs and diverge on extreme ones.

Pass the exported file to `export()` along with the torch policy, and the parity gate
extends across the ONNX graph too:

```python
export(env, "./my_policy", policy_path="policy.onnx", torch_policy=policy)
```

This is worth doing. A normalizer that silently failed to make it into the exported
graph is the classic sim-to-real failure, and it is invisible to every other check.

## Installing on a Pi or Jetson

- A **64-bit OS is required** — `onnxruntime` publishes no 32-bit ARM wheels.
- `CPUExecutionProvider` is the supported baseline. A small MLP policy at control
  rates does not need a GPU.
- On Jetson, CUDA/TensorRT execution providers exist but come from NVIDIA's own
  builds rather than PyPI. They are optional and unsupported here.

## Action managers and what their targets mean

Every built-in action manager is deployable out of the box. The bundle records which
one produced each target, because that determines what you do with the number:

| Manager | `deploy_type` | Targets are |
|---------|---------------|-------------|
| `PositionActionManager` | `position` | Joint positions |
| `PositionWithinLimitsActionManager` | `position_within_limits` | Joint positions, mapped into each joint's limits |
| `VelocityActionManager` | `velocity` | Joint/wheel velocities |

The first two go to a position command, the third to a velocity command — the
arithmetic that produces them is identical, which is exactly why the bundle names the
type rather than leaving you to infer it. `targets.by_joint` is keyed by joint name
either way; what the *value* means is what changes, so check the type before wiring
it to a motor call:

```python
for spec in bundle.manifest.actions:
    print(spec.deploy_type, spec.joint_names)
```

`PositionActionManager` and `VelocityActionManager` both inherit their deployment
contract from `AffineDofActionManager`, so any future affine manager is deployable
without new runtime code — it only declares its own `deploy_type`.

## Custom action managers

If you have written your own `BaseActionManager` subclass, it participates in
deployment by describing its decode as plain data and shipping a decoder that replays
it. On the training side:

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
from genesis_forge_deploy import ManagerDecoder

class CartesianImpedanceDecoder(ManagerDecoder):
    def decode(self, actions):
        stiffness = self.spec.config["stiffness"]
        return np.asarray(actions, dtype=np.float32) * stiffness
```

If your manager is an affine one (scale, offset, clip), subclass
`AffineDofActionManager` instead and you inherit the contract — just set
`deploy_type`, and the runtime's built-in decoder handles it with no extra code.

Nothing in Genesis Forge needs to know your type exists. The parity gate checks your
decoder against your `process_actions` like any other, so the two cannot drift apart
unnoticed.

!!! note "This contract is provisional"
    The deployment contract has not yet been through a real hardware deployment. It
    may change once it has. The bundle carries a `schema_version` so an out-of-date
    bundle fails loudly rather than misbehaving.

## Trust model

**A bundle is trusted input, equivalent to executable code.** Loading one may import
decoder classes it names, so only load bundles you produced yourself. Treat a bundle
from someone else the way you would treat an unpickled checkpoint from someone else.
