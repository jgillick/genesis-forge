# Deploy to your robot

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

| Package                 | Where it runs         | Depends on                               |
| ----------------------- | --------------------- | ---------------------------------------- |
| `genesis-forge`         | Your training machine | Genesis, torch, the RL framework         |
| `genesis-forge-runtime` | The robot             | numpy (plus `onnxruntime` if you use it) |

The robot never installs the simulator. That is the point of the split: a Raspberry
Pi has no business downloading a physics engine.

## Exporting

After your environment is built, export it:

```python
from genesis_forge.deployment import export

env = MyEnv(num_envs=1)
env.build()

bundle = export(env, "./my_policy", policy_path="policy.onnx")
print(bundle.describe())
```

`export()` returns the bundle it wrote, carrying the manifest and golden samples in
memory — so describing or checking what you just exported reads nothing back off
disk, and never unpacks the archive.

It writes `my_policy.gfb`, a **bundle**, holding:

```
manifest.json     # the deployment contract, human readable
golden.npz        # recorded input/output pairs, for an on-robot smoke test
policy.onnx       # your exported policy, if you passed one
policy.onnx.data  # its weights, when the export put them in a separate file
```

Re-exporting replaces a bundle already at that path, since doing it after every
training run is the normal thing to do. It will only ever replace a bundle, though:
a path holding anything else is refused, so a mistyped destination cannot cost you
a file.

The policy keeps whatever extension you handed it, and the manifest records what
kind of file it is — `policy.pt` and `format: "torchscript"` if that is what you
exported. Nothing here requires ONNX.

A policy is not always one file. ONNX keeps tensors above a size threshold in a
companion file, so `policy.onnx` can be a small skeleton whose weights live beside
it; OpenVINO always splits into `.xml` and `.bin`. Hand over everything the export
produced:

```python
export(env, "./my_policy", policy_path=["policy.onnx", "policy.onnx.data"])
```

The first entry is the one the runtime loads, and gets renamed to `policy.<ext>`.
The rest keep their own names, because a graph refers to its companions by the
filename recorded inside it.

Genesis Forge does not try to work out which files belong together — the naming
differs per format, and a wrong guess produces a bundle whose policy loads on your
machine, where the companion is still next door, and fails on the robot. Verifying
the _bundled_ policy, as below, is what catches a file left behind.

### One file instead of a folder

By default a bundle is written as a single `my_policy.gfb` file — one thing to
`scp`, and a transfer that either arrives whole or fails, rather than a folder that
can arrive missing a file and look fine until the robot loads it.

While you are working, a plain directory is easier to poke at — you can read
`manifest.json` straight out of it:

```python
export(env, "./my_policy", policy_path="policy.onnx", archive=False)
```

`load_bundle` reads either form, so nothing in your control loop changes:

```python
bundle = load_bundle("./my_policy.gfb")
```

An archive is unpacked beside itself into `.my_policy/` and reused on later loads,
so the files sit in one predictable place for as long as anything needs them —
`bundle.policy_path` stays valid, and nothing disappears from under a running
robot. Replacing the archive unpacks it again: the bundle you just deployed is
never masked by the one that was there before.

A `.gfb` is a zip. `unzip my_policy.gfb` works if you would rather unpack it
yourself and load the directory, and the format is recognised by content rather
than by name, so a bundle renamed to `.zip` still loads. Expect the archive to be
about the same size as the directory — the bulk is model weights and `golden.npz`,
both of which are already dense — so this is about handling, not space. And note
that keeping both the archive and its extraction roughly doubles the disk it uses,
which is worth knowing on a small SD card.

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

What each half of the gate covers is worth being precise about, because they are not
equally strong:

- **Action decoding is verified end to end.** The numpy decoder is compared against
  the manager's own `process_actions`, so the scale, offset, clipping, and joint
  order cannot drift apart unnoticed.
- **Observation assembly is verified as far as the layout.** The gate feeds both
  sides the same values, so it proves the ordering, per-entry scaling, and history
  stacking match. It does _not_ run your observation functions — supplying the
  values is what bypasses them — so it cannot tell you that the `dof_pos` your robot
  reads means the same thing as the `dof_pos` training computed. Matching units and
  frames on the robot is still yours to get right, and `bundle.describe()` prints the
  units it recorded to help.

### Recording where the bundle came from

Every bundle carries a `provenance` block. The exporter stamps what it can measure
for itself — when the export ran, and the Genesis Forge and torch versions it ran
under. Everything else depends on how you train, so you state it rather than the
library guessing:

```python
export(
    env,
    "./my_policy",
    policy_path="policy.onnx",
    additional_provenance={
        "checkpoint": "logs/my_run/model_500.pt",
        "framework": "rsl_rl",
        "framework_version": "5.4.2",
    },
)
```

```json
"provenance": {
  "exported_at": "2026-08-27T18:02:27+00:00",
  "genesis_forge_version": "1.0.0",
  "torch_version": "2.13.0",
  "additional": {
    "checkpoint": "logs/my_run/model_500.pt",
    "framework": "rsl_rl",
    "framework_version": "5.4.2"
  }
}
```

Your entries stay under `additional` rather than being merged in, so a reader can
tell a version the tooling observed from a value a person typed — and so a key of
yours can never quietly overwrite one of the measured ones.

`checkpoint`, `framework` and `framework_version` are the conventional keys and are
worth recording; beyond those the field is open, and a git commit, a robot serial,
or a dataset version are all reasonable things to put there. Values must survive the
trip to JSON — strings, numbers, bools, and lists or dicts of those. Paths are
converted for you; anything else that cannot be written is rejected up front, before
the parity gate runs.

## Running on the robot

Install just the runtime:

```bash
pip install genesis-forge-runtime
```

Copy the bundle over, and ask it what to wire up:

```python
from genesis_forge_runtime import load_bundle

bundle = load_bundle("./my_policy")
print(bundle.describe())
```

```
Bundle: my_policy
  control rate: 50.0 Hz (dt=0.02)
  observation vector: 45 values (15 per tick x 3 history)
  values you supply each tick:
    - robot_ang_vel (3 values) in rad/s, scaled by 0.25 -- Body-frame angular velocity
    - dof_pos (12 values) in rad -- Joint positions relative to the default pose
    - actions (12 values) -- Previous policy output
  joint targets produced (12):
    - [position] FL_hip, FL_thigh, FL_calf, ...
```

Then the control loop:

```python
import numpy as np
from genesis_forge_runtime import load_bundle

bundle = load_bundle("./my_policy")
observation_assembler = bundle.create_observation_assembler()
action_decoder = bundle.create_action_decoder()
policy = ...  # see "Running the policy" below

while True:
    observation = observation_assembler.assemble({
        "robot_ang_vel": imu.gyro,        # rad/s, body frame
        "dof_pos": joints.positions,      # rad, relative to default pose
        # This one is not a sensor -- see "Feeding the policy's output back" below.
        "actions": action_decoder.last_raw_actions,
    })

    targets = action_decoder.decode(policy(observation))

    for joint_name, target in targets.by_joint.items():
        motors[joint_name].set_position(target)   # a position manager -> position

    sleep_until_next_tick(bundle.manifest.dt)
```

### Feeding the policy's output back

That `"actions"` entry is not a sensor reading — it is the policy's own previous
output fed back in, a common input in locomotion policies. You read it off the
decoder instead of off your hardware, and pass it exactly like any other value.

The bundle does not mark these entries out, because nothing about _supplying_ them
differs: every entry in `bundle.describe()` is a value you pass each tick. What
differs is where you get it, and that follows from how you wrote the observation in
training:

| In training                           | On the robot                                           |
| ------------------------------------- | ------------------------------------------------------ |
| `current_actions()`                   | `action_decoder.last_raw_actions`                      |
| `current_actions(action_manager=mgr)` | `action_decoder.last_raw_actions_by_manager["<name>"]` |

The per-manager form matters once you have more than one action manager, since the
flat property holds the whole policy vector. The manager name is the attribute you
assigned it to in `config()`, and `bundle.describe()` lists those under the joint
targets.

If you leave one out, the assembler raises and names it — it never quietly feeds
zeros — so a forgotten feedback wire fails on the bench rather than on the robot.

A few things the runtime does for you:

- **Names, not indices.** You supply values by name and get targets back by joint
  name, so there is no index arithmetic to get wrong.
- **It refuses bad input.** A missing entry or a wrong-length value raises
  immediately and says which one — silence there would mean a misaligned vector. An
  unrecognized name raises too: it could not corrupt the vector, since entries are
  read by name, but it means your loop and the bundle disagree about what this
  policy consumes.
- **It refuses bad output.** If the policy emits `NaN` or infinity, the decoder raises
  rather than passing it to your motors.
- **Nothing is filled in behind your back.** Every entry is supplied by you,
  including the ones fed back from the decoder. A wire you forget raises
  immediately instead of quietly feeding zeros forever.

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
episode -- so it is a state the policy knows well. Call `observation_assembler.reset()` and
`action_decoder.reset()` whenever you (re)start the control loop, so the robot begins from
the same state an episode began from in training.

The first `history_length` ticks still carry less information than a full buffer, so
it is worth holding the robot in a safe posture until it fills.

## Running the policy

The runtime does not care how you run inference — hand `assemble()`'s output to
anything that takes a float32 vector. The bundle carries the policy file and records
its format; running it is yours to choose.

ONNX with `onnxruntime` is the usual choice on a Pi or Jetson, because it installs
without pulling in torch:

```bash
pip install genesis-forge-runtime[onnx]
```

```python
import onnxruntime

session = onnxruntime.InferenceSession("my_policy/policy.onnx",
                                       providers=["CPUExecutionProvider"])

def policy(observation):
    return session.run(None, {"obs": observation[None, :].astype("float32")})[0].ravel()
```

TorchScript works too, if you would rather keep torch on the robot:

```python
import torch

module = torch.jit.load("my_policy/policy.pt").eval()

def policy(observation):
    with torch.no_grad():
        return module(torch.from_numpy(observation)[None, :]).numpy().ravel()
```

That is a real trade: torch is a large install and slower to start, but it removes
the export step as a place for things to go wrong. ONNX is the recommendation on
small boards; TorchScript is reasonable on a Jetson or an x86 robot PC.

### Exporting the policy

**rsl_rl** ships an ONNX exporter:

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

For **TorchScript**, trace or script the same wrapped policy and save it with
`torch.jit.save`. The normalizer has to be inside what you trace, exactly as it does
for ONNX.

### Verifying the exported policy

Genesis Forge packages the policy file and records what format it is, but it does
not open it. Confirming the exported file still computes what the trained policy
computes is yours to do, in the same script that exported it — that code already
knows which framework produced the file and how to run it.

It is worth doing. An observation normalizer that silently failed to make it into
the graph is the classic sim-to-real failure, and no other check would see it.

`export()` hands back the bundle it wrote, with everything you need already in
memory: `golden["observations"]` holds the vectors the parity gate ran on, which are
exactly the right inputs to compare against.

Check the copy _inside the bundle_, not the file you passed in. That is the artifact
going to the robot, and it is the only way to notice a companion file you forgot to
list — the original graph would load perfectly, with its weights still sitting beside
it. `bundle.unpacked()` gives you the contents in a temporary directory and clears it
afterwards, so nothing is left next to your archive:

```python
import numpy as np, onnxruntime, torch

bundle = export(env, "./my_policy", policy_path="policy.onnx")

with bundle.unpacked() as directory:
    session = onnxruntime.InferenceSession(
        str(directory / bundle.policy_file), providers=["CPUExecutionProvider"]
    )
    name = session.get_inputs()[0].name

    for observation in bundle.golden["observations"]:
        batched = observation[None, :].astype("float32")
        exported = np.asarray(session.run(None, {name: batched})[0]).ravel()
        with torch.no_grad():
            reference = policy(torch.from_numpy(batched)).cpu().numpy().ravel()
        assert np.allclose(exported, reference, rtol=1e-4, atol=1e-5), "graph differs"
```

Compare **relatively**, not with a fixed absolute bound. Exporting reorders
floating-point accumulation, and that drift grows with how large your actions are —
a policy emitting wheel velocities around 50 drifts roughly fifty times further than
one emitting joint angles around 1, for exactly the same graph. On a trained
wheeled-robot policy the drift measures ~3e-05, while the _closest_ wrong checkpoint
diverges by at least 2e-01, so real faults sit orders of magnitude clear of rounding.

`examples/wheeled_robot/deploy.py` does this end to end, including the error message
worth printing when it fails.

## Installing on a Pi or Jetson

- A **64-bit OS is required** — `onnxruntime` publishes no 32-bit ARM wheels.
- `CPUExecutionProvider` is the supported baseline. A small MLP policy at control
  rates does not need a GPU.
- On Jetson, CUDA/TensorRT execution providers exist but come from NVIDIA's own
  builds rather than PyPI. They are optional and unsupported here.

## Action managers and what their targets mean

Every built-in action manager is deployable out of the box. The bundle records which
one produced each target, because that determines what you do with the number:

| Manager                             | `deploy_type`            | Targets are                                      |
| ----------------------------------- | ------------------------ | ------------------------------------------------ |
| `PositionActionManager`             | `position`               | Joint positions                                  |
| `PositionWithinLimitsActionManager` | `position_within_limits` | Joint positions, mapped into each joint's limits |
| `VelocityActionManager`             | `velocity`               | Joint/wheel velocities                           |

The first two go to a position command, the third to a velocity command — the
arithmetic that produces them is identical, which is exactly why the bundle names the
type rather than leaving you to infer it. `targets.by_joint` is keyed by joint name
either way; what the _value_ means is what changes, so check the type before wiring
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
from genesis_forge_runtime import ManagerDecoder

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
