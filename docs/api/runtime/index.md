# Runtime

`genesis-forge-runtime` is the package installed on the robot. It sits between your
policy and your hardware: hand it sensor readings and it returns the observation
vector the policy expects; hand it the policy's output and it returns named joint
targets for your motors.

It does that using the ordering, scaling and history the policy was trained with,
and numpy alone — no Genesis, no torch, no simulator.

```bash
pip install genesis-forge-runtime
```

The [deployment guide](../../guide/deployment.md) walks through the whole flow. In
brief, a control loop needs three things from here: [`load_bundle`](bundle.md) to
read what training recorded, an [`ObservationAssembler`](observations.md) to build
the policy's input vector, and an [`ActionDecoder`](actions.md) to turn its output
into joint targets.

```python
from genesis_forge_runtime import load_bundle

bundle = load_bundle("./go2_walk.gfb")
print(bundle.describe())

observation_assembler = bundle.create_observation_assembler()
action_decoder = bundle.create_action_decoder()
```
