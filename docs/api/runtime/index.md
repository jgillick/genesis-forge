# Runtime

`genesis-forge-runtime` is the package installed on the robot. It sits between your
policy and your hardware and handles packaging observations for the model, as then
processing the returned actions into target values for your actuators.

It defaults to using numpy, and does not have any dependency on pytorch,
or the Genesis simulator.

```bash
pip install genesis-forge-runtime
```

The [deployment guide](../../guide/deployment.md) walks through the whole flow. In
brief, a control loop needs three things from here: [`load_bundle`](bundle.md) to
read what training recorded, an [`ObservationAssembler`](observations.md) to build
the policy's input vector, and an [`ActionProcessor`](actions.md) to turn its output
into joint targets.

```python
from genesis_forge_runtime import load_bundle

bundle = load_bundle("./go2_walk.gfb")
print(bundle.describe())

observation_assembler = bundle.create_observation_assembler()
action_processor = bundle.create_action_processor()

while True:
  # Package observations and get actions from your policy
  observation = observation_assembler.assemble({...})
  actions = policy(observation)

  # Convert raw actions to joint values that are sent to your motors
  targets = action_processor.process(actions)
  send_to_motors(targets.by_joint)
```
