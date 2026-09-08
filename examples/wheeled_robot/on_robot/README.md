# Running the policy on the car

This directory contains the program that runs the trained policy on the [Freenove 4WD car](https://store.freenove.com/products/fnk0043).

If you haven't trained the [wheeled_robot environment](../README.md) yet, do that first, and then run the deploy script.

## What you need

- [Freenove 4WD car](https://store.freenove.com/products/fnk0043)
- [A Raspberry Pi 4 or above](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/)
- [Two 18650 batteries](https://www.18650batterystore.com/collections/18650-batteries)
- [Logitech F710 gamepad](https://www.logitechg.com/en-us/shop/p/f710-wireless-gamepad)

## Instructions

Run the training program (if you haven't already), and then run the deploy script:

```bash
cd examples/wheeled_robot
uv run ./train.py
uv run ./deploy.py
```

Then, copy the entire `on_robot` directory to your Raspberry Pi.

Now login to your Raspberry Pi and setup the python environment:

```bash
cd ~/on_robot # <~~ or wherever you copied the `on_robot` directory

sudo apt install libsdl2-2.0-0
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Finally, plug the Logitech F710 gamepad USB dongle into the Raspberry Pi, and fire up the run program:

```bash
python ./run.py
```

If everything worked, you should be able to drive your car around using the Logitech gamepad.
The left stick is forwards/backwards, and use the right stick to turn left/right.

## How it works

Here are the important bits from [run.py](./run.py).

Load the deployed `trained_bundle.gfb` bundle file containing the trained policy and important genesis forge metadata.

```python
bundle = load_bundle(args.bundle)
```

Create functions to assemble observations and convert the raw policy actions into actuator velocities.

```python
obs_assembler = bundle.create_observation_assembler()
action_decoder = bundle.create_action_decoder()
```

Here we construct the observations which will soon be passed to the onnx policy runtime.

```python
observation = obs_assembler.assemble({
      "velocity_cmd": gamepad.command(),
      "actions": action_decoder.last_raw_actions,
})
```

Pass the observations to your policy, which then returns the raw actions.
Those actions are decoded by the `action_decoder` to convert them into
velocity inputs (using the same algorithms as `VelocityActionManager`)

```python
raw_action = session.run(
    None, {input_name: observation[None, :].astype("float32")}
)[0]
action_targets = action_decoder.decode(np.ravel(raw_action))
```

Finally, we fetch the motor velocity values by joint name, and convert
them to PWM values used by the motors.

```python
pwm = (
    calculate_motor_pwm(action_targets.by_joint[name], joint_clip[name])
    for name in [
        "TT_Motor-1_axel",
        "TT_Motor-2_axel",
        "TT_Motor-3_axel",
        "TT_Motor-4_axel",
    ]
)
motors.set_wheels_pwm(*pwm)
```
