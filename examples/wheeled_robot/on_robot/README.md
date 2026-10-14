# Running the policy on the car

This directory contains the program that runs the trained policy on the [Freenove 4WD car](https://store.freenove.com/products/fnk0043).

If you haven't trained the [wheeled_robot environment](../README.md) yet, do that first, and then run the deploy script.

## What you need

- [Freenove 4WD car](https://store.freenove.com/products/fnk0043)
- [A Raspberry Pi 4 or above](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/)
- [Two 18650 batteries](https://www.18650batterystore.com/collections/18650-batteries)
- [Logitech F710 gamepad](https://www.logitechg.com/en-us/shop/p/f710-wireless-gamepad)

## Train/Export

If you haven't already, on your main computer, train the model and export the bundle to the `on_robot` directory:

```bash
cd examples/wheeled_robot
uv run ./train.py
uv run ./deploy.py
```

## Setup your Raspberry Pi

Before you can run this on the car, you need to make sure the Rapsberry Pi is setup.
There are only a few steps:

1. Install the [Raspberry Pi Base OS](https://www.raspberrypi.com/documentation/computers/getting-started.html).
   I set mine up as headless (Raspberry Pi Lite 64-bit) with SSH enabled.
2. [Configure Raspberry Pi](https://www.raspberrypi.com/documentation/computers/configuration.html#config-methods) and
   enable I2C (under Interfaces or Interface Options)
3. Run this command from the terminal: `sudo apt install python3-dev python3-smbus libsdl2-2.0-0`

## Setup robot env and run

Copy the entire `on_robot` directory to your Raspberry Pi. We'll assume you put it in your home directory at `~/on_robot/`

Setup the python environment with the following shell commands:

```bash
cd ~/on_robot

# Create a python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the required python packages
pip install -r requirements.txt
```

Now, plug the Logitech F710 gamepad USB dongle into the Raspberry Pi, and run the program:

```bash
python ./run.py
```

If everything worked, the program will prompt you to press the gamepad's start button.
Once pressed, the left stick controls the robot's movement.

## How it works

Here are the important bits from [run.py](./run.py).

Load the deployed `trained_bundle.gfb` bundle file containing the trained policy and important genesis forge metadata.

```python
bundle = load_bundle("./trained_bundle.gfb")
```

Create functions to assemble observations and convert the raw policy actions into actuator velocities.

```python
obs_assembler = bundle.create_observation_assembler()
action_processor = bundle.create_action_processor()
```

Here we construct the observations which will soon be passed to the onnx policy runtime.

```python
observation = obs_assembler.assemble(
    {
        "velocity_cmd": gamepad.command(),
        "actions": action_processor.last_raw_actions,
    }
)
```

When we pass the observations to the policy, it returns the raw actions.
then `action_processor` converts them into velocity inputs (using the same algorithms as `VelocityActionManager`)

```python
raw_action = session.run(None, {input_name: observation[None, :].astype("float32")})[0]
action_targets = action_processor.process(np.ravel(raw_action))
```

Finally, we fetch the motor velocity values by joint name, and convert
them to PWM values used by the motors.

```python
pwm = (
    calculate_motor_pwm(
        action_targets.by_joint[name],
        action_processor.clip_range_by_joint[name],
    )
    for name in [
        "TT_Motor-1_axel",
        "TT_Motor-2_axel",
        "TT_Motor-3_axel",
        "TT_Motor-4_axel",
    ]
)
motors.set_wheels_pwm(*pwm)
```
