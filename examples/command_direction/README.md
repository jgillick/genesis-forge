# Go2 - Commanded direction

Train a Go2 robot to move in a commanded direction, controlled programmatically or through a gamepad controller.

This builds on the [simple example](../simple/) example. The only thing necessary to convert that example from a robot that walks in a straight line
on the X-axis, is adding the `VelocityCommandManager`, and related rewards and observations.

```python
def config(self):
    # ...

    # Control manager
    # Sample random directions from the X/Y ranges, as well as a rotation velocity around the Z axis.
    self.velocity_command = VelocityCommandManager(
        self,
        range={
            "lin_vel_x": [-1.0, 1.0],
            "lin_vel_y": [-1.0, 1.0],
            "ang_vel_z": [-1.0, 1.0],
        },
        standing_probability=0.05,
        resample_time_sec=5.0,
        debug_visualizer=True,
        debug_visualizer_cfg={
            "envs_idx": [0],
        },
    )

    # Add command tracking to the reward manager
    RewardManager(
        self,
        logging_enabled=True,
        cfg={
            "tracking_lin_vel": {
                "weight": 1.0,
                "fn": rewards.command_tracking_lin_vel,
                "params": {
                    "vel_cmd_manager": self.velocity_command,
                    "entity_manager": self.robot_manager,
                },
            },
            "tracking_ang_vel": {
                "weight": 0.2,
                "fn": rewards.command_tracking_ang_vel,
                "params": {
                    "vel_cmd_manager": self.velocity_command,
                    "entity_manager": self.robot_manager,
                },
            },
            # ... other rewards ...
        },
    )

    # Add command to observations
    ObservationManager(
        self,
        cfg={
            "velocity_cmd": {"fn": self.velocity_command.observation, "scale": 0.5},
            # ... other observations ...
        },
    )
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

The Genesis Forge training environment will also save videos while training that can be viewed in `./logs/go2-command/videos`.

## Evaluation

Now you can view the trained policy:

```bash
python ./eval.py ./logs/go2-command/
```

## Gamepad control

If you have a logitech [F310](https://www.logitechg.com/en-us/shop/p/f310-gamepad.940-000110?sp=1&searchclick=Logitech%20G) or [F710](https://www.logitechg.com/en-us/shop/p/f710-wireless-gamepad) you can control the robot in the trained policy yourself.

First, you need to make sure that HIDAPI is installed on your machine.
https://github.com/libusb/hidapi?tab=readme-ov-file#installing-hidapi

Then, connect your gamepad, and run the following command:

```python
python ./gamepad.py
```

You should now be able to use the joysticks to control the Go2 robot.

### Troubleshooting

If you have trouble connecting to the gamepad on linux, you might need to update the udev rules:

Create the file: `/etc/udev/rules.d/100-hidapi.rules`

```
SUBSYSTEM=="usb", ATTR{idVendor}=="046d", ATTR{idProduct}=="c216", MODE="0660", GROUP="plugdev", TAG+="uaccess", TAG+="udev-acl", SYMLINK+="logitech_f310%n"
KERNEL=="hidraw*", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c216", MODE="0660", GROUP="plugdev", TAG+="uaccess", TAG+="udev-acl"
SUBSYSTEM=="usb", ATTR{idVendor}=="046d", ATTR{idProduct}=="c219", MODE="0660", GROUP="plugdev", TAG+="uaccess", TAG+="udev-acl", SYMLINK+="logitech_f710%n"
KERNEL=="hidraw*", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c219", MODE="0660", GROUP="plugdev", TAG+="uaccess", TAG+="udev-acl"
```

Then

```bash
sudo chmod 644 /etc/udev/rules.d/00-hidapi.rules
sudo udevadm control --reload-rules
```
