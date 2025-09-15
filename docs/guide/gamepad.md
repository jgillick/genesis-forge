# 🎮 Gamepad Controllers

Genesis Forge currently integrates with the Logitech [F310](https://www.logitechg.com/en-us/shop/p/f310-gamepad.940-000110?sp=1&searchclick=Logitech%20G) and [F710](https://www.logitechg.com/en-us/shop/p/f710-wireless-gamepad) controllers.

```{figure} _images/f710.webp
:alt: F10 controller
:width: 350
:align: center
:class: dark-light
The Logitech F710 gamepad controller
```

## Installation

To use these controllers, you need to install [HIDAPI](https://github.com/libusb/hidapi) on your computer:

### Mac

With [Homebrew](https://brew.sh/)

```shell
brew install hidapi
```

### Windows

Download the windows files from [here](https://github.com/libusb/hidapi/releases) and then place them in `Windows/System32`.

### Linux

Use your package manager to install `libhidapi-dev`, for example:

```shell
sudo apt install libhidapi-dev
```

You'll likely also need to add udev rules for the controllers. Create the file: `/etc/udev/rules.d/100-hidapi.rules`

```{code-block}
:caption: /etc/udev/rules.d/100-hidapi.rules
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

## Usage

(see the [Command Manager](managers/command) guide for more information)

If you have command managers defined in your environment, you can setup an eval program to connect the gamepad to the commands in the environment.

For example, let's say you have both a velocity command and a target height command defined in your environment:

```{code-block} python
:caption: environment.py

self.velocity_command = VelocityCommandManager(
    self,
    range={
        "lin_vel_x": [-1.0, 1.0],
        "lin_vel_y": [-1.0, 1.0],
        "ang_vel_z": [-1.0, 1.0],
    },
)
self.height_command = CommandManager(self, range=(0.2, 0.4))
```

Now let's create an eval script that:

```{code-block} python
:caption: eval.py
EXPERIMENT_DIR = "./logs/experiment"
TRAINED_MODEL = f"{EXPERIMENT_DIR}/model_100.pt"

gs.init(logging_level="warning", backend=gs.gpu)

# Setup environment
env = MyEnv(num_envs=1, headless=False)
env.build()

# Connect to gamepad
gamepad = Gamepad()
env.velocity_command.use_gamepad(
    gamepad, lin_vel_y_axis=0, lin_vel_x_axis=1, ang_vel_z_axis=2
)
env.height_command.use_gamepad(gamepad, range_axis=3)

# Setup policy runner
env = RslRlWrapper(env)
[cfg] = pickle.load(open(f"{EXPERIMENT_DIR}/cfgs.pkl", "rb"))
runner = OnPolicyRunner(env, cfg, EXPERIMENT_DIR, device=gs.device)
runner.load(TRAINED_MODEL)
policy = runner.get_inference_policy(device=gs.device)

obs, _ = env.reset()
with torch.no_grad():
    while True:
        actions = policy(obs)
        obs, _rews, _dones, _infos = env.step(actions)
```

This will attach the left joystick to X/Y movements, and on the right joystick, left/right will control rotation around the Z axis, and up/down will control height.
