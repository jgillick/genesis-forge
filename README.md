<p align="center">
<img src="./media/logo_text.png" width="250" />
</p>

# Genesis Forge

A robotics RL training frameworks for Genesis inspired by Isaac Lab and Gymnasium.
The goal of Genesis Forge is to give developers the tools they need to get training quickly, with less of the boilerplate setup.
Genesis Forge is a modular framework, made up of managers and wrappers, each dedicated to specific areas of your robotics training program.

Features:

- 🦿 Action manager - Control your joints and actuators, within limits and with domain randomization
- 🏆 Reward/Termination managers - Simple and extensible reward/termination setup with automatic logging
- ↪️ Command managers - Generates random command values with debug visualization
- 🏔️ Terrain manager - Randomize locations across terrains and curriculum-based placement
- 💥 Contact manager - Comprehensive contact/collision detection and reward/termination functions
- 🎬 Video Wrapper - Automatically records videos at regular intervals during training
- 🕹️ Gamepad interface - Control trained policies directly with a physical gamepad controller.
- And more...

Learn more in the [documentation](https://genesis-forge.readthedocs.io/en/latest/)

<img src="media/cmd_locomotion.gif" alt="Massively parallel locomotion training" height="240" />
<img src="media/gamepad.gif" alt="Gamepad controller interface" height="240" />
<img src="media/terrain.gif" alt="Rough terrain" height="240" />
<img src="media/spider.gif" alt="Complex robots" height="240" />

## Install

Before installing Genesis Forge, ensure you have:

- Python >=3.10,<3.14
- pip package manager

(Optional) CUDA-compatible GPU for faster training

```shell
pip install genesis-forge
```

## Example

Here's an example of a environment to teach the Go2 robot how to follow direction commands. See the full runnable example [here](./examples/simple/).

```python
class Go2CEnv(ManagedEnvironment):
    def __init__(self, num_envs: int = 1):
        super().__init__(num_envs=num_envs)

        # Construct the scene
        self.scene = gs.Scene(show_viewer=False)
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=[0.0, 0.0, 0.35],
                quat=[1.0, 0.0, 0.0, 0.0],
            ),
        )

    def config(self):
        # Robot manager - Reset the robot's initial position on reset
        self.robot_manager = EntityManager(
            self,
            entity_attr="robot",
            on_reset={
                "position": {
                    "fn": reset.position,
                    "params": {
                        "position": [0.0, 0.0, 0.35],
                        "quat": [1.0, 0.0, 0.0, 0.0],
                    },
                },
            },
        )

        # Joint Actions
        self.action_manager = PositionActionManager(
            self,
            joint_names=[".*"],
            default_pos={
                ".*_hip_joint": 0.0,
                ".*_thigh_joint": 0.8,
                ".*_calf_joint": -1.5,
            },
            scale=0.25,
            use_default_offset=True,
            pd_kp=20,
            pd_kv=0.5,
        )

        # Commanded direction
        self.velocity_command = VelocityCommandManager(
            self,
            range={
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_z": [-1.0, 1.0],
            },
        )

        # Rewards
        RewardManager(
            self,
            cfg={
                "base_height_target": {
                    "weight": -50.0,
                    "fn": rewards.base_height,
                    "params": {
                        "target_height": 0.3,
                    },
                },
                "tracking_lin_vel": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_lin_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                    },
                },
                "tracking_ang_vel": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_ang_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                    },
                },
                "lin_vel_z": {
                    "weight": -1.0,
                    "fn": rewards.lin_vel_z_l2,
                },
            },
        )

        # Termination conditions
        self.termination_manager = TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={
                # The episode ended
                "timeout": {
                    "fn": terminations.timeout,
                    "time_out": True,
                },
                # Terminate if the robot's pitch and yaw angles are too large
                "fall_over": {
                    "fn": terminations.bad_orientation,
                    "params": {
                        "limit_angle": 10, # degrees
                    },
                },
            },
        )

        # Observations
        ObservationManager(
            self,
            cfg={
                "velocity_cmd": { "fn": self.velocity_command.observation },
                "angle_velocity": {
                    "fn": lambda env: self.robot_manager.get_angular_velocity(),
                },
                "linear_velocity": {
                    "fn": lambda env: self.robot_manager.get_linear_velocity(),
                },
                "projected_gravity": {
                    "fn": lambda env: self.robot_manager.get_projected_gravity(),
                },
                "dof_position": {
                    "fn": lambda env: self.action_manager.get_dofs_position(),
                },
                "dof_velocity": {
                    "fn": lambda env: self.action_manager.get_dofs_velocity(),
                    "scale": 0.05,
                },
                "actions": {
                    "fn": lambda env: self.action_manager.get_actions(),
                },
            },
        )
```

## Learn More

Check out the [user guide](https://genesis-forge.readthedocs.io/en/latest/guide/index.html) and [API reference](https://genesis-forge.readthedocs.io/en/latest/api/index.html)

## Citation

If you used Genesis Forge in your research, we would appreciate it if you could cite it.

```
@misc{Genesis,
  author = {Jeremy Gillick},
  title = {Genesis Forge: A modular framework for RL robot environments},
  month = {September},
  year = {2025},
  url = {https://github.com/jgillick/genesis-forge}
}
```
