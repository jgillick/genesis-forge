# Contact Manager

:::{important}
It's recommended to install [Genesis Simulator](https://github.com/Genesis-Embodied-AI/Genesis) from source, in order to get this [bug fix](https://github.com/Genesis-Embodied-AI/Genesis/issues/1727), which affects the contact manager.
:::

The Contact Manager tracks collisions and contacts between your robot and the environment. It's essential for detecting foot contacts, illegal collisions, and computing contact-based rewards or terminations.

## Basic Usage

```python
from genesis_forge.managers import ContactManager

class MyEnv(ManagedEnvironment):
    def config(self):
        # Detect the body hitting the ground
        self.contact_manager = ContactManager(
            self,
            entity_attr="robot",
            sensor_links=["body"],
            with_entity_attr="terrain"
        )

        # Terminate when the body touches the floor with more than 10N
        TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={
                "body_contact": {
                    "fn": terminations.contact_force,
                    "params": {
                        "threshold": 10.0,
                        "contact_manager": self.contact_manager,
                    },
                },
            },
        )
```

## Foot air-time rewards

To encourage your robot to take longer steps, use air time tracking and rewards:

```python
self.foot_contact_manager = ContactManager(
    self,
    link_names=[".*_foot"],
    track_air_time=True, # Whether to track the air/contact time of the links
    air_time_contact_threshold=5.0, # How much contact force is considered a step
)

RewardManager(
    self,
    logging_enabled=True,
    cfg={
        "foot_air_time": {
            "weight": 1.0,
            "fn": rewards.feet_air_time,
            "params": {
                "time_threshold": 0.5, # Target air-time, in seconds
                "contact_manager": self.foot_contact_manager,
                "vel_cmd_manager": self.velocity_command, # reduces the penalty if the the velocity command is close to zero
            },
        },
    }
)
```

## Self-contacts

Penalize or terminate on the robot hitting itself.

```python
class MyEnv(ManagedEnvironment):

    def config(self):
        # Detect the body links colliding with other body liks
        self.contact_manager = ContactManager(
            self,
            entity_attr="robot",
            with_entity_attr="robot"
        )

        RewardManager(
            self,
            cfg={
                "self_contact": {
                    "weight": -1.0,
                    "fn": rewards.contact_force,
                    "params": {
                        "threshold": 1.0, # Only collisions that are above 1.0N
                        "contact_manager": self.self_contact,
                    },
                },
            },
        )
```

## Contact Visualization

<video autoplay="" muted="" loop="" playsinline="" controls="" src="../../_static/contacts_debug.webm" width="100%"></video>

<p align="center">
<em>Foot contacts are marked in red</em>
</p>

To visualize which contacts are being registered, you can enable debugging, with the `debug_visualizer` param, and red spheres will appear where the contacts happen.

```python
self.contact_manager = ContactManager(
    self,
    entity_attr="robot",
    sensor_links=["body"],
    with_entity_attr="terrain"
    debug_visualizer=True,
    debug_visualizer_cfg={
        "envs_idx": [0],
    },
)
```

:::{caution}
This can slow down the simulation since the debug spheres need to be calculated and rendered for each environment on every step.

It's recommended to only enable them for a small number of environments at a time with the `envs_idx` configuration setting.
:::
