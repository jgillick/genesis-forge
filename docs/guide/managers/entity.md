# Entity Manager

The Entity Manager handles robot spawning, resets, and state management. It provides a clean interface for resetting robot positions, velocities, and other properties when episodes end.

You can see a full example using the entity manager in [examples/simple](https://github.com/jgillick/genesis-forge/tree/main/examples/simple).

## Basic Usage

```python
from genesis_forge.managers import EntityManager
from genesis_forge.mdp import reset

class MyEnv(ManagedEnvironment):
    def __init__(self):
        super().__init__()

        # Construct the scene
        self.scene = gs.Scene(
            # ... scene settings ...
        )

        # Robot entity
        self.robot = self.scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf"))

    def config(self):
        self.robot_manager = EntityManager(
            self,
            entity_attr="robot", # references self.robot
            on_reset={
                "position": {
                    "fn": reset.position, # resets the robot to the same position and rotation at each reset
                    "params": {
                        "position": [0.0, 0.0, 0.4],  # X, Y, Z
                        "quat": [1.0, 0.0, 0.0, 0.0], # W, X, Y, Z quaternion
                    },
                },
            },
        )
```

## Reset Configuration

Each reward config item has the following possible values:

- **fn**: A function that handles the reset
- **params** (optional): Additional parameters which will be passed to the function

```python
EntityManager(
    self,
    entity_attr="robot", # references self.robot
    on_reset={
        "position": {
            "fn": reset.position, # resets the robot to the same position and rotation at each reset
            "params": {
                "position": [0.0, 0.0, 0.4],  # X, Y, Z
                "quat": [1.0, 0.0, 0.0, 0.0], # W, X, Y, Z quaternion
            },
        },
    },
)
```

## Built-in Reset Functions

Genesis Forge provides many common reset functions in [`genesis_forge.mdp.reset`](project:/api/mdp/reset.md):

## Custom Reset Functions

It's easy to define your own reset function. The first three params of any reset function are: the environment, the entity, and the environment indices that are being reset. Additionally, any params defined for that reset item in RewardManager will be passed by name.

For example, let's create a simple reset function that will randomly add mass to the entity's links:

```python
def add_mass_on_reset(
    env: GenesisEnv,
    entity: RigidEntity,
    envs_idx: list[int],
    link_name: string,
    mass_range: tuple[float, float]
):
    """
    Randomly add/subtract mass to links of the robot
    """
    link = entity.get_link(link_name)
    mass_shift = torch.tensor((env.num_envs, len(links_ids)), device=gs.device).uniform_(*mass_range)
    entity.set_mass_shift(
            mass_shift
            links_idx_local=[link.idx],
            envs_idx=envs_idx,
        )


EntityManager(
    self,
    entity_attr="robot",
    on_reset={
        "random_mass": {
            "fn": add_mass_on_reset,
            "params": {
                "link_name": "body",
                "mass_range": [-0.5, 1.0],
            },
        },
    },
)
```

You can see a more advanced, class-based, version of this reset method, by looking at the source to [randomize_link_mass_shift](project:/api/mdp/reset.md#genesis_forge.managers.mdp.reset.randomize_link_mass_shift):
