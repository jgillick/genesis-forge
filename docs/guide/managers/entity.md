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
        self.scene = gs.Scene()
        self.robot = self.scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf"))

    def config(self):
        self.robot_manager = EntityManager(
            self,
            entity=self.robot,
            on_reset={
                "position": {
                    # resets the robot to the same position and rotation at each reset
                    "fn": reset.position(
                        position=[0.0, 0.0, 0.4],  # X, Y, Z
                        quat=[1.0, 0.0, 0.0, 0.0],  # W, X, Y, Z quaternion
                    ),
                },
            },
        )
```

## Reset Configuration

Each reset config item has the following possible values:

- **fn**: A function that handles the reset, constructed with its params

```python
EntityManager(
    self,
    entity=self.robot,
    on_reset={
        "position": {
            # resets the robot to the same position and rotation at each reset
            "fn": reset.position(
                position=[0.0, 0.0, 0.4],  # X, Y, Z
                quat=[1.0, 0.0, 0.0, 0.0],  # W, X, Y, Z quaternion
            ),
        },
    },
)
```

## Built-in Reset Functions

Genesis Forge provides many common reset functions in [`genesis_forge.mdp.reset`](project:/api/mdp/reset.md):

## Custom Reset Functions

A custom reset function is defined as a simple dataclass with a `__call__` method that
performs the reset. `__call__` always receives the environment, the entity, and the
environment ids being reset; any other fields you declare become the function's params.

For example, let's create a simple reset function that will randomly add mass to one of the entity's links:

```python
@dataclass(kw_only=True, eq=False)
class add_mass_on_reset(ResetMdpFn):
    """Randomly add/subtract mass to a link of the robot."""

    link_name: str = None
    mass_range: tuple[float, float] = (-0.5, 1.0)

    def build(self):
        self._link = self.entity.get_link(self.link_name)

    def __call__(self, env: GenesisEnv, entity: RigidEntity, envs_idx: torch.Tensor):
        mass_shift = torch.empty(len(envs_idx), device=gs.device).uniform_(*self.mass_range)
        entity.set_mass_shift(
            mass_shift,
            links_idx_local=[self._link.idx_local],
            envs_idx=envs_idx,
        )

class MyEnv(ManagedEnvironment):
    # ...

    def config(self):
        EntityManager(
            self,
            entity=self.robot,
            on_reset={
                "random_mass": {
                    "fn": add_mass_on_reset(link_name="body"),
                },
            },
        )
```

You can see a more advanced version of this reset function by looking at the source to [randomize_link_mass_shift](project:/api/mdp/reset.md#genesis_forge.mdp.reset.randomize_link_mass_shift):
