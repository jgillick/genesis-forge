# Entity Manager

The Entity Manager handles robot spawning, resets, and state management. It provides a clean interface for resetting robot positions, velocities, and other properties when episodes end.

You can see a full example using the entity manager in [examples/simple](https://github.com/jgillick/genesis-forge/tree/main/examples/simple).

## Basic Usage

```python
from genesis_forge.managers import EntityManager
from genesis_forge.managers.entity import reset

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
