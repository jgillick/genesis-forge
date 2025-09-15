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

## Reset Functions

### Position Reset

Reset robot position and orientation:

```python
EntityManager(
    self,
    entity_attr="robot",
    on_reset={
        "position": {
            "fn": reset.position,
            "params": {
                "position": [0.0, 0.0, 0.5],  # X, Y, Z
                "quat": [1.0, 0.0, 0.0, 0.0],  # W, X, Y, Z quaternion
                "zero_velocity": True,  # Also zero linear/angular velocity
            },
        },
    },
)
```

### Position with Randomization

Add randomization for robustness:

```python
EntityManager(
    self,
    entity_attr="robot",
    on_reset={
        "position": {
            "fn": reset.position,
            "params": {
                "position": [0.0, 0.0, 0.5],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "position_noise": 0.1,  # ±0.1m random offset
                "rotation_noise": 0.2,  # ±0.2 rad random rotation
                "zero_velocity": True,
            },
        },
    },
)
```

### Velocity Reset

Reset velocities without changing position:

```python
EntityManager(
    self,
    entity_attr="robot",
    on_reset={
        "velocity": {
            "fn": reset.velocity,
            "params": {
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
                "velocity_noise": 0.5,  # Add random velocity
            },
        },
    },
)
```

## Multiple Reset Functions

Chain multiple reset operations:

```python
EntityManager(
    self,
    entity_attr="robot",
    on_reset={
        # First reset position
        "position": {
            "fn": reset.position,
            "params": {
                "position": [0.0, 0.0, 0.5],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
        },
        # Then apply random push
        "push": {
            "fn": reset.random_push,
            "params": {
                "force_range": [50, 100],
                "duration": 0.1,
            },
        },
        # Custom reset function
        "custom": {
            "fn": self.custom_reset_fn,
            "params": {"some_param": 1.0},
        },
    },
)
```

## Custom Reset Functions

### Simple Custom Reset

```python
def reset_to_random_height(env, entity_manager, envs_idx):
    """Reset robot to random height."""
    num_resets = len(envs_idx)
    heights = torch.rand(num_resets, device=gs.device) * 0.3 + 0.3  # 0.3-0.6m

    positions = torch.zeros((num_resets, 3), device=gs.device)
    positions[:, 2] = heights

    entity = getattr(env, entity_manager.entity_attr)
    entity.set_pos(positions, envs_idx)

EntityManager(
    self,
    entity_attr="robot",
    on_reset={
        "random_height": {
            "fn": reset_to_random_height,
        },
    },
)
```

### Complex Custom Reset

```python
def terrain_aware_reset(env, entity_manager, envs_idx):
    """Reset robot based on terrain difficulty."""
    num_resets = len(envs_idx)
    entity = getattr(env, entity_manager.entity_attr)

    for i, idx in enumerate(envs_idx):
        # Get terrain difficulty at this environment
        difficulty = env.terrain.get_difficulty(idx)

        # Easy terrain: normal spawn
        if difficulty < 0.3:
            pos = torch.tensor([0.0, 0.0, 0.4], device=gs.device)
            quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)

        # Medium terrain: random orientation
        elif difficulty < 0.7:
            pos = torch.tensor([0.0, 0.0, 0.5], device=gs.device)
            # Random yaw
            yaw = torch.rand(1, device=gs.device) * 2 * 3.14159
            quat = euler_to_quat(0, 0, yaw)

        # Hard terrain: random position and orientation
        else:
            pos = torch.rand(3, device=gs.device) * 2 - 1  # [-1, 1]
            pos[2] = 0.6  # Higher starting height
            # Random orientation
            rpy = torch.rand(3, device=gs.device) * 0.2 - 0.1  # Small tilts
            quat = euler_to_quat(rpy[0], rpy[1], rpy[2])

        entity.set_pos(pos.unsqueeze(0), [idx])
        entity.set_quat(quat.unsqueeze(0), [idx])
```

## State Access Methods

The EntityManager provides convenient methods for accessing entity state:

### Position and Orientation

```python
# Get position
pos = self.robot_manager.get_position()  # (num_envs, 3)

# Get orientation
quat = self.robot_manager.get_quaternion()  # (num_envs, 4)
rpy = self.robot_manager.get_euler_angles()  # (num_envs, 3)

# Get projected gravity (useful for observations)
gravity = self.robot_manager.get_projected_gravity()  # (num_envs, 3)
```

### Velocities

```python
# Linear velocity
lin_vel = self.robot_manager.get_linear_velocity()  # World frame
robot_lin_vel = self.robot_manager.get_linear_velocity_robot_frame()  # Robot frame

# Angular velocity
ang_vel = self.robot_manager.get_angular_velocity()  # World frame
robot_ang_vel = self.robot_manager.get_angular_velocity_robot_frame()  # Robot frame
```

### Transformations

```python
# Transform world coordinates to robot frame
world_point = torch.tensor([1.0, 0.0, 0.0], device=gs.device)
robot_point = self.robot_manager.world_to_robot_frame(world_point)

# Transform robot coordinates to world frame
robot_vec = torch.tensor([1.0, 0.0, 0.0], device=gs.device)
world_vec = self.robot_manager.robot_to_world_frame(robot_vec)
```

## Managing Multiple Entities

Handle multiple robots or objects:

```python
class MyEnv(ManagedEnvironment):
    def config(self):
        # Main robot
        self.robot_manager = EntityManager(
            self,
            entity_attr="robot",
            on_reset={
                "position": {
                    "fn": reset.position,
                    "params": {"position": [0, 0, 0.5]},
                },
            },
        )

        # Target object
        self.target_manager = EntityManager(
            self,
            entity_attr="target",
            on_reset={
                "random_position": {
                    "fn": self.randomize_target_position,
                },
            },
        )

        # Obstacles
        self.obstacle_manager = EntityManager(
            self,
            entity_attr="obstacles",
            on_reset={
                "scatter": {
                    "fn": self.scatter_obstacles,
                },
            },
        )
```

## Curriculum-Based Resets

Adjust reset difficulty during training:

```python
class MyEnv(ManagedEnvironment):
    def __init__(self, ...):
        super().__init__(...)
        self.curriculum_level = 0.0

    def adaptive_reset(self, env, entity_manager, envs_idx):
        """Reset with increasing difficulty."""
        entity = getattr(env, entity_manager.entity_attr)

        # Base position
        pos = torch.zeros((len(envs_idx), 3), device=gs.device)
        pos[:, 2] = 0.4

        # Add noise based on curriculum
        if self.curriculum_level > 0.3:
            # Add position noise
            pos[:, :2] += torch.randn_like(pos[:, :2]) * 0.1 * self.curriculum_level

        if self.curriculum_level > 0.6:
            # Add random orientation
            rpy = torch.randn((len(envs_idx), 3), device=gs.device)
            rpy *= 0.2 * self.curriculum_level
            quats = euler_to_quat_batch(rpy)
            entity.set_quat(quats, envs_idx)

        entity.set_pos(pos, envs_idx)

    def config(self):
        self.robot_manager = EntityManager(
            self,
            entity_attr="robot",
            on_reset={"adaptive": {"fn": self.adaptive_reset}},
        )
```

## Common Configurations

### Quadruped Robot

```python
EntityManager(
    self,
    entity_attr="robot",
    on_reset={
        "position": {
            "fn": reset.position,
            "params": {
                "position": [0.0, 0.0, 0.35],  # Standing height
                "quat": [1.0, 0.0, 0.0, 0.0],  # Upright
                "position_noise": 0.02,  # Small position variation
                "rotation_noise": 0.1,  # Small rotation variation
                "zero_velocity": True,
            },
        },
    },
)
```

### Manipulator

```python
EntityManager(
    self,
    entity_attr="arm",
    on_reset={
        "home_position": {
            "fn": reset.position,
            "params": {
                "position": [0.0, 0.0, 0.8],  # Base position
                "quat": [1.0, 0.0, 0.0, 0.0],
                "zero_velocity": True,
            },
        },
        "joint_positions": {
            "fn": reset.joint_positions,
            "params": {
                "positions": [0, -1.57, 1.57, -1.57, -1.57, 0],  # Home pose
                "noise": 0.1,
            },
        },
    },
)
```

### Flying Robot

```python
EntityManager(
    self,
    entity_attr="drone",
    on_reset={
        "hover": {
            "fn": reset.position,
            "params": {
                "position": [0.0, 0.0, 2.0],  # Hover height
                "quat": [1.0, 0.0, 0.0, 0.0],
                "position_noise": 0.5,  # Larger spawn area
                "zero_velocity": False,  # Keep some velocity
            },
        },
        "initial_velocity": {
            "fn": reset.velocity,
            "params": {
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
                "velocity_noise": 0.2,  # Small initial velocities
            },
        },
    },
)
```

## Best Practices

### 1. Always Zero Velocities

Unless specifically needed, zero velocities on reset:

```python
"position": {
    "fn": reset.position,
    "params": {"zero_velocity": True},
}
```

### 2. Add Small Randomization

Even small randomization improves robustness:

```python
"position": {
    "fn": reset.position,
    "params": {
        "position_noise": 0.02,  # 2cm variation
        "rotation_noise": 0.05,  # ~3 degree variation
    },
}
```

### 3. Use Entity Manager Methods

Prefer EntityManager methods over direct entity access:

```python
# Good: Use manager methods
vel = self.robot_manager.get_linear_velocity()

# Less preferred: Direct access
vel = self.robot.get_vel()
```

### 4. Chain Reset Operations

Order matters when chaining resets:

```python
on_reset={
    "1_position": {...},  # First: set position
    "2_velocity": {...},  # Second: set velocity
    "3_forces": {...},    # Third: apply forces
}
```

### 5. Test Reset Consistency

Ensure resets are deterministic when needed:

```python
# For reproducible resets
torch.manual_seed(42)
self.robot_manager.reset([0])

# For random resets
self.robot_manager.reset([0])
```

## Debugging

### Verify Reset Positions

```python
def reset(self, envs_idx):
    super().reset(envs_idx)

    # Check reset worked
    pos = self.robot_manager.get_position()[envs_idx]
    print(f"Reset positions: {pos}")

    # Verify velocities are zero
    vel = self.robot_manager.get_linear_velocity()[envs_idx]
    assert torch.allclose(vel, torch.zeros_like(vel), atol=1e-4)
```

### Visualize Reset Distribution

```python
# Collect reset positions over many episodes
reset_positions = []
for _ in range(100):
    self.reset()
    reset_positions.append(self.robot_manager.get_position())

positions = torch.stack(reset_positions)
print(f"Position mean: {positions.mean(dim=0)}")
print(f"Position std: {positions.std(dim=0)}")
```
