# Go2 Gait Learning with Periodic Reward Composition

This example implements the method from ["Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition"](https://arxiv.org/pdf/2011.01387) (Siekmann et al., 2020) to teach the Go2 quadruped robot various natural gaits including walking, trotting, galloping, and hopping.

## Overview

The key innovation of this approach is **periodic reward composition** - using phase-dependent rewards that encourage different behaviors during swing and stance phases of each foot. This leads to natural gait emergence without requiring reference trajectories or motion capture data.

### Supported Gaits

- **Stand**: Stable standing posture with minimal movement
- **Walk**: Four-beat gait with each foot contacting separately
- **Trot**: Diagonal pairs of feet move together
- **Gallop**: Front feet move together, then rear feet
- **Hop**: All four feet synchronized, with clear aerial phase

## Method

### Periodic Phases

Each foot cycles through two phases:

- **Swing Phase**: Foot lifts and moves forward
- **Stance Phase**: Foot contacts ground and provides support

### Phase-Specific Rewards

The reward function changes based on the current phase:

**During Swing Phase:**

- Penalize foot contact forces (encourage lifting)
- Reward foot velocity (encourage movement)

**During Stance Phase:**

- Reward firm ground contact
- Maintain stability

**Always Active:**

- Forward progress tracking
- Gait symmetry
- Body stability
- Action smoothness

### Gait Parameters

Each gait is defined by:

- **Frequency**: How fast the gait cycles (Hz)
- **Swing Ratio**: Percentage of cycle in swing phase
- **Foot Offsets**: Phase offset for each foot
- **Target Velocity**: Desired forward/lateral speed

## Installation

Ensure you have genesis-forge installed with the training dependencies:

```bash
pip install genesis-forge[train]
```

## Training

Train a specific gait using the provided script:

```bash
# Train walking gait
python train.py --gait walk --num_envs 4096

# Train trotting with randomized parameters for robustness
python train.py --gait trot --randomize --num_envs 4096

# Train galloping with custom hyperparameters
python train.py --gait gallop --lr 3e-4 --num_steps 32
```

### Training Arguments

- `--gait`: Gait type to train (stand/walk/trot/gallop/hop)
- `--randomize`: Randomize gait parameters for robust learning
- `--num_envs`: Number of parallel environments (default: 4096)
- `--max_iterations`: Training iterations (default: 5000)
- `--lr`: Learning rate (default: 5e-4)
- `--headless`: Run without visualization
- `--experiment_name`: Custom name for the experiment
- `--resume`: Resume from checkpoint path

### Training Tips

1. **Walking** typically trains fastest (2000-3000 iterations)
2. **Trotting** needs moderate training (3000-4000 iterations)
3. **Galloping** requires longer training (4000-5000 iterations)
4. **Hopping** can be challenging - use lower learning rates

Monitor training progress with tensorboard:

```bash
tensorboard --logdir logs/
```

## Evaluation

Evaluate a trained model:

```bash
# Basic evaluation
python eval.py logs/go2-gait-walk-*/final_model.pt --gait walk

# With metrics display
python eval.py logs/go2-gait-trot-*/final_model.pt --gait trot --show_metrics

# Multiple environments
python eval.py logs/go2-gait-gallop-*/final_model.pt --gait gallop --num_envs 4

# Slow motion visualization
python eval.py logs/go2-gait-hop-*/final_model.pt --gait hop --slow_motion 2.0
```

### Evaluation Arguments

- `model_path`: Path to trained model checkpoint
- `--gait`: Gait type (should match training)
- `--num_episodes`: Episodes to evaluate (default: 10)
- `--show_metrics`: Display gait quality metrics
- `--deterministic`: Use deterministic policy
- `--slow_motion`: Slow motion factor for visualization
- `--record`: Record evaluation videos

### Gait Quality Metrics

The evaluation script provides several metrics:

- **Phase Sync**: How well foot contacts align with gait phases
- **Swing/Contact Ratio**: Time spent in each phase
- **Velocity Tracking**: How well the robot matches target speed
- **Gait Symmetry**: Left-right and front-rear coordination

## Understanding the Implementation

### Key Components

1. **`gait_command_manager.py`**: Manages gait parameters and phase calculations
2. **`periodic_rewards.py`**: Implements phase-specific reward functions
3. **`environment.py`**: Main environment with periodic reward composition
4. **`train.py`**: Training script with PPO
5. **`eval.py`**: Evaluation and analysis script

### Gait Parameter Presets

Located in `gait_command_manager.py`:

```python
"walk": {
    "frequency": 2.0,        # 2 Hz gait cycle
    "swing_ratio": 0.5,      # 50% swing, 50% stance
    "fl_offset": 0.0,        # Front left at 0°
    "fr_offset": 0.5,        # Front right at 180°
    "rl_offset": 0.75,       # Rear left at 270°
    "rr_offset": 0.25,       # Rear right at 90°
    "velocity_x": 0.5,       # 0.5 m/s forward
}
```

### Customizing Gaits

You can create custom gaits by modifying the presets in `GaitCommandManager.GAIT_PRESETS`:

```python
"custom_gait": {
    "frequency": 2.5,
    "swing_ratio": 0.4,
    "fl_offset": 0.0,
    "fr_offset": 0.25,
    "rl_offset": 0.5,
    "rr_offset": 0.75,
    "velocity_x": 0.7,
    "velocity_y": 0.0,
}
```

Then train with:

```bash
python train.py --gait custom_gait
```

## Curriculum Learning

For challenging gaits, use curriculum learning by gradually increasing difficulty:

```python
# In environment.py, modify reward weights over time:
def _setup_rewards(self):
    # Start with stability, gradually increase movement rewards
    progress = min(1.0, self.step_count / 1000000)

    reward_cfg["forward_progress"]["weight"] = 0.5 + 1.5 * progress
    reward_cfg["swing_velocity"]["weight"] = 0.2 + 0.8 * progress
```

## Troubleshooting

### Robot Falls Frequently

- Reduce `forward_progress` weight
- Increase `stability` weight
- Lower learning rate

### Gait Looks Unnatural

- Increase `phase_timing` weight
- Ensure proper foot offset values
- Check if `swing_ratio` matches gait type

### Poor Velocity Tracking

- Increase `forward_progress` weight
- Reduce action penalties
- Check if target velocity is realistic

### Training Unstable

- Reduce `num_envs` if out of memory
- Lower learning rate
- Increase `num_mini_batches`

## Advanced Usage

### Multi-Gait Policy

Train a single policy for multiple gaits:

```python
# In environment.py constructor:
self.gait_schedule = ["walk", "trot", "gallop"]
self.gait_change_interval = 1000  # steps

# In step():
if self.step_count % self.gait_change_interval == 0:
    new_gait = random.choice(self.gait_schedule)
    self.gait_command.gait_type = new_gait
```

### Terrain Adaptation

Combine with terrain for robust outdoor gaits:

```python
from genesis_forge.managers import TerrainManager

# In config():
self.terrain_manager = TerrainManager(
    self,
    terrain_type="random",
    difficulty=0.5,
)
```

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{siekmann2021sim,
  title={Sim-to-real learning of all common bipedal gaits via periodic reward composition},
  author={Siekmann, Jonah and Godse, Yesh and Fern, Alan and Hurst, Jonathan},
  journal={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021}
}
```

## References

- [Original Paper](https://arxiv.org/pdf/2011.01387)
- [Genesis Forge Documentation](https://genesis-forge.readthedocs.io)
- [RSL_RL Documentation](https://github.com/leggedrobotics/rsl_rl)
