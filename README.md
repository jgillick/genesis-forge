<p align="center">
<img src="https://raw.githubusercontent.com/jgillick/genesis-forge/main/docs/media/logo_text.png" width="250" />
</p>

# Genesis Forge

A powerful and easy to use robotics reinforcement learning framework built on the [Genesis](https://genesis-world.readthedocs.io/en/latest/) physics simulator, plus the runtime that connects a trained policy to a real robot.

Genesis Forge is the fastest path from simulation to robot!

👋 **[Quick Start](https://docs.genesisforge.io/en/latest/guide/quick_start.html)** &nbsp;·&nbsp; 📖 **[Documentation](https://genesis-forge.readthedocs.io/en/latest/guide/index.html)** &nbsp;·&nbsp; 🤖 **[Examples](https://github.com/jgillick/genesis-forge/tree/main/examples)** &nbsp;·&nbsp; 🚀 **[Deployment guide](https://genesis-forge.readthedocs.io/en/latest/guide/deployment.html)**

<center>
<div>
<img src="https://raw.githubusercontent.com/jgillick/genesis-forge/main/docs/media/cmd_locomotion.gif" alt="Massively parallel locomotion training" width="48%" />
<img src="https://raw.githubusercontent.com/jgillick/genesis-forge/main/docs/media/gamepad.gif" alt="Gamepad controller interface" width="48%" />
<img src="https://raw.githubusercontent.com/jgillick/genesis-forge/main/docs/media/terrain.gif" alt="Rough terrain" width="33%" />
<img src="https://raw.githubusercontent.com/jgillick/genesis-forge/main/docs/media/spider.gif" alt="Complex robots" width="33%" />
<img src="https://raw.githubusercontent.com/jgillick/genesis-forge/main/docs/media/cmd_pose2d.gif" alt="Obstacle avoidance" width="33%" />
</div>
</center>

## Packages

This repository publishes two packages, released together at the same version.

| Package                                                       | Install on            | What it does                                                                                                |
| ------------------------------------------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------- |
| **[genesis-forge](./packages/genesis-forge)**                 | your training machine | The framework: managers, MDP functions, environments, and the deployment exporter. Needs Genesis and torch. |
| **[genesis-forge-runtime](./packages/genesis-forge-runtime)** | the robot             | Replays the trained observation and action pipelines with **numpy only** — no simulator, no torch.          |

```bash
pip install genesis-forge            # to train
pip install genesis-forge-runtime    # deploy on the robot
```

See **[packages/genesis-forge/README.md](./packages/genesis-forge/README.md)** for the framework's features and quickstart.

## License

Apache-2.0. See [LICENSE](./LICENSE).
