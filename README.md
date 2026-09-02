<p align="center">
<img src="https://raw.githubusercontent.com/jgillick/genesis-forge/main/docs/media/logo_text.png" width="250" />
</p>

# Genesis Forge

A robotics reinforcement learning framework built on the [Genesis](https://genesis-world.readthedocs.io/en/latest/) physics simulator, plus the runtime that replays a trained policy on the robot itself.

👋 **[Quick Start](https://docs.genesisforge.io/en/latest/guide/quick_start.html)** &nbsp;·&nbsp; 📖 **[Documentation](https://genesis-forge.readthedocs.io/en/latest/guide/index.html)** &nbsp;·&nbsp; 🤖 **[Examples](./examples)** &nbsp;·&nbsp; 🚀 **[Deployment guide](https://genesis-forge.readthedocs.io/en/latest/guide/deployment/)**

## Packages

This repository publishes two packages, released together at the same version.

| Package                                                       | Install on            | What it does                                                                                                |
| ------------------------------------------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------- |
| **[genesis-forge](./packages/genesis-forge)**                 | your training machine | The framework: managers, MDP functions, environments, and the deployment exporter. Needs Genesis and torch. |
| **[genesis-forge-runtime](./packages/genesis-forge-runtime)** | the robot             | Replays the trained observation and action pipelines with **numpy only** — no simulator, no torch.          |

```bash
pip install genesis-forge            # to train
pip install genesis-forge-runtime    # on the robot
```

See **[packages/genesis-forge/README.md](./packages/genesis-forge/README.md)** for the framework's features and quickstart.

## Working on this repo

It is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/); both packages are developed together.

```bash
uv sync           # install both packages and dev tools
uv run pytest     # the full suite, both packages
make lint         # ruff across everything
make docs         # build the documentation site
make build        # build both distributions into dist/
```

Each example under [`examples/`](./examples) is its own project with its own lockfile, so they pin
their dependencies independently of the workspace.

## License

Apache-2.0. See [LICENSE](./LICENSE).
