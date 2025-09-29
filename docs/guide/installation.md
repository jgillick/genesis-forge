# ⚙️ Installation

This guide will help you install Genesis Forge and its dependencies.

## Prerequisites

Before installing Genesis Forge, ensure you have:

- Python >=3.10,<3.14
- pip package manager

(Optional) CUDA-compatible GPU for faster training

## Installing Genesis Forge

### From PyPI

The easiest way to install Genesis Forge is via pip:

```shell
pip install genesis-forge
```

Currently, due to some recent [bug fixes](https://github.com/Genesis-Embodied-AI/Genesis/issues/1727) with Genesis Simulator, it's recommended to also install Genesis Simulator from source (this is in addition to installing `genesis-force`).

```shell
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip install .
```

### From Source

To install the latest development version from source:

```shell
git clone https://github.com/yourusername/genesis-forge.git
cd genesis-forge
pip install -e .
```
