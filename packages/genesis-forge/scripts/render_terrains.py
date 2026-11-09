"""
Render a picture of every built-in subterrain type, for the terrain guide.

Each subterrain is generated with its default settings (except a few in PICTURE_PARAMS,
whose defaults are too subtle to see) and saved as
``docs/media/terrains/<ClassName>.png``. Run it again whenever a subterrain
type is added or its defaults change:

    uv run python packages/genesis-forge/scripts/render_terrains.py
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import genesis as gs
from PIL import Image

from genesis_forge import terrain

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "docs" / "media" / "terrains"

SUBTERRAIN_SIZE = 8.0  # meters per side
SPACING = 40.0  # meters between subterrains, so each picture shows only one

# Params that differ from a subterrain's defaults, so its shape shows up in the picture.
# The guide's table notes each one.
PICTURE_PARAMS = {
    "PyramidSloped": {"slope": 0.3},
    "Stairs": {"step_height": 0.2, "step_width": 0.5 },
    "RandomGrid": { "grid_width": 0.4, "grid_height": 0.15 },
}


def built_in_subterrains() -> list[type[terrain.Subterrain]]:
    """Every subterrain class in genesis_forge.terrain that works with its defaults."""
    return [
        cls
        for _, cls in inspect.getmembers(terrain, inspect.isclass)
        # HeightField has no defaults: it needs your own heights
        if issubclass(cls, terrain.Subterrain)
        and cls not in (terrain.Subterrain, terrain.HeightField)
    ]


def main():
    """Build one scene with every built-in subterrain, and save a picture of each."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cpu", action="store_true", help="Render on the CPU")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=420)
    args = parser.parse_args()

    gs.init(logging_level="warning", backend=gs.cpu if args.cpu else gs.gpu)
    scene = gs.Scene(show_viewer=False)
    # Lay the subterrains out in a row, far enough apart to photograph one at a time
    classes = built_in_subterrains()
    centers = []
    for i, cls in enumerate(classes):
        x_min = i * SPACING
        terrain.TerrainLayout(
            scene,
            grid=[[cls(**PICTURE_PARAMS.get(cls.__name__, {}))]],
            subterrain_size=(SUBTERRAIN_SIZE, SUBTERRAIN_SIZE),
            horizontal_scale=0.1,
            pos=(x_min, 0.0, 0.0),
            seed=2
        )
        centers.append((x_min + SUBTERRAIN_SIZE / 2, SUBTERRAIN_SIZE / 2))

    camera = scene.add_camera(res=(args.width, args.height), fov=50, GUI=False)
    scene.build()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for cls, (x, y) in zip(classes, centers, strict=True):
        # Look down at the subterrain from one corner
        camera.set_pose(
            pos=(x - 6.5, y - 6.5, 5.5), lookat=(x, y, -0.6), up=(0.0, 0.0, 1.0)
        )
        rgb, *_ = camera.render(rgb=True)
        path = OUTPUT_DIR / f"{cls.__name__}.png"
        Image.fromarray(rgb).save(path)
        print(f"{cls.__name__:<20} {cls.type:<28} -> {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
