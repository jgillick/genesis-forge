# Meshes

Trimesh builders for drawing custom debug visuals with `scene.draw_debug_mesh`.

Each builder returns a `trimesh.Trimesh` built at the origin: arrows along the +Z axis,
arcs in the XY plane starting from the +X axis. The pose helpers build the 4x4 transform
that places a mesh in the scene, which is passed to `draw_debug_mesh` as `T`.

```python
from genesis_forge.meshes import arrow_mesh, z_aligned_pose

mesh = arrow_mesh(length=0.2, radius=0.01, color=(0.0, 0.5, 0.0, 1.0))
scene.draw_debug_mesh(mesh, T=z_aligned_pose(pos=(0.0, 0.0, 0.5), direction=(1.0, 0.0, 0.0)))
```

::: genesis_forge.meshes
