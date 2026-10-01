# Meshes

Trimesh builders for drawing custom debug visuals with `scene.draw_debug_mesh`.

Each builder returns a `trimesh.Trimesh` already placed in the scene, so it can be
drawn directly. `arrow_mesh` builds a round arrow, and `flat_arc_arrow_mesh` builds a
thin extruded arc that reads clearly from above.

```python
from genesis_forge.meshes import arrow_mesh

mesh = arrow_mesh(pos=(0.0, 0.0, 0.5), vec=(0.2, 0.0, 0.0), radius=0.01, color=(0.0, 0.5, 0.0, 1.0))
scene.draw_debug_mesh(mesh)
```

::: genesis_forge.meshes
