# Mesh completion

Geometry-based mesh completion: fill holes in a partial mesh using a full template.

## complete.py

Fills holes in a defective/partial mesh by aligning it to a template (ICP), closing holes with PyMeshLab, identifying the filled region, transferring the corresponding geometry from the template, and merging.

**Usage:**

```bash
python Mesh_completion/complete.py -template path/to/template.ply -input path/to/partial.ply -output path/to/completed.ply
```

Requires: `trimesh`, `pymeshlab`, `scipy`. Install with `pip install trimesh pymeshlab scipy`.
