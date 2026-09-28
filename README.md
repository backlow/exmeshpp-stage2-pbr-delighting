# exmeshpp-stage2

Minimal Python / PyTorch / nvdiffrast UV rasterization smoke test and a standalone
OBJ + PNG asset loader. The loader is independent of the renderer; PBR and
optimization are not included.

`V` holds geometry vertices in clip-space xyz, `F` holds triangle geometry
vertex indices, `U` holds UV coordinates, and `Phi` holds triangle corner UV
indices. The smoke quad has 4 geometry vertices and 6 UV coordinates. Its two
triangles share geometry vertices while their UV corners use separate indices.

Run in the existing `exmesh` conda environment (Python 3.11, PyTorch
2.8.0+cu128, nvdiffrast 0.4.0, and working CUDA):

```powershell
conda activate exmesh
python scripts/smoke_uv.py
python -m unittest discover -s tests -v
```

The script writes `outputs/smoke_uv.png`. Use `--size` and `--output` to change
the resolution or destination. The renderer returns face IDs, barycentric
weights, per-pixel UVs, colors, and a foreground mask for inspection.

Load an asset with `stage2.io.load_obj_with_texture(obj_path, texture_path,
device="cpu")`, or inspect the sample from the repository root:

```powershell
python scripts/inspect_asset.py data/sample/xatlas_version.obj data/sample/xatlas_version_texture.png
```

The loader uses PyTorch, Pillow, and the Python standard library. It returns a
`MeshUVCarrier` with separate geometry and UV face indices, converted from
positive OBJ indices to zero-based int32 tensors. Positions and UVs are float32;
the texture is float32 RGB `[H, W, 3]` in `[0, 1]`, with alpha discarded.
All tensors use the requested device (`--device` in the inspection script).

Supported faces are triangles with `v/vt` or `v/vt/vn` corners. Missing UVs,
polygons, relative/negative indices, out-of-range indices, and empty meshes raise
errors. Normals, materials, and other directives are ignored. Vertex entries use
their first three coordinates and UV entries their first two; UV values and
image row order are preserved without flipping.
