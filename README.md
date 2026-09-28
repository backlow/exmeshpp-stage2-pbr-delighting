# exmeshpp-stage2

OBJ + RGB texture loading, independently indexed UV rendering, and fixed-view
texture-only Adam optimization. Stage II PBR optimization is not implemented.

## Code map

Graphics primitives live in `src/stage2`; experiment flow lives in `scripts`.
Scripts do not import one another, and production modules do not import demos.

```text
scripts/
  inspect_asset.py                 # Load and print tensor metadata
  smoke_uv.py                      # Demo orchestration only
  render_real_asset.py             # Load -> fixed view -> render -> save/log
  optimize_single_view_texture.py  # Explicit GT, loss, backward, Adam loop
src/stage2/
  io/mesh_uv.py                    # MeshUVCarrier and OBJ + texture loader
  camera/fixed.py                  # Center/scale mesh and build +Z clip view
  renderer/rasterization.py        # Rasterization, barycentrics, UV interpolation
  renderer/uv.py                   # UVRender and linear/clamp texture rendering
  image/png.py                    # PNG export and iteration previews
  optimization/texture.py          # Gray parameter initialization, gradient checks
  demos/uv.py                      # Smoke quad and checkerboard, demos/tests only
```

Previously, `smoke_uv.py` owned PNG writing and the quad fixture;
`render_real_asset.py` owned `test_clip_vertices`; optimization imported both
scripts. Rasterization, texture rendering, and checkerboard rendering shared
`renderer/uv.py`. The camera helper is now `stage2.camera.fixed_clip_vertices`;
the checkerboard is now `stage2.demos.uv.render_checkerboard`.

## Reading the optimization experiment

Start at `scripts/optimize_single_view_texture.py:main`:

1. `stage2.io.load_obj_with_texture` loads the asset.
2. `stage2.camera.fixed_clip_vertices` builds the fixed view.
3. `stage2.renderer.render_texture` builds GT under `torch.no_grad()`.
4. `stage2.optimization.create_learnable_texture` initializes gray RGB texels.
5. The script loops through render, full-image L1, `loss.backward()`, gradient
   checks, `optimizer.step()`, and texture clamping to `[0, 1]`.
6. `stage2.image` saves images; the script writes the loss CSV and logs.

The local `render` function only binds the fixed geometry, resolution, and UV
convention. Adam, the loss expression, iteration order, and logging remain
visible in the script. There is no trainer class or generic experiment framework.

Tensor notation: `V [Nv, 3]` is clip-space geometry; `F [Nf, 3]` indexes its
triangles; `U [Nu, 2]` stores UV coordinates; `Phi [Nf, 3]` independently indexes
each triangle's UV corners. Texture is `[Ht, Wt, 3]`; rendered RGB is `[H, W, 3]`.
The loader returns world-space vertices, which the fixed camera transforms.

## Preserved conventions

- Fixed camera: bounds center, longest side fitted to 1.6, negated Z, `w=1`.
- Renderer: independent UV indices, linear filtering, clamp boundary, background
  RGB 0.25, optional `1-v` sampling, bottom-up framebuffer rows.
- Optimization: texture only, gray 0.5 initialization, full-image L1, Adam,
  post-update clamping, and pre-update loss logging plus a final evaluation.
- Final experiment PNGs flip framebuffer rows and truncate RGB to bytes.
  Iteration previews retain torchvision rounding and their original row order.
- All optimization outputs live under `--output-dir` (default: `outputs`).
  Preview directories `iter_renders` and `iter_textures` are created automatically.
  Every two updates, the clamped texture is rendered again under `no_grad`;
  both previews show that post-update state and use the completed update count
  in their filenames, such as `render_0002.png` and `texture_0002.png`.

## Run and test

Use the existing `exmesh` environment with CUDA, PyTorch, nvdiffrast, Pillow,
and torchvision. Run from the repository root:

```powershell
conda activate exmesh
python -m unittest discover -s tests -v
python scripts/smoke_uv.py
python scripts/render_real_asset.py
python scripts/optimize_single_view_texture.py
```

The tests cover loading, the fixed camera, image row/quantization conventions,
independent UV interpolation, texture gradients, and loss reduction. CUDA tests
are skipped if CUDA is unavailable.
