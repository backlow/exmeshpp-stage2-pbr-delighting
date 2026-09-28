"""Render one OBJ + PNG with a deterministic orthographic test transform."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stage2.io import load_obj_with_texture
from stage2.renderer import render_texture
from smoke_uv import write_png


def test_clip_vertices(vertices: torch.Tensor) -> torch.Tensor:
    """Fixed +Z orthographic view; center and fit the longest side to 1.6."""
    bounds_min = vertices.amin(dim=0)
    bounds_max = vertices.amax(dim=0)
    extent = (bounds_max - bounds_min).max()
    if extent.item() <= 0:
        raise ValueError("Mesh must have nonzero spatial extent")
    clip_vertices = (vertices - (bounds_min + bounds_max) / 2) * (1.6 / extent)
    # Larger world Z is nearer; the renderer supplies homogeneous w=1.
    clip_vertices[:, 2] = -clip_vertices[:, 2]
    return clip_vertices


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, default=Path("data/sample/xatlas_version.obj"))
    parser.add_argument("--texture", type=Path, default=Path("data/sample/xatlas_version_texture.png"))
    parser.add_argument("--output", type=Path, default=Path("outputs/real_asset_render.png"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--flip-v", action=argparse.BooleanOptionalAction, default=True,
        help="sample at 1-v for bottom-origin OBJ UVs and top-down PNG rows (default: enabled)",
    )
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this nvdiffrast connectivity test")

    # The existing loader places all five MeshUVCarrier tensors on CUDA.
    mesh = load_obj_with_texture(args.obj, args.texture, device="cuda")
    bounds_min = mesh.vertices.amin(dim=0)
    bounds_max = mesh.vertices.amax(dim=0)
    center = (bounds_min + bounds_max) / 2
    extent = (bounds_max - bounds_min).max()
    clip_vertices = test_clip_vertices(mesh.vertices)
    result = render_texture(
        clip_vertices, mesh.faces, mesh.uv_coords, mesh.uv_indices,
        mesh.rgb_texture, args.size, args.size, flip_v=args.flip_v,
    )
    visible_pixels = result.mask.sum().item()
    if visible_pixels == 0:
        raise RuntimeError("The test transform produced no visible mesh pixels")

    # Framebuffer conversion is independent of texture V: bottom-up -> PNG top-down.
    write_png(args.output, result.rgb.flip(0))
    print(f"Rasterized resolution: {args.size} x {args.size}")
    print(f"Visible pixel count: {visible_pixels}")
    print(f"Mesh bounds before normalization: min={bounds_min.tolist()}, max={bounds_max.tolist()}")
    print(f"UV min/max: {mesh.uv_coords.min().item()} / {mesh.uv_coords.max().item()}")
    print(f"Texture V flipped: {args.flip_v}")
    print(f"Transform: center={center.tolist()}, uniform scale={(1.6 / extent).item()}, view from +Z")
    print(f"Saved {args.output} (framebuffer rows flipped for PNG)")


if __name__ == "__main__":
    main()
