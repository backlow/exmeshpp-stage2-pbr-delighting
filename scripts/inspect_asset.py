"""Inspect tensor shapes and index ranges of an OBJ plus PNG texture."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stage2.io import load_obj_with_texture


def main() -> None:
    """Load an asset and print its tensor shapes and index ranges."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("obj_path", type=Path)
    parser.add_argument("texture_path", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    mesh = load_obj_with_texture(args.obj_path, args.texture_path, device=args.device)
    for name in ("vertices", "faces", "uv_coords", "uv_indices", "rgb_texture"):
        tensor = getattr(mesh, name)
        print(f"{name}: shape={list(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")
    for label, tensor in (
        ("UV", mesh.uv_coords),
        ("geometry face index", mesh.faces),
        ("UV face index", mesh.uv_indices),
    ):
        print(f"{label} min/max: {tensor.min().item()} / {tensor.max().item()}")


if __name__ == "__main__":
    main()
