"""Render a two-triangle quad with a separate UV index for every corner."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stage2.demos.uv import render_checkerboard, smoke_mesh
from stage2.image import write_png


def main() -> None:
    """Render the demo quad and save its checkerboard image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke_uv.png"))
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    V, F, U, Phi = smoke_mesh()
    result = render_checkerboard(V, F, U, Phi, args.size, args.size)
    write_png(args.output, result.rgb)
    print(f"Saved {args.output} ({args.size}x{args.size}); faces={torch.unique(result.face_index[result.mask]).tolist()}")


if __name__ == "__main__":
    main()
