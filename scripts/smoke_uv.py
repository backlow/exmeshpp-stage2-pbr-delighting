"""Render a two-triangle quad with a separate UV index for every corner."""

import argparse
import struct
import sys
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stage2.renderer import render_checkerboard


def smoke_mesh(device: str = "cuda") -> tuple[torch.Tensor, ...]:

    V = torch.tensor(
        [[-0.8, -0.8, 0], [0.8, -0.8, 0], [0.8, 0.8, 0], [-0.8, 0.8, 0]],
        dtype=torch.float32, device=device,
    )

    F = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device=device)

    U = torch.tensor(
        [[0, 0], [1, 0], [1, 1], [0.125, 0], [1.125, 1], [0.125, 1]],
        dtype=torch.float32, device=device,
    )

    Phi = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=device)
    
    return V, F, U, Phi


def write_png(path: Path, rgb: torch.Tensor) -> None:
    """Write an RGB8 PNG using only Python's standard library."""
    pixels = (rgb.clamp(0, 1) * 255).byte().cpu().contiguous()
    height, width, channels = pixels.shape
    if channels != 3:
        raise ValueError("Expected RGB image")

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(
            ">I", zlib.crc32(kind + data) & 0xFFFFFFFF
        )

    raw = b"".join(b"\x00" + pixels[row].numpy().tobytes() for row in range(height))
    encoded = b"\x89PNG\r\n\x1a\n"
    encoded += chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0))
    encoded += chunk(b"IDAT", zlib.compress(raw))
    encoded += chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> None:
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
