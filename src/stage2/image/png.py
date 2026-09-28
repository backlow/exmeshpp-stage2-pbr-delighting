"""RGB image export with explicit row and quantization conventions."""

from pathlib import Path
import struct
import zlib

import torch


def write_png(path: Path, rgb: torch.Tensor) -> None:
    """Write float RGB [H, W, 3] as RGB8, truncating after clamping to [0, 1].

    Rows are saved as supplied; callers explicitly flip framebuffer rows.
    Create the parent directory if needed."""
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


def save_preview(path: Path, rgb: torch.Tensor) -> None:
    """Save RGB [H, W, 3] with torchvision's rounding and unchanged rows.

    Match the existing iteration previews, including requiring the parent
    directory to exist. Detach here so image export never retains gradients.
    """
    from torchvision.utils import save_image

    save_image(rgb.detach().permute(2, 0, 1), str(path))
