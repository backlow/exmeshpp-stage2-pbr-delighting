"""Load an OBJ mesh with separate geometry and face-corner UV indices."""

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image


@dataclass
class MeshUVCarrier:
    """Geometry and UV tables with independent face indices and an RGB texture.

    Positions [Nv, 3], UVs [Nu, 2], and texture [H, W, 3] are float32.
    Geometry and UV face indices are int32 [Nf, 3], on the same device.
    """

    vertices: torch.Tensor
    faces: torch.Tensor
    uv_coords: torch.Tensor
    uv_indices: torch.Tensor
    rgb_texture: torch.Tensor


def _positive_obj_index(value: str, kind: str, line_number: int) -> int:
    try:
        index = int(value)
    except ValueError as exc:
        raise ValueError(f"OBJ line {line_number}: invalid {kind} index {value!r}") from exc
    if index <= 0:
        raise ValueError(f"OBJ line {line_number}: {kind} index must be positive, got {index}")
    return index - 1


def load_obj_with_texture(
    obj_path: str | Path,
    texture_path: str | Path,
    device: str | torch.device = "cpu",
) -> MeshUVCarrier:
    """Load positive-indexed triangular OBJ faces and an explicit RGB texture.

    Accept v/vt and v/vt/vn corners; ignore normals and material directives.
    Relative (negative) indices and empty meshes are unsupported. Read the
    first three vertex and first two UV components, without flipping UVs or
    image rows. Discard image alpha and normalize RGB bytes to [0, 1].
    """
    vertices: list[list[float]] = []
    uv_coords: list[list[float]] = []
    faces: list[list[int]] = []
    uv_indices: list[list[int]] = []

    with Path(obj_path).open("r", encoding="utf-8") as obj_file:
        for line_number, line in enumerate(obj_file, start=1):
            parts = line.split("#", 1)[0].split()
            if not parts:
                continue

            if parts[0] == "v":
                if len(parts) < 4:
                    raise ValueError(f"OBJ line {line_number}: vertex needs three coordinates")
                vertices.append([float(value) for value in parts[1:4]])

            elif parts[0] == "vt":
                if len(parts) < 3:
                    raise ValueError(f"OBJ line {line_number}: UV needs two coordinates")
                uv_coords.append([float(value) for value in parts[1:3]])

            elif parts[0] == "f":
                if len(parts) != 4:
                    raise ValueError(f"OBJ line {line_number}: only triangular faces are supported")
                face: list[int] = []
                face_uv: list[int] = []
                for corner in parts[1:]:
                    fields = corner.split("/")
                    if len(fields) not in (2, 3) or not fields[1]:
                        raise ValueError(f"OBJ line {line_number}: every face corner needs a UV index")
                    face.append(_positive_obj_index(fields[0], "geometry", line_number))
                    face_uv.append(_positive_obj_index(fields[1], "UV", line_number))
                faces.append(face)
                uv_indices.append(face_uv)

    if not vertices or not uv_coords or not faces:
        raise ValueError("OBJ must contain vertices, UV coordinates, and triangular faces")
    if max(max(face) for face in faces) >= len(vertices):
        raise ValueError("OBJ face references a geometry index outside the vertex list")
    if max(max(face_uv) for face_uv in uv_indices) >= len(uv_coords):
        raise ValueError("OBJ face references a UV index outside the UV list")

    with Image.open(texture_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        pixels = bytearray(rgb.tobytes())
    rgb_texture = torch.frombuffer(pixels, dtype=torch.uint8).reshape(height, width, 3)

    return MeshUVCarrier(
        vertices=torch.tensor(vertices, dtype=torch.float32, device=device),
        faces=torch.tensor(faces, dtype=torch.int32, device=device),
        uv_coords=torch.tensor(uv_coords, dtype=torch.float32, device=device),
        uv_indices=torch.tensor(uv_indices, dtype=torch.int32, device=device),
        rgb_texture=rgb_texture.to(device=device, dtype=torch.float32).div_(255),
    )
