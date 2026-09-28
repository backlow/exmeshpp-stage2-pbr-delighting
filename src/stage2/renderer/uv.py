"""Minimal CUDA rasterization and independently indexed UV interpolation."""

from dataclasses import dataclass

import nvdiffrast.torch as dr
import torch


@dataclass
class UVRender:
    face_index: torch.Tensor  # [H, W], zero based; -1 for background
    barycentric: torch.Tensor  # [H, W, 3], in F corner order
    uv: torch.Tensor  # [H, W, 2]
    rgb: torch.Tensor  # [H, W, 3], in [0, 1]
    mask: torch.Tensor  # [H, W]


def _rasterize_uv(
    V: torch.Tensor,
    F: torch.Tensor,
    U: torch.Tensor,
    Phi: torch.Tensor,
    height: int = 512,
    width: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render clip-space V with geometry faces F and independent UV faces Phi.

    V is [vertex, xyz], F and Phi are [face, corner], and U is [uv, xy].
    All tensors reside on the same CUDA device. Phi may address more UVs than
    there are geometry vertices, including distinct UVs at a shared vertex.
    """
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("V must have shape [vertex, 3]")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("F must have shape [face, 3]")
    if U.ndim != 2 or U.shape[1] != 2:
        raise ValueError("U must have shape [uv, 2]")
    if Phi.shape != F.shape:
        raise ValueError("Phi must have the same [face, 3] shape as F")
    if not (V.device == F.device == U.device == Phi.device) or V.device.type != "cuda":
        raise ValueError("V, F, U, and Phi must be on the same CUDA device")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    positions = torch.cat((V.float(), torch.ones_like(V[:, :1])), dim=1)[None]
    context = dr.RasterizeCudaContext(device=V.device)
    rast, _ = dr.rasterize(context, positions, F.int().contiguous(), (height, width))

    face_index = rast[0, :, :, 3].long() - 1
    mask = face_index >= 0
    barycentric = torch.cat(
        (rast[0, :, :, :2], 1.0 - rast[0, :, :, :2].sum(dim=-1, keepdim=True)),
        dim=-1,
    )
    barycentric = torch.where(mask[..., None], barycentric, 0.0)

    # nvdiffrast uses Phi to select UV corners independently from F's vertices.
    uv, _ = dr.interpolate(U.float()[None].contiguous(), rast, Phi.int().contiguous())
    uv = uv[0]
    return face_index, barycentric, uv, mask


def render_checkerboard(
    V: torch.Tensor,
    F: torch.Tensor,
    U: torch.Tensor,
    Phi: torch.Tensor,
    height: int = 512,
    width: int = 512,
    checks: int = 8,
) -> UVRender:
    """Render the smoke checkerboard using independently indexed UVs."""
    if checks <= 0:
        raise ValueError("checks must be positive")
    face_index, barycentric, uv, mask = _rasterize_uv(V, F, U, Phi, height, width)
    texels = torch.arange(checks, device=V.device)
    texel_y, texel_x = torch.meshgrid(texels, texels, indexing="ij")
    shade = torch.where((texel_x + texel_y).remainder(2) == 0, 0.9, 0.15)
    texture = shade[..., None].expand(-1, -1, 3)
    sample_xy = torch.floor(uv * checks).long().remainder(checks)
    sampled = texture[sample_xy[..., 1], sample_xy[..., 0]]
    rgb = torch.where(mask[..., None], sampled, 0.25)
    return UVRender(face_index, barycentric, uv, rgb, mask)


def render_texture(
    V: torch.Tensor,
    F: torch.Tensor,
    U: torch.Tensor,
    Phi: torch.Tensor,
    rgb_texture: torch.Tensor,
    height: int = 512,
    width: int = 512,
    *,
    flip_v: bool = False,
) -> UVRender:
    """Sample RGB with linear filtering; return nvdiffrast's bottom-up image.

    Set flip_v for bottom-origin OBJ UVs with a top-down Pillow texture.
    The returned UVs retain the original OBJ convention.
    """
    if rgb_texture.ndim != 3 or rgb_texture.shape[2] != 3:
        raise ValueError("rgb_texture must have shape [height, width, 3]")
    if rgb_texture.device != V.device or rgb_texture.dtype != torch.float32:
        raise ValueError("rgb_texture must be float32 on the same CUDA device as V")
    face_index, barycentric, uv, mask = _rasterize_uv(V, F, U, Phi, height, width)
    sample_uv = uv.clone()
    if flip_v:
        sample_uv[..., 1] = 1.0 - sample_uv[..., 1]
    sampled = dr.texture(
        rgb_texture[None].contiguous(), sample_uv[None].contiguous(),
        filter_mode="linear", boundary_mode="clamp",
    )[0]
    rgb = torch.where(mask[..., None], sampled, 0.25)
    return UVRender(face_index, barycentric, uv, rgb, mask)
