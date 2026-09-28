"""Differentiable RGB texture sampling and render result buffers."""

from dataclasses import dataclass

import nvdiffrast.torch as dr
import torch

from .rasterization import rasterize_uv


@dataclass
class UVRender:
    """Bottom-up render buffers; H/W are framebuffer height/width."""

    face_index: torch.Tensor  # [H, W], zero based; -1 for background
    barycentric: torch.Tensor  # [H, W, 3], in F corner order
    uv: torch.Tensor  # [H, W, 2]
    rgb: torch.Tensor  # [H, W, 3], in [0, 1]
    mask: torch.Tensor  # [H, W]


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

    V [Nv, 3], F/Phi [Nf, 3], U [Nu, 2], and rgb_texture [Ht, Wt, 3]
    reside on one CUDA device. Return UVRender with RGB [height, width, 3].
    Set flip_v for bottom-origin OBJ UVs with a top-down Pillow texture.
    The returned UVs retain the original OBJ convention.
    """
    if rgb_texture.ndim != 3 or rgb_texture.shape[2] != 3:
        raise ValueError("rgb_texture must have shape [height, width, 3]")
    if rgb_texture.device != V.device or rgb_texture.dtype != torch.float32:
        raise ValueError("rgb_texture must be float32 on the same CUDA device as V")
    face_index, barycentric, uv, mask = rasterize_uv(V, F, U, Phi, height, width)
    sample_uv = uv.clone()
    if flip_v:
        sample_uv[..., 1] = 1.0 - sample_uv[..., 1]
    sampled = dr.texture(
        rgb_texture[None].contiguous(), sample_uv[None].contiguous(),
        filter_mode="linear", boundary_mode="clamp",
    )[0]
    rgb = torch.where(mask[..., None], sampled, 0.25)
    return UVRender(face_index, barycentric, uv, rgb, mask)
