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
    #차원수고 3이 아니거나 마지막 원소의 개수가 3이 아닐때,rgb
    if rgb_texture.ndim != 3 or rgb_texture.shape[2] != 3:
        raise ValueError("rgb_texture must have shape [height, width, 3]")
    
    #cuda 위에 있어야 함, float32여야 함 
    if rgb_texture.device != V.device or rgb_texture.dtype != torch.float32:
        raise ValueError("rgb_texture must be float32 on the same CUDA device as V")

    #input data 정보 넣어서 face index, barycentric 좌표, uv, rendered image의 배경 오브젝트 분리
    face_index, barycentric, uv, mask = rasterize_uv(V, F, U, Phi, height, width)

    #원본 uv를 바꾸지 않음 
    sample_uv = uv.clone()  

    #pillow로 읽었을 때 이미지 시작점이 왼쪽 원점에서 오른쪽 위쪽으로 될 수 있음 
    if flip_v:
        sample_uv[..., 1] = 1.0 - sample_uv[..., 1]

    #UV 좌표를 이용해서 texture에서 색을 읽어오는 함수
    #각 pixel의 UV 위치에 해당하는 texture RGB
    #앞에 none을 붙여서 [h,w,3]을 [1,h,w,3]으로 만듬 
    #여러 texture/image를 동시에 GPU에서 처리할 수 있게 하려는 목적 
    sampled = dr.texture(rgb_texture[None].contiguous(), sample_uv[None].contiguous(),filter_mode="linear", boundary_mode="clamp",)[0]

    #object가 있는 영역에 대해서만 
    rgb = torch.where(mask[..., None], sampled, 0.25)

    return UVRender(face_index, barycentric, uv, rgb, mask)
