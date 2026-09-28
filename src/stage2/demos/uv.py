"""Smoke-only mesh and checkerboard; production rendering does not import this."""

import torch

from stage2.renderer import UVRender, rasterize_uv


def smoke_mesh(device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    """Return demo V [4, 3], F [2, 3], U [6, 2], Phi [2, 3] on device."""
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


def render_checkerboard(
    V: torch.Tensor,
    F: torch.Tensor,
    U: torch.Tensor,
    Phi: torch.Tensor,
    height: int = 512,
    width: int = 512,
    checks: int = 8,
) -> UVRender:
    """Render a demo checkerboard to RGB [height, width, 3].

    V [Nv, 3], F/Phi [Nf, 3], and U [Nu, 2] are CUDA tensors.
    Use independent UV indices and a repeating nearest-neighbor checkerboard."""
    if checks <= 0:
        raise ValueError("checks must be positive")
    face_index, barycentric, uv, mask = rasterize_uv(V, F, U, Phi, height, width)
    texels = torch.arange(checks, device=V.device)
    texel_y, texel_x = torch.meshgrid(texels, texels, indexing="ij")
    shade = torch.where((texel_x + texel_y).remainder(2) == 0, 0.9, 0.15)
    texture = shade[..., None].expand(-1, -1, 3)
    sample_xy = torch.floor(uv * checks).long().remainder(checks)
    sampled = texture[sample_xy[..., 1], sample_xy[..., 0]]
    rgb = torch.where(mask[..., None], sampled, 0.25)
    return UVRender(face_index, barycentric, uv, rgb, mask)


