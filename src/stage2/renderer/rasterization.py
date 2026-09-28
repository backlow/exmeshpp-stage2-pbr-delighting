"""CUDA rasterization, barycentric coordinates, and independent UV interpolation."""

import nvdiffrast.torch as dr
import torch


def rasterize_uv(
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
    Return face_index [H, W], barycentric [H, W, 3], uv [H, W, 2],
    and mask [H, W], with bottom-up framebuffer rows.
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


