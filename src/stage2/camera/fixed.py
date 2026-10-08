"""Deterministic clip transform for single-view experiments."""

import torch

#orthographic view in this stage
def fixed_clip_vertices(vertices: torch.Tensor) -> torch.Tensor:
    """Map world vertices [Nv, 3] to a fixed +Z orthographic view [Nv, 3].

    Center the bounds, fit the longest side to 1.6, and negate Z.
    The renderer supplies homogeneous w=1; dtype and device are preserved."""

    ##aabb
    bounds_min = vertices.amin(dim=0)
    bounds_max = vertices.amax(dim=0)
    extent = (bounds_max - bounds_min).max()

    if extent.item() <= 0:
        raise ValueError("Mesh must have nonzero spatial extent")

    #0.2% margin
    clip_vertices = (vertices - (bounds_min + bounds_max) / 2) * (1.6 / extent)

    # Larger world Z is nearer; the renderer supplies homogeneous w=1.
    clip_vertices[:, 2] = -clip_vertices[:, 2]

    return clip_vertices


