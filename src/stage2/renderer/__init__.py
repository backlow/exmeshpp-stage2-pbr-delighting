"""Rasterization, UV interpolation, and differentiable texture rendering."""

from .rasterization import rasterize_uv
from .uv import UVRender, render_texture

__all__ = ["UVRender", "rasterize_uv", "render_texture"]
