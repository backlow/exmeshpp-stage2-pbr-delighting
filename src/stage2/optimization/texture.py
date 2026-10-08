"""Texture initialization and backward diagnostics for single-view fitting."""

import math

import torch

#neutral texture (0.5,0.5,0.5)
def create_learnable_texture(reference: torch.Tensor) -> torch.nn.Parameter:
    """Create a gray (0.5) parameter matching reference RGB [Ht, Wt, 3].

    Preserve the reference dtype/device without connecting its gradient graph.
    """
    #tensor가 gradient를 추적해야 하는 변수임을 알려줌 
    return torch.nn.Parameter(torch.full_like(reference, 0.5))

#solve tensor gradient with torch
def texture_gradient_norm(texture: torch.Tensor, loss: torch.Tensor) -> float:
    """Check RGB texture [Ht, Wt, 3] gradients and scalar loss after backward.

    Return the gradient L2 norm; reject missing gradients or non-finite values.
    """
    if texture.grad is None:
        raise RuntimeError("Texture gradient is missing")
    grad_norm = texture.grad.norm().item()
    if not math.isfinite(grad_norm) or not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite loss or texture gradient")
    return grad_norm
