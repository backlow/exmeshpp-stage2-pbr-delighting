"""Check texture gradients through the existing rasterize/interpolate/sample path."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from smoke_uv import smoke_mesh
from stage2.renderer import render_texture


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by nvdiffrast")
class TextureOptimizationTest(unittest.TestCase):
    def test_texture_backward_and_loss_reduction(self) -> None:
        V, F, U, Phi = smoke_mesh()
        fixed = [value.clone() for value in (V, F, U, Phi)]
        target_texture = torch.linspace(0.1, 0.9, 8 * 8 * 3, device="cuda").reshape(8, 8, 3)

        def render(texture):
            return render_texture(V, F, U, Phi, texture, 32, 32, flip_v=True).rgb


        with torch.no_grad():
            target = render(target_texture)

        texture = torch.nn.Parameter(torch.full_like(target_texture, 0.5))
        optimizer = torch.optim.Adam([texture], lr=0.03)
        initial_loss = (render(texture) - target).abs().mean().item()

        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            loss = (render(texture) - target).abs().mean()
            loss.backward()
            self.assertIsNotNone(texture.grad)
            self.assertTrue(torch.isfinite(texture.grad).all().item())
            self.assertGreater(texture.grad.norm().item(), 0)
            optimizer.step()

        with torch.no_grad():
            final_loss = (render(texture) - target).abs().mean().item()

        self.assertLess(final_loss, initial_loss)
        self.assertTrue(texture.requires_grad)
        
        for value, original in zip((V, F, U, Phi), fixed):
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad)
            self.assertTrue(torch.equal(value, original))


if __name__ == "__main__":
    unittest.main()
