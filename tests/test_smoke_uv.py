"""Exercise geometry/UV indexing and the CUDA UV rasterization path."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2.demos.uv import smoke_mesh
from stage2.demos.uv import render_checkerboard


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by nvdiffrast")
class SmokeUVTest(unittest.TestCase):
    def test_independent_uv_indices_and_interpolation(self) -> None:
        V, F, U, Phi = smoke_mesh()
        self.assertEqual((V.shape[0], U.shape[0]), (4, 6))
        self.assertTrue(torch.equal(F, torch.tensor([[0, 1, 2], [0, 2, 3]], device="cuda")))
        self.assertTrue(torch.equal(Phi, torch.tensor([[0, 1, 2], [3, 4, 5]], device="cuda")))
        self.assertNotEqual(F[1, 0].item(), Phi[1, 0].item())
        self.assertFalse(torch.equal(U[Phi[0, 0]], U[Phi[1, 0]]))

        result = render_checkerboard(V, F, U, Phi, 128, 128)
        self.assertEqual(set(torch.unique(result.face_index[result.mask]).tolist()), {0, 1})
        for face in (0, 1):
            ys, xs = torch.where(result.face_index == face)
            y, x = ys[len(ys) // 2], xs[len(xs) // 2]
            weights = result.barycentric[y, x]
            self.assertAlmostEqual(weights.sum().item(), 1.0, places=5)
            expected_uv = (weights[:, None] * U[Phi[face].long()]).sum(dim=0)
            torch.testing.assert_close(result.uv[y, x], expected_uv, atol=1e-5, rtol=0)
            texel_xy = torch.floor(expected_uv * 8).long().remainder(8)
            expected_shade = 0.9 if texel_xy.sum().item() % 2 == 0 else 0.15
            torch.testing.assert_close(result.rgb[y, x], torch.full((3,), expected_shade, device="cuda"))
            if face == 1:
                wrong_uv = (weights[:, None] * U[F[face].long()]).sum(dim=0)
                self.assertGreater((result.uv[y, x] - wrong_uv).abs().max().item(), 0.1)

        self.assertGreater(torch.unique(result.rgb[result.mask], dim=0).shape[0], 1)


if __name__ == "__main__":
    unittest.main()
