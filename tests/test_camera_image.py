"""Check the fixed view and PNG export conventions independently of CUDA."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stage2.camera import fixed_clip_vertices
from stage2.image import save_preview, write_png


class CameraImageTest(unittest.TestCase):
    def test_fixed_view_centers_scales_and_reverses_depth(self) -> None:
        vertices = torch.tensor([[2., 4., 6.], [6., 6., 8.]])
        original = vertices.clone()
        clip = fixed_clip_vertices(vertices)
        torch.testing.assert_close(clip, torch.tensor([[-.8, -.4, .4], [.8, .4, -.4]]))
        self.assertTrue(torch.equal(vertices, original))

    def test_zero_extent_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonzero spatial extent"):
            fixed_clip_vertices(torch.ones(3, 3))

    def test_png_quantization_and_row_order(self) -> None:
        rgb = torch.tensor([[[-1., .5, 2.]], [[1., 0., .25]]], requires_grad=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "image.png"
            write_png(path, rgb)
            with Image.open(path) as image:
                self.assertEqual(image.size, (1, 2))
                self.assertEqual([image.getpixel((0, y)) for y in range(2)], [(0, 127, 255), (255, 0, 63)])
            save_preview(path, rgb)
            with Image.open(path) as image:
                self.assertEqual([image.getpixel((0, y)) for y in range(2)], [(0, 128, 255), (255, 0, 64)])


if __name__ == "__main__":
    unittest.main()
