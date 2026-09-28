"""Small, independent OBJ and PNG fixtures for the asset loader."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stage2.io import load_obj_with_texture


class MeshUVLoaderTest(unittest.TestCase):
    def test_separate_indices_and_rgb_texture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            obj_path = Path(directory) / "mesh.obj"
            texture_path = Path(directory) / "texture.png"
            obj_path.write_text(
                "v 0 0 0\n"
                "v 1 0 0\n"
                "v 1 1 0\n"
                "v 0 1 0\n"
                "vt 0 0\n"
                "vt 1 0\n"
                "vt 1 1\n"
                "vt 0 1\n"
                "vt 0.5 0.5\n"
                "vn 0 0 1\n"
                "usemtl ignored_material\n"
                "f 1/3 2/4 3/5\n"
                "f 1/1/1 3/2/1 4/3/1 # normals are ignored\n",
                encoding="utf-8",
            )
            image = Image.new("RGBA", (2, 1))
            image.putdata([(255, 0, 0, 0), (0, 128, 255, 64)])
            image.save(texture_path)

            mesh = load_obj_with_texture(obj_path, texture_path)

        torch.testing.assert_close(
            mesh.vertices,
            torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            mesh.uv_coords,
            torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], dtype=torch.float32),
        )
        self.assertTrue(torch.equal(mesh.faces, torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)))
        self.assertTrue(torch.equal(mesh.uv_indices, torch.tensor([[2, 3, 4], [0, 1, 2]], dtype=torch.int32)))
        self.assertFalse(torch.equal(mesh.faces, mesh.uv_indices))
        self.assertEqual(mesh.rgb_texture.shape, (1, 2, 3))
        self.assertEqual(mesh.rgb_texture.dtype, torch.float32)
        self.assertEqual(mesh.faces.dtype, torch.int32)
        self.assertEqual(mesh.uv_indices.dtype, torch.int32)
        for tensor in (mesh.vertices, mesh.faces, mesh.uv_coords, mesh.uv_indices, mesh.rgb_texture):
            self.assertEqual(tensor.device, torch.device("cpu"))
        torch.testing.assert_close(
            mesh.rgb_texture,
            torch.tensor([[[1, 0, 0], [0, 128 / 255, 1]]], dtype=torch.float32),
        )
        self.assertGreaterEqual(mesh.rgb_texture.min().item(), 0)
        self.assertLessEqual(mesh.rgb_texture.max().item(), 1)

    def test_non_triangle_face_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            obj_path = Path(directory) / "quad.obj"
            texture_path = Path(directory) / "texture.png"
            obj_path.write_text("f 1/1 2/2 3/3 4/4\n", encoding="utf-8")
            Image.new("RGB", (1, 1)).save(texture_path)
            with self.assertRaisesRegex(ValueError, "only triangular faces"):
                load_obj_with_texture(obj_path, texture_path)

    def test_face_without_uv_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            obj_path = Path(directory) / "no_uv.obj"
            texture_path = Path(directory) / "texture.png"
            Image.new("RGB", (1, 1)).save(texture_path)
            for face in ("f 1 2 3\n", "f 1//1 2//1 3//1\n"):
                with self.subTest(face=face):
                    obj_path.write_text(face, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "every face corner needs a UV index"):
                        load_obj_with_texture(obj_path, texture_path)


if __name__ == "__main__":
    unittest.main()
