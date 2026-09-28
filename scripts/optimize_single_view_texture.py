"""Texture-only Adam smoke test against a synthetic, fixed-view RGB target."""

import argparse
import csv
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stage2.io import load_obj_with_texture
from stage2.renderer import render_texture
from render_real_asset import test_clip_vertices
from smoke_uv import write_png
from torchvision.utils import save_image

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, default=Path("data/sample/xatlas_version.obj"))
    parser.add_argument("--texture", type=Path, default=Path("data/sample/xatlas_version_texture.png"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument(
        "--flip-v", action=argparse.BooleanOptionalAction, default=True,
        help="sample at 1-v for bottom-origin OBJ UVs and top-down PNG rows",
    )

    args = parser.parse_args()
    if args.size <= 0 or args.iterations <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("size, iterations, and learning rate must be positive and finite")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this nvdiffrast optimization test")

    mesh = load_obj_with_texture(args.obj, args.texture, device="cuda")
    V = test_clip_vertices(mesh.vertices)
    F, U, Phi = mesh.faces, mesh.uv_coords, mesh.uv_indices
    for name, tensor in (("V", V), ("F", F), ("U", U), ("Phi", Phi)):
        print(f"{name}.requires_grad: {tensor.requires_grad}")
        assert not tensor.requires_grad

    def render(texture: torch.Tensor):
        return render_texture(V, F, U, Phi, texture, args.size, args.size, flip_v=args.flip_v)

    # Same full-resolution texture shape, deliberately different gray initialization.
    texture = torch.nn.Parameter(torch.full_like(mesh.rgb_texture, 0.5))
    optimizer = torch.optim.Adam([texture], lr=args.lr) # texture 하나만 adam에게 넘김
    print(f"Learnable texture.requires_grad: {texture.requires_grad}")
    print(f"Texture V flipped: {args.flip_v}")

#get GT img
    with torch.no_grad():
        gt_render = render(mesh.rgb_texture)
        if not gt_render.mask.any().item():
            raise RuntimeError("The fixed view has no visible mesh pixels")
        
        gt = gt_render.rgb
        initial = render(texture).rgb

        #getloss
        initial_loss = (initial - gt).abs().mean().item()

        # Convert bottom-up framebuffer rows to top-down PNG rows on export only.
        write_png(args.output_dir / "single_view_gt.png", gt.flip(0))
        write_png(args.output_dir / "single_view_initial.png", initial.flip(0))
    print(f"Initial L1 loss (full image): {initial_loss:.9f}")

    losses = []
    first_grad_norm = None

#optimization
    for step in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)

        #pred
        prediction = render(texture).rgb

        #loss function gt, pred
        loss = (prediction - gt).abs().mean()

        #gradient 계산
        loss.backward()

        if texture.grad is None:
            raise RuntimeError("Texture gradient is missing")
        
        grad_norm = texture.grad.norm().item()

        if not math.isfinite(grad_norm) or not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite loss or texture gradient")

        #grad normalize가 아무것도 초기화 안되어 있으면 grad_norm으로
        if first_grad_norm is None:
            first_grad_norm = grad_norm
            if grad_norm == 0:
                raise RuntimeError("Initial texture gradient is zero")
            
        losses.append((step, loss.item()))  # Loss before this step's update.

        #optimizing
        optimizer.step()

        with torch.no_grad():
            texture.clamp_(0, 1)

        if(step+1)%2==0:
            save_image(prediction.detach().permute(2, 0, 1),f"outputs/iter_renders/render_{step:04d}.png")
            save_image(texture.detach().permute(2, 0, 1),f"outputs/iter_textures/texture_{step:04d}.png")

        #25iteration
        if (step + 1) % 25 == 0:
            print(f"Step {step + 1}/{args.iterations}: pre-update loss={loss.item():.9f}", flush=True)
            

    # Evaluate after the final update, rather than reporting the last pre-update loss.
    with torch.no_grad():
        optimized = render(texture).rgb
        final_loss = (optimized - gt).abs().mean().item()
        write_png(args.output_dir / "single_view_optimized.png", optimized.flip(0))

    losses.append((args.iterations, final_loss))

    with (args.output_dir / "single_view_loss.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("completed_updates", "l1_loss"))
        writer.writerows(losses)

    print(f"Final L1 loss: {final_loss:.9f}")
    print(f"texture.grad is not None: {texture.grad is not None}")
    print(f"Texture gradient L2 norm: first={first_grad_norm:.9g}, last backward={grad_norm:.9g}")
    print(f"Optimization iterations: {args.iterations}; optimizer parameters: texture only")
    print(f"Saved GT, initial, optimized PNGs and loss CSV in {args.output_dir}")
    if not final_loss < initial_loss:
        raise RuntimeError("Texture optimization did not reduce the loss")


if __name__ == "__main__":
    main()
