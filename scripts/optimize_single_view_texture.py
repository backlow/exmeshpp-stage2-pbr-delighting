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
from stage2.camera import fixed_clip_vertices
from stage2.image import save_preview, write_png
from stage2.optimization import create_learnable_texture, texture_gradient_norm


def main() -> None:
    """Run the fixed-view texture experiment and save images and loss history."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, default=Path("data/sample/xatlas_version.obj"))
    parser.add_argument("--texture", type=Path, default=Path("data/sample/xatlas_version_texture.png"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--flip-v", action=argparse.BooleanOptionalAction, default=True,help="sample at 1-v for bottom-origin OBJ UVs and top-down PNG rows",)

    args = parser.parse_args()

    if args.size <= 0 or args.iterations <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("size, iterations, and learning rate must be positive and finite")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this nvdiffrast optimization test")


    render_preview_dir = args.output_dir / "iter_renders"
    texture_preview_dir = args.output_dir / "iter_textures"
    render_preview_dir.mkdir(parents=True, exist_ok=True)
    texture_preview_dir.mkdir(parents=True, exist_ok=True)


    # 1. Load the asset and build the fixed view.
    mesh = load_obj_with_texture(args.obj, args.texture, device="cuda")
    #return  MeshUVCarrier:
    V = fixed_clip_vertices(mesh.vertices)  # [Nv, 3], clip-space xyz
    F = mesh.faces       # [Nf, 3], geometry indices
    U = mesh.uv_coords   # [Nu, 2], UV coordinates
    Phi = mesh.uv_indices  # [Nf, 3], independent UV corner indices

    for name, tensor in (("V", V), ("F", F), ("U", U), ("Phi", Phi)):
        print(f"{name}.requires_grad: {tensor.requires_grad}")
        #assert -> condition이 true 여야 지속됨 
        assert not tensor.requires_grad

    
    def render(texture: torch.Tensor):
        return render_texture(V, F, U, Phi, texture, args.size, args.size, flip_v=args.flip_v)



    # 2. Build a fixed GT image [H, W, 3] from the original texture.
    with torch.no_grad():
        gt_render = render(mesh.rgb_texture)
        if not gt_render.mask.any().item():
            raise RuntimeError("The fixed view has no visible mesh pixels")
        gt = gt_render.rgb



    # 3. Initialize only the texture [Ht, Wt, 3] as a learnable parameter.
    texture = create_learnable_texture(mesh.rgb_texture)
    #texture에 대해서만 update

    #미분 변수 지정 - > texture
    optimizer = torch.optim.Adam([texture], lr=args.lr)
    print(f"Learnable texture.requires_grad: {texture.requires_grad}")
    print(f"Texture V flipped: {args.flip_v}")

    #initial loss
    with torch.no_grad():
        initial = render(texture).rgb

        initial_loss = (initial - gt).abs().mean().item()

        # Convert bottom-up framebuffer rows to top-down PNG rows on export only.
        write_png(args.output_dir / "single_view_gt.png", gt)
        write_png(args.output_dir / "single_view_initial.png", initial)
    print(f"Initial L1 loss (full image): {initial_loss:.9f}")

    losses = []
    first_grad_norm = None



    # 4. Render -> full-image L1 -> backward -> Adam update -> save/log.
    for step in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        # texure.grad 를 0으로 지정함, optimizer 가 texture를 미분변수로 둠

        prediction = render(texture).rgb

        loss = (prediction - gt).abs().mean()

        #
        loss.backward()
        grad_norm = texture_gradient_norm(texture, loss)

        if first_grad_norm is None:
            first_grad_norm = grad_norm
            if grad_norm == 0:
                raise RuntimeError("Initial texture gradient is zero")

        losses.append((step, loss.item()))  # Loss before this step's update.

        #실제 texture tensor 업데이트, gd, torch optim adam
        optimizer.step()

        with torch.no_grad():
            texture.clamp_(0, 1)

        completed_updates = step + 1
        if completed_updates % 2 == 0:
            # Both previews show the same clamped, post-update texture state.
            with torch.no_grad():
                #render view
                preview_prediction = render(texture).rgb
            save_preview(render_preview_dir / f"render_{completed_updates:04d}.png", preview_prediction)
            save_preview(texture_preview_dir / f"texture_{completed_updates:04d}.png", texture)

        if completed_updates % 25 == 0:
            print(f"Step {completed_updates}/{args.iterations}: pre-update loss={loss.item():.9f}", flush=True)



    # 5. Evaluate and save after the final update.
    #optimize 되고 난 결과
    with torch.no_grad():
        optimized = render(texture).rgb
        final_loss = (optimized - gt).abs().mean().item()
        write_png(args.output_dir / "single_view_optimized.png", optimized)

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
