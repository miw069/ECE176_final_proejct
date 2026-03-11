import argparse
import os

import numpy as np
import torch
import trimesh
from tqdm import tqdm
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from dataset_pix3d import Pix3DDataset
from model_occupancy import ImplicitOccNet


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    ta = cKDTree(a)
    tb = cKDTree(b)
    da, _ = tb.query(a, k=1)
    db, _ = ta.query(b, k=1)
    return float((da ** 2).mean() + (db ** 2).mean())


@torch.no_grad()
def reconstruct_mesh(model, image_tensor, device, grid_res=64, bounds=0.6, thresh=0.5):
    model.eval()
    img = image_tensor.unsqueeze(0).to(device)
    z = model.encode_image(img)

    lin = np.linspace(-bounds, bounds, grid_res, dtype=np.float32)
    grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), axis=-1)
    pts = torch.from_numpy(grid.reshape(-1, 3)).unsqueeze(0).to(device)

    logits = model.occ_decoder(pts, z)
    occ = torch.sigmoid(logits).reshape(grid_res, grid_res, grid_res).cpu().numpy()

    occ_min, occ_max = float(occ.min()), float(occ.max())

    # Adaptively pick a threshold that is guaranteed to be inside the data range.
    # Falls back to the midpoint of the actual value range if thresh is out of range.
    if occ_min >= thresh or occ_max <= thresh:
        adaptive_thresh = (occ_min + occ_max) / 2.0
        print(f"[WARN] thresh={thresh} outside occ range [{occ_min:.3f}, {occ_max:.3f}]. "
              f"Using adaptive thresh={adaptive_thresh:.3f}")
    else:
        adaptive_thresh = thresh

    try:
        verts, faces, normals, values = marching_cubes(occ, level=adaptive_thresh)
        verts = verts / (grid_res - 1) * (2 * bounds) - bounds
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    except ValueError:
        # If marching cubes still fails, return an empty mesh — chamfer will use random pts
        print("[WARN] marching_cubes failed even with adaptive thresh. Returning empty mesh.")
        mesh = trimesh.Trimesh()

    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pix3d_root", type=str, default="data/pix3d")
    ap.add_argument("--ann_json", type=str, default="pix3d.json")
    ap.add_argument("--category", type=str, default="chair")
    ap.add_argument("--ckpt", type=str, default="outputs/checkpoints/best.pt")
    ap.add_argument("--out_dir", type=str, default="outputs/renders")
    ap.add_argument("--grid_res", type=int, default=64)
    ap.add_argument("--bounds", type=float, default=0.6)
    ap.add_argument("--n_eval", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = Pix3DDataset(
        pix3d_root=args.pix3d_root,
        ann_json=args.ann_json,
        split="test",
        image_size=224,
        n_occ_points=2048,
        n_cad_points=4096,
        use_mask=True,
        categories=[args.category],
        cache_dir="cache_occ",
        use_cache=True,
    )

    ckpt = torch.load(args.ckpt, map_location="cpu")
    z_dim = ckpt["args"]["z_dim"]
    model = ImplicitOccNet(z_dim=z_dim, pretrained_encoder=False, use_cad_encoder=True)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)

    cds = []
    for i in tqdm(range(min(args.n_eval, len(ds))), desc="eval_recon"):
        sample = ds[i]
        img = sample["image"]
        cad_pts = sample["cad_points"].numpy()

        pred_mesh = reconstruct_mesh(model, img, device, grid_res=args.grid_res, bounds=args.bounds, thresh=0.5)

        if pred_mesh.faces.shape[0] > 0:
            pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, 4096)
        else:
            pred_pts = np.random.uniform(-0.5, 0.5, size=(4096, 3))

        cd = chamfer_distance(pred_pts.astype(np.float32), cad_pts.astype(np.float32))
        cds.append(cd)

        pred_mesh.export(os.path.join(args.out_dir, f"pred_{i:03d}.obj"))

    print(f"Chamfer Distance mean over {len(cds)}: {np.mean(cds):.6f}")
    print(f"Meshes saved to: {args.out_dir}")


if __name__ == "__main__":
    main()