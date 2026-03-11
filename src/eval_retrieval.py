import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_pix3d import Pix3DDataset
from model_occupancy import ImplicitOccNet


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0).float(),
        "cad_points": torch.stack([b["cad_points"] for b in batch], dim=0).float(),
        "meta": [b["meta"] for b in batch],
    }


@torch.no_grad()
def compute_embeddings(model, loader, device):
    model.eval()
    img_embs = []
    cad_embs = []

    for batch in tqdm(loader, desc="embed", leave=False):
        img = batch["image"].to(device)
        cad = batch["cad_points"].to(device)

        z_img = torch.nn.functional.normalize(model.encode_image(img), dim=-1)
        z_cad = torch.nn.functional.normalize(model.encode_cad(cad), dim=-1)

        img_embs.append(z_img.cpu())
        cad_embs.append(z_cad.cpu())

    return torch.cat(img_embs, dim=0).numpy(), torch.cat(cad_embs, dim=0).numpy()


def recall_at_k(ranks, k):
    ranks = np.asarray(ranks)
    return float((ranks < k).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pix3d_root", type=str, default="data/pix3d")
    ap.add_argument("--ann_json", type=str, default="pix3d.json")
    ap.add_argument("--category", type=str, default="chair")
    ap.add_argument("--ckpt", type=str, default="outputs/checkpoints/best.pt")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = Pix3DDataset(
        pix3d_root=args.pix3d_root,
        ann_json=args.ann_json,
        split="test",
        image_size=224,
        n_occ_points=2048,
        n_cad_points=2048,
        use_mask=True,
        categories=[args.category],
        cache_dir="cache_occ",
        use_cache=True,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    z_dim = ckpt["args"]["z_dim"]
    model = ImplicitOccNet(z_dim=z_dim, pretrained_encoder=False, use_cad_encoder=True)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)

    img_embs, cad_embs = compute_embeddings(model, loader, device)
    sims = img_embs @ cad_embs.T

    ranks = []
    for i in range(sims.shape[0]):
        order = np.argsort(-sims[i])
        ranks.append(int(np.where(order == i)[0][0]))

    print("Retrieval (paired image->CAD):")
    print(f"  Recall@1   = {recall_at_k(ranks, 1):.3f}")
    print(f"  Recall@5   = {recall_at_k(ranks, 5):.3f}")
    print(f"  Recall@10  = {recall_at_k(ranks, 10):.3f}")
    print(f"  Median rank = {float(np.median(ranks)):.1f}")


if __name__ == "__main__":
    main()