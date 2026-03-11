import argparse
import os
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_pix3d import Pix3DDataset
from model_occupancy import ImplicitOccNet


def collate_fn(batch):
    out: Dict = {}
    out["image"] = torch.stack([b["image"] for b in batch], dim=0).float()
    out["points"] = torch.stack([b["points"] for b in batch], dim=0).float()
    out["occ"] = torch.stack([b["occ"] for b in batch], dim=0).float()
    out["cad_points"] = torch.stack([b["cad_points"] for b in batch], dim=0).float()
    out["meta"] = [b["meta"] for b in batch]
    return out


def _make_loss_fn(pos_weight: Optional[float] = None, device: str = "cpu"):
    if pos_weight is None:
        return torch.nn.BCEWithLogitsLoss()
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))


def _print_occ_stats_once(occ: torch.Tensor, tag: str):
    occ_mean = float(occ.mean().item())
    occ_min = float(occ.min().item())
    occ_max = float(occ.max().item())
    print(f"[SANITY] {tag}: occ mean={occ_mean:.4f} min={occ_min:.1f} max={occ_max:.1f}")
    if occ_max == 0.0:
        print("[WARN] occ is all zeros. Check rtree install: pip install rtree")
    if occ_min == 1.0:
        print("[WARN] occ is all ones. Sampling bounds may be wrong.")


def warmup_cache(dataset, desc: str = "building cache"):
    """
    Iterate every sample once (batch_size=1, num_workers=0) so all .npz
    cache files are written to disk before training begins.  Without this,
    the first real epoch blocks silently on mesh processing and tqdm shows 0%.
    """
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,   # must be 0 so file writes are visible immediately
        collate_fn=collate_fn,
    )
    for _ in tqdm(loader, desc=desc, leave=True):
        pass


def train_one_epoch(model, loader, opt, device, loss_fn, sanity: bool = False):
    model.train()
    total = 0.0
    count = 0

    for batch_i, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        img = batch["image"].to(device)
        pts = batch["points"].to(device)
        occ = batch["occ"].to(device)

        if sanity and batch_i == 0:
            _print_occ_stats_once(occ, "train batch 0")

        opt.zero_grad(set_to_none=True)

        # Forward pass for occupancy
        logits, z_img = model(img, pts)

        # Forward pass for CAD retrieval
        z_cad = model.encode_cad(batch["cad_points"].to(device))

        # 1. Occupancy loss
        loss_occ = loss_fn(logits, occ)

        # 2. Cosine similarity loss (image <-> CAD embedding alignment)
        z_img_norm = torch.nn.functional.normalize(z_img, dim=-1)
        z_cad_norm = torch.nn.functional.normalize(z_cad, dim=-1)
        loss_sim = 1.0 - (z_img_norm * z_cad_norm).sum(dim=-1).mean()

        loss = loss_occ + loss_sim
        loss.backward()
        opt.step()

        total += float(loss.item())
        count += 1

    return total / max(count, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, device, loss_fn, sanity: bool = False):
    model.eval()
    total = 0.0
    count = 0

    for batch_i, batch in enumerate(tqdm(loader, desc="val", leave=False)):
        img = batch["image"].to(device)
        pts = batch["points"].to(device)
        occ = batch["occ"].to(device)

        if sanity and batch_i == 0:
            _print_occ_stats_once(occ, "val batch 0")

        logits, z_img = model(img, pts)
        z_cad = model.encode_cad(batch["cad_points"].to(device))

        # 1. Occupancy loss
        loss_occ = loss_fn(logits, occ)

        # 2. Cosine similarity loss
        z_img_norm = torch.nn.functional.normalize(z_img, dim=-1)
        z_cad_norm = torch.nn.functional.normalize(z_cad, dim=-1)
        loss_sim = 1.0 - (z_img_norm * z_cad_norm).sum(dim=-1).mean()

        loss = loss_occ + loss_sim

        total += float(loss.item())
        count += 1

    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pix3d_root", type=str, default="data/pix3d")
    ap.add_argument("--ann_json", type=str, default="pix3d.json")
    ap.add_argument("--category", type=str, default="chair")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--occ_points", type=int, default=8192)
    ap.add_argument("--cad_points", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--z_dim", type=int, default=256)
    ap.add_argument("--out_dir", type=str, default="outputs/checkpoints")
    ap.add_argument("--pos_weight", type=float, default=None)

    # caching controls
    ap.add_argument("--cache_dir", type=str, default="cache_occ")
    ap.add_argument("--no_cache", action="store_true")

    # set to 2-4 on Linux/Mac; keep 0 on Windows to avoid multiprocessing issues
    ap.add_argument("--num_workers", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    train_ds = Pix3DDataset(
        pix3d_root=args.pix3d_root,
        ann_json=args.ann_json,
        split="train",
        image_size=args.image_size,
        n_occ_points=args.occ_points,
        n_cad_points=args.cad_points,
        use_mask=True,
        categories=[args.category],
        cache_dir=args.cache_dir,
        use_cache=(not args.no_cache),
    )
    val_ds = Pix3DDataset(
        pix3d_root=args.pix3d_root,
        ann_json=args.ann_json,
        split="val",
        image_size=args.image_size,
        n_occ_points=args.occ_points,
        n_cad_points=args.cad_points,
        use_mask=True,
        categories=[args.category],
        cache_dir=args.cache_dir,
        use_cache=(not args.no_cache),
    )

    print(f"[INFO] train size: {len(train_ds)}")
    print(f"[INFO]   val size: {len(val_ds)}")

    # ------------------------------------------------------------------ #
    # Cache warmup — build all .npz files BEFORE training starts so that  #
    # tqdm progress bars during actual epochs are not stuck at 0%.         #
    # This is a no-op for samples already cached.                          #
    # ------------------------------------------------------------------ #
    if not args.no_cache:
        cache_path = os.path.join(args.pix3d_root, args.cache_dir)
        print(f"[INFO] cache dir: {cache_path}")
        print("[INFO] Warming up cache (one-time per sample, skips already-cached)...")
        warmup_cache(train_ds, desc="cache train")
        warmup_cache(val_ds,   desc="cache val  ")
        print("[INFO] Cache ready. Starting training.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    model = ImplicitOccNet(z_dim=args.z_dim, pretrained_encoder=True, use_cad_encoder=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = _make_loss_fn(args.pos_weight, device=device)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, device, loss_fn, sanity=(epoch == 1))
        va_loss = eval_one_epoch(model, val_loader, device, loss_fn, sanity=(epoch == 1))

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt"))

        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))

        print(f"Epoch {epoch:03d} | train {tr_loss:.4f} | val {va_loss:.4f} | best {best_val:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()