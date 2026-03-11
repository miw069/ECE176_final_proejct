import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import trimesh


@dataclass
class Pix3DSample:
    image_path: str
    mask_path: Optional[str]
    model_path: str
    category: str
    bbox: Optional[Tuple[int, int, int, int]]


def _safe_join(root: str, rel_or_abs: Optional[str]) -> Optional[str]:
    if rel_or_abs is None:
        return None
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.normpath(os.path.join(root, rel_or_abs))


def _find_first(d: Dict, keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _parse_bbox(b):
    if b is None:
        return None
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return int(b[0]), int(b[1]), int(b[2]), int(b[3])
    if isinstance(b, dict):
        xmin = b.get("xmin", None)
        ymin = b.get("ymin", None)
        xmax = b.get("xmax", None)
        ymax = b.get("ymax", None)
        if None not in (xmin, ymin, xmax, ymax):
            return int(xmin), int(ymin), int(xmax), int(ymax)
    return None


def _load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_mask(path: Optional[str]) -> Optional[Image.Image]:
    if path is None or not os.path.exists(path):
        return None
    return Image.open(path).convert("L")


def _apply_bbox_crop(img: Image.Image, mask: Optional[Image.Image], bbox):
    if bbox is None:
        return img, mask
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = max(xmin + 1, xmax)
    ymax = max(ymin + 1, ymax)
    img = img.crop((xmin, ymin, xmax, ymax))
    if mask is not None:
        mask = mask.crop((xmin, ymin, xmax, ymax))
    return img, mask


def _resize(img: Image.Image, mask: Optional[Image.Image], size: int):
    img = img.resize((size, size), resample=Image.BILINEAR)
    if mask is not None:
        mask = mask.resize((size, size), resample=Image.NEAREST)
    return img, mask


def _image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def _normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    if mesh.vertices.shape[0] == 0:
        return mesh
    center = mesh.bounding_box.centroid
    mesh.vertices -= center
    ext = mesh.bounding_box.extents
    scale = float(np.max(ext)) if np.max(ext) > 0 else 1.0
    mesh.vertices /= scale
    return mesh


def _sample_points_for_occupancy(
    mesh: trimesh.Trimesh,
    n_points: int,
    near_surface_ratio: float = 0.7,
    surface_sigma: float = 0.02,
    uniform_padding: float = 0.6,
    fallback_surface_dist_thresh: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns pts (N,3) and occ (N,) in {0,1}.

    Fast/accurate order:
    1) mesh.contains() (true occupancy)  [needs rtree often]
    2) signed_distance() (approx inside/outside)
    3) closest_point distance threshold (weak but non-zero signal)
    """
    n_near = int(n_points * near_surface_ratio)
    n_uni = n_points - n_near

    if n_near > 0 and mesh.faces.shape[0] > 0:
        surf_pts, _ = trimesh.sample.sample_surface(mesh, n_near)
        near_pts = surf_pts + np.random.normal(scale=surface_sigma, size=surf_pts.shape)
    else:
        near_pts = np.random.uniform(-0.5, 0.5, size=(n_near, 3))

    uni_pts = np.random.uniform(-uniform_padding, uniform_padding, size=(n_uni, 3))
    pts = np.concatenate([near_pts, uni_pts], axis=0).astype(np.float32)

    # 1) True occupancy
    try:
        inside = mesh.contains(pts)
        occ = inside.astype(np.float32)
        if occ.max() > 0 and occ.min() < 1:
            return pts, occ
        # If degenerate, keep going
    except Exception:
        pass

    # 2) Signed distance (negative often means inside)
    try:
        sd = trimesh.proximity.signed_distance(mesh, pts)
        occ = (sd < 0).astype(np.float32)
        if occ.max() > 0 and occ.min() < 1:
            return pts, occ
    except Exception:
        pass

    # 3) Closest-point distance threshold (not true occupancy, but prevents all-zeros)
    try:
        _, dist, _ = trimesh.proximity.closest_point(mesh, pts)
        occ = (dist < fallback_surface_dist_thresh).astype(np.float32)
        return pts, occ
    except Exception as e:
        raise RuntimeError(f"Occupancy labeling failed for this mesh: {e}")


def _sample_surface_pointcloud(mesh: trimesh.Trimesh, n_points: int) -> np.ndarray:
    if mesh.faces.shape[0] == 0:
        return np.random.uniform(-0.5, 0.5, size=(n_points, 3)).astype(np.float32)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return pts.astype(np.float32)


def _cache_key(image_path: str, model_path: str, n_occ: int, n_cad: int) -> str:
    s = f"{image_path}|{model_path}|{n_occ}|{n_cad}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()


class Pix3DDataset(Dataset):
    """
    Caching:
      - First time, will compute pts/occ/cad_pts and save .npz under:
        data/pix3d/cache_occ/<hash>.npz
      - Next epochs: loads instantly from cache => training no longer stuck at 0%.
    """

    def __init__(
        self,
        pix3d_root: str,
        ann_json: str,
        split: str = "train",
        image_size: int = 224,
        n_occ_points: int = 8192,
        n_cad_points: int = 2048,
        use_mask: bool = True,
        categories: Optional[List[str]] = None,
        seed: int = 0,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        cache_dir: str = "cache_occ",
        use_cache: bool = True,
    ):
        self.pix3d_root = pix3d_root
        self.image_size = image_size
        self.n_occ_points = n_occ_points
        self.n_cad_points = n_cad_points
        self.use_mask = use_mask

        self.use_cache = use_cache
        self.cache_dir = os.path.join(self.pix3d_root, cache_dir)
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        ann_path = _safe_join(pix3d_root, ann_json)
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        if not isinstance(ann, list):
            raise ValueError("Expected Pix3D annotation JSON to be a list of records.")

        all_samples: List[Pix3DSample] = []
        for r in ann:
            category = str(_find_first(r, ["category", "class", "cls"], default="unknown"))
            if categories is not None and category not in categories:
                continue

            img_rel = _find_first(r, ["img", "image", "image_path", "img_path"])
            model_rel = _find_first(r, ["model", "model_path", "mesh", "mesh_path"])
            if img_rel is None or model_rel is None:
                continue

            mask_rel = None
            if self.use_mask:
                mask_rel = _find_first(r, ["mask", "mask_path", "segmentation", "seg_path"], default=None)

            bbox = _parse_bbox(_find_first(r, ["bbox", "box", "bndbox"], default=None))

            s = Pix3DSample(
                image_path=_safe_join(pix3d_root, img_rel),
                mask_path=_safe_join(pix3d_root, mask_rel) if mask_rel else None,
                model_path=_safe_join(pix3d_root, model_rel),
                category=category,
                bbox=bbox,
            )
            if os.path.exists(s.image_path) and os.path.exists(s.model_path):
                all_samples.append(s)

        if len(all_samples) == 0:
            raise RuntimeError("No valid samples found. Check pix3d_root/ann_json paths and JSON field names.")

        rng = random.Random(seed)
        rng.shuffle(all_samples)

        n = len(all_samples)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])

        if split == "train":
            self.samples = all_samples[:n_train]
        elif split == "val":
            self.samples = all_samples[n_train : n_train + n_val]
        elif split == "test":
            self.samples = all_samples[n_train + n_val :]
        else:
            raise ValueError("split must be train/val/test")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # image (fast)
        img = _load_image_rgb(s.image_path)
        mask = _load_mask(s.mask_path) if self.use_mask else None
        img, mask = _apply_bbox_crop(img, mask, s.bbox)
        img, mask = _resize(img, mask, self.image_size)
        img_t = _image_to_tensor(img)

        # cache key
        key = _cache_key(s.image_path, s.model_path, self.n_occ_points, self.n_cad_points)
        cache_path = os.path.join(self.cache_dir, f"{key}.npz") if self.use_cache else None

        if self.use_cache and os.path.exists(cache_path):
            data = np.load(cache_path)
            pts = data["pts"].astype(np.float32)
            occ = data["occ"].astype(np.float32)
            cad_pts = data["cad_pts"].astype(np.float32)
        else:
            # expensive part (only happens once per sample if caching enabled)
            mesh = trimesh.load(s.model_path, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
            mesh = _normalize_mesh(mesh)

            pts, occ = _sample_points_for_occupancy(mesh, self.n_occ_points)
            cad_pts = _sample_surface_pointcloud(mesh, self.n_cad_points)

            if self.use_cache:
                # Give the temp file a name that still ends in .npz 
                # so numpy doesn't automatically append anything to it
                tmp_path = cache_path.replace(".npz", "_tmp.npz")
                np.savez_compressed(tmp_path, pts=pts, occ=occ, cad_pts=cad_pts)
                os.replace(tmp_path, cache_path)

        return {
            "image": img_t,
            "points": torch.from_numpy(pts),
            "occ": torch.from_numpy(occ),
            "cad_points": torch.from_numpy(cad_pts),
            "meta": {
                "category": s.category,
                "image_path": s.image_path,
                "model_path": s.model_path,
            },
        }