"""STM defect segmentation data: load scans, tile into patches, build the two protocols.

Each .npy scan is one source image. Masks are binary and stored at half resolution.
Patches (128x128, stride 64) are the "variants"; the parent scan is the "source".
"""
from __future__ import annotations

from config import STM_ROOT

import glob
import os
from pathlib import Path

import numpy as np

RAW = STM_ROOT
PATCH = 64
STRIDE = 32


def loads():
    """Return list of dicts: {name, substrate, img (float32 HxW), mask (uint8 HxW)}."""
    out = []
    for d in sorted(RAW.iterdir()):
        if not d.is_dir():
            continue
        masks = {}
        for mf in glob.glob(str(d / "masks" / "*.npy")):
            base = os.path.basename(mf)[:-4]
            for suf in ("_mask_full", "_mask"):
                if base.endswith(suf):
                    masks[base[: -len(suf)]] = mf
                    break
        for f in sorted(glob.glob(str(d / "images" / "*.npy"))):
            stem = os.path.basename(f)[:-4]
            mf = masks.get(stem)
            if mf is None:
                continue
            img = np.load(f).astype(np.float32)
            m = np.load(mf)
            if m.ndim == 3:
                m = m[..., 0]
            m = (m > 0.5).astype(np.uint8)
            if m.shape != img.shape:
                # masks come at half resolution -> nearest-neighbour upsample
                k = img.shape[0] // m.shape[0]
                m = np.repeat(np.repeat(m, k, axis=0), k, axis=1)
                if m.shape != img.shape:
                    from PIL import Image as I
                    m = np.array(I.fromarray(m * 255).resize((img.shape[1], img.shape[0]), I.NEAREST)) > 127
                    m = m.astype(np.uint8)
            # robust per-scan normalisation: 1st/99th percentile
            lo, hi = np.percentile(img, 1), np.percentile(img, 99)
            img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)
            out.append({"name": stem, "substrate": d.name, "img": img, "mask": m})
    return out


def tiles(scan, patch=PATCH, stride=STRIDE):
    h, w = scan["img"].shape
    out = []
    for y in range(0, max(h - patch, 0) + 1, stride):
        for x in range(0, max(w - patch, 0) + 1, stride):
            im = scan["img"][y:y + patch, x:x + patch]
            mk = scan["mask"][y:y + patch, x:x + patch]
            if im.shape != (patch, patch):
                continue
            out.append((im, mk))
    return out


def build(k: int, seed: int = 0, n_train: int = 400, test_per_substrate: int = 1):
    """Fixed test scans; k training scans nested across k; 400 training patches."""
    rng = np.random.default_rng(seed)
    scans = loads()
    by_sub = {}
    for s in scans:
        by_sub.setdefault(s["substrate"], []).append(s)

    test_scans, train_pool = [], []
    for sub, lst in sorted(by_sub.items()):
        idx = rng.permutation(len(lst))
        test_scans += [lst[i] for i in idx[:test_per_substrate]]
        train_pool += [lst[i] for i in idx[test_per_substrate:]]

    order = [train_pool[i] for i in rng.permutation(len(train_pool))]
    train_scans = order[:k]

    t_tiles = [t for s in test_scans for t in tiles(s)]
    ti = rng.permutation(len(t_tiles))[:n_train]
    test_set = [t_tiles[i] for i in ti]

    b_pool = [t for s in train_scans for t in tiles(s)]
    bi = rng.permutation(len(b_pool))[:min(n_train, len(b_pool))]
    B_train = [b_pool[i] for i in bi]

    a_pool = [t for s in train_scans + test_scans for t in tiles(s)]
    ai = rng.permutation(len(a_pool))[:n_train]
    A_train = [a_pool[i] for i in ai]

    return {"k": k, "seed": seed, "A_train": A_train, "B_train": B_train, "test": test_set,
            "test_scans": [s["name"] for s in test_scans],
            "train_scans": [s["name"] for s in train_scans],
            "n_test": len(test_set)}


if __name__ == "__main__":
    scans = loads()
    print(f"scans: {len(scans)}")
    for s in scans:
        n = len(tiles(s))
        print(f"  {s['substrate'][:14]:14s} {s['img'].shape} tiles={n:3d} "
              f"defect={s['mask'].mean()*100:5.2f}%  {s['name'][:52]}")
    for k in (2, 3, 5):
        s = build(k)
        print(f"k={k}: A_train={len(s['A_train'])} B_train={len(s['B_train'])} test={s['n_test']} "
              f"test_scans={s['test_scans']}")

