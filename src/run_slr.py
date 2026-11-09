"""Recompute the sibling-leakage statistic on the final k-source design."""
from __future__ import annotations

from config import OUT_ROOT

import json
from pathlib import Path

import numpy as np
from PIL import Image

from neuseg_data import IMGS, extract_boxes, make_curve_splits, variant

OUT = OUT_ROOT
KS = (2, 3, 5, 10)
SEED = 0


def dhash(im: np.ndarray, size: int = 8) -> np.ndarray:
    g = np.array(Image.fromarray(im).convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def pack(h: np.ndarray) -> np.ndarray:
    return np.packbits(h)


def main():
    boxes = extract_boxes()
    cache: dict[tuple[str, int], np.ndarray] = {}

    def h(name: str, vid: int) -> np.ndarray:
        key = (name, vid)
        if key not in cache:
            img = np.array(Image.open(IMGS / name).convert("RGB"))
            b = np.array(boxes[name], dtype=np.float32)
            a, _, _ = variant(img, b, vid)
            cache[key] = pack(dhash(a))
        return cache[key]

    rows = []
    for k in KS:
        s = make_curve_splits(k, SEED)
        test = s["test"]
        src_of_test = {n for n, _ in test}
        out = {"k": k}
        for proto in ("A", "B"):
            train = s[f"{proto}_train"]
            # nearest training neighbour for each test item
            T = np.stack([h(*i) for i in test])
            R = np.stack([h(*i) for i in train])
            d = np.unpackbits(T[:, None, :] ^ R[None, :, :], axis=2).sum(2)
            nn = d.min(1)
            # sibling if the nearest training item shares a source image
            train_src = np.array([n for n, _ in train])
            arg = d.argmin(1)
            sib = np.array([train_src[j] == test[i][0] for i, j in enumerate(arg)])
            out[proto] = {
                "slr_le4": float(((nn <= 4) & sib).mean()),
                "frac_le4": float((nn <= 4).mean()),
                "median_d": float(np.median(nn)),
                "dists": nn.tolist(),
                "sibling_nn": float(sib.mean()),
            }
        rows.append(out)
        print(f"k={k}: A SLR={out['A']['slr_le4']:.3f} median={out['A']['median_d']:.0f} | "
              f"B SLR={out['B']['slr_le4']:.3f} median={out['B']['median_d']:.0f}", flush=True)

    (OUT / "slr_final.json").write_text(json.dumps(rows, indent=1))
    print("saved slr_final.json")


if __name__ == "__main__":
    main()
