"""NEU-Seg loading: masks -> boxes, deterministic dihedral+photometric variants, split protocols.

A "source" is one original 200x200 image. A "variant" is one deterministic
transformation of a source (8 dihedral x 5 photometric = 40).

Protocol A (leaky): split the variant pool at random -> test variants come from
                    sources that are also in train.
Protocol B (grouped): hold out whole sources -> test sources never seen in train.
"""
from __future__ import annotations

from config import NEUSEG_ROOT

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = NEUSEG_ROOT
IMGS = ROOT / "images" / "test"
MASKS = ROOT / "annotations" / "test"
CACHE = Path(__file__).parent / "neuseg_boxes.json"
N_DIHEDRAL = 8
PHOTO = [(1.0, 0.0), (0.85, 0.0), (1.15, 0.0), (1.0, 12.0), (1.0, -12.0),
         (0.90, 6.0), (1.10, -6.0), (1.05, 0.0), (0.95, 0.0), (1.0, 6.0)]


def extract_boxes():
    """Per image: list of [x1, y1, x2, y2, label] from connected components per class."""
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    out = {}
    for p in sorted(IMGS.glob("*.jpg")):
        m = np.array(Image.open(MASKS / (p.stem + ".png")))
        if m.ndim == 3:
            m = m[..., 0]
        boxes = []
        for cls in (1, 2, 3):
            lab, n = ndimage.label(m == cls)
            if n == 0:
                continue
            slices = ndimage.find_objects(lab)
            for sl in slices:
                if sl is None:
                    continue
                y0, y1 = sl[0].start, sl[0].stop
                x0, x1 = sl[1].start, sl[1].stop
                if (y1 - y0) * (x1 - x0) < 25:
                    continue
                boxes.append([float(x0), float(y0), float(x1), float(y1), cls - 1])
        if boxes:
            out[p.name] = boxes
    CACHE.write_text(json.dumps(out))
    return out


def variant(img: np.ndarray, boxes: np.ndarray, vid: int):
    """Deterministic variant vid in [0, 40). Returns (uint8 HxWx3, boxes, labels)."""
    dih, ph = divmod(vid, len(PHOTO))
    flip = bool(dih % 2)
    k = dih // 2  # 0..3
    arr = img
    b = boxes.copy()
    h, w = arr.shape[:2]
    if flip:
        arr = arr[:, ::-1]
        x1 = b[:, 0].copy()
        b[:, 0] = w - b[:, 2]
        b[:, 2] = w - x1
    if k:
        arr = np.rot90(arr, k)
        for _ in range(k):
            x1, y1 = b[:, 0].copy(), b[:, 1].copy()
            x2, y2 = b[:, 2].copy(), b[:, 3].copy()
            b[:, 0], b[:, 2] = y1, y2
            b[:, 1], b[:, 3] = (h - 1) - x2, (h - 1) - x1
            h, w = w, h
    scale, shift = PHOTO[ph]
    a = np.clip(arr.astype(np.float32) * scale + shift, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return a, b, b[:, 4].astype(np.int64)


def make_splits(n_source: int, seed: int = 0, n_train: int | None = None, n_test: int = 120):
    """Return dict with protocol A and B item lists: [(image_name, variant_id), ...].

    n_train fixes the annotation budget so that A and B (and different n_source
    values) are compared at equal training size.
    """
    rng = np.random.default_rng(seed)
    boxes = extract_boxes()
    names = sorted(boxes)
    # group sources by majority label
    src_of_class = {0: [], 1: [], 2: []}
    for n in names:
        b = np.array(boxes[n])
        areas = {}
        for row in b:
            areas[int(row[4])] = areas.get(int(row[4]), 0) + (row[2] - row[0]) * (row[3] - row[1])
        src_of_class[max(areas, key=areas.get)].append(n)

    chosen = {}
    holdout = {}
    for c, lst in src_of_class.items():
        idx = rng.permutation(len(lst))[:n_source]
        chosen[c] = [lst[i] for i in idx]
        holdout[c] = chosen[c][-1]  # last source is unseen in protocol B

    all_items = [(n, v) for c in chosen for n in chosen[c] for v in range(N_DIHEDRAL * len(PHOTO))]
    n_train = n_train or int(len(all_items) * 2 / 3)

    # Protocol A: random train/test split of the whole variant pool (leaky)
    perm = rng.permutation(len(all_items))
    A_train = [all_items[i] for i in perm[:n_train]]
    A_test = [all_items[i] for i in perm[n_train:n_train + n_test]]

    # Protocol B: held-out sources -> test, equal train budget from the rest
    B_test = [(n, v) for c in chosen for n in [holdout[c]] for v in range(N_DIHEDRAL * len(PHOTO))][:n_test]
    pool = [(n, v) for c in chosen for n in chosen[c] if n not in holdout.values()
            for v in range(N_DIHEDRAL * len(PHOTO))]
    p = rng.permutation(len(pool))
    B_train = [pool[i] for i in p[:min(n_train, len(pool))]]

    # Protocol B2: same train sources, but test on the ORIGINAL image only (vid 0),
    # i.e. the realistic deployment sample -- one image per class per held-out source.
    B_orig_test = [(n, 0) for c in chosen for n in [holdout[c]]]

    return {"A_train": A_train, "A_test": A_test, "B_train": B_train, "B_test": B_test,
            "B_orig_test": B_orig_test, "n_source": n_source, "seed": seed,
            "sources": {str(k): v for k, v in chosen.items()}, "holdout": holdout}


def make_curve_splits(k: int, seed: int = 0, n_train: int = 240, n_test: int = 120):
    """Nested design for the source-count curve.

    One source per class is reserved for testing and never used in B's training.
    For each k, B trains on the first k-1 remaining sources per class; A draws its
    training set from those same sources plus the reserved ones. The test set is
    FIXED across k, so the curve isolates source diversity.
    """
    assert k >= 2
    rng = np.random.default_rng(seed)
    boxes = extract_boxes()
    names = sorted(boxes)
    per_class = {0: [], 1: [], 2: []}
    for n in names:
        b = np.array(boxes[n])
        areas = {}
        for row in b:
            areas[int(row[4])] = areas.get(int(row[4]), 0) + (row[2] - row[0]) * (row[3] - row[1])
        per_class[max(areas, key=areas.get)].append(n)

    order = {}
    for c, lst in per_class.items():
        p = rng.permutation(len(lst))
        order[c] = [lst[i] for i in p]
    reserved = {c: order[c][0] for c in order}
    train_src = {c: order[c][1:k] for c in order}

    V = N_DIHEDRAL * len(PHOTO)
    test_pool = [(n, v) for c in order for n in [reserved[c]] for v in range(V)]
    tp = rng.permutation(len(test_pool))
    test = [test_pool[i] for i in tp[:n_test]]

    a_pool = [(n, v) for c in order for n in order[c][:k] for v in range(V)]
    ap = rng.permutation(len(a_pool))
    A_train = [a_pool[i] for i in ap[:n_train]]

    b_pool = [(n, v) for c in order for n in train_src[c] for v in range(V)]
    bp = rng.permutation(len(b_pool))
    B_train = [b_pool[i] for i in bp[:n_train]]

    # class coverage of the fixed test set
    cov = {c: sum(1 for n, _ in test for b in boxes[n] if int(b[4]) == c) for c in order}
    return {"A_train": A_train, "B_train": B_train, "test": test,
            "k": k, "seed": seed, "reserved": reserved,
            "train_sources": {str(c): train_src[c] for c in train_src},
            "test_class_coverage": cov, "n_test": len(test)}


class VariantDataset:
    """Regenerates a deterministic variant on every __getitem__."""

    def __init__(self, items, boxes):
        self.items = items
        self.boxes = boxes
        self._cache = {}

    def __len__(self):
        return len(self.items)

    def _src(self, name):
        if name not in self._cache:
            self._cache[name] = np.array(Image.open(IMGS / name).convert("RGB"))
        return self._cache[name]

    def get(self, i):
        name, vid = self.items[i]
        img = self._src(name)
        b = np.array(self.boxes[name], dtype=np.float32)
        a, bb, lab = variant(img, b, vid)
        return a, bb, lab

    def __getitem__(self, i):
        import torch
        a, bb, lab = self.get(i)
        t = torch.from_numpy(a).permute(2, 0, 1).float() / 255.0
        order = np.argsort(lab)
        bb = bb[order]
        lab = lab[order]
        target = {"boxes": torch.from_numpy(bb[:, :4].copy()).float(),
                  "labels": torch.from_numpy(lab.copy() + 1).long()}  # 1..3 (0 = background)
        return t, target
