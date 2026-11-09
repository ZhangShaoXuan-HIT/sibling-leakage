"""Source-count curve: fixed test set, nested training sources, k in {2,3,5,10}."""
from __future__ import annotations

from config import OUT_ROOT

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from neuseg_data import VariantDataset, extract_boxes, make_curve_splits
from models import build_model, evaluate, predict, train

OUT = OUT_ROOT


def one(k, proto, s, boxes, epochs, seed, dev):
    torch.manual_seed(seed)
    tr = VariantDataset(s[f"{proto}_train"], boxes)
    te = VariantDataset(s["test"], boxes)
    print(f"[k={k} {proto}] train={len(tr)} test={len(te)}", flush=True)
    model = build_model(dev)
    t0 = time.time()
    hist = train(model, tr, epochs, dev)
    preds = predict(model, te, dev)
    res = evaluate(preds)
    res.update({"k": k, "protocol": proto, "seed": seed, "n_train": len(tr), "n_test": len(te),
                "epochs": epochs, "train_minutes": round((time.time() - t0) / 60, 1),
                "loss_final": round(hist[-1], 3)})
    print(f"[k={k} {proto}] mAP@0.5={res['mAP@0.5']:.4f} mAP@0.5:0.95={res['mAP@0.5:0.95']:.4f} "
          f"({res['train_minutes']} min)", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="2,3,5,10")
    ap.add_argument("--protocols", default="A,B")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="curve")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boxes = extract_boxes()
    results = []
    out_file = OUT / f"results_{a.tag}.json"
    for k in [int(x) for x in a.ks.split(",")]:
        s = make_curve_splits(k, a.seed)
        print(f"k={k} coverage={s['test_class_coverage']}", flush=True)
        for proto in a.protocols.split(","):
            r = one(k, proto.strip(), s, boxes, a.epochs, a.seed, dev)
            r["test_class_coverage"] = s["test_class_coverage"]
            results.append(r)
            out_file.write_text(json.dumps(results, indent=2))
    print("CURVE DONE", flush=True)
