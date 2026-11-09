"""Train Faster R-CNN under two split protocols and report mAP.

Protocol A: random split of the variant pool (near-duplicate test set).
Protocol B: grouped split, whole source images held out.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

from neuseg_data import VariantDataset, extract_boxes, make_splits

OUT = Path(".")
NUM_CLASSES = 4  # background + 3 defect classes


def build_model(device):
    m = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", weights_backbone=None,
                                   min_size=256, max_size=256, box_detections_per_img=100)
    in_feat = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES)
    return m.to(device)


def collate(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


def train(model, ds, epochs, device, bs=4, lr=0.005, log_every=10):
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate, num_workers=0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(epochs * .6), int(epochs * .85)], gamma=0.1)
    model.train()
    hist = []
    for ep in range(epochs):
        t0 = time.time()
        run = 0.0
        n = 0
        for i, (imgs, targets) in enumerate(dl):
            imgs = [x.to(device) for x in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(imgs, targets).values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            run += loss.item()
            n += 1
            if i % log_every == 0:
                print(f"    ep{ep} it{i}/{len(dl)} loss={loss.item():.3f}", flush=True)
        sched.step()
        hist.append(run / max(n, 1))
        print(f"  epoch {ep+1}/{epochs} loss={hist[-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return hist


@torch.no_grad()
def predict(model, ds, device, bs=4):
    dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate, num_workers=0)
    model.eval()
    out = []
    for imgs, targets in dl:
        imgs = [x.to(device) for x in imgs]
        preds = model(imgs)
        for p, t in zip(preds, targets):
            out.append({"boxes": p["boxes"].cpu(), "scores": p["scores"].cpu(),
                        "labels": p["labels"].cpu(),
                        "gt_boxes": t["boxes"], "gt_labels": t["labels"]})
    return out


def ap_at_iou(preds, iou_thr):
    aps = {}
    for cls in range(1, NUM_CLASSES):
        scores, tps, n_gt = [], [], 0
        for r in preds:
            gb = r["gt_boxes"][r["gt_labels"] == cls]
            pb = r["boxes"][r["labels"] == cls]
            ps = r["scores"][r["labels"] == cls]
            n_gt += len(gb)
            if len(pb) == 0:
                continue
            order = torch.argsort(-ps)
            pb, ps = pb[order], ps[order]
            matched = torch.zeros(len(gb), dtype=torch.bool)
            ious = box_iou(pb, gb) if len(gb) else torch.zeros(len(pb), 0)
            for j in range(len(pb)):
                tp = 0
                if len(gb):
                    row = ious[j].clone()
                    row[matched] = -1
                    best, idx = row.max(0)
                    if best >= iou_thr:
                        matched[idx] = True
                        tp = 1
                scores.append(ps[j].item())
                tps.append(tp)
        if n_gt == 0:
            continue
        if not scores:
            aps[cls] = 0.0
            continue
        order = np.argsort(-np.array(scores))
        tp = np.array(tps)[order]
        ctp = np.cumsum(tp)
        cfp = np.cumsum(1 - tp)
        rec = ctp / n_gt
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        # all-point interpolation
        mrec = np.concatenate([[0.0], rec, [1.0]])
        mpre = np.concatenate([[0.0], prec, [0.0]])
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        aps[cls] = float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())
    return aps


def evaluate(preds):
    out = {}
    for thr in (0.5, 0.75):
        aps = ap_at_iou(preds, thr)
        out[f"mAP@{thr}"] = float(np.mean(list(aps.values()))) if aps else 0.0
        out[f"AP@{thr}"] = {int(k): round(v, 4) for k, v in aps.items()}
    aps_all = [ap_at_iou(preds, t) for t in np.arange(0.5, 1.0, 0.05)]
    keys = set().union(*[set(a) for a in aps_all]) if aps_all else set()
    per = {k: float(np.mean([a.get(k, 0.0) for a in aps_all])) for k in keys}
    out["mAP@0.5:0.95"] = float(np.mean([v for v in per.values()])) if per else 0.0
    out["AP@0.5:0.95"] = {int(k): round(v, 4) for k, v in per.items()}
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def run(protocol, n_source, epochs, seed, device, n_train=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    boxes = extract_boxes()
    s = make_splits(n_source, seed, n_train=n_train)
    tr = VariantDataset(s[f"{protocol}_train"], boxes)
    te = VariantDataset(s[f"{protocol}_test"], boxes)
    print(f"[{protocol}] n_source={n_source} train={len(tr)} test={len(te)} epochs={epochs}", flush=True)
    model = build_model(device)
    t0 = time.time()
    hist = train(model, tr, epochs, device)
    preds = predict(model, te, device)
    res = evaluate(preds)
    res.update({"protocol": protocol, "n_source": n_source, "seed": seed,
                "n_train": len(tr), "n_test": len(te), "epochs": epochs,
                "train_minutes": round((time.time() - t0) / 60, 1), "loss_curve": [round(x, 3) for x in hist]})
    print(f"[{protocol}] {res['mAP@0.5']:.4f} @0.5 | {res['mAP@0.5:0.95']:.4f} @0.5:0.95 "
          f"| {res['train_minutes']} min", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", default="A,B")
    ap.add_argument("--n-source", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--n-train", type=int, default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, flush=True)
    results = []
    for p in a.protocols.split(","):
        results.append(run(p.strip(), a.n_source, a.epochs, a.seed, dev, a.n_train))
        (OUT / f"results_{a.tag}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps([{k: v for k, v in r.items() if k != "loss_curve"} for r in results], indent=2))
