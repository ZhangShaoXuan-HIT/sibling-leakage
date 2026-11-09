"""STM defect segmentation: U-Net trained under protocol A (random patch split)
and protocol B (scan-level split). Fixed 400 training patches, 400 test patches.
"""
from __future__ import annotations

from config import OUT_ROOT

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stm_data import build

OUT = OUT_ROOT


class Double(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                               nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))

    def forward(self, x):
        return self.b(x)


class UNet(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.e1 = Double(1, base)
        self.e2 = Double(base, base * 2)
        self.e3 = Double(base * 2, base * 4)
        self.bn = Double(base * 4, base * 8)
        self.d3 = Double(base * 8 + base * 4, base * 4)
        self.d2 = Double(base * 4 + base * 2, base * 2)
        self.d1 = Double(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.bn(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        return self.out(d1)


def to_arrays(pairs):
    x = np.stack([p[0] for p in pairs])[:, None, :, :].astype(np.float32)
    y = np.stack([p[1] for p in pairs])[:, None, :, :].astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def evaluate(model, x, y, dev):
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(x.to(dev))) > 0.5
    p = pred.cpu().numpy().astype(np.uint8)
    t = y.numpy().astype(np.uint8)
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    acc = float((p == t).mean())
    return {"dice": round(dice, 4), "iou": round(iou, 4), "pixel_acc": round(acc, 4)}


def run(proto, k, seed, epochs, dev):
    s = build(k, seed)
    tr = s[f"{proto}_train"]
    te = s["test"]
    xt, yt = to_arrays(tr)
    xv, yv = to_arrays(te)
    torch.manual_seed(seed)
    model = UNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(xt)
    bs = 32
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = xt[idx].to(dev), yt[idx].to(dev)
            logit = model(xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb) + dice_loss(logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
    res = evaluate(model, xv, yv, dev)
    res.update({"protocol": proto, "k": k, "seed": seed, "n_train": n, "n_test": len(xv),
                "epochs": epochs, "minutes": round((time.time() - t0) / 60, 2),
                "train_scans": len(s["train_scans"])})
    print(f"[{proto} k={k} s={seed}] dice={res['dice']:.4f} iou={res['iou']:.4f} ({res['minutes']} min)", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="2,3,5")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tag", default="stm")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, flush=True)
    out, f = [], OUT / f"results_{a.tag}.json"
    for seed in [int(x) for x in a.seeds.split(",")]:
        for k in [int(x) for x in a.ks.split(",")]:
            for proto in ("A", "B"):
                out.append(run(proto, k, seed, a.epochs, dev))
                f.write_text(json.dumps(out, indent=1))
    print("STM DONE")

