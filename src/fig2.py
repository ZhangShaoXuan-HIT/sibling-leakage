"""Figure 2 - a random split overstates detection by 30-55 points on a surface-defect benchmark."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from style import A_COL, B_COL, NEUTRAL, panel_label, save_figure, soft_grid

OUT = Path(__file__).resolve().parent.parent / "out"
runs = {}
for f in ("results_curve_s0.json", "results_curve_s1.json", "results_curve_s2.json"):
    for r in json.loads((OUT / f).read_text()):
        runs[(r["k"], r["protocol"], r["seed"])] = r
seeds = sorted({s for _, _, s in runs})
ks = sorted({k for k, _, _ in runs})


def series(proto, metric):
    m = [st.mean([runs[(k, proto, s)][metric] for s in seeds]) for k in ks]
    sd = [st.pstdev([runs[(k, proto, s)][metric] for s in seeds]) for k in ks]
    return np.array(m), np.array(sd)


A, Asd = series("A", "mAP@0.5:0.95")
B, Bsd = series("B", "mAP@0.5:0.95")
gap95 = A - B
gap95sd = np.array([st.pstdev([runs[(k, "A", s)]["mAP@0.5:0.95"] - runs[(k, "B", s)]["mAP@0.5:0.95"]
                               for s in seeds]) for k in ks])
gap50 = np.array([st.mean([runs[(k, "A", s)]["mAP@0.5"] - runs[(k, "B", s)]["mAP@0.5"] for s in seeds])
                  for k in ks])
gap50sd = np.array([st.pstdev([runs[(k, "A", s)]["mAP@0.5"] - runs[(k, "B", s)]["mAP@0.5"] for s in seeds])
                    for k in ks])

q = np.load(OUT / "qual_neuseg.npz")
slr = {r["k"]: r["A"]["slr_le4"] for r in json.loads((OUT / "slr_final.json").read_text())}


def plate(ax, picks=(0, 1), half=58):
    """Cropped detection plate: the crop is centred on the largest ground-truth box."""
    ax.axis("off")
    ax.set_aspect("equal")
    W, GAP, ROM = 200, 26, 20
    cols = [("ground truth", "#F2B134"), ("A  random split", A_COL), ("B  by acquisition", B_COL)]
    size = 2 * half
    for c, (title, colour) in enumerate(cols):
        for r, j in enumerate(picks):
            img = q["A_img_%d" % j]
            gts = q["A_gt_%d" % j]
            areas = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
            big = gts[int(np.argmax(areas))]
            cx, cy = (big[0] + big[2]) / 2, (big[1] + big[3]) / 2
            x0c = int(np.clip(cx - half, 0, img.shape[1] - size))
            y0c = int(np.clip(cy - half, 0, img.shape[0] - size))
            crop = img[y0c:y0c + size, x0c:x0c + size]
            s = W / size
            X = c * (W + GAP)
            Y = (1 - r) * (W + ROM)
            ax.imshow(crop, extent=(X, X + W, Y, Y + W), aspect="auto", zorder=1)
            ax.add_patch(Rectangle((X, Y), W, W, fill=False, ec="white", lw=0.6, zorder=3))
            if c == 0:
                boxes, key_col = gts[:, :4], colour
            else:
                key = "A" if c == 1 else "B"
                boxes, key_col = q["%s_pb_%d" % (key, j)], colour
            for b in boxes:
                x1, y1 = (b[0] - x0c) * s, (b[1] - y0c) * s
                x2, y2 = (b[2] - x0c) * s, (b[3] - y0c) * s
                x1, x2 = max(0.0, x1), min(W, x2)
                y1, y2 = max(0.0, y1), min(W, y2)
                if x2 - x1 < 1.5 or y2 - y1 < 1.5:
                    continue
                ax.add_patch(Rectangle((X + x1, Y + (W - y2)), x2 - x1, y2 - y1,
                                       fill=False, ec=key_col, lw=0.9, zorder=4))
        ax.text(c * (W + GAP) + W / 2, 2 * (W + ROM) - ROM + 16, title, ha="center", va="bottom",
                fontsize=6.4, color=colour)
    ax.set_xlim(-8, 3 * (W + GAP) - GAP + 8)
    ax.set_ylim(-8, 2 * (W + ROM) - ROM + W + 48)


def main():
    fig = plt.figure(figsize=(7.09, 5.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.02], width_ratios=[0.90, 1.10],
                          left=0.090, right=0.975, top=0.950, bottom=0.080,
                          wspace=0.36, hspace=0.44)

    ax_a = fig.add_subplot(gs[0, 0])
    x = np.array(ks, dtype=float)
    ax_a.fill_between(x, A - Asd, A + Asd, color=A_COL, alpha=0.15, lw=0)
    ax_a.fill_between(x, B - Bsd, B + Bsd, color=B_COL, alpha=0.15, lw=0)
    ax_a.fill_between(x, B, A, color=A_COL, alpha=0.09, lw=0)
    ax_a.plot(x, A, "-o", color=A_COL, ms=4.0, lw=1.1, zorder=3)
    ax_a.plot(x, B, "-o", color=B_COL, ms=4.0, lw=1.1, zorder=3)
    i5 = ks.index(5)
    ax_a.text(6.1, 0.66, "mAP that leakage adds", fontsize=6.0, color=A_COL, va="center")
    ax_a.text(10.25, A[-1] + 0.034, "A", color=A_COL, fontsize=7.5, fontweight="bold", va="center")
    ax_a.text(10.25, B[-1] - 0.034, "B", color=B_COL, fontsize=7.5, fontweight="bold", va="center")
    ax_a.set_xticks(ks)
    ax_a.set_xticklabels([str(k) for k in ks])
    ax_a.set_xlim(1.5, 11.3)
    ax_a.set_ylim(0.2, 1.05)
    ax_a.set_xlabel("source images per class, $k$")
    ax_a.set_ylabel("mAP@0.5:0.95")
    soft_grid(ax_a, "y")
    panel_label(fig, ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    xx = np.arange(len(ks))
    w = 0.36
    ax_b.bar(xx - w / 2, gap95, w, yerr=gap95sd, capsize=2.0, color=A_COL, label="mAP@0.5:0.95",
             error_kw=dict(lw=0.6, capthick=0.6))
    ax_b.bar(xx + w / 2, gap50, w, yerr=gap50sd, capsize=2.0, color="#9E9E9E", label="mAP@0.5",
             error_kw=dict(lw=0.6, capthick=0.6))
    ax_b.set_xticks(xx)
    ax_b.set_xticklabels([str(k) for k in ks])
    ax_b.set_xlabel("source images per class, $k$")
    ax_b.set_ylabel("A $-$ B (mAP points)")
    ax_b.set_ylim(0, 0.68)
    ax_b.legend(loc="upper right", handlelength=1.0, borderaxespad=0.2, fontsize=6.0)
    panel_label(fig, ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 0])
    cls = ["1", "2", "3"]
    kk = 3
    pa = [st.mean([runs[(kk, "A", s)]["AP@0.5:0.95"][c] for s in seeds]) for c in cls]
    pb = [st.mean([runs[(kk, "B", s)]["AP@0.5:0.95"][c] for s in seeds]) for c in cls]
    xc = np.arange(3)
    w = 0.34
    ax_c.bar(xc - w / 2, pa, w, color=A_COL)
    ax_c.bar(xc + w / 2, pb, w, color=B_COL)
    for xi, v in zip(xc - w / 2, pa):
        ax_c.text(xi, v + 0.03, "%.2f" % v, ha="center", fontsize=6.0, color=A_COL)
    for xi, v in zip(xc + w / 2, pb):
        ax_c.text(xi, v + 0.03, "%.2f" % v, ha="center", fontsize=6.0, color=B_COL)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels(["class %s" % c for c in cls], fontsize=6.5)
    ax_c.set_ylabel("AP@0.5:0.95 at $k$ = 3")
    ax_c.set_ylim(0, 1.12)
    panel_label(fig, ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    plate(ax_d)
    panel_label(fig, ax_d, "d", dx=0.050)

    save_figure(fig, "fig2_neuseg")


if __name__ == "__main__":
    main()
