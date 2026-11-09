"""Figure 3 - the same effect on real STM scans: segmentation under a held-out-acquisition split."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from style import A_COL, B_COL, NEUTRAL, panel_label, save_figure, soft_grid

OUT = Path(__file__).resolve().parent.parent / "out"
rows = json.loads((OUT / "results_stm_s012.json").read_text())
q = np.load(OUT / "qual_stm.npz")

d = {}
for r in rows:
    d.setdefault((r["k"], r["protocol"]), []).append(r["dice"])
ks = sorted({k for k, _ in d})
A = [st.mean(d[(k, "A")]) for k in ks]
Asd = [st.pstdev(d[(k, "A")]) for k in ks]
B = [st.mean(d[(k, "B")]) for k in ks]
Bsd = [st.pstdev(d[(k, "B")]) for k in ks]

GT_COL = "#2E7D32"
OVER = {"A": A_COL, "B": B_COL}


def overlay(ax, img, mask, colour, extent):
    ax.imshow(img, cmap="gray", extent=extent, aspect="auto", zorder=1)
    rgba = np.zeros(mask.shape + (4,))
    rgba[..., :3] = plt.matplotlib.colors.to_rgb(colour)
    rgba[..., 3] = mask * 0.45
    ax.imshow(rgba, extent=extent, aspect="auto", zorder=2)


def plate(ax):
    ax.axis("off")
    W, GAP, Y0 = 512, 46, 0
    tiles = [
        ("STM image", None, None),
        ("ground truth", q["gt_2"], GT_COL),
        ("A  random split", q["A_2"], A_COL),
        ("B  by acquisition", q["B_2"], B_COL),
    ]
    for j, (title, mask, colour) in enumerate(tiles):
        x = j * (W + GAP)
        ext = (x, x + W, Y0, Y0 + W)
        if mask is None:
            ax.imshow(q["img_2"], cmap="gray", extent=ext, aspect="auto")
        else:
            overlay(ax, q["img_2"], mask, colour, ext)
        ax.add_patch(plt.Rectangle((x, Y0), W, W, fill=False, ec="white", lw=0.8, zorder=4))
        ax.text(x + W / 2, Y0 + W + 26, title, ha="center", va="bottom", fontsize=6.6,
                color=colour if colour else "black")
    ax.set_xlim(-10, 3 * (W + GAP) + W + 10)
    ax.set_ylim(-40, W + 90)


def main():
    fig = plt.figure(figsize=(7.09, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.22], width_ratios=[1.15, 1.0],
                          left=0.075, right=0.975, top=0.945, bottom=0.095,
                          wspace=0.30, hspace=0.36)

    ax_a = fig.add_subplot(gs[0, 0])
    x = np.array(ks)
    ax_a.fill_between(x, np.array(A) - np.array(Asd), np.array(A) + np.array(Asd), color=A_COL, alpha=0.16, lw=0)
    ax_a.fill_between(x, np.array(B) - np.array(Bsd), np.array(B) + np.array(Bsd), color=B_COL, alpha=0.16, lw=0)
    ax_a.plot(x, A, "-o", color=A_COL, ms=4.0, lw=1.1, zorder=3)
    ax_a.plot(x, B, "-o", color=B_COL, ms=4.0, lw=1.1, zorder=3)
    for xi, a, b in zip(x, A, B):
        ax_a.vlines(xi, b, a, color=NEUTRAL, lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
    ax_a.text(x[-1] + 0.18, A[-1], "A", color=A_COL, fontsize=7.5, fontweight="bold", va="center")
    ax_a.text(x[-1] + 0.18, B[-1], "B", color=B_COL, fontsize=7.5, fontweight="bold", va="center")
    ax_a.set_xticks(ks)
    ax_a.set_xticklabels([str(k) for k in ks])
    ax_a.set_xlabel("training scans, $k$")
    ax_a.set_ylabel("Dice on held-out scans")
    ax_a.set_ylim(0, 1.0)
    ax_a.set_xlim(1.6, 5.6)
    soft_grid(ax_a, "y")
    panel_label(fig, ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    scans = ["Ge(001)", "Si(001)", "TiO2(110)"]
    xs = np.arange(3)
    w = 0.26
    gt = [q[f"gt_{j}"].mean() * 100 for j in range(3)]
    pa = [q[f"A_{j}"].mean() * 100 for j in range(3)]
    pb = [q[f"B_{j}"].mean() * 100 for j in range(3)]
    ax_b.bar(xs - w, gt, w, color="#BDBDBD", label="ground truth")
    ax_b.bar(xs, pa, w, color=A_COL, label="A  random split")
    ax_b.bar(xs + w, pb, w, color=B_COL, label="B  by acquisition")
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(scans, fontsize=6.3)
    ax_b.set_ylabel("defect area (%)")
    ax_b.legend(loc="upper left", handlelength=0.9, borderaxespad=0.2, fontsize=6.0)
    ax_b.set_ylim(0, 30)
    soft_grid(ax_b, "y")
    panel_label(fig, ax_b, "b")

    ax_c = fig.add_subplot(gs[1, :])
    plate(ax_c)
    panel_label(fig, ax_c, "c")

    save_figure(fig, "fig3_stm")


if __name__ == "__main__":
    main()
