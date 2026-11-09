"""Figure 1 - the construction manufactures siblings; the statistic measures it without training."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

from neuseg_data import IMGS, make_curve_splits
from style import (A_COL, B_COL, NEUTRAL, SRC_COLS, panel_label, save_figure, soft_grid)

OUT = Path(__file__).resolve().parent.parent / "out"
slr = json.loads((OUT / "slr_final.json").read_text())
ks = [r["k"] for r in slr]
A = [r["A"]["slr_le4"] for r in slr]
B = [r["B"]["slr_le4"] for r in slr]

PITCH, W, H = 0.98, 0.84, 1.30
X0 = 24.5
YA, YB = 19.4, 15.2


def draw_schematic(ax):
    ax.axis("off")
    s = make_curve_splits(3, 0)
    names = [s["reserved"][c] for c in sorted(s["reserved"])]

    # acquisitions
    xs = 1.0
    for i, n in enumerate(names):
        img = np.array(Image.open(IMGS / n).convert("L"))
        ax.imshow(img, extent=(xs, xs + 4.4, 23.0, 27.4), cmap="gray", zorder=2, aspect="auto")
        ax.add_patch(Rectangle((xs, 23.0), 4.4, 4.4, fill=False, ec=NEUTRAL, lw=0.6, zorder=3))
        ax.text(xs + 2.2, 22.4, "acq. %d" % (i + 1), ha="center", va="top", fontsize=6.2, color=NEUTRAL)
        xs += 5.3
    ax.text(8.6, 28.0, "source images", ha="center", va="bottom", fontsize=6.5, color="black")

    ax.add_patch(FancyArrowPatch((17.4, 25.2), (20.9, 25.2), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.8, color=NEUTRAL))
    ax.text(19.2, 25.8, "x80 variants", ha="center", va="bottom", fontsize=6.0, color=NEUTRAL)

    # assignments
    n_src, per = 3, 8
    rng = np.random.default_rng(3)
    for label, y, kind in [("random split", YA, "A"), ("by acquisition", YB, "B")]:
        srcs = [c for c in range(n_src) for _ in range(per)]
        if kind == "A":
            idx = rng.choice(len(srcs), size=per, replace=False)
            is_test = np.zeros(len(srcs), dtype=bool)
            is_test[idx] = True
        else:
            is_test = np.array([s == n_src - 1 for s in srcs])
        for j, (sr, te) in enumerate(zip(srcs, is_test)):
            x = X0 + j * PITCH
            ax.add_patch(Rectangle((x, y), W, H, fc=SRC_COLS[sr], ec="white", lw=0.4, alpha=0.85, zorder=2))
            if te:
                ax.add_patch(Rectangle((x, y), W, H, fc="none", ec=A_COL, lw=0.9, zorder=3))
        ax.text(X0 - 1.0, y + H / 2 + 0.34, label, ha="right", va="center", fontsize=6.5, color="black")
        ax.text(X0 - 1.0, y + H / 2 - 0.86, kind, ha="right", va="center", fontsize=7.5,
                fontweight="bold", color=A_COL if kind == "A" else B_COL)

    ax.text(X0 + 11.8, YA + 1.5, "one variant pool, two assignments", ha="center", va="bottom",
            fontsize=6.3, color=NEUTRAL)

    # held-out block bracket
    xb = X0 + 16 * PITCH
    xe = X0 + 24 * PITCH - (PITCH - W)
    yb = YB - 0.9
    ax.plot([xb, xe], [yb, yb], color=B_COL, lw=0.7)
    for xx in (xb, xe):
        ax.plot([xx, xx], [yb, yb + 0.40], color=B_COL, lw=0.7)
    ax.text((xb + xe) / 2, yb - 0.40, "one acquisition held out", ha="center", va="top",
            fontsize=6.0, color=B_COL)

    ax.text(X0 + 24 * PITCH + 0.5, YA + H / 2, "siblings\nin training", ha="left", va="center",
            fontsize=6.2, color=A_COL, linespacing=1.25)
    ax.text(X0 + 24 * PITCH + 0.5, YB + H / 2, "no siblings", ha="left", va="center",
            fontsize=6.2, color=B_COL)

    # key: one row under the assignment strips
    ly = YB - 3.95
    items = [(SRC_COLS[0], "acquisition 1", 8.6), (SRC_COLS[1], "acquisition 2", 18.2),
             (SRC_COLS[2], "acquisition 3", 27.8), (None, "variant assigned to test", 36.8)]
    for col, txt, x in items:
        ax.add_patch(Rectangle((x, ly), 0.85, 1.0, fc=col if col else "none",
                               ec="none" if col else A_COL, lw=0 if col else 0.9, alpha=0.85 if col else 1))
        ax.text(x + 1.1, ly + 0.5, txt, va="center", fontsize=6.0,
                color=NEUTRAL if col else A_COL)

    ax.set_xlim(0, 55)
    ax.set_ylim(10.3, 28.7)


def main():
    fig = plt.figure(figsize=(7.09, 3.80))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.22, 1.0], width_ratios=[1.0, 1.0],
                          left=0.075, right=0.975, top=0.955, bottom=0.115,
                          wspace=0.30, hspace=0.32)

    ax_a = fig.add_subplot(gs[0, :])
    draw_schematic(ax_a)
    panel_label(fig, ax_a, "a")

    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.plot(ks, A, "-o", color=A_COL, ms=4.2, lw=1.1, zorder=3)
    ax_b.plot(ks, B, "-o", color=B_COL, ms=4.2, lw=1.1, zorder=3)
    ax_b.text(10.35, A[-1] + 0.03, "A", color=A_COL, fontsize=7.5, fontweight="bold", va="center")
    ax_b.text(10.35, 0.04, "B", color=B_COL, fontsize=7.5, fontweight="bold", va="center")
    ax_b.set_xticks(ks)
    ax_b.set_xticklabels([str(k) for k in ks])
    ax_b.set_xlabel("source images per class, $k$")
    ax_b.set_ylabel("sibling leakage rate")
    ax_b.set_ylim(-0.06, 1.10)
    ax_b.set_xlim(1.2, 10.9)
    soft_grid(ax_b, "y")
    panel_label(fig, ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 1])
    bins = np.arange(-0.5, 33.5, 1)
    da = np.concatenate([np.array(r["A"]["dists"]) for r in slr])
    db = np.concatenate([np.array(r["B"]["dists"]) for r in slr])
    ax_c.hist(da, bins=bins, color=A_COL, alpha=0.55, lw=0, edgecolor="none")
    ax_c.hist(db, bins=bins, color=B_COL, alpha=0.55, lw=0, edgecolor="none")
    ax_c.text(2.2, 232, "A", color=A_COL, fontsize=7.5, fontweight="bold", ha="left", va="center")
    ax_c.text(16.2, 78, "B", color=B_COL, fontsize=7.5, fontweight="bold", ha="left", va="center")
    ax_c.axvline(4, color=NEUTRAL, lw=0.7, ls=(0, (2, 1.5)))
    ax_c.text(4.8, 0.94, "4-bit\nthreshold", fontsize=5.8, color=NEUTRAL, va="top",
              linespacing=1.2, transform=ax_c.get_xaxis_transform())
    ax_c.set_xlabel("distance to nearest training image (bits)")
    ax_c.set_ylabel("test images")
    ax_c.set_xlim(-0.5, 32)
    soft_grid(ax_c, "y")
    panel_label(fig, ax_c, "c")

    save_figure(fig, "fig1_construction")


if __name__ == "__main__":
    main()
