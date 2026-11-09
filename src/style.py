"""Plotting style shared by the manuscript figures (Nature-style defaults)."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

A_COL = "#C2452D"
B_COL = "#2E6F95"
A_FILL = "#F2C9BE"
B_FILL = "#BFD7E4"
NEUTRAL = "#4A4A4A"
GRID = "#D9D9D9"
ACCENT = "#E0A458"
SRC_COLS = ["#8AB17D", "#E9C46A", "#6D9DC5"]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "axes.grid": False,
})


def panel_label(fig, ax, letter, dx=0.042, dy=0.010, size=8):
    p = ax.get_position()
    fig.text(p.x0 - dx, p.y1 + dy, letter, fontsize=size, fontweight="bold",
             ha="left", va="bottom", color="black")


def soft_grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig, name, dpi=600):
    stem = OUT / name
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    print("saved", stem)
