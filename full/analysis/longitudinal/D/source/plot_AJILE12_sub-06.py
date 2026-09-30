import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from resources import path as resource_path




import os
import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, default=resource_path('ajile_repr'))
parser.add_argument("--output", type=Path, default=Path(HERE).parent / "outputs")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
DATA = str(args.input)
OUT = str(args.output / "AJILE12_sub-06")
DIM_K, DIM_SEEDS = 5, 40
CMAP = "viridis"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def dim_sim_5x5(X, k=DIM_K, n_seed=DIM_SEEDS):

    X = X.astype(np.float64)
    acc = np.zeros((k, k), dtype=np.float64)
    for seed in range(n_seed):
        projection = np.random.default_rng(seed).standard_normal((X.shape[1], k))
        projected = X @ projection
        normalized = projected / (np.linalg.norm(projected, axis=0) + 1e-12)
        acc += np.abs(normalized.T @ normalized)
    return acc / n_seed


def off_diagonal_mean(matrix):
    mask = ~np.eye(len(matrix), dtype=bool)
    return float(matrix[mask].mean())


def annotate_off_diagonal(ax, matrix, text_color=None):
    cmap = mpl.colormaps[CMAP]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if row == col:
                continue
            value = float(matrix[row, col])
            red, green, blue, _ = cmap(norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color or ("black" if luminance > 0.52 else "white"),
            )


data = np.load(DATA)
panels = [
    ("SPaRCNet", dim_sim_5x5(data["sparc"])),
    ("Ours", dim_sim_5x5(data["ours"])),
]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
for ax, (title, matrix) in zip(axes, panels):
    image = ax.imshow(
        matrix,
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xticks(range(DIM_K))
    ax.set_yticks(range(DIM_K))
    ax.set_xticklabels(["D0", "D1", "D2", "D3", "D4"])
    ax.set_yticklabels(["D0", "D1", "D2", "D3", "D4"])
    ax.set_title(
        f"AJILE12 sub-06 - {title}\nmean = {off_diagonal_mean(matrix):.2f}",
        fontsize=9,
        pad=6,
    )
    annotate_off_diagonal(ax, matrix, text_color="black" if title == "SPaRCNet" else None)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("|cosine|", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(f"{OUT}.svg", bbox_inches="tight")
fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"WROTE {OUT}.svg")
for title, matrix in panels:
    np.savetxt(args.output / (title + "_recomputed_matrix.csv"), matrix, delimiter=",")
    print(f"  {title}: off-diagonal mean |cosine| = {off_diagonal_mean(matrix):.3f}")
