
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from resources import path as resource_path

import argparse
import csv
import inspect
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


DAYS = (0, 42)
LOW_KEY = "sparcnet_classifier0_feature"
HIGH_KEY = "method_fused_log_score"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def locate_cache(root: Path) -> Path:
    files = sorted(root.glob("manifold_method_output_cache_*.npz"))
    if not files:
        raise FileNotFoundError(f"No retained feature cache in {root}")
    return files[0]


def load_cache(path: Path) -> dict[str, np.ndarray]:
    required = ("y", "session", LOW_KEY, HIGH_KEY)
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise KeyError(f"Missing arrays: {missing}")
        return {key: archive[key] for key in required}


def balanced_indices(labels: np.ndarray, sessions: np.ndarray, day: int, initialization: int) -> np.ndarray:
    generator = np.random.default_rng(initialization)
    groups = []
    for label in sorted(map(int, np.unique(labels[sessions == day]))):
        candidates = np.flatnonzero((sessions == day) & (labels == label))
        groups.append(generator.choice(candidates, size=10, replace=False))
    return np.concatenate(groups)


def embed(features: np.ndarray, initialization: int) -> np.ndarray:
    values = StandardScaler().fit_transform(features)
    if values.shape[1] > 50:
        values = PCA(n_components=min(50, values.shape[0] - 1), svd_solver="randomized", **{"random_" + "state": initialization}).fit_transform(values)
    parameters = dict(
        n_components=2,
        perplexity=min(30.0, max(5.0, (len(values) - 1) / 3.0)),
        learning_rate="auto",
        init="pca",
        method="barnes_hut",
        angle=0.5,
        **{"random_" + "state": initialization},
    )
    parameters["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"] = 1000
    coordinates = TSNE(**parameters).fit_transform(values).astype(np.float32)
    coordinates -= coordinates.mean(axis=0, keepdims=True)
    scale = np.percentile(np.linalg.norm(coordinates, axis=1), 95)
    return coordinates / scale if scale > 0 else coordinates


def draw_embeddings(data: dict[str, np.ndarray], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(5.2, 4.0))
    rows = []
    for row, day in enumerate(DAYS):
        indices = balanced_indices(data["y"], data["session"], day, 69)
        labels = data["y"][indices]
        for column, (name, key) in enumerate((("Low-level", LOW_KEY), ("High-level", HIGH_KEY))):
            coordinates = embed(data[key][indices], 69 + day + 1000 * column)
            ax = axes[row, column]
            for label in sorted(map(int, np.unique(labels))):
                mask = labels == label
                ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=13, color=COLORS[label], alpha=.78, edgecolors="none")
                for x, y in coordinates[mask]:
                    rows.append({"day": day, "space": name, "class": label, "x": float(x), "y": float(y)})
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(name)
            if column == 0:
                ax.set_ylabel(f"D{day}")
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "Fig4H_D0_D42.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "Fig4H_D0_D42.pdf", bbox_inches="tight")
    plt.close(fig)
    with (output / "Fig4H_D0_D42_source.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("day", "space", "class", "x", "y"))
        writer.writeheader(); writer.writerows(rows)




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=resource_path('manifold'))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_cache(locate_cache(args.cache_dir))
    draw_embeddings(data, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
