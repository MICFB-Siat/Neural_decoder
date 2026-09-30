from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def compute_coordinates(config):
    previous = {key: os.environ.get(key) for key in config["environment"]}
    os.environ.update(config["environment"])
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        module = importlib.import_module("plot_real_r3_three_classes")
        full, _ = module.base.prepare()
        full["obs"] = module.projected_observation_coordinates(
            config["projection_dimensions"], config["projection_initialization"]
        )
        ids = [module.ALL.index(name) for name in config["classes"]]
        coordinates = {key: value[:, ids] for key, value in full.items()}
        correlations = {
            key: [float(np.corrcoef(value[0, :, :, m].ravel(),
                                   value[1, :, :, m].ravel())[0, 1])
                  for m in range(3)]
            for key, value in coordinates.items()
        }
        return module.base, coordinates, correlations
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def render(base, coordinates, correlations, config, output):
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})
    base.CLASSES = config["classes"]
    base.COLORS = ["#13A7B7", "#6650A4", "#E67E22"]
    fig = plt.figure(figsize=(7.2, 4.15), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 2.35],
                         wspace=.12, hspace=.19, left=.065, right=.99,
                         top=.95, bottom=.17)
    for row, key in enumerate(("obs", "prior")):
        limits = []
        for m in range(3):
            values = coordinates[key][:, :, :, m]
            lo, hi = float(values.min()), float(values.max())
            limits.append((lo - .06*(hi-lo), hi + .06*(hi-lo)))
        for day in range(2):
            ax = fig.add_subplot(gs[row, day], projection="3d")
            base.draw_3d(ax, coordinates[key][day], limits,
                         coordinates[key][0] if day else None)
            ax.set_title("D0" if day == 0 else "D42",
                         fontsize=8, weight="bold", pad=-1)
        base.draw_modes(fig, gs[row, 2], coordinates[key], correlations[key])
    for y, label in ((.735, "Neural observation"), (.325, "Meta-neural semantic")):
        fig.text(.020, y, label, rotation=90, ha="center", va="center",
                 fontsize=7, weight="semibold")
    for x, label in ((.058, "a"), (.505, "b")):
        fig.text(x, .965, label, fontsize=9, weight="bold", va="top")
    handles = [Line2D([0], [0], c=c, lw=2, label=name)
               for c, name in zip(base.COLORS, config["classes"])]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=6.5, bbox_to_anchor=(.31, .035),
               handlelength=1.6, columnspacing=1.6)
    for extension in ("png", "pdf", "svg", "tiff"):
        fig.savefig(output / f"Fig4G.{extension}", bbox_inches="tight",
                    facecolor="white", dpi=600 if extension == "tiff" else 320)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_suffix(".json"))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "output" / "fig4g")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    base, coordinates, correlations = compute_coordinates(config)
    render(base, coordinates, correlations, config, args.output)
    with (args.output / "Fig4G_correlations.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["representation", "pc", "pearson_r", "n_centroids"])
        for key, values in correlations.items():
            for pc, value in enumerate(values, 1):
                writer.writerow([key, pc, value, 24])
    with (args.output / "Fig4G_coordinates.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["representation", "day", "class", "token_index",
                         "nominal_time_s", "PC1", "PC2", "PC3"])
        for key, values in coordinates.items():
            for di, day in enumerate((0, 42)):
                for ci, name in enumerate(config["classes"]):
                    for t in range(8):
                        writer.writerow([key, day, name, t, .125+.25*t,
                                         *values[di, ci, t]])
    (args.output / "Fig4G_processing.json").write_text(
        json.dumps({"configuration": config, "correlations": correlations,
                    "input": "retained token features and trained Gaussian heads",
                    "training_performed": False}, indent=2))
    print(json.dumps(correlations, indent=2))


if __name__ == "__main__":
    main()
