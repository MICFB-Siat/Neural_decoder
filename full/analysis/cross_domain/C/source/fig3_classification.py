
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("BCIC", "FACED", "SEEDV", "MOTOR")
DISPLAY = {"BCIC": "BCIC", "FACED": "FACED", "SEEDV": "SEED-V", "MOTOR": "Motor Imagery"}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(payload: dict, key: str = "poe_temp") -> float:
    block = payload.get("overall", payload).get(key)
    if isinstance(block, dict):
        return float(block["mean"])
    return float(block)


def within_rows(root: Path) -> list[dict[str, object]]:
    rows = []
    for dataset in DATASETS:
        folder = root / dataset / "BT-ND+BrainOmni_within5fold"
        grouped: dict[str, list[float]] = {}
        for path in sorted(folder.glob("sub-*/fold_*/metrics.json")):
            value = read(path).get("poe_temp")
            if value is not None:
                grouped.setdefault(path.parent.parent.name, []).append(float(value))
        for participant, values in grouped.items():
            rows.append({"panel": "3C", "dataset": DISPLAY[dataset], "participant": participant, "accuracy": float(np.mean(values)), "folds": len(values), "protocol": "within-participant five-fold"})
    return rows


def held_out_rows(root: Path) -> list[dict[str, object]]:
    rows = []
    for dataset in DATASETS:
        alternatives = sorted((root / dataset).glob("BT-ND+BrainOmni_LOSO_rerun_*/*/summary.json"))
        for index, path in enumerate(alternatives, start=1):
            payload = read(path)
            values = payload.get("overall", payload)["poe_temp"]["values"]
            subjects = payload["meta"]["subjects"]
            if len(values) != len(subjects) or payload["meta"]["n_folds"] != len(subjects):
                raise ValueError(f"Incomplete leave-one-participant-out result: {path}")
            for participant, value in zip(subjects, values):
                rows.append({"panel": "3E", "dataset": DISPLAY[dataset], "participant": participant, "run": index, "accuracy": float(value), "protocol": "leave-one-participant-out"})
        if not alternatives:
            print(f"{DISPLAY[dataset]}: leave-one-participant-out results not located")
            folder = root / dataset / "BT-ND+BrainOmni_cross5fold"
            for index, path in enumerate(sorted(folder.glob("*/summary.json")), start=1):
                rows.append({"panel": "3E historical", "dataset": DISPLAY[dataset], "run": index, "accuracy": metric(read(path)), "protocol": "held-out-participant five-fold"})
    return rows


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def draw(rows: list[dict[str, object]], output: Path, panels=("3C", "3E")) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.1), sharey=True)
    axes = np.atleast_1d(axes)
    colors = "#E66B6B"
    titles = {"3C": "Within-participant", "3E": "Leave-one-participant-out"}
    for ax, panel in zip(axes, panels):
        title = titles[panel]
        subset = [row for row in rows if row["panel"] == panel]
        arrays = [np.asarray([float(row["accuracy"]) for row in subset if row["dataset"] == DISPLAY[name]]) for name in DATASETS]
        boxes = ax.boxplot(arrays, patch_artist=True, showfliers=False)
        for box in boxes["boxes"]:
            box.set_facecolor("white"); box.set_edgecolor(colors)
        generator = np.random.default_rng(0)
        for index, values in enumerate(arrays, start=1):
            ax.scatter(index + generator.uniform(-.08, .08, len(values)), values, s=10, color=colors, alpha=.55)
            if not len(values):
                ax.text(index, .5, "Pending", ha="center", fontsize=8, rotation=90)
        ax.set_xticks(range(1, 5), [DISPLAY[name] for name in DATASETS], rotation=25, ha="right")
        ax.set_title(title); ax.set_ylim(0, 1); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy")
    fig.tight_layout()
    stem = "Fig" + panels[0] if len(panels) == 1 else "Fig3C_Fig3E_checkpoint_reproduction"
    fig.savefig(output / (stem + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/media/wsqlab/data/gy/code/Eigen_brain_decoding/data_check_20260507/Exp_tyf_Results/Exp_Classification/result"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows = within_rows(args.root) + held_out_rows(args.root)
    save_csv(args.output / "Fig3C_Fig3E_checkpoint_metrics.csv", rows)
    draw(rows, args.output)
    for panel in ("3C", "3E", "3E historical"):
        chosen = [row for row in rows if row["panel"] == panel]
        if chosen:
            print(panel, {name: round(float(np.mean([row["accuracy"] for row in chosen if row["dataset"] == name])), 4) for name in sorted({row["dataset"] for row in chosen})})


if __name__ == "__main__":
    raise SystemExit('Use ../../run.py --panels C for fresh weight inference; archived metric-only entry is disabled.')
