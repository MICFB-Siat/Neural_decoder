


from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATE = Path(__file__).resolve().parents[1]
SOURCE = (
    DATE
    / "figures"
    / "group_individual_spatial_band_subject_boxplot"
    / "source_data_mode_level_n10.csv"
)
OUT = DATE / "figures" / "group_individual_spatial_curve_n10"
STEM = "Alice_group_individual_spatial_correspondence_n10"

SUBJECTS = (
    "sub-18", "sub-22", "sub-23", "sub-24", "sub-26",
    "sub-30", "sub-31", "sub-35", "sub-36", "sub-37",
)
ANCHORS = (1, 5, 20, 40, 100, 200, 500, 1000)
BAND_BOUNDARIES = (10, 50, 200)

GREY = "#555555"
RAW_GREY = "#BEBEBE"
IQR_GREY = "#8E8E8E"
GREEN = "#35A765"
AMBER = "#D79B2E"
RED = "#D9544D"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adaptive_smooth(values: np.ndarray, fraction: float = 0.035) -> np.ndarray:

    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    for index in range(values.size):
        rank = index + 1
        half_width = int(np.clip(np.rint(fraction * rank), 2, 30))
        lo = max(0, index - half_width)
        hi = min(values.size, index + half_width + 1)
        output[index] = np.mean(values[lo:hi])
    return output


def load_matrix() -> np.ndarray:
    source = pd.read_csv(SOURCE)
    required = {
        "subject", "hemisphere", "mode_rank", "same_rank_abs_spatial_r",
    }
    if not required.issubset(source.columns):
        raise ValueError(f"Missing source columns: {sorted(required - set(source.columns))}")
    source = source[source["subject"].isin(SUBJECTS)].copy()
    if source.shape[0] != 10000 or source["subject"].nunique() != 10:
        raise ValueError(f"Expected 10,000 rows from 10 participants; got {source.shape}")
    matrix = (
        source.pivot(index="subject", columns="mode_rank", values="same_rank_abs_spatial_r")
        .loc[list(SUBJECTS), list(range(1, 1001))]
        .to_numpy(dtype=np.float64)
    )
    if matrix.shape != (10, 1000) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"Invalid matrix: {matrix.shape}")
    if np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("Spatial correlations outside [0, 1]")
    return matrix


def anchor_color(value: float) -> str:
    if value >= 0.60:
        return GREEN
    if value >= 0.30:
        return AMBER
    return RED


def write_source_data(
    raw: np.ndarray,
    smoothed: np.ndarray,
    raw_median: np.ndarray,
    raw_q25: np.ndarray,
    raw_q75: np.ndarray,
    smooth_median: np.ndarray,
    smooth_q25: np.ndarray,
    smooth_q75: np.ndarray,
) -> tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    participant_path = OUT / "source_data_participant_mode_n10.csv"
    curve_path = OUT / "source_data_curve_summary_n10.csv"

    with participant_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["dataset", "subject", "hemisphere", "group_template", "mode_rank",
             "same_rank_abs_spatial_r", "adaptively_smoothed_abs_spatial_r"]
        )
        for subject_index, subject in enumerate(SUBJECTS):
            for mode_index in range(1000):
                writer.writerow(
                    ["Alice ds002322", subject, "left", "HCP S1200", mode_index + 1,
                     raw[subject_index, mode_index], smoothed[subject_index, mode_index]]
                )

    with curve_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["mode_rank", "n_subjects", "raw_across_subject_median",
             "raw_across_subject_q25", "raw_across_subject_q75",
             "smoothed_across_subject_median", "smoothed_across_subject_q25",
             "smoothed_across_subject_q75", "smoothing_half_window"]
        )
        for index in range(1000):
            rank = index + 1
            half_width = int(np.clip(np.rint(0.035 * rank), 2, 30))
            writer.writerow(
                [rank, len(SUBJECTS), raw_median[index], raw_q25[index], raw_q75[index],
                 smooth_median[index], smooth_q25[index], smooth_q75[index], half_width]
            )
    return participant_path, curve_path


def draw(
    raw_median: np.ndarray,
    raw_q25: np.ndarray,
    raw_q75: np.ndarray,
    smooth_median: np.ndarray,
    smooth_q25: np.ndarray,
    smooth_q75: np.ndarray,
) -> list[Path]:
    x = np.arange(1, 1001)
    fig, ax = plt.subplots(figsize=(96 / 25.4, 68 / 25.4), facecolor="white")
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.22, top=0.87)

    for boundary in BAND_BOUNDARIES:
        ax.axvline(boundary, color="#D7D7D7", linewidth=0.55, linestyle=(0, (2, 2)), zorder=0)

    ax.fill_between(
        x,
        smooth_q25,
        smooth_q75,
        color=IQR_GREY,
        alpha=0.20,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        x,
        smooth_median,
        color=GREY,
        linewidth=1.75,
        solid_capstyle="round",
        zorder=3,
    )

    offsets = {
        1: (4, -14, "left"),
        5: (0, 8, "center"),
        20: (0, 8, "center"),
        40: (0, 8, "center"),
        100: (0, 8, "center"),
        200: (0, 8, "center"),
        500: (-1, 8, "right"),
        1000: (-1, 8, "right"),
    }
    for rank in ANCHORS:
        index = rank - 1
        value = float(raw_median[index])
        ax.vlines(
            rank,
            raw_q25[index],
            raw_q75[index],
            color=anchor_color(value),
            linewidth=1.0,
            zorder=4,
        )
        ax.scatter(
            rank,
            value,
            s=30,
            facecolor=anchor_color(value),
            edgecolor="#202020",
            linewidth=0.5,
            zorder=5,
        )
        dx, dy, align = offsets[rank]
        ax.annotate(
            rf"$\psi_{{{rank}}}$",
            (rank, value),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=5.6,
        )

    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(ANCHORS)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xlabel("Eigenmode index")
    ax.set_ylabel("Group–individual\nspatial correlation  $|r|$")
    ax.set_title(
        "Group–individual spatial correspondence",
        loc="left",
        fontweight="bold",
        pad=7,
    )
    ax.text(
        -0.16,
        1.12,
        "b",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )
    ax.text(
        0.98,
        0.95,
        "n = 10 participants\nsmoothed median (IQR); points: exact ranks",
        transform=ax.transAxes,
        fontsize=5.0,
        color="#6A6A6A",
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )
    ax.tick_params(width=0.8, length=3, pad=2)

    outputs = []
    for extension, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 450}),
        ("tiff", {"dpi": 600}),
    ):
        path = OUT / f"{STEM}.{extension}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    raw = load_matrix()
    smoothed = np.vstack([adaptive_smooth(row) for row in raw])
    raw_median = np.median(raw, axis=0)
    raw_q25 = np.percentile(raw, 25, axis=0)
    raw_q75 = np.percentile(raw, 75, axis=0)
    smooth_median = np.median(smoothed, axis=0)
    smooth_q25 = np.percentile(smoothed, 25, axis=0)
    smooth_q75 = np.percentile(smoothed, 75, axis=0)

    participant_path, curve_path = write_source_data(
        raw, smoothed, raw_median, raw_q25, raw_q75,
        smooth_median, smooth_q25, smooth_q75,
    )
    figure_paths = draw(
        raw_median, raw_q25, raw_q75,
        smooth_median, smooth_q25, smooth_q75,
    )
    outputs = [participant_path, curve_path, *figure_paths]
    manifest = {
        "backend": "Python/matplotlib",
        "dataset": "Alice Datasets EEG-fMRI (ds002322)",
        "subjects": list(SUBJECTS),
        "n_subjects": len(SUBJECTS),
        "hemisphere": "left",
        "n_cortical_vertices": 29696,
        "group_template": "HCP S1200",
        "metric": "same-rank absolute Pearson spatial correlation across matched cortical vertices",
        "main_line": "across-subject median of participant-wise adaptive running means",
        "ribbon": "25th to 75th percentile of participant-wise adaptive running means",
        "anchor_points": "unsmoothed across-subject medians at exact eigenmode ranks",
        "anchor_error_bars": "unsmoothed participant IQR at exact eigenmode ranks",
        "smoothing": "centered mean; half-window=clip(round(0.035*rank),2,30)",
        "band_boundaries": list(BAND_BOUNDARIES),
        "source_sha256": sha256(SOURCE),
        "outputs": [path.name for path in outputs],
        "output_sha256": {path.name: sha256(path) for path in outputs},
        "anchor_raw_medians": {
            str(rank): float(raw_median[rank - 1]) for rank in ANCHORS
        },
    }
    temporary = OUT / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(OUT / "manifest.json")

    print(f"Created: {OUT / (STEM + '.png')}")
    print("Exact-rank across-subject medians:")
    for rank in ANCHORS:
        print(f"  psi_{rank}: {raw_median[rank - 1]:.6f}")


if __name__ == "__main__":
    main()
