














from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "figures_alice_individual_modes_1000").exists())
SPATIAL_CSV = (
    REPO
    / "figures_alice_individual_modes_1000"
    / "revised_spatial"
    / "source_data_spatial_1000.csv"
)
EIGENVALUE_CSV = (
    REPO
    / "paper_main"
    / "Eigen_brain_gy"
    / "Fig素材"
    / "Fig5"
    / "eigenmode_difference_options"
    / "spectral_band_boxplot_source_data.csv"
)

SUBJECTS = ("sub-22", "sub-24")
SELECTED_SUBJECT = "sub-22"
N_MODES = 999
ANCHORS = (1, 5, 20, 40, 100, 200, 500, 999)
BANDS = (
    ("1–10", 1, 10),
    ("11–50", 11, 50),
    ("51–200", 51, 200),
    ("201–999", 201, 999),
)

SUBJECT_COLORS = {"sub-22": "#3B8FB8", "sub-24": "#D66B55"}
GREY = "#444444"
LIGHT_GREY = "#D8D8D8"
GREEN = "#2E9E44"
AMBER = "#D79A24"
RED = "#D84A43"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)


def rank_band(rank: int) -> str:
    return next(name for name, lo, hi in BANDS if lo <= rank <= hi)


def adaptive_smooth(values: np.ndarray, fraction: float = 0.035) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    for idx in range(len(values)):
        rank = idx + 1
        half = int(np.clip(np.rint(fraction * rank), 2, 30))
        lo = max(0, idx - half)
        hi = min(len(values), idx + half + 1)
        out[idx] = np.mean(values[lo:hi])
    return out


def load_source_data() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    spatial = {s: np.full(N_MODES, np.nan, dtype=float) for s in SUBJECTS}
    with SPATIAL_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            subject = row["subject"]
            if subject not in spatial:
                continue
            rank = int(row["reference_rank"])
            if 1 <= rank <= N_MODES:
                spatial[subject][rank - 1] = float(row["same_rank_abs_spatial_r"])

    eigdiff = {s: np.full(N_MODES, np.nan, dtype=float) for s in SUBJECTS}
    with EIGENVALUE_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            subject = row["subject_or_pair"]
            if row["comparison"] != "group-individual" or subject not in eigdiff:
                continue
            rank = int(row["mode_rank"])
            if 1 <= rank <= N_MODES:
                eigdiff[subject][rank - 1] = float(
                    row["area_normalized_eigenvalue_difference_percent"]
                )

    for label, values in [("spatial", spatial), ("eigenvalue", eigdiff)]:
        for subject in SUBJECTS:
            if np.count_nonzero(np.isfinite(values[subject])) != N_MODES:
                raise ValueError(f"Incomplete {label} data for {subject}")
    return spatial, eigdiff


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.12,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def correlation_color(value: float) -> str:
    if value >= 0.60:
        return GREEN
    if value >= 0.30:
        return AMBER
    return RED


def draw_spatial_panel(ax: plt.Axes, spatial: dict[str, np.ndarray]) -> None:
    x = np.arange(1, N_MODES + 1)
    subject_smoothed = np.vstack([adaptive_smooth(spatial[s]) for s in SUBJECTS])
    mean = subject_smoothed.mean(axis=0)



    ax.axhspan(0.80, 1.02, color="#EFF8F0", zorder=0)
    ax.axhspan(0.50, 0.80, color="#FCF8EC", zorder=0)
    ax.axhspan(0.00, 0.50, color="#FCEFEF", zorder=0)
    ax.plot(x, mean, color=GREY, lw=1.8, zorder=3)

    for rank in ANCHORS:
        value = float(mean[rank - 1])
        ax.scatter(
            rank,
            value,
            s=25,
            facecolor=correlation_color(value),
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
        if rank == 1:
            offset, align = (3, -12), "left"
        elif rank in (5, 20, 40):
            offset, align = (0, 7), "center"
        else:
            offset, align = (-1, 7), "right"
        ax.annotate(
            rf"$\psi_{{{rank}}}$",
            (rank, value),
            xytext=offset,
            textcoords="offset points",
            ha=align,
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
    ax.set_title("Eigenfunction correspondence", fontsize=7.8, fontweight="bold", pad=6)
    ax.text(
        0.04,
        0.05,
        "n = 2 individuals\nline: smoothed mean",
        transform=ax.transAxes,
        fontsize=5.1,
        color="#666666",
    )
    panel_label(ax, "b")


def draw_spatial_panel_single(ax: plt.Axes, spatial: dict[str, np.ndarray]) -> None:

    x = np.arange(1, N_MODES + 1)
    smooth = adaptive_smooth(spatial[SELECTED_SUBJECT])

    ax.axhspan(0.80, 1.02, color="#EFF8F0", zorder=0)
    ax.axhspan(0.50, 0.80, color="#FCF8EC", zorder=0)
    ax.axhspan(0.00, 0.50, color="#FCEFEF", zorder=0)
    ax.plot(x, smooth, color=GREY, lw=1.8, zorder=3)

    for rank in ANCHORS:
        value = float(smooth[rank - 1])
        ax.scatter(
            rank,
            value,
            s=25,
            facecolor=correlation_color(value),
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
        if rank == 1:
            offset, align = (3, -12), "left"
        elif rank in (5, 20, 40):
            offset, align = (0, 7), "center"
        else:
            offset, align = (-1, 7), "right"
        ax.annotate(
            rf"$\psi_{{{rank}}}$",
            (rank, value),
            xytext=offset,
            textcoords="offset points",
            ha=align,
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
    ax.set_title("Eigenfunction correspondence", fontsize=7.8, fontweight="bold", pad=6)
    panel_label(ax, "b")


def band_arrays(values: np.ndarray) -> list[np.ndarray]:
    return [values[lo - 1 : hi] for _, lo, hi in BANDS]


def subject_band_medians(eigdiff: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(
        [[np.median(a) for a in band_arrays(eigdiff[s])] for s in SUBJECTS],
        dtype=float,
    )


def draw_eigenvalue_subject_panel(ax: plt.Axes, eigdiff: dict[str, np.ndarray]) -> None:

    x = np.arange(len(BANDS), dtype=float)
    values = subject_band_medians(eigdiff)
    offsets = (-0.07, 0.07)

    for idx, subject in enumerate(SUBJECTS):
        ax.scatter(
            x + offsets[idx],
            values[idx],
            s=31,
            color=SUBJECT_COLORS[subject],
            edgecolor="white",
            linewidth=0.55,
            label=subject,
            zorder=4,
        )

    center = np.median(values, axis=0)
    lo = values.min(axis=0)
    hi = values.max(axis=0)
    ax.vlines(x, lo, hi, color=LIGHT_GREY, lw=1.1, zorder=1)
    ax.scatter(
        x,
        center,
        marker="D",
        s=22,
        color=GREY,
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
        label="Median",
    )
    for idx, value in enumerate(center):
        ax.text(
            x[idx],
            value * 1.18,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=5.5,
            color=GREY,
        )

    ax.set_yscale("log")
    ax.set_ylim(0.22, 12.0)
    ax.set_yticks([0.3, 1, 3, 10])
    ax.get_yaxis().set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _, _ in BANDS])
    ax.set_xlabel("Eigenmode rank band")
    ax.set_ylabel("Area-normalized eigenvalue\ndifference (%)  [log scale]")
    ax.set_title("Eigenvalue deviation", fontsize=7.8, fontweight="bold", pad=6)
    ax.legend(loc="upper right", fontsize=5.6, ncol=3, handletextpad=0.25, columnspacing=0.7)
    ax.text(
        0.03,
        0.045,
        "Each point: one individual's median across modes",
        transform=ax.transAxes,
        fontsize=5.1,
        color="#666666",
    )
    panel_label(ax, "c")


def styled_boxplot(
    ax: plt.Axes,
    arrays: list[np.ndarray],
    positions: np.ndarray,
    color: str,
) -> None:
    ax.boxplot(
        arrays,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        whis=(0, 100),
        medianprops={"color": "#262626", "linewidth": 1.0},
        boxprops={"facecolor": color, "edgecolor": color, "linewidth": 0.8, "alpha": 0.58},
        whiskerprops={"color": color, "linewidth": 0.8},
        capprops={"color": color, "linewidth": 0.8},
    )


def draw_eigenvalue_box_panel(ax: plt.Axes, eigdiff: dict[str, np.ndarray]) -> None:

    x = np.arange(len(BANDS), dtype=float)
    offsets = (-0.17, 0.17)
    for idx, subject in enumerate(SUBJECTS):
        arrays = band_arrays(eigdiff[subject])
        styled_boxplot(ax, arrays, x + offsets[idx], SUBJECT_COLORS[subject])
        medians = [np.median(a) for a in arrays]
        ax.scatter(
            x + offsets[idx],
            medians,
            s=15,
            color=SUBJECT_COLORS[subject],
            edgecolor="white",
            linewidth=0.45,
            zorder=5,
            label=subject,
        )

    counts = [hi - lo + 1 for _, lo, hi in BANDS]
    labels = [f"{name}\n$n_{{mode}}$={count}" for (name, _, _), count in zip(BANDS, counts)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Eigenmode rank band")
    ax.set_ylabel("Area-normalized eigenvalue\ndifference (%)")
    ax.set_title("Eigenvalue deviation", fontsize=7.8, fontweight="bold", pad=6)
    ax.legend(loc="upper right", fontsize=5.7)
    panel_label(ax, "c")


def draw_eigenvalue_box_single(ax: plt.Axes, eigdiff: dict[str, np.ndarray]) -> None:

    x = np.arange(len(BANDS), dtype=float)
    arrays = band_arrays(eigdiff[SELECTED_SUBJECT])
    color = SUBJECT_COLORS[SELECTED_SUBJECT]
    ax.boxplot(
        arrays,
        positions=x,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        whis=(0, 100),
        medianprops={"color": "#262626", "linewidth": 1.05},
        boxprops={"facecolor": color, "edgecolor": color, "linewidth": 0.85, "alpha": 0.58},
        whiskerprops={"color": color, "linewidth": 0.85},
        capprops={"color": color, "linewidth": 0.85},
    )
    medians = np.asarray([np.median(a) for a in arrays])
    ax.scatter(
        x,
        medians,
        s=19,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
    )
    for idx, value in enumerate(medians):
        ax.text(
            x[idx],
            value + (0.42 if idx == 0 else 0.18),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=5.7,
            color="#245F7A",
        )

    counts = [hi - lo + 1 for _, lo, hi in BANDS]
    labels = [f"{name}\n$n_{{mode}}$={count}" for (name, _, _), count in zip(BANDS, counts)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ymax = max(float(np.max(a)) for a in arrays)
    ax.set_ylim(0, ymax * 1.08)
    ax.set_xlabel("Eigenmode rank band")
    ax.set_ylabel("Area-normalized eigenvalue\ndifference (%)")
    ax.set_title("Eigenvalue deviation", fontsize=7.8, fontweight="bold", pad=6)
    ax.text(
        0.98,
        0.96,
        SELECTED_SUBJECT,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        color=color,
        fontweight="bold",
    )
    panel_label(ax, "c")


def build_figure(
    spatial: dict[str, np.ndarray],
    eigdiff: dict[str, np.ndarray],
    right_drawer,
    stem: str,
    footer: str,
) -> None:
    fig = plt.figure(figsize=(183 / 25.4, 68 / 25.4), facecolor="white")
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 1.5],
        left=0.09,
        right=0.985,
        bottom=0.24,
        top=0.88,
        wspace=0.40,
    )
    ax_b = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    draw_spatial_panel(ax_b, spatial)
    right_drawer(ax_c, eigdiff)
    fig.text(0.985, 0.045, footer, ha="right", va="bottom", fontsize=4.9, color="#777777")
    base = HERE / stem
    fig.savefig(base.with_suffix(".svg"), facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=450, facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)


def build_single_subject_figure(
    spatial: dict[str, np.ndarray], eigdiff: dict[str, np.ndarray]
) -> None:
    fig = plt.figure(figsize=(183 / 25.4, 68 / 25.4), facecolor="white")
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 1.5],
        left=0.09,
        right=0.985,
        bottom=0.24,
        top=0.88,
        wspace=0.40,
    )
    ax_b = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    draw_spatial_panel_single(ax_b, spatial)
    draw_eigenvalue_box_single(ax_c, eigdiff)
    fig.text(
        0.985,
        0.045,
        "Representative individual: sub-22; c, boxes summarize eigenmodes (median/IQR; whiskers, minimum–maximum), not biological replicates",
        ha="right",
        va="bottom",
        fontsize=4.9,
        color="#777777",
    )
    base = HERE / "group_individual_modes_eigenvalues_sub22_box"
    fig.savefig(base.with_suffix(".svg"), facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=450, facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)


def write_single_subject_box_statistics(eigdiff: dict[str, np.ndarray]) -> None:

    path = HERE / "source_data_sub22_box_statistics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "rank_band", "n_modes", "minimum", "q1", "median", "q3", "maximum"])
        for (name, lo, hi), values in zip(BANDS, band_arrays(eigdiff[SELECTED_SUBJECT])):
            minimum, q1, median, q3, maximum = np.percentile(values, [0, 25, 50, 75, 100])
            writer.writerow(
                [SELECTED_SUBJECT, name, hi - lo + 1, minimum, q1, median, q3, maximum]
            )


def write_single_subject_source_data(
    spatial: dict[str, np.ndarray], eigdiff: dict[str, np.ndarray]
) -> None:
    path = HERE / "source_data_sub22_modes_eigenvalues_verified.csv"


    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subject",
                "mode_rank",
                "rank_band",
                "same_rank_abs_spatial_r",
                "area_normalized_eigenvalue_difference_percent",
            ]
        )
        for idx in range(N_MODES):
            rank = idx + 1
            writer.writerow(
                [
                    SELECTED_SUBJECT,
                    rank,
                    rank_band(rank),
                    spatial[SELECTED_SUBJECT][idx],
                    eigdiff[SELECTED_SUBJECT][idx],
                ]
            )
    temporary.replace(path)


def write_source_data(spatial: dict[str, np.ndarray], eigdiff: dict[str, np.ndarray]) -> None:
    path = HERE / "source_data_group_individual_modes_eigenvalues.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subject",
                "mode_rank",
                "rank_band",
                "same_rank_abs_spatial_r",
                "area_normalized_eigenvalue_difference_percent",
            ]
        )
        for subject in SUBJECTS:
            for idx in range(N_MODES):
                rank = idx + 1
                writer.writerow(
                    [subject, rank, rank_band(rank), spatial[subject][idx], eigdiff[subject][idx]]
                )


def write_qa(spatial: dict[str, np.ndarray], eigdiff: dict[str, np.ndarray]) -> None:
    med = subject_band_medians(eigdiff)
    payload = {
        "backend": "Python/matplotlib",
        "subjects": list(SUBJECTS),
        "n_individuals": len(SUBJECTS),
        "bands": [
            {"label": name, "lo": lo, "hi": hi, "n_modes": hi - lo + 1}
            for name, lo, hi in BANDS
        ],
        "panel_b_metric": "absolute Pearson r across 29696 matched cortical vertices",
        "panel_c_metric": "symmetric percent difference of area-normalized eigenvalues",
        "panel_c_subject_summary": {
            subject: {BANDS[i][0]: float(med[s_idx, i]) for i in range(len(BANDS))}
            for s_idx, subject in enumerate(SUBJECTS)
        }
    }
    (HERE / "qa_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    spatial, eigdiff = load_source_data()
    write_source_data(spatial, eigdiff)
    write_single_subject_source_data(spatial, eigdiff)
    write_single_subject_box_statistics(eigdiff)
    write_qa(spatial, eigdiff)
    build_figure(
        spatial,
        eigdiff,
        draw_eigenvalue_subject_panel,
        "group_individual_modes_eigenvalues_recommended",
        "b, |r| across 29,696 matched vertices; c, one band median per individual; n = 2; descriptive only",
    )
    build_figure(
        spatial,
        eigdiff,
        draw_eigenvalue_box_panel,
        "group_individual_modes_eigenvalues_box_descriptive",
        "b, |r| across matched vertices; c, boxes summarize eigenmodes (median/IQR; whiskers, minimum–maximum), not biological replicates",
    )
    build_single_subject_figure(spatial, eigdiff)


if __name__ == "__main__":
    main()
