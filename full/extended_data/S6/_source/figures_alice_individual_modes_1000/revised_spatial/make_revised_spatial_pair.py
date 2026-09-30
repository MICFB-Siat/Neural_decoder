












from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "zhf_exp/zhf_exp_fmri_eigen/data/generated"
OUT = Path(__file__).resolve().parent
SUBJECTS = ("sub-22", "sub-24")
N_MODES = 999
ANCHORS = (1, 5, 20, 40, 100, 200, 500, 999)
BANDS = (
    ("1-20", 1, 20),
    ("21-100", 21, 100),
    ("101-300", 101, 300),
    ("301-999", 301, 999),
)

GREY = "#4D4D4D"
LIGHT_GREY = "#CFCFCF"
GREEN = "#2E9E44"
YELLOW = "#E0A030"
RED = "#E53935"
LOCAL = "#C4473D"
SUBJECT_COLORS = {"sub-22": "#6F85A8", "sub-24": "#A88A6F"}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)


def mode_path(key: str) -> Path:
    return DATA_ROOT / key / f"{key}_L_eigenmodes_1000_with_mode0_masked.npy"


def eigenvalue_path(key: str) -> Path:
    return DATA_ROOT / key / f"{key}_L_eigenmodes_1000_with_mode0_evals.npy"


def load_normalized_modes(key: str) -> np.ndarray:

    path = mode_path(key)
    modes = np.array(np.load(path, mmap_mode="r")[:, 1:1000], dtype=np.float32, copy=True)
    if modes.shape != (29696, N_MODES):
        raise ValueError(f"Unexpected mode shape for {key}: {modes.shape}")
    modes -= modes.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(modes, axis=0, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError(f"Zero-norm eigenfunction for {key}")
    modes /= norm
    return modes


def adaptive_smooth(y: np.ndarray, fraction: float = 0.035) -> np.ndarray:






    y = np.asarray(y, dtype=np.float64)
    out = np.empty_like(y)
    n = len(y)
    for idx in range(n):
        rank = idx + 1
        half = int(np.clip(np.rint(fraction * rank), 2, 30))
        lo = max(0, idx - half)
        hi = min(n, idx + half + 1)
        out[idx] = np.nanmean(y[lo:hi])
    return out


def dot_color(value: float) -> str:
    return GREEN if value >= 0.60 else (YELLOW if value >= 0.30 else RED)


def compute_correspondence() -> tuple[dict[str, dict[str, np.ndarray]], list[dict]]:






    group = load_normalized_modes("template")
    group_eval = np.load(eigenvalue_path("template"))[1:1000]
    results: dict[str, dict[str, np.ndarray]] = {}
    source_rows: list[dict] = []

    for subject in SUBJECTS:
        individual = load_normalized_modes(subject)
        subject_eval = np.load(eigenvalue_path(subject))[1:1000]


        corr = np.abs(group.T @ individual)
        same = np.diag(corr).astype(np.float64)
        local = np.empty(N_MODES, dtype=np.float64)
        matched = np.empty(N_MODES, dtype=np.int32)

        for idx in range(N_MODES):
            rank = idx + 1
            radius = max(3, int(np.ceil(0.08 * rank)))
            lo = max(0, idx - radius)
            hi = min(N_MODES - 1, idx + radius)
            offset = int(np.argmax(corr[idx, lo : hi + 1]))
            match_idx = lo + offset
            local[idx] = float(corr[idx, match_idx])
            matched[idx] = match_idx + 1
            source_rows.append(
                {
                    "subject": subject,
                    "hemisphere": "left",
                    "reference_rank": rank,
                    "same_rank_abs_spatial_r": float(same[idx]),
                    "best_local_rank": int(match_idx + 1),
                    "best_local_abs_spatial_r": float(local[idx]),
                    "local_rank_shift": int(match_idx - idx),
                    "template_eigenvalue": float(group_eval[idx]),
                    "subject_same_rank_eigenvalue": float(subject_eval[idx]),
                    "subject_best_local_eigenvalue": float(subject_eval[match_idx]),
                    "local_radius": radius,
                    "metric": "absolute Pearson r across 29696 matched cortical vertices"
                }
            )

        results[subject] = {"same": same, "local": local, "matched": matched}
    return results, source_rows


def cohort_summary(results: dict[str, dict[str, np.ndarray]], key: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.vstack([results[s][key] for s in SUBJECTS])
    smooth = np.vstack([adaptive_smooth(v) for v in values])
    return smooth.mean(axis=0), smooth.std(axis=0, ddof=1) / np.sqrt(len(SUBJECTS))


def panel_label(ax: plt.Axes, label: str, x: float = -0.14) -> None:
    ax.text(x, 1.13, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", clip_on=False)


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(labelsize=6.5, length=3, pad=2)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def draw_panel_b(ax: plt.Axes, results: dict[str, dict[str, np.ndarray]], log_x: bool = True) -> None:
    x = np.arange(1, N_MODES + 1)
    raw = np.vstack([results[s]["same"] for s in SUBJECTS])
    raw_mean = raw.mean(axis=0)
    mean, sem = cohort_summary(results, "same")


    ax.plot(x, raw_mean, color=LIGHT_GREY, lw=0.45, alpha=0.55, zorder=1)
    ax.fill_between(x, np.clip(mean - sem, 0, 1), np.clip(mean + sem, 0, 1),
                    color=GREY, alpha=0.17, linewidth=0, zorder=2)
    ax.plot(x, mean, color=GREY, lw=2.0, zorder=4)

    for rank in ANCHORS:
        value = float(mean[rank - 1])
        ax.scatter(rank, value, s=33, facecolor=dot_color(value), edgecolor="black",
                   linewidth=0.5, zorder=8)
        if rank == 1:
            offset, align = (4, -14), "left"
        elif rank in (5, 20, 40):
            offset, align = (0, 8), "center"
        elif rank in (100, 200):
            offset, align = (0, 7), "center"
        else:
            offset, align = (-2, 7), "right"
        ax.annotate(rf"$\psi_{{{rank}}}$", (rank, value), xytext=offset,
                    textcoords="offset points", ha=align, fontsize=6.1)

    ax.axhspan(0.80, 1.02, color="#EAF6EC", zorder=0)
    ax.axhspan(0.50, 0.80, color="#FBF6E9", zorder=0)
    ax.axhspan(0.00, 0.50, color="#FBECEC", zorder=0)
    ax.axhline(0.80, color=GREEN, lw=0.55, ls=":", zorder=3)
    ax.axhline(0.50, color=RED, lw=0.55, ls=":", zorder=3)

    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(1, 1000)
        ax.set_xticks([1, 5, 20, 40, 100, 200, 500, 1000])
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    else:
        ax.set_xlim(0, 1000)
        ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xlabel("Eigenmode index")
    ax.set_ylabel("Group-individual\nspatial correlation  $|r|$")
    ax.set_title("Spatial correspondence declines with rank", fontsize=7.7,
                 fontweight="bold", pad=7, loc="left")
    ax.text(0.97, 0.97, "shared\nglobal shape", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8, color=GREEN, fontweight="bold")
    ax.text(0.97, 0.38, "increasingly\ndivergent", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8, color=RED, fontweight="bold")
    ax.text(0.025, 0.54, "n = 2 individuals\nline: smoothed mean; shade: SEM",
            transform=ax.transAxes, fontsize=5.0, color="#666666", va="bottom")
    panel_label(ax, "b")
    style_axis(ax)


def band_values(results: dict[str, dict[str, np.ndarray]], key: str) -> np.ndarray:
    rows = []
    for subject in SUBJECTS:
        rows.append([np.median(results[subject][key][lo - 1 : hi]) for _, lo, hi in BANDS])
    return np.asarray(rows, dtype=np.float64)


def draw_panel_c_summary(ax: plt.Axes, results: dict[str, dict[str, np.ndarray]]) -> None:
    same = band_values(results, "same")
    local = band_values(results, "local")
    x = np.arange(len(BANDS), dtype=np.float64)
    dx = 0.14
    subject_jitter = (-0.025, 0.025)

    for s_idx, subject in enumerate(SUBJECTS):
        for band_idx in range(len(BANDS)):
            xx_same = x[band_idx] - dx + subject_jitter[s_idx]
            xx_local = x[band_idx] + dx + subject_jitter[s_idx]
            ax.plot([xx_same, xx_local], [same[s_idx, band_idx], local[s_idx, band_idx]],
                    color="#B9B9B9", lw=0.75, zorder=1)
            ax.scatter(xx_same, same[s_idx, band_idx], s=19,
                       facecolor=SUBJECT_COLORS[subject], edgecolor="white", lw=0.45, zorder=3)
            ax.scatter(xx_local, local[s_idx, band_idx], s=19,
                       facecolor=LOCAL, edgecolor="white", lw=0.45, zorder=3)

    same_med = np.median(same, axis=0)
    local_med = np.median(local, axis=0)
    ax.scatter(x - dx, same_med, marker="_", s=120, color=GREY, linewidth=2.0,
               label="Same rank", zorder=5)
    ax.scatter(x + dx, local_med, marker="_", s=120, color=LOCAL, linewidth=2.0,
               label="Best local pattern", zorder=5)

    for idx, value in enumerate(local_med):
        ax.text(x[idx] + dx, value + 0.045, f"{value:.2f}", ha="center", va="bottom",
                fontsize=5.7, color=LOCAL, fontweight="bold")

    ax.axhspan(0.00, 0.20, color="#FBECEC", alpha=0.70, zorder=0)
    ax.set_xlim(-0.48, len(BANDS) - 0.52)
    ax.set_ylim(0, 0.86)
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\psi_{{{lo}}}$-$\psi_{{{hi}}}$" for _, lo, hi in BANDS])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Spectral band")
    ax.set_ylabel("Median spatial correlation  $|r|$")
    ax.set_title("Local rank search only partly restores correspondence", fontsize=7.7,
                 fontweight="bold", pad=5)
    ax.legend(loc="upper right", fontsize=5.9, handletextpad=0.4, borderaxespad=0.2)
    ax.text(0.025, 0.025,
            "Search window: +/- max(3, 8% of rank)\nHigh-order local matches remain weak; dots: sub-22/sub-24",
            transform=ax.transAxes, fontsize=5.1, color="#666666", va="bottom")
    panel_label(ax, "c", x=-0.03)
    style_axis(ax)


def draw_panel_c_curves(ax: plt.Axes, results: dict[str, dict[str, np.ndarray]]) -> None:
    x = np.arange(1, N_MODES + 1)
    same, same_sem = cohort_summary(results, "same")
    local, local_sem = cohort_summary(results, "local")
    ax.fill_between(x, np.clip(local - local_sem, 0, 1), np.clip(local + local_sem, 0, 1),
                    color=LOCAL, alpha=0.12, linewidth=0)
    ax.plot(x, same, color=GREY, lw=1.65, label="Same rank")
    ax.plot(x, local, color=LOCAL, lw=2.0, label="Best local pattern")
    ax.fill_between(x, same, local, where=local >= same, color="#F4C6C1", alpha=0.42, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 5, 20, 40, 100, 200, 500, 1000])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel("Eigenmode index")
    ax.set_ylabel("Spatial correlation  $|r|$")
    ax.set_title("Local search helps, but high-order modes remain different",
                 fontsize=7.2, fontweight="bold", pad=6)
    ax.legend(loc="upper right", fontsize=6.0)
    ax.text(0.025, 0.025, "Best local pattern is a descriptive ceiling, not a one-to-one permutation",
            transform=ax.transAxes, fontsize=5.0, color="#666666", va="bottom")
    panel_label(ax, "c", x=-0.03)
    style_axis(ax)


def save_pair(stem_name: str, results: dict[str, dict[str, np.ndarray]],
              right_panel, log_x_left: bool = True) -> None:
    fig = plt.figure(figsize=(183 / 25.4, 2.52), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, width_ratios=[1.0, 1.62], left=0.085, right=0.985,
        bottom=0.22, top=0.90, wspace=0.30
    )
    ax_b = fig.add_subplot(grid[0, 0])
    ax_c = fig.add_subplot(grid[0, 1])
    draw_panel_b(ax_b, results, log_x=log_x_left)
    right_panel(ax_c, results)
    fig.text(0.985, 0.035,
             "Left cortex, fsLR32k; absolute Pearson r across 29,696 matched vertices; constant mode excluded",
             ha="right", va="bottom", fontsize=5.0, color="#777777")
    stem = OUT / stem_name
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)


def save_panel_b(results: dict[str, dict[str, np.ndarray]], log_x: bool, name: str) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.45), facecolor="white")
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.20, top=0.88)
    draw_panel_b(ax, results, log_x=log_x)
    stem = OUT / name
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)


def write_source_data(rows: list[dict]) -> None:
    path = OUT / "source_data_spatial_1000.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(results: dict[str, dict[str, np.ndarray]]) -> None:
    summary = {
        "backend": "Python",
        "subjects": list(SUBJECTS),
        "available_nontrivial_modes": [1, 999],
        "metric": "absolute Pearson correlation across 29696 matched cortical vertices",
        "smoothing": "centered adaptive running mean: 5 points at low order, up to 61 points",
        "right_panel": "same-rank versus best-local spatial correspondence by spectral band",
        "local_search": "plus/minus max(3, ceil(0.08*rank)); candidate reuse allowed; descriptive ceiling",
        "band_medians": {},
    }
    same = band_values(results, "same")
    local = band_values(results, "local")
    for idx, (name, lo, hi) in enumerate(BANDS):
        summary["band_medians"][name] = {
            "mode_range": [lo, hi],
            "same_rank_across_subject_median": float(np.median(same[:, idx])),
            "best_local_across_subject_median": float(np.median(local[:, idx])),
        }
    (OUT / "qa_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results, rows = compute_correspondence()
    write_source_data(rows)
    write_summary(results)
    save_pair("recommended_spatial_pair", results, draw_panel_c_summary, log_x_left=True)
    save_pair("alternative_spatial_pair_curves", results, draw_panel_c_curves, log_x_left=True)
    save_panel_b(results, log_x=True, name="panel_b_spatial_1000_log")
    save_panel_b(results, log_x=False, name="panel_b_spatial_1000_linear")


if __name__ == "__main__":
    main()
