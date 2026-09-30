
















from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr


ROOT = Path("/home/guoyi/a800/share/code/Eigen_brain_decoding")
EXP = Path(
    "/home/guoyi/nas2/share/Dataset/ds005256-download/experiments/"
    "alignvideo_mae_epochwise_rsa17_two_init_20260724"
)
DEFAULT_FEATURE_ROOT = EXP / "epochwise_features_incohort17"
DEFAULT_OUTPUT_ROOT = EXP / "figures_epochwise_k_l_rsa17"
VIDEOS_CSV = ROOT / (
    "lizhuo_exp/lizhuo_exp_NSD/test_class2_0715/"
    "39_alignvideo_oldmethod_17sub_common35_20260723/"
    "source_data/K_PER_VIDEO_17sub_pooled.csv"
)

RSA_SUBJECTS = [
    "sub-0031", "sub-0034", "sub-0081", "sub-0043", "sub-0046",
    "sub-0058", "sub-0060", "sub-0070", "sub-0035", "sub-0078",
    "sub-0003", "sub-0004", "sub-0099", "sub-0052", "sub-0073",
    "sub-0116", "sub-0069",
]
INITIALIZATIONS = ("crossdataset", "random")
EPOCHS = tuple(range(21))
RATING_SETS = {
    "four_happy_sad_afraid_disgusted": (1, 2, 3, 4),
    "five_happy_sad_afraid_disgusted_engaged": (1, 2, 3, 4, 6),
}
RATING_LABELS = {
    "four_happy_sad_afraid_disgusted": "4 ratings",
    "five_happy_sad_afraid_disgusted_engaged": "5 ratings",
}
METRIC_LABELS = {
    "pearson_r": "Pearson RSA",
    "spearman_r": "Spearman RSA",
}
INITIALIZATION_LABELS = {
    "crossdataset": "Cross-dataset init., Low-level Obs (512D)",
    "random": "Random init., Low-level Obs (512D)",
}
COLORS = {"crossdataset": "#3B6FB6", "random": "#D47A28"}
MARKERS = {"crossdataset": "D", "random": "^"}


def finite_corr(left: np.ndarray, right: np.ndarray, metric: str) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if (
        int(valid.sum()) < 3
        or float(left[valid].std()) <= 1e-12
        or float(right[valid].std()) <= 1e-12
    ):
        return math.nan
    if metric == "spearman_r":
        return float(spearmanr(left[valid], right[valid]).statistic)
    if metric == "pearson_r":
        return float(pearsonr(left[valid], right[valid]).statistic)
    raise ValueError(metric)


def fisher_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan
    return float(
        np.tanh(np.mean(np.arctanh(np.clip(values, -0.999999, 0.999999))))
    )


def load_feature(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def fit_rating_scalers(
    labels: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:

    labels = np.asarray(labels, dtype=np.float64)
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, indices in RATING_SETS.items():
        selected = labels[:, list(indices)]
        complete = np.all(np.isfinite(selected) & (selected >= 0), axis=1)
        stacked = selected[complete]
        if len(stacked) < 2:
            raise RuntimeError(f"{name}: insufficient complete training labels")
        scale = stacked.std(axis=0)
        scale[scale <= 1e-12] = 1.0
        output[name] = (stacked.mean(axis=0), scale)
    return output


def select_exact30(
    data: dict[str, np.ndarray],
    exact35: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subjects = data["subject"].astype(str)
    sessions = data["session"].astype(int)
    objectives = data["objective"].astype(int)
    ratings = data["labels_subjective"].astype(np.float64)
    obs = data["obs"].astype(np.float64)
    obs35 = []
    rating35 = []
    for subject in RSA_SUBJECTS:
        subject_indices = np.flatnonzero(subjects == subject)
        lookup = {
            (int(sessions[index]), int(objectives[index])): int(index)
            for index in subject_indices
        }
        if len(lookup) != len(subject_indices):
            raise RuntimeError(f"{subject}: duplicate session x objective key")
        missing = [key for key in exact35 if key not in lookup]
        if missing:
            raise RuntimeError(f"{subject}: missing exact35 keys {missing}")
        indices = np.asarray([lookup[key] for key in exact35], dtype=int)
        obs35.append(obs[indices])
        rating35.append(ratings[indices])
    obs35_array = np.stack(obs35)
    rating35_array = np.stack(rating35)
    five = list(RATING_SETS["five_happy_sad_afraid_disgusted_engaged"])
    valid = np.all(
        np.isfinite(rating35_array[:, :, five])
        & (rating35_array[:, :, five] >= 0),
        axis=2,
    ).all(axis=0)
    if int(valid.sum()) != 30:
        raise RuntimeError(f"expected 30 common complete videos, got {valid.sum()}")
    return obs35_array[:, valid], rating35_array[:, valid], valid


def validate_feature_file(
    data: dict[str, np.ndarray],
    initialization: str,
    epoch: int,
) -> None:
    required = {
        "subject", "session", "objective", "labels_subjective", "obs", "indiv"
    }
    if not required.issubset(data):
        raise RuntimeError(
            f"{initialization}/epoch {epoch}: missing {sorted(required - set(data))}"
        )
    subjects = sorted(set(data["subject"].astype(str)))
    if subjects != sorted(RSA_SUBJECTS):
        raise RuntimeError(
            f"{initialization}/epoch {epoch}: cohort differs from fixed 17"
        )
    if data["obs"].shape != (len(data["subject"]), 512):
        raise RuntimeError(
            f"{initialization}/epoch {epoch}: unexpected Obs shape {data['obs'].shape}"
        )
    if not np.isfinite(data["obs"]).all():
        raise RuntimeError(
            f"{initialization}/epoch {epoch}: non-finite Obs feature"
        )


def compute_epochwise(
    feature_root: Path,
    exact35: list[tuple[int, int]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    subject_rows: list[dict] = []
    common_masks: dict[str, list[int]] = {}
    label_reference: np.ndarray | None = None
    for initialization in INITIALIZATIONS:
        for epoch in EPOCHS:
            feature_path = (
                feature_root / initialization / f"epoch_{epoch:03d}.npz"
            )
            if not feature_path.is_file():
                raise FileNotFoundError(feature_path)
            data = load_feature(feature_path)
            validate_feature_file(data, initialization, epoch)
            obs30, rating30, common_mask = select_exact30(data, exact35)
            common_masks[f"{initialization}/epoch_{epoch:03d}"] = (
                common_mask.astype(int).tolist()
            )
            if label_reference is None:
                label_reference = rating30.copy()
            elif not np.array_equal(
                label_reference.astype(np.float32),
                rating30.astype(np.float32),
            ):
                raise RuntimeError(
                    f"{initialization}/epoch {epoch}: evaluation ratings changed"
                )
            scalers = fit_rating_scalers(data["labels_subjective"])
            for subject_index, subject in enumerate(RSA_SUBJECTS):
                neural_edge = pdist(
                    obs30[subject_index], metric="correlation"
                )
                if not np.isfinite(neural_edge).all():
                    raise RuntimeError(
                        f"{initialization}/epoch {epoch}/{subject}: invalid Obs RDM"
                    )
                for rating_set, indices in RATING_SETS.items():
                    center, scale = scalers[rating_set]
                    rating_edge = pdist(
                        (
                            rating30[subject_index][:, list(indices)]
                            - center
                        )
                        / scale,
                        metric="euclidean",
                    )
                    subject_rows.append({
                        "initialization": initialization,
                        "mae_epoch": epoch,
                        "subject": subject,
                        "rating_set": rating_set,
                        "n_videos": 30,
                        "n_rdm_edges": int(len(neural_edge)),
                        "feature": "raw Obs encoder representation",
                        "feature_dim": 512,
                        "neural_distance": "correlation",
                        "rating_distance": "euclidean after in-cohort rating standardization",
                        "pearson_r": finite_corr(
                            neural_edge, rating_edge, "pearson_r"
                        ),
                        "spearman_r": finite_corr(
                            neural_edge, rating_edge, "spearman_r"
                        ),
                        "feature_path": str(feature_path),
                    })

    per_subject = pd.DataFrame(subject_rows)
    expected_subject_rows = 2 * 21 * 17 * 2
    if len(per_subject) != expected_subject_rows:
        raise RuntimeError(
            f"expected {expected_subject_rows} per-subject rows, got {len(per_subject)}"
        )
    summary_rows = []
    grouping = ["initialization", "mae_epoch", "rating_set"]
    for keys, frame in per_subject.groupby(grouping, sort=True):
        initialization, epoch, rating_set = keys
        for metric in ("pearson_r", "spearman_r"):
            values = frame[metric].to_numpy(dtype=float)
            summary_rows.append({
                "initialization": initialization,
                "mae_epoch": int(epoch),
                "rating_set": rating_set,
                "metric": metric,
                "n_subjects": int(len(values)),
                "fisher_mean_r": fisher_mean(values),
                "arithmetic_mean_r": float(np.mean(values)),
                "std_across_subjects": float(np.std(values, ddof=1)),
                "median_r": float(np.median(values)),
                "n_positive": int(np.sum(values > 0)),
                "feature": "Low-level raw Obs (512D)",
                "fusion": "none",
                "seed_selection": "none",
                "epoch_selection": "none",
            })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["rating_set", "metric", "initialization", "mae_epoch"]
    ).reset_index(drop=True)
    expected_summary_rows = 2 * 21 * 2 * 2
    if len(summary) != expected_summary_rows:
        raise RuntimeError(
            f"expected {expected_summary_rows} summary rows, got {len(summary)}"
        )
    audit = {
        "common_exact35_masks": common_masks,
        "all_evaluation_ratings_identical_across_initialization_and_epoch": True,
    }
    return per_subject, summary, audit


def validate_against_existing_qap(
    per_subject: pd.DataFrame,
    output_root: Path,
) -> dict:

    checks = [
        (
            "crossdataset",
            13,
            output_root / "L_NONZERO_PEARSON_MAX_EPOCH013_HIGH_LOW_PER_SUBJECT_QAP.csv",
        ),
        (
            "crossdataset",
            4,
            output_root / "L_NONZERO_SPEARMAN_MAX_EPOCH004_HIGH_LOW_PER_SUBJECT_QAP.csv",
        ),
    ]
    results = []
    for initialization, epoch, path in checks:
        if not path.is_file():
            results.append({
                "path": str(path),
                "status": "not_available",
            })
            continue
        reference = pd.read_csv(path)[
            ["subject", "low_pearson_r", "low_spearman_rho"]
        ]
        current = per_subject[
            (per_subject["initialization"] == initialization)
            & (per_subject["mae_epoch"] == epoch)
            & (
                per_subject["rating_set"]
                == "five_happy_sad_afraid_disgusted_engaged"
            )
        ][["subject", "pearson_r", "spearman_r"]]
        merged = reference.merge(current, on="subject", validate="one_to_one")
        pearson_max_abs = float(
            np.max(np.abs(merged["low_pearson_r"] - merged["pearson_r"]))
        )
        spearman_max_abs = float(
            np.max(
                np.abs(merged["low_spearman_rho"] - merged["spearman_r"])
            )
        )
        if pearson_max_abs > 1e-10 or spearman_max_abs > 1e-10:
            raise RuntimeError(
                f"Low-level reproduction mismatch for {initialization}/"
                f"epoch {epoch}: Pearson={pearson_max_abs}, "
                f"Spearman={spearman_max_abs}"
            )
        results.append({
            "path": str(path),
            "status": "exact_within_1e-10",
            "n_subjects": int(len(merged)),
            "pearson_max_abs_difference": pearson_max_abs,
            "spearman_max_abs_difference": spearman_max_abs,
        })
    return {"checks": results}


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv.tmp",
        dir="/tmp",
        delete=False,
        encoding="utf-8",
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    staged = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    shutil.copyfile(temporary, staged)
    temporary.unlink()
    os.replace(staged, destination)


def atomic_json(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json.tmp",
        dir="/tmp",
        delete=False,
        encoding="utf-8",
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    staged = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    shutil.copyfile(temporary, staged)
    temporary.unlink()
    os.replace(staged, destination)


def export_figure(fig: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    with tempfile.NamedTemporaryFile(
        suffix=".tiff", dir="/tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        fig.savefig(
            temporary,
            dpi=600,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"},
        )
        final = destination.with_suffix(".tiff")
        staged = final.with_name(f".{final.name}.tmp.{os.getpid()}")
        shutil.copyfile(temporary, staged)
        os.replace(staged, final)
    finally:
        temporary.unlink(missing_ok=True)


def plot_summary(summary: pd.DataFrame, destination: Path) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })
    rating_order = (
        "four_happy_sad_afraid_disgusted",
        "five_happy_sad_afraid_disgusted_engaged",
    )
    metric_order = ("pearson_r", "spearman_r")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 4.85),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    for row_index, rating_set in enumerate(rating_order):
        for column_index, metric in enumerate(metric_order):
            ax = axes[row_index, column_index]
            panel = summary[
                (summary["rating_set"] == rating_set)
                & (summary["metric"] == metric)
            ]
            for initialization in INITIALIZATIONS:
                line = panel[
                    panel["initialization"] == initialization
                ].sort_values("mae_epoch")
                ax.plot(
                    line["mae_epoch"],
                    line["fisher_mean_r"],
                    color=COLORS[initialization],
                    linestyle="-",
                    marker=MARKERS[initialization],
                    markersize=3.2,
                    markerfacecolor="white",
                    markeredgewidth=0.75,
                    linewidth=1.2,
                    label=INITIALIZATION_LABELS[initialization],
                )
            ax.axhline(0.0, color="#777777", linewidth=0.75, zorder=0)
            ax.set_title(
                f"{RATING_LABELS[rating_set]} · {METRIC_LABELS[metric]}",
                fontsize=7,
                pad=4,
            )
            ax.set_xticks([0, 5, 10, 15, 20])
            ax.grid(axis="y", color="#E7E7E7", linewidth=0.55)
            ax.tick_params(length=3, width=0.7)
            if row_index == 1:
                ax.set_xlabel("MAE continual-pretraining epoch")
            if column_index == 0:
                ax.set_ylabel("Low-level L RSA Fisher mean r")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        columnspacing=1.2,
        handlelength=2.2,
        handletextpad=0.5,
    )
    fig.text(
        0.5,
        0.005,
        (
            "Fixed 17-participant cohort and 30 shared videos; raw 512-D Obs "
            "features; no PoE, seed selection, or selection across MAE epochs."
        ),
        ha="center",
        va="bottom",
        fontsize=6,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        bottom=0.14,
        top=0.86,
        wspace=0.22,
        hspace=0.28,
    )
    export_figure(fig, destination)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    videos = pd.read_csv(VIDEOS_CSV)
    exact35 = [
        (int(row.session), int(row.objective_id))
        for row in videos.itertuples()
    ]
    if len(exact35) != 35 or len(set(exact35)) != 35:
        raise RuntimeError(f"expected 35 unique video keys, got {len(exact35)}")

    per_subject, summary, audit = compute_epochwise(
        args.feature_root, exact35
    )
    reproduction = validate_against_existing_qap(
        per_subject, args.output_root
    )
    per_subject_path = (
        args.output_root
        / "L_LOWLEVEL_OBS_RSA_EPOCHWISE_PER_SUBJECT_SOURCE_DATA.csv"
    )
    summary_path = (
        args.output_root / "L_LOWLEVEL_OBS_RSA_EPOCHWISE_SOURCE_DATA.csv"
    )
    atomic_csv(per_subject, per_subject_path)
    atomic_csv(summary, summary_path)

    protocol = {
        "status": "complete",
        "figure_contract": {
            "core_conclusion": (
                "Track how the raw Obs representation's rating geometry "
                "changes across MAE continual-pretraining epochs, separately "
                "from supervised PoE fusion and seed selection."
            ),
            "archetype": "quantitative grid",
            "backend": "Python matplotlib only",
            "exports": ["SVG", "PDF", "TIFF 600 DPI", "PNG 300 DPI"],
        },
        "n_initializations": 2,
        "epochs": list(EPOCHS),
        "n_subjects": 17,
        "n_common_videos": 30,
        "rating_sets": {
            name: list(indices) for name, indices in RATING_SETS.items()
        },
        "lowlevel_definition": {
            "feature": "raw epoch-specific Obs representation, 512 dimensions",
            "neural_rdm": "correlation distance across Obs dimensions",
            "rating_rdm": (
                "Euclidean distance after standardization by rating mean/std "
                "from the same fixed 17-participant in-cohort training set"
            ),
            "subject_statistic": (
                "Pearson or Spearman correlation between 435 upper-triangle "
                "video-pair distances"
            ),
            "group_statistic": "Fisher-z mean over the 17 subjects",
            "poe": "not used",
            "seed_selection": "not applicable",
            "epoch_selection": "none",
        },
        "row_counts": {
            "per_subject": int(len(per_subject)),
            "summary": int(len(summary)),
        },
        "reproduction_audit": reproduction,
        "data_alignment_audit": audit,
        "feature_root": str(args.feature_root),
        "per_subject_csv": str(per_subject_path),
        "summary_csv": str(summary_path),
    }
    atomic_json(
        protocol,
        args.output_root / "L_LOWLEVEL_OBS_RSA_EPOCHWISE_PROTOCOL.json",
    )
    plot_summary(
        summary,
        args.output_root / "L_LOWLEVEL_OBS_RSA_EPOCHWISE",
    )
    print(summary.to_string(index=False))
    print(f"saved Low-level epochwise L-RSA to {args.output_root}")


if __name__ == "__main__":
    main()
