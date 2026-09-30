
















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
import torch
from scipy.spatial.distance import pdist

import compute_plot_alignvideo_epochwise_lowlevel_l_rsa as low
from decode_narratives_shortvideo_poe import GaussianPoERegressor


EXP = Path(
    "/home/guoyi/nas2/share/Dataset/ds005256-download/experiments/"
    "alignvideo_mae_epochwise_rsa17_two_init_20260724"
)
DEFAULT_FEATURE_ROOT = EXP / "epochwise_features_incohort17"
DEFAULT_RUN_ROOT = EXP / "poe5seed_rsa17_incohort/runs"
DEFAULT_OUTPUT_ROOT = EXP / "figures_epochwise_k_l_rsa17"
SEEDS = tuple(range(5))

REPRESENTATION_LABELS = {
    "raw_obs": "Raw Obs (512D)",
    "obs_expert_prepoe": "Obs expert post-MLP / pre-PoE (128D)",
}
COLORS = low.COLORS
INITIALIZATION_LABELS = {
    "crossdataset": "Cross-dataset init.",
    "random": "Random init.",
}
LINESTYLES = {"raw_obs": "--", "obs_expert_prepoe": "-"}
MARKERS = {"raw_obs": "s", "obs_expert_prepoe": "o"}


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_model(checkpoint: dict) -> GaussianPoERegressor:
    config = checkpoint["model_config"]
    model = GaussianPoERegressor(
        int(config["input_dim"]),
        int(config["latent_dim"]),
        int(config["hidden"]),
        int(config["d_out"]),
        float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


@torch.inference_mode()
def extract_mu_obs(
    obs30: np.ndarray,
    checkpoint: dict,
) -> np.ndarray:
    scaler = checkpoint["obs_scaler"]
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    standardized = (
        (obs30.astype(np.float32) - mean[None, None, :])
        / std[None, None, :]
    )
    flat = torch.as_tensor(
        standardized.reshape(-1, standardized.shape[-1]),
        dtype=torch.float32,
    )
    model = build_model(checkpoint)
    mu_obs, _ = model.obs_expert(flat)
    return (
        mu_obs.cpu()
        .numpy()
        .astype(np.float64)
        .reshape(obs30.shape[0], obs30.shape[1], -1)
    )


def validate_checkpoint(
    checkpoint: dict,
    *,
    initialization: str,
    epoch: int,
    seed: int,
    feature_path: Path,
) -> None:
    expected = {
        "initialization": initialization,
        "mae_epoch": epoch,
        "fusion": "gaussian_poe128",
        "seed": seed,
        "poe_training_regime": "incohort17",
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(
                f"{key} mismatch: expected {value!r}, "
                f"got {checkpoint.get(key)!r}"
            )
    config = checkpoint["model_config"]
    if (
        int(config["input_dim"]) != 512
        or int(config["latent_dim"]) != 128
        or int(config["d_out"]) != 7
    ):
        raise RuntimeError(f"unexpected model config: {config}")
    recorded = Path(checkpoint["feature_source"]).resolve()
    if recorded != feature_path.resolve():
        raise RuntimeError(
            f"checkpoint feature source mismatch: {recorded} != {feature_path}"
        )
    if list(checkpoint["rsa_subjects"]) != low.RSA_SUBJECTS:
        raise RuntimeError("checkpoint RSA cohort/order mismatch")


def rating_scalers(
    checkpoint: dict,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output = {}
    for rating_set in low.RATING_SETS:
        state = checkpoint["rating_rdm_scalers"][rating_set]
        output[rating_set] = (
            np.asarray(state["mean"], dtype=np.float64),
            np.asarray(state["std"], dtype=np.float64),
        )
    return output


def compute_candidates(
    feature_root: Path,
    run_root: Path,
    exact35: list[tuple[int, int]],
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    scaler_audit: dict[str, dict] = {}
    label_reference: np.ndarray | None = None
    for initialization in low.INITIALIZATIONS:
        for epoch in low.EPOCHS:
            feature_path = (
                feature_root / initialization / f"epoch_{epoch:03d}.npz"
            )
            data = low.load_feature(feature_path)
            low.validate_feature_file(data, initialization, epoch)
            obs30, rating30, common_mask = low.select_exact30(data, exact35)
            if int(common_mask.sum()) != 30:
                raise RuntimeError("common-video mask changed")
            if label_reference is None:
                label_reference = rating30.copy()
            elif not np.array_equal(
                label_reference.astype(np.float32),
                rating30.astype(np.float32),
            ):
                raise RuntimeError("evaluation ratings changed across epochs")

            reference_scalers = None
            for seed in SEEDS:
                checkpoint_path = (
                    run_root / initialization / f"epoch_{epoch:03d}"
                    / "gaussian_poe128" / f"seed_{seed:02d}"
                    / "checkpoint.pt"
                )
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(checkpoint_path)
                checkpoint = load_checkpoint(checkpoint_path)
                validate_checkpoint(
                    checkpoint,
                    initialization=initialization,
                    epoch=epoch,
                    seed=seed,
                    feature_path=feature_path,
                )
                current_scalers = rating_scalers(checkpoint)
                if reference_scalers is None:
                    reference_scalers = current_scalers
                else:
                    for rating_set in low.RATING_SETS:
                        for current, reference in zip(
                            current_scalers[rating_set],
                            reference_scalers[rating_set],
                        ):
                            if not np.array_equal(current, reference):
                                raise RuntimeError(
                                    f"rating scaler changed across seeds: "
                                    f"{initialization}/epoch {epoch}/"
                                    f"{rating_set}"
                                )
                mu_obs = extract_mu_obs(obs30, checkpoint)
                if mu_obs.shape != (17, 30, 128):
                    raise RuntimeError(
                        f"unexpected mu_obs shape {mu_obs.shape}"
                    )
                if not np.isfinite(mu_obs).all():
                    raise RuntimeError("non-finite mu_obs")
                for subject_index, subject in enumerate(low.RSA_SUBJECTS):
                    neural_edge = pdist(
                        mu_obs[subject_index], metric="correlation"
                    )
                    for rating_set, indices in low.RATING_SETS.items():
                        center, scale = current_scalers[rating_set]
                        rating_edge = pdist(
                            (
                                rating30[subject_index][:, list(indices)]
                                - center
                            )
                            / scale,
                            metric="euclidean",
                        )
                        rows.append({
                            "initialization": initialization,
                            "mae_epoch": epoch,
                            "seed": seed,
                            "subject": subject,
                            "rating_set": rating_set,
                            "n_videos": 30,
                            "n_rdm_edges": int(len(neural_edge)),
                            "representation": "obs_expert_prepoe",
                            "representation_label": (
                                REPRESENTATION_LABELS[
                                    "obs_expert_prepoe"
                                ]
                            ),
                            "feature_dim": 128,
                            "pearson_r": low.finite_corr(
                                neural_edge, rating_edge, "pearson_r"
                            ),
                            "spearman_r": low.finite_corr(
                                neural_edge, rating_edge, "spearman_r"
                            ),
                            "checkpoint": str(checkpoint_path),
                            "feature_path": str(feature_path),
                        })
            assert reference_scalers is not None
            scaler_audit[
                f"{initialization}/epoch_{epoch:03d}"
            ] = {
                "all_5_seed_rating_scalers_identical": True,
            }
    candidates = pd.DataFrame(rows)
    expected = 2 * 21 * 5 * 17 * 2
    if len(candidates) != expected:
        raise RuntimeError(
            f"expected {expected} candidate rows, got {len(candidates)}"
        )
    return candidates, scaler_audit


def select_best_seed(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    grouping = [
        "initialization", "mae_epoch", "subject", "rating_set"
    ]
    for keys, frame in candidates.groupby(grouping, sort=True):
        for metric in ("pearson_r", "spearman_r"):
            row = (
                frame.sort_values(
                    [metric, "seed"],
                    ascending=[False, True],
                )
                .iloc[0]
                .to_dict()
            )
            row["selection_metric"] = metric
            row["selected_value"] = float(row[metric])
            row["selection"] = (
                "per-subject maximum post-MLP/pre-PoE L-RSA over "
                "5 supervised Gaussian-Obs-expert seeds; exploratory _"
            )
            selected_rows.append(row)
    selected = pd.DataFrame(selected_rows)
    expected = 2 * 21 * 17 * 2 * 2
    if len(selected) != expected:
        raise RuntimeError(
            f"expected {expected} selected rows, got {len(selected)}"
        )

    summary_rows = []
    for keys, frame in selected.groupby(
        ["initialization", "mae_epoch", "rating_set", "selection_metric"],
        sort=True,
    ):
        initialization, epoch, rating_set, metric = keys
        values = frame["selected_value"].to_numpy(dtype=float)
        summary_rows.append({
            "initialization": initialization,
            "mae_epoch": int(epoch),
            "rating_set": rating_set,
            "metric": metric,
            "n_subjects": int(len(values)),
            "fisher_mean_r": low.fisher_mean(values),
            "arithmetic_mean_r": float(np.mean(values)),
            "std_across_subjects": float(np.std(values, ddof=1)),
            "median_r": float(np.median(values)),
            "n_positive": int(np.sum(values > 0)),
            "representation": "obs_expert_prepoe",
            "representation_label": REPRESENTATION_LABELS[
                "obs_expert_prepoe"
            ],
            "feature_dim": 128,
            "fusion": "none; stopped before PoE",
            "seed_selection": (
                "per-subject maximum over 5 supervised Obs-expert seeds"
            ),
            "epoch_selection": "none",
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["rating_set", "metric", "initialization", "mae_epoch"]
    )
    if len(summary) != 168:
        raise RuntimeError(
            f"expected 168 summary rows, got {len(summary)}"
        )

    mean_seed_rows = []
    for keys, frame in candidates.groupby(grouping, sort=True):
        for metric in ("pearson_r", "spearman_r"):
            values = frame[metric].to_numpy(dtype=float)
            mean_seed_rows.append({
                "initialization": keys[0],
                "mae_epoch": int(keys[1]),
                "subject": keys[2],
                "rating_set": keys[3],
                "metric": metric,
                "n_seeds": int(len(values)),
                "seed_fisher_mean_r": low.fisher_mean(values),
            })
    seed_mean = pd.DataFrame(mean_seed_rows)
    return selected, summary, seed_mean


def build_comparison_source(
    post_mlp: pd.DataFrame,
    raw_path: Path,
) -> pd.DataFrame:
    raw = pd.read_csv(raw_path).copy()
    raw["representation"] = "raw_obs"
    raw["representation_label"] = REPRESENTATION_LABELS["raw_obs"]
    raw["feature_dim"] = 512
    common = [
        "initialization", "mae_epoch", "rating_set", "metric",
        "n_subjects", "fisher_mean_r", "arithmetic_mean_r",
        "std_across_subjects", "median_r", "n_positive",
        "representation", "representation_label", "feature_dim",
    ]
    comparison = pd.concat(
        [raw[common], post_mlp[common]],
        ignore_index=True,
    )
    expected = 2 * 168
    if len(comparison) != expected:
        raise RuntimeError(
            f"expected {expected} comparison rows, got {len(comparison)}"
        )
    return comparison.sort_values(
        [
            "rating_set", "metric", "initialization",
            "representation", "mae_epoch",
        ]
    )


def configure_plotting() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial", "Helvetica", "DejaVu Sans", "sans-serif"
        ],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })


def plot_post_mlp(
    summary: pd.DataFrame,
    destination: Path,
) -> None:
    configure_plotting()
    rating_order = tuple(low.RATING_SETS)
    metric_order = ("pearson_r", "spearman_r")
    fig, axes = plt.subplots(
        2, 2, figsize=(7.15, 4.85), sharex=True, sharey=True
    )
    for row_index, rating_set in enumerate(rating_order):
        for column_index, metric in enumerate(metric_order):
            ax = axes[row_index, column_index]
            panel = summary[
                (summary["rating_set"] == rating_set)
                & (summary["metric"] == metric)
            ]
            for initialization in low.INITIALIZATIONS:
                line = panel[
                    panel["initialization"] == initialization
                ].sort_values("mae_epoch")
                ax.plot(
                    line["mae_epoch"],
                    line["fisher_mean_r"],
                    color=COLORS[initialization],
                    linestyle="-",
                    marker="o",
                    markersize=3.0,
                    markerfacecolor="white",
                    markeredgewidth=0.7,
                    linewidth=1.15,
                    label=(
                        f"{INITIALIZATION_LABELS[initialization]} "
                        "Obs expert post-MLP / pre-PoE (128D)"
                    ),
                )
            ax.axhline(0, color="#777777", linewidth=0.75, zorder=0)
            ax.set_title(
                f"{low.RATING_LABELS[rating_set]} · "
                f"{low.METRIC_LABELS[metric]}",
                fontsize=7,
                pad=4,
            )
            ax.set_xticks([0, 5, 10, 15, 20])
            ax.grid(axis="y", color="#E7E7E7", linewidth=0.55)
            if row_index == 1:
                ax.set_xlabel("MAE continual-pretraining epoch")
            if column_index == 0:
                ax.set_ylabel("Obs-expert L RSA Fisher mean r")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        columnspacing=1.1,
        handlelength=2.2,
    )
    fig.text(
        0.5,
        0.005,
        (
            "Fixed 17-participant cohort and 30 shared videos; supervised "
            "Obs-expert MLP output before PoE; best of 5 seeds per participant "
            "within each epoch; no selection across MAE epochs."
        ),
        ha="center",
        va="bottom",
        fontsize=6,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.10, right=0.99, bottom=0.14, top=0.86,
        wspace=0.22, hspace=0.28,
    )
    low.export_figure(fig, destination)
    plt.close(fig)


def plot_raw_vs_post_mlp(
    comparison: pd.DataFrame,
    destination: Path,
) -> None:
    configure_plotting()
    rating_order = tuple(low.RATING_SETS)
    metric_order = ("pearson_r", "spearman_r")
    fig, axes = plt.subplots(
        2, 2, figsize=(7.15, 4.85), sharex=True, sharey=True
    )
    for row_index, rating_set in enumerate(rating_order):
        for column_index, metric in enumerate(metric_order):
            ax = axes[row_index, column_index]
            panel = comparison[
                (comparison["rating_set"] == rating_set)
                & (comparison["metric"] == metric)
            ]
            for initialization in low.INITIALIZATIONS:
                for representation in (
                    "raw_obs", "obs_expert_prepoe"
                ):
                    line = panel[
                        (panel["initialization"] == initialization)
                        & (panel["representation"] == representation)
                    ].sort_values("mae_epoch")
                    ax.plot(
                        line["mae_epoch"],
                        line["fisher_mean_r"],
                        color=COLORS[initialization],
                        linestyle=LINESTYLES[representation],
                        marker=MARKERS[representation],
                        markersize=2.8,
                        markerfacecolor="white",
                        markeredgewidth=0.65,
                        linewidth=1.1,
                        label=(
                            f"{INITIALIZATION_LABELS[initialization]} "
                            f"{REPRESENTATION_LABELS[representation]}"
                        ),
                    )
            ax.axhline(0, color="#777777", linewidth=0.75, zorder=0)
            ax.set_title(
                f"{low.RATING_LABELS[rating_set]} · "
                f"{low.METRIC_LABELS[metric]}",
                fontsize=7,
                pad=4,
            )
            ax.set_xticks([0, 5, 10, 15, 20])
            ax.grid(axis="y", color="#E7E7E7", linewidth=0.55)
            if row_index == 1:
                ax.set_xlabel("MAE continual-pretraining epoch")
            if column_index == 0:
                ax.set_ylabel("L RSA Fisher mean r")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        columnspacing=1.0,
        handlelength=2.2,
        fontsize=6.2,
    )
    fig.text(
        0.5,
        0.005,
        (
            "Raw Obs uses no seed selection; post-MLP/pre-PoE is supervised "
            "and uses the existing exploratory best-of-5 seed rule."
        ),
        ha="center",
        va="bottom",
        fontsize=6,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.10, right=0.99, bottom=0.14, top=0.84,
        wspace=0.22, hspace=0.28,
    )
    low.export_figure(fig, destination)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT
    )
    parser.add_argument(
        "--run-root", type=Path, default=DEFAULT_RUN_ROOT
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    videos = pd.read_csv(low.VIDEOS_CSV)
    exact35 = [
        (int(row.session), int(row.objective_id))
        for row in videos.itertuples()
    ]
    if len(exact35) != 35 or len(set(exact35)) != 35:
        raise RuntimeError("invalid exact35 video list")

    candidates, scaler_audit = compute_candidates(
        args.feature_root, args.run_root, exact35
    )
    selected, summary, seed_mean = select_best_seed(candidates)
    raw_path = (
        args.output_root / "L_LOWLEVEL_OBS_RSA_EPOCHWISE_SOURCE_DATA.csv"
    )
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"raw Low-level source is required for comparison: {raw_path}"
        )
    comparison = build_comparison_source(summary, raw_path)

    paths = {
        "candidates": (
            args.output_root
            / "L_OBS_EXPERT_PREPOE_ALL_SEED_CANDIDATES.csv"
        ),
        "selected": (
            args.output_root
            / "L_OBS_EXPERT_PREPOE_BEST5_PER_SUBJECT_SOURCE_DATA.csv"
        ),
        "summary": (
            args.output_root
            / "L_OBS_EXPERT_PREPOE_BEST5_EPOCHWISE_SOURCE_DATA.csv"
        ),
        "seed_mean": (
            args.output_root
            / "L_OBS_EXPERT_PREPOE_SEED_MEAN_PER_SUBJECT.csv"
        ),
        "comparison": (
            args.output_root
            / "L_RAW_OBS_VS_OBS_EXPERT_PREPOE_EPOCHWISE_SOURCE_DATA.csv"
        ),
    }
    low.atomic_csv(candidates, paths["candidates"])
    low.atomic_csv(selected, paths["selected"])
    low.atomic_csv(summary, paths["summary"])
    low.atomic_csv(seed_mean, paths["seed_mean"])
    low.atomic_csv(comparison, paths["comparison"])

    protocol = {
        "status": "complete",
        "representation": {
            "name": "Obs expert post-MLP / pre-PoE",
            "input": "checkpoint-standardized raw Obs, 512D",
            "transform": (
                "LayerNorm(512) -> Linear(512,256) -> GELU -> "
                "Dropout(disabled at inference) -> Linear mu(256,128)"
            ),
            "output": "mu_obs, 128D",
            "poe_or_indiv_used": False
        },
        "rsa": {
            "subjects": 17,
            "videos": 30,
            "video_pairs_per_subject": 435,
            "rating_sets": {
                key: list(value)
                for key, value in low.RATING_SETS.items()
            },
            "metrics": ["pearson_r", "spearman_r"],
            "group_aggregation": "Fisher-z mean over 17 subjects",
        },
        "selection": {
            "rule": (
                "per subject x epoch x initialization x rating set x metric, "
                "maximum L-RSA among Gaussian Obs-expert seeds 0-4; "
                "ties use smallest seed"
            ),
            "no_selection_across_mae_epochs": True,
        },
        "row_counts": {
            "candidate": int(len(candidates)),
            "selected": int(len(selected)),
            "summary": int(len(summary)),
            "seed_mean": int(len(seed_mean)),
            "comparison": int(len(comparison)),
        },
        "scaler_audit": scaler_audit,
        "files": {key: str(value) for key, value in paths.items()},
    }
    low.atomic_json(
        protocol,
        args.output_root / "L_OBS_EXPERT_PREPOE_PROTOCOL.json",
    )
    plot_post_mlp(
        summary,
        args.output_root / "L_OBS_EXPERT_PREPOE_RSA_EPOCHWISE",
    )
    plot_raw_vs_post_mlp(
        comparison,
        args.output_root
        / "L_RAW_OBS_VS_OBS_EXPERT_PREPOE_RSA_EPOCHWISE",
    )
    print(summary.to_string(index=False))
    print(f"saved post-MLP/pre-PoE L-RSA to {args.output_root}")


if __name__ == "__main__":
    main()
