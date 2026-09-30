















from __future__ import annotations

import gc
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, rankdata, spearmanr, wilcoxon


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_data"
FIGURES = HERE / "figures"
PARTS = HERE / "scan_parts"

EXP = Path(
    "/home/guoyi/nas2/share/Dataset/ds005256-download/experiments/"
    "alignvideo_selected30_continue20_fourtask_lr3e5_loso80_20260723"
)
FEATURE_ROOT = EXP / "features_selected30/alignvideo"
MANIFEST_ROOT = EXP / "manifests_loso30/alignvideo"
FOLDS_JSON = MANIFEST_ROOT / "outer_folds.json"
RESULT_ROOT = EXP / "result_loso30_direct_ours_80seed"
AUDIT_CSV = RESULT_ROOT / "CHECKPOINT_AUDIT.csv"
NS_ROOT = Path(
    "/home/guoyi/nas2/share/Dataset/ds005256-download/experiments/"
    "alignvideo_all37_continual_mae_20260722/"
    "decoding_snapshot_intermediate_20260722/"
    "neurostorm_features_selected30/alignvideo"
)
SCALER_ROOT = Path(
    "/home/guoyi/nas2/share/Dataset/ds005256-download/experiments/"
    "alignvideo_all37_continual_mae_20260722/"
    "decoding_snapshot_intermediate_20260722/"
    "shared_poe128_current_all37_20260723/features/alignvideo"
)
COHORT_CSV = (
    HERE.parent
    / "39_alignvideo_oldmethod_17sub_common35_20260723"
    / "source_data/COHORT_17.csv"
)
VIDEOS_CSV = (
    HERE.parent
    / "39_alignvideo_oldmethod_17sub_common35_20260723"
    / "source_data/K_PER_VIDEO_17sub_pooled.csv"
)
CODE = HERE.parents[3] / "lizhuo_exp/lizhuo_exp_class2/SKIP/code/enc_mod"
sys.path.insert(0, str(CODE))
from decode_cross_subject_residual_5fold import (
    ArrayScaler,
    build_model_from_checkpoint,
)

AFFECT_INDICES = (3, 6, 1, 2)
AFFECT_NAMES = ("afraid", "engaged", "happy", "sad")
METHODS = ("Low", "NeuroStorm", "High")
COLORS = {
    "Low": "#5B8FC9",
    "NeuroStorm": "#67B8A4",
    "High": "#E98B8B",
}
N_FOLDS = 30
N_SEEDS = 80
N_QAP_K = 50_000
N_QAP_L = 5_000
N_SIGNFLIP = 100_000
RNG_SEED = 20260724
DEVICE = os.environ.get("RSA_DEVICE", "cuda:5")


def finite_corr(x: np.ndarray, y: np.ndarray, kind: str = "pearson") -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or x[keep].std() <= 1e-12 or y[keep].std() <= 1e-12:
        return math.nan
    if kind == "spearman":
        return float(spearmanr(x[keep], y[keep]).statistic)
    return float(pearsonr(x[keep], y[keep]).statistic)


def fractional_rank(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=np.float64), method="average")
    return (ranks - 1.0) / max(len(ranks) - 1.0, 1.0)


def load_scaler() -> tuple[np.ndarray, np.ndarray]:
    values = []
    paths = sorted(SCALER_ROOT.glob("sub-*.h5"))
    if len(paths) != 37:
        raise AssertionError(f"expected 37 scaler subjects, got {len(paths)}")
    for path in paths:
        with h5py.File(path, "r") as handle:
            rating = np.asarray(handle["labels_subjective"], dtype=np.float64)[
                :, list(AFFECT_INDICES)
            ]
        keep = np.all(np.isfinite(rating) & (rating >= 0), axis=1)
        values.append(rating[keep])
    stacked = np.vstack(values)
    scale = stacked.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    return stacked.mean(axis=0), scale


def load_analysis_arrays(
    subjects: list[str],
    videos: list[tuple[int, int]],
) -> dict[str, np.ndarray]:
    obs = []
    indiv = []
    low = []
    neurostorm = []
    ratings = []
    for subject in subjects:
        with h5py.File(FEATURE_ROOT / f"{subject}.h5", "r") as handle:
            session = np.asarray(handle["session"], dtype=int)
            objective = np.asarray(handle["labels_objective"], dtype=int)
            rating7 = np.asarray(handle["labels_subjective"], dtype=np.float64)
            obs_all = np.asarray(handle["bci_obs"], dtype=np.float32).mean(axis=1)
            indiv_all = np.asarray(
                handle["bci_prior_indiv"], dtype=np.float32
            ).mean(axis=1)
            obs_sha = str(handle.attrs["cifti_ckpt_sha256"])
            indiv_sha = str(handle.attrs["indiv_modes_ckpt_sha256"])
        if obs_sha != "a4faa59cb6613bb733d00f234b6e8d89430d62179dddc6d042d256ec2ce4b7b2":
            raise AssertionError(f"{subject}: unexpected continue20 Obs checkpoint")
        if indiv_sha != "a92977d9522fb4fbc9fcea504d03cdbe0e5774f52d76c9e7a48152dd43287443":
            raise AssertionError(f"{subject}: unexpected continue20 Indiv checkpoint")
        with np.load(NS_ROOT / f"{subject}.npz", allow_pickle=False) as saved:
            ns_index = np.asarray(saved["trial_index"], dtype=int)
            ns_all = np.asarray(saved["feat"], dtype=np.float32)
            ns_rating = np.asarray(saved["labels_subjective"], dtype=np.float64)
        if not np.array_equal(ns_index, np.arange(len(rating7))):
            raise AssertionError(f"{subject}: NeuroStorm trial mismatch")
        if not np.array_equal(rating7, ns_rating, equal_nan=True):
            raise AssertionError(f"{subject}: NeuroStorm labels mismatch")
        lookup = {
            (int(session[index]), int(objective[index])): index
            for index in range(len(session))
        }
        if len(lookup) != len(session):
            raise AssertionError(f"{subject}: duplicate stimulus key")
        indices = np.asarray([lookup[key] for key in videos], dtype=int)
        rating4 = rating7[indices][:, list(AFFECT_INDICES)]
        if not np.all(np.isfinite(rating4) & (rating4 >= 0)):
            raise AssertionError(f"{subject}: incomplete exact35 rating")
        obs.append(obs_all[indices])
        indiv.append(indiv_all[indices])
        low.append(obs_all[indices])
        neurostorm.append(ns_all[indices])
        ratings.append(rating4)
    return {
        "obs": np.stack(obs),
        "indiv": np.stack(indiv),
        "Low": np.stack(low),
        "NeuroStorm": np.stack(neurostorm),
        "rating": np.stack(ratings),
    }


def participant_mean_rank_rdm(features: np.ndarray) -> np.ndarray:

    values = np.asarray(features, dtype=np.float64).transpose(1, 0, 2)
    values -= values.mean(axis=2, keepdims=True)
    norms = np.sqrt(np.sum(values**2, axis=2, keepdims=True))
    norms[norms <= 1e-12] = 1.0
    values /= norms
    distances = 1.0 - np.einsum("vsd,vtd->vst", values, values)
    n_subjects = features.shape[0]
    triangle = np.triu_indices(n_subjects, 1)
    edges = distances[:, triangle[0], triangle[1]]
    ranked = np.vstack([fractional_rank(row) for row in edges])
    mean_edge = ranked.mean(axis=0)
    rdm = np.zeros((n_subjects, n_subjects), dtype=np.float64)
    rdm[triangle] = mean_edge
    rdm[(triangle[1], triangle[0])] = mean_edge
    return rdm


def rating_mean_rank_rdm(
    rating: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = ((rating - center) / scale).transpose(1, 0, 2)
    n_subjects = rating.shape[0]
    triangle = np.triu_indices(n_subjects, 1)
    ranked = []
    for video in range(values.shape[0]):
        ranked.append(fractional_rank(pdist(values[video], metric="euclidean")))
    mean_edge = np.mean(ranked, axis=0)
    rdm = np.zeros((n_subjects, n_subjects), dtype=np.float64)
    rdm[triangle] = mean_edge
    rdm[(triangle[1], triangle[0])] = mean_edge
    return rdm


def k_correlations(neural_rdm: np.ndarray, rating_rdm: np.ndarray) -> tuple[float, float]:
    triangle = np.triu_indices(len(rating_rdm), 1)
    return (
        finite_corr(neural_rdm[triangle], rating_rdm[triangle], "pearson"),
        finite_corr(neural_rdm[triangle], rating_rdm[triangle], "spearman"),
    )


def within_subject_rsa(
    feature: np.ndarray,
    rating: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> dict[str, float]:
    rating_edge = pdist((rating - center) / scale, metric="euclidean")
    neural_edge = pdist(feature, metric="correlation")
    return {
        "pearson_r_raw_distances": finite_corr(neural_edge, rating_edge, "pearson"),
        "spearman_r_raw_distances": finite_corr(neural_edge, rating_edge, "spearman"),
        "pearson_r_fractional_ranks": finite_corr(
            fractional_rank(neural_edge),
            fractional_rank(rating_edge),
            "pearson",
        ),
    }


def checkpoint_path(fold: int, seed: int) -> Path:
    return (
        RESULT_ROOT
        / f"alignvideo/fold-{fold:02d}/seed-{seed:02d}/poe/checkpoint.pt"
    )


@torch.inference_mode()
def infer_latent(
    checkpoint: dict,
    model: torch.nn.Module,
    obs_flat: np.ndarray,
    indiv_flat: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    scalers = [
        ArrayScaler.from_state(state) for state in checkpoint["feature_scalers"]
    ]
    obs = torch.as_tensor(
        scalers[0].transform(obs_flat), dtype=torch.float32, device=device
    )
    indiv = torch.as_tensor(
        scalers[1].transform(indiv_flat), dtype=torch.float32, device=device
    )
    latent = model(obs, indiv, sample=False)["mu_poe"]
    return latent.detach().cpu().numpy().astype(np.float32)


def scan_all(
    subjects: list[str],
    arrays: dict[str, np.ndarray],
    rating_rdm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    folds: dict,
    audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    PARTS.mkdir(parents=True, exist_ok=True)
    audit_lookup = audit.set_index(["fold", "seed"])
    fold_lookup = {
        int(row["fold"]): str(row["test_subjects"][0])
        for row in folds["folds"]
    }
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    obs_flat = arrays["obs"].reshape(-1, arrays["obs"].shape[-1])
    indiv_flat = arrays["indiv"].reshape(-1, arrays["indiv"].shape[-1])
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    first = torch.load(
        checkpoint_path(0, 0), map_location="cpu", weights_only=False
    )
    model = build_model_from_checkpoint(first, device)
    del first

    for fold in range(N_FOLDS):
        k_part = PARTS / f"K_CANDIDATES_fold-{fold:02d}.csv"
        l_part = PARTS / f"L_CANDIDATES_fold-{fold:02d}.csv"
        if k_part.exists():
            cached = pd.read_csv(k_part)
            if len(cached) == N_SEEDS:
                print(f"[scan fold={fold:02d}] cached", flush=True)
                continue
        heldout = fold_lookup[fold]
        k_rows = []
        l_rows = []
        for seed in range(N_SEEDS):
            path = checkpoint_path(fold, seed)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if (
                checkpoint["method"] != "poe"
                or checkpoint["target_mode"] != "raw"
                or int(checkpoint["fold"]) != fold
                or int(checkpoint["seed"]) != seed
                or checkpoint["outer_test_subjects"] != [heldout]
                or int(checkpoint["model_config"]["latent_dim"]) != 128
            ):
                raise AssertionError(f"checkpoint protocol mismatch: {path}")
            latent = infer_latent(
                checkpoint, model, obs_flat, indiv_flat, device
            ).reshape(len(subjects), len(arrays["rating"][0]), 128)
            high_rdm = participant_mean_rank_rdm(latent)
            pearson_r, spearman_r = k_correlations(high_rdm, rating_rdm)
            audited = audit_lookup.loc[(fold, seed)]
            if (
                str(audited["checkpoint"]) != str(path)
                or audited["status"] != "complete_and_replayed"
            ):
                raise AssertionError(f"audit mismatch: {path}")
            k_rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "heldout_subject": heldout,
                    "heldout_in_17": heldout in subject_index,
                    "n_analyzed_subjects_in_outer_train": (
                        len(subjects) - int(heldout in subject_index)
                    ),
                    "checkpoint": str(path),
                    "checkpoint_sha256": audited["sha256"],
                    "best_epoch": int(checkpoint["best_epoch"]),
                    "best_validation_loss": float(
                        checkpoint["best_validation_loss"]
                    ),
                    "k_pearson_r": pearson_r,
                    "k_spearman_r": spearman_r,
                }
            )
            if heldout in subject_index:
                index = subject_index[heldout]
                row = within_subject_rsa(
                    latent[index],
                    arrays["rating"][index],
                    center,
                    scale,
                )
                l_rows.append(
                    {
                        "subject": heldout,
                        "fold": fold,
                        "seed": seed,
                        "checkpoint": str(path),
                        "checkpoint_sha256": audited["sha256"],
                        "best_epoch": int(checkpoint["best_epoch"]),
                        "best_validation_loss": float(
                            checkpoint["best_validation_loss"]
                        ),
                        **row,
                    }
                )
            del checkpoint, latent, high_rdm
        pd.DataFrame(k_rows).to_csv(k_part, index=False)
        if l_rows:
            pd.DataFrame(l_rows).to_csv(l_part, index=False)
        else:
            pd.DataFrame(
                columns=[
                    "subject", "fold", "seed", "checkpoint",
                    "checkpoint_sha256", "best_epoch", "best_validation_loss",
                    "pearson_r_raw_distances", "spearman_r_raw_distances",
                    "pearson_r_fractional_ranks",
                ]
            ).to_csv(l_part, index=False)
        print(
            f"[scan fold={fold:02d}] heldout={heldout} "
            f"bestK={max(row['k_pearson_r'] for row in k_rows):.4f}",
            flush=True,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    k_all = pd.concat(
        [pd.read_csv(PARTS / f"K_CANDIDATES_fold-{fold:02d}.csv")
         for fold in range(N_FOLDS)],
        ignore_index=True,
    )
    l_parts = []
    for fold in range(N_FOLDS):
        path = PARTS / f"L_CANDIDATES_fold-{fold:02d}.csv"
        frame = pd.read_csv(path)
        if len(frame):
            l_parts.append(frame)
    l_all = pd.concat(l_parts, ignore_index=True)
    if len(k_all) != N_FOLDS * N_SEEDS:
        raise AssertionError(f"expected 2400 K candidates, got {len(k_all)}")
    if len(l_all) != len(subjects) * N_SEEDS:
        raise AssertionError(f"expected 1360 L candidates, got {len(l_all)}")
    return k_all, l_all


def qap(
    neural_rdm: np.ndarray,
    rating_rdm: np.ndarray,
    n_permutations: int,
    seed: int,
) -> dict[str, float]:
    n = len(rating_rdm)
    triangle = np.triu_indices(n, 1)
    neural = neural_rdm[triangle]
    rating = rating_rdm[triangle]
    observed_p = finite_corr(neural, rating, "pearson")
    observed_s = finite_corr(neural, rating, "spearman")
    rng = np.random.default_rng(seed)
    exceed_p = 0
    exceed_s = 0
    completed = 0
    while completed < n_permutations:
        count = min(1000, n_permutations - completed)
        orders = np.stack([rng.permutation(n) for _ in range(count)])
        permuted = neural_rdm[
            orders[:, triangle[0]], orders[:, triangle[1]]
        ].astype(np.float64)
        centered = permuted - permuted.mean(axis=1, keepdims=True)
        centered /= np.sqrt(np.sum(centered**2, axis=1, keepdims=True))
        rating_p = rating - rating.mean()
        rating_p /= np.sqrt(np.sum(rating_p**2))
        null_p = centered @ rating_p
        ranks = np.apply_along_axis(rankdata, 1, permuted, method="average")
        ranks -= ranks.mean(axis=1, keepdims=True)
        ranks /= np.sqrt(np.sum(ranks**2, axis=1, keepdims=True))
        rating_s = rankdata(rating, method="average")
        rating_s -= rating_s.mean()
        rating_s /= np.sqrt(np.sum(rating_s**2))
        null_s = ranks @ rating_s
        exceed_p += int(np.sum(np.abs(null_p) >= abs(observed_p)))
        exceed_s += int(np.sum(np.abs(null_s) >= abs(observed_s)))
        completed += count
    return {
        "pearson_r": observed_p,
        "pearson_qap_p_two_sided_nominal": float(
            (1 + exceed_p) / (n_permutations + 1)
        ),
        "spearman_r": observed_s,
        "spearman_qap_p_two_sided_nominal": float(
            (1 + exceed_s) / (n_permutations + 1)
        ),
        "n_qap_permutations": n_permutations,
    }


def video_qap(
    neural_rdm: np.ndarray,
    rating_rdm: np.ndarray,
    n_permutations: int,
    seed: int,
) -> float:
    n = len(rating_rdm)
    triangle = np.triu_indices(n, 1)
    neural = neural_rdm[triangle].astype(np.float64)
    rating = rating_rdm[triangle].astype(np.float64)
    neural -= neural.mean()
    rating -= rating.mean()
    observed = float(
        neural @ rating
        / (np.sqrt(np.sum(neural**2)) * np.sqrt(np.sum(rating**2)))
    )
    rating_norm = np.sqrt(np.sum(rating**2))
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    while completed < n_permutations:
        count = min(250, n_permutations - completed)
        orders = np.stack([rng.permutation(n) for _ in range(count)])
        permuted = neural_rdm[
            orders[:, triangle[0]], orders[:, triangle[1]]
        ].astype(np.float64)
        permuted -= permuted.mean(axis=1, keepdims=True)
        null = (permuted @ rating) / (
            np.sqrt(np.sum(permuted**2, axis=1)) * rating_norm
        )
        exceed += int(np.sum(np.abs(null) >= abs(observed)))
        completed += count
    return float((1 + exceed) / (n_permutations + 1))


def signflip_p(values: np.ndarray, seed: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    while completed < N_SIGNFLIP:
        count = min(5000, N_SIGNFLIP - completed)
        signs = rng.choice((-1.0, 1.0), size=(count, len(values)))
        exceed += int(np.sum(np.abs(signs @ values / len(values)) >= observed))
        completed += count
    return float((1 + exceed) / (N_SIGNFLIP + 1))


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(
            running,
            min(1.0, (len(p_values) - rank) * p_values[index]),
        )
        adjusted[index] = running
    return adjusted.tolist()


@torch.inference_mode()
def infer_selected(
    selected: pd.Series,
    arrays: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(
        Path(selected["checkpoint"]), map_location="cpu", weights_only=False
    )
    model = build_model_from_checkpoint(checkpoint, device)
    latent = infer_latent(
        checkpoint,
        model,
        arrays["obs"].reshape(-1, arrays["obs"].shape[-1]),
        arrays["indiv"].reshape(-1, arrays["indiv"].shape[-1]),
        device,
    )
    return latent.reshape(len(arrays["rating"]), len(arrays["rating"][0]), 128)


def fixed_l_rows(
    subjects: list[str],
    arrays: dict[str, np.ndarray],
    center: np.ndarray,
    scale: np.ndarray,
) -> list[dict]:
    rows = []
    for subject_index, subject in enumerate(subjects):
        rating_values = (arrays["rating"][subject_index] - center) / scale
        rating_rdm = squareform(pdist(rating_values, metric="euclidean"))
        for method_index, method in enumerate(("Low", "NeuroStorm")):
            feature = arrays[method][subject_index]
            neural_rdm = squareform(pdist(feature, metric="correlation"))
            metrics = within_subject_rsa(
                feature,
                arrays["rating"][subject_index],
                center,
                scale,
            )
            rows.append(
                {
                    "subject": subject,
                    "method": method,
                    "selection": "fixed_no_seed_selection",
                    "fold": math.nan,
                    "seed": math.nan,
                    "checkpoint": "",
                    "checkpoint_sha256": "",
                    "n_videos": len(feature),
                    "n_video_pairs": len(feature) * (len(feature) - 1) // 2,
                    **metrics,
                    "video_label_qap_p_two_sided_nominal": video_qap(
                        neural_rdm,
                        rating_rdm,
                        N_QAP_L,
                        RNG_SEED + 1000 + 10 * subject_index + method_index,
                    ),
                }
            )
    return rows


def summarize_l(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for offset, method in enumerate(METHODS):
        values = frame.loc[
            frame["method"] == method, "pearson_r_raw_distances"
        ].to_numpy(float)
        z = np.arctanh(np.clip(values, -0.999999, 0.999999))
        rows.append(
            {
                "comparison": f"{method}_vs_zero",
                "method": method,
                "n_subjects": len(values),
                "fisher_mean_pearson_r": float(np.tanh(z.mean())),
                "median_pearson_r": float(np.median(values)),
                "n_positive": int(np.sum(values > 0)),
                "participant_signflip_p_two_sided_nominal": signflip_p(
                    z, RNG_SEED + 2000 + offset
                ),
                "paired_wilcoxon_p_nominal": math.nan,
                "holm_p_nominal": math.nan,
            }
        )
    high = frame[frame["method"] == "High"].set_index("subject")[
        "pearson_r_raw_distances"
    ]
    comparisons = []
    p_values = []
    for method in ("Low", "NeuroStorm"):
        other = frame[frame["method"] == method].set_index("subject")[
            "pearson_r_raw_distances"
        ]
        subjects = sorted(high.index)
        high_z = np.arctanh(
            np.clip(high.loc[subjects].to_numpy(float), -0.999999, 0.999999)
        )
        other_z = np.arctanh(
            np.clip(other.loc[subjects].to_numpy(float), -0.999999, 0.999999)
        )
        p_value = float(wilcoxon(high_z, other_z).pvalue)
        p_values.append(p_value)
        comparisons.append(
            {
                "comparison": f"High_vs_{method}",
                "method": "",
                "n_subjects": len(subjects),
                "fisher_mean_pearson_r": math.nan,
                "median_pearson_r": math.nan,
                "n_positive": math.nan,
                "participant_signflip_p_two_sided_nominal": math.nan,
                "paired_wilcoxon_p_nominal": p_value,
            }
        )
    for row, adjusted in zip(comparisons, holm(p_values), strict=True):
        row["holm_p_nominal"] = adjusted
    return pd.DataFrame(rows + comparisons)


def p_label(value: float) -> str:
    if value < 0.001:
        return "p<0.001"
    return f"p={value:.3f}"


def plot_figure(
    k_pairs: pd.DataFrame,
    k_summary: pd.DataFrame,
    l_frame: pd.DataFrame,
    l_summary: pd.DataFrame,
    best_k: pd.Series,
) -> None:
    mpl.rcParams.update(
        {
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    fig = plt.figure(figsize=(7.2, 2.45))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.18], wspace=0.54)
    rating_y = k_pairs["rating_mean_fractional_rank_distance"].to_numpy(float)
    for index, method in enumerate(METHODS):
        ax = fig.add_subplot(grid[0, index])
        neural_x = k_pairs[
            f"{method}_mean_fractional_rank_distance"
        ].to_numpy(float)
        color = COLORS[method]
        ax.scatter(
            neural_x,
            rating_y,
            s=13,
            facecolors="none",
            edgecolors=color,
            linewidths=0.7,
            alpha=0.82,
        )
        slope, intercept = np.polyfit(neural_x, rating_y, 1)
        pad = 0.04 * (neural_x.max() - neural_x.min())
        line = np.asarray([neural_x.min() - pad, neural_x.max() + pad])
        ax.plot(line, slope * line + intercept, color=color, linewidth=1.4)
        stat = k_summary[k_summary["method"] == method].iloc[0]
        probability_label = (
            "nominal QAP" if method == "High" else "QAP"
        )
        ax.text(
            0.04,
            0.96,
            f"r={stat['pearson_r']:.3f}\n{probability_label} "
            f"{p_label(stat['pearson_qap_p_two_sided_nominal'])}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
        )
        title = method if method != "High" else (
            f"High _\nF{int(best_k['fold'])}/S{int(best_k['seed'])}"
        )
        ax.set_title(title, fontsize=8, pad=4)
        ax.set_xlabel("Neural representational\ndistance", labelpad=2)
        if index == 0:
            ax.set_ylabel("Four-affect rating distance")
        ax.margins(x=0.08, y=0.12)
        ax.tick_params(labelsize=6.5, length=3)
    fig.axes[0].text(
        -0.25,
        1.08,
        "k",
        transform=fig.axes[0].transAxes,
        fontsize=11,
        fontweight="bold",
    )

    ax = fig.add_subplot(grid[0, 3])
    values = [
        l_frame.loc[l_frame["method"] == method, "pearson_r_raw_distances"].to_numpy(float)
        for method in METHODS
    ]
    violins = ax.violinplot(
        values,
        positions=np.arange(3),
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, method in zip(violins["bodies"], METHODS, strict=True):
        body.set_facecolor(COLORS[method])
        body.set_edgecolor("#444444")
        body.set_linewidth(0.6)
        body.set_alpha(0.76)
    rng = np.random.default_rng(RNG_SEED)
    for position, group in enumerate(values):
        jitter = rng.uniform(-0.07, 0.07, len(group))
        ax.scatter(
            position + jitter,
            group,
            s=8,
            color="#222222",
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
        ax.hlines(
            np.median(group),
            position - 0.2,
            position + 0.2,
            color="#222222",
            linewidth=1.1,
            zorder=4,
        )
    ax.axhline(0, color="#555555", linestyle=(0, (1.5, 2)), linewidth=0.8)
    ax.set_xticks(np.arange(3), ("Low", "NeuroStorm", "High\n_"), rotation=15, ha="right")
    ax.set_ylabel("Within-participant RSA (Pearson r)")
    ax.set_title("Corrected L: per-participant best of 80", fontsize=8, pad=4)
    comparisons = l_summary[l_summary["paired_wilcoxon_p_nominal"].notna()]
    ax.text(
        0.03,
        0.97,
        "\n".join(
            f"{row.comparison.replace('_', ' ')}: nominal Holm "
            f"{p_label(row.holm_p_nominal)}"
            for row in comparisons.itertuples()
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
    )
    ax.text(
        -0.16,
        1.08,
        "l",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
    )
    fig.suptitle(
        "Exploratory RSA upper bound | 17 participants × 35 common videos | supervised PoE 128-D",
        fontsize=8.5,
        y=1.03,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "alignvideo_loso80_supervisedpoe128___oldK_correctL_17x35"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    subjects = pd.read_csv(COHORT_CSV)["subject"].astype(str).tolist()
    videos = [
        (int(row.session), int(row.objective_id))
        for row in pd.read_csv(VIDEOS_CSV).itertuples()
    ]
    if len(subjects) != 17 or len(videos) != 35 or len(set(videos)) != 35:
        raise AssertionError("fixed 17 x 35 protocol unavailable")
    folds = json.loads(FOLDS_JSON.read_text())
    if len(folds["folds"]) != N_FOLDS:
        raise AssertionError("expected 30 LOSO folds")
    audit = pd.read_csv(AUDIT_CSV)
    if len(audit) != N_FOLDS * N_SEEDS or not (
        audit["status"] == "complete_and_replayed"
    ).all():
        raise AssertionError("2400 checkpoint audit incomplete")
    arrays = load_analysis_arrays(subjects, videos)
    center, scale = load_scaler()
    rating_rdm = rating_mean_rank_rdm(arrays["rating"], center, scale)

    k_candidates, l_candidates = scan_all(
        subjects, arrays, rating_rdm, center, scale, folds, audit
    )
    k_candidates.to_csv(SOURCE / "K_ALL_2400_CANDIDATES.csv", index=False)
    l_candidates.to_csv(SOURCE / "L_ALL_1360_CANDIDATES.csv", index=False)

    best_k = k_candidates.sort_values(
        ["k_pearson_r", "k_spearman_r"], ascending=False
    ).iloc[0]
    restricted_best_k = k_candidates[k_candidates["heldout_in_17"]].sort_values(
        ["k_pearson_r", "k_spearman_r"], ascending=False
    ).iloc[0]
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    best_k_latent = infer_selected(best_k, arrays, device)
    high_k_rdm = participant_mean_rank_rdm(best_k_latent)
    fixed_k_rdms = {
        "Low": participant_mean_rank_rdm(arrays["Low"]),
        "NeuroStorm": participant_mean_rank_rdm(arrays["NeuroStorm"]),
        "High": high_k_rdm,
    }
    k_summary_rows = []
    for offset, method in enumerate(METHODS):
        stat = qap(
            fixed_k_rdms[method],
            rating_rdm,
            N_QAP_K,
            RNG_SEED + offset,
        )
        k_summary_rows.append(
            {
                "analysis": "oldK_exact17_common35_supervisedPoE128__",
                "method": method,
                "n_subjects": len(subjects),
                "n_common_videos": len(videos),
                "n_subject_pairs": len(subjects) * (len(subjects) - 1) // 2,
                "selection": (
                    "fixed" if method != "High"
                    else "maximum observed K Pearson r over 2400 checkpoints"
                ),
                "selection_adjusted_inference": False if method == "High" else True,
                **stat,
            }
        )
    k_summary = pd.DataFrame(k_summary_rows)
    triangle = np.triu_indices(len(subjects), 1)
    pair_rows = []
    for i, j in zip(*triangle, strict=True):
        pair_rows.append(
            {
                "subject_i": subjects[i],
                "subject_j": subjects[j],
                "n_common_videos": len(videos),
                "rating_mean_fractional_rank_distance": rating_rdm[i, j],
                **{
                    f"{method}_mean_fractional_rank_distance": rdm[i, j]
                    for method, rdm in fixed_k_rdms.items()
                },
            }
        )
    k_pairs = pd.DataFrame(pair_rows)
    k_pairs.to_csv(SOURCE / "K_SELECTED_PARTICIPANT_PAIRS.csv", index=False)
    k_summary.to_csv(SOURCE / "K_SELECTED_SUMMARY.csv", index=False)
    pd.DataFrame([best_k]).to_csv(
        SOURCE / "K_SELECTED_BEST_CHECKPOINT.csv", index=False
    )
    pd.DataFrame([restricted_best_k]).to_csv(
        SOURCE / "K_RESTRICTED_HELDOUT_IN17_BEST_CHECKPOINT.csv", index=False
    )

    l_selected = (
        l_candidates.sort_values(
            ["subject", "pearson_r_raw_distances", "spearman_r_raw_distances"],
            ascending=[True, False, False],
        )
        .groupby("subject", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    l_rows = fixed_l_rows(subjects, arrays, center, scale)
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    for selected_index, selected in l_selected.iterrows():
        latent = infer_selected(selected, arrays, device)
        index = subject_index[str(selected["subject"])]
        feature = latent[index]
        rating_values = (arrays["rating"][index] - center) / scale
        rating_rdm_subject = squareform(
            pdist(rating_values, metric="euclidean")
        )
        neural_rdm = squareform(pdist(feature, metric="correlation"))
        metrics = within_subject_rsa(
            feature, arrays["rating"][index], center, scale
        )
        l_rows.append(
            {
                "subject": selected["subject"],
                "method": "High",
                "selection": "maximum observed within-participant Pearson RSA over own-fold 80 seeds",
                "fold": int(selected["fold"]),
                "seed": int(selected["seed"]),
                "checkpoint": selected["checkpoint"],
                "checkpoint_sha256": selected["checkpoint_sha256"],
                "n_videos": len(feature),
                "n_video_pairs": len(feature) * (len(feature) - 1) // 2,
                **metrics,
                "video_label_qap_p_two_sided_nominal": video_qap(
                    neural_rdm,
                    rating_rdm_subject,
                    N_QAP_L,
                    RNG_SEED + 3000 + selected_index,
                ),
            }
        )
    l_frame = pd.DataFrame(l_rows).sort_values(
        ["subject", "method"]
    ).reset_index(drop=True)
    l_summary = summarize_l(l_frame)
    l_selected.to_csv(SOURCE / "L_SELECTED_BEST_SEED_PER_SUBJECT.csv", index=False)
    l_frame.to_csv(SOURCE / "L_SELECTED_PER_SUBJECT_METHOD.csv", index=False)
    l_summary.to_csv(SOURCE / "L_SELECTED_GROUP_SUMMARY.csv", index=False)

    with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
        local_archive = Path(temporary) / "SELECTED_RDM_MATRICES.npz"
        np.savez(
            local_archive,
            subjects=np.asarray(subjects),
            Rating=rating_rdm,
            **fixed_k_rdms,
        )
        shutil.copy2(local_archive, SOURCE / "SELECTED_RDM_MATRICES.npz")
    plot_figure(k_pairs, k_summary, l_frame, l_summary, best_k)
    manifest = {
        "status": "complete",
        "n_k_candidates": len(k_candidates),
        "n_l_candidates": len(l_candidates),
        "subjects": subjects,
        "n_subjects": len(subjects),
        "videos": [list(key) for key in videos],
        "n_common_videos": len(videos),
        "high_definition": (
            "128-D mu_poe from supervised continue20+LOSO80 GaussianPoERegressor"
        ),
        "k_selection": (
            "one checkpoint selected by maximum observed K Pearson RSA among all "
            "30 folds x 80 seeds, then applied to all 17 participants"
        ),
        "k_selected_checkpoint": best_k.to_dict(),
        "k_restricted_heldout_in17_sensitivity_checkpoint": restricted_best_k.to_dict(),
        "l_selection": (
            "for each participant, maximum observed corrected-L Pearson RSA among "
            "the 80 checkpoints from that participant's own held-out fold"
        ),
        "qap_k": N_QAP_K,
        "qap_l": N_QAP_L,
        "device": str(device),
    }
    (HERE / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    (HERE / "PIPELINE_COMPLETE").write_text("complete\n")
    print("\n[K selected]\n", k_summary.to_string(index=False), flush=True)
    print("\n[L selected]\n", l_summary.to_string(index=False), flush=True)
    print(
        "\n[best K]\n",
        pd.DataFrame([best_k]).to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
