















from __future__ import annotations

import importlib.util
import itertools
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "run_loso80_supervised_poe128__.py"
OUT = HERE / "affect45_pearson_spearman"
PARTS = OUT / "parts"
OUT.mkdir(parents=True, exist_ok=True)
PARTS.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("rsa44", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {BASE_SCRIPT}")
rsa44 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rsa44)

SETS = {
    "four_happy_sad_afraid_disgusted": (1, 2, 3, 4),
    "five_happy_sad_afraid_disgusted_engaged": (1, 2, 3, 4, 6),
}
CURRENT_K_FOLD = 23
CURRENT_K_SEED = 70


def finite_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return math.nan
    if kind == "pearson":
        return float(pearsonr(x[keep], y[keep]).statistic)
    return float(spearmanr(x[keep], y[keep]).statistic)


def load_rating_scaler(indices: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for path in sorted(rsa44.SCALER_ROOT.glob("sub-*.h5")):
        with h5py.File(path, "r") as handle:
            rating = np.asarray(handle["labels_subjective"], dtype=np.float64)[
                :, list(indices)
            ]
        keep = np.all(np.isfinite(rating) & (rating >= 0), axis=1)
        values.append(rating[keep])
    stacked = np.vstack(values)
    scale = stacked.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    return stacked.mean(axis=0), scale


def load_rating7(
    subjects: list[str], videos: list[tuple[int, int]]
) -> np.ndarray:
    rows = []
    for subject in subjects:
        with h5py.File(rsa44.FEATURE_ROOT / f"{subject}.h5", "r") as handle:
            session = np.asarray(handle["session"], dtype=int)
            objective = np.asarray(handle["labels_objective"], dtype=int)
            rating = np.asarray(handle["labels_subjective"], dtype=np.float64)
        lookup = {
            (int(session[index]), int(objective[index])): index
            for index in range(len(session))
        }
        rows.append(rating[np.asarray([lookup[key] for key in videos], dtype=int)])
    return np.stack(rows)


def rating_k_rdm(
    rating: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    values = ((rating - center) / scale).transpose(1, 0, 2)
    n_subjects = rating.shape[0]
    triangle = np.triu_indices(n_subjects, 1)
    ranked_edges = np.vstack(
        [
            rsa44.fractional_rank(pdist(video_values, metric="euclidean"))
            for video_values in values
        ]
    )
    mean_edge = ranked_edges.mean(axis=0)
    rdm = np.zeros((n_subjects, n_subjects), dtype=np.float64)
    rdm[triangle] = mean_edge
    rdm[(triangle[1], triangle[0])] = mean_edge
    return rdm


def l_corr(
    feature: np.ndarray,
    rating: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> tuple[float, float]:
    neural = pdist(feature, metric="correlation")
    behavior = pdist((rating - center) / scale, metric="euclidean")
    return (
        finite_corr(neural, behavior, "pearson"),
        finite_corr(neural, behavior, "spearman"),
    )


def infer_all(
    checkpoint: dict,
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    obs = arrays["obs"].reshape(-1, arrays["obs"].shape[-1])
    indiv = arrays["indiv"].reshape(-1, arrays["indiv"].shape[-1])
    return rsa44.infer_latent(checkpoint, model, obs, indiv, device).reshape(
        arrays["obs"].shape[0], arrays["obs"].shape[1], 128
    )


def scan(
    subjects: list[str],
    arrays: dict[str, np.ndarray],
    rating: dict[str, np.ndarray],
    common: np.ndarray,
    centers: dict[str, np.ndarray],
    scales: dict[str, np.ndarray],
    rating_k: dict[str, np.ndarray],
    folds: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    heldout = {
        int(row["fold"]): str(row["test_subjects"][0]) for row in folds["folds"]
    }
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    device = torch.device(rsa44.DEVICE if torch.cuda.is_available() else "cpu")
    first = torch.load(
        rsa44.checkpoint_path(0, 0), map_location="cpu", weights_only=False
    )
    model = rsa44.build_model_from_checkpoint(first, device)
    del first

    for fold in range(rsa44.N_FOLDS):
        k_path = PARTS / f"K_fold-{fold:02d}.csv"
        l_path = PARTS / f"L_fold-{fold:02d}.csv"
        if k_path.exists() and len(pd.read_csv(k_path)) == rsa44.N_SEEDS:
            print(f"[fold={fold:02d}] cached", flush=True)
            continue
        k_rows = []
        l_rows = []
        test_subject = heldout[fold]
        for seed in range(rsa44.N_SEEDS):
            path = rsa44.checkpoint_path(fold, seed)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            latent = infer_all(checkpoint, model, arrays, device)
            neural_k = rsa44.participant_mean_rank_rdm(latent[:, common])
            row = {
                "fold": fold,
                "seed": seed,
                "heldout_subject": test_subject,
                "checkpoint": str(path),
            }
            for name in SETS:
                pearson_value, spearman_value = rsa44.k_correlations(
                    neural_k, rating_k[name]
                )
                row[f"{name}_pearson"] = pearson_value
                row[f"{name}_spearman"] = spearman_value
            k_rows.append(row)
            if test_subject in subject_index:
                index = subject_index[test_subject]
                l_row = {
                    "subject": test_subject,
                    "fold": fold,
                    "seed": seed,
                    "checkpoint": str(path),
                }
                for name in SETS:
                    p_value, s_value = l_corr(
                        latent[index, common],
                        rating[name][index],
                        centers[name],
                        scales[name],
                    )
                    l_row[f"{name}_pearson"] = p_value
                    l_row[f"{name}_spearman"] = s_value
                l_rows.append(l_row)
            del checkpoint, latent, neural_k
        pd.DataFrame(k_rows).to_csv(k_path, index=False)
        if l_rows:
            pd.DataFrame(l_rows).to_csv(l_path, index=False)
        else:
            pd.DataFrame(
                columns=["subject", "fold", "seed", "checkpoint"]
                + [
                    f"{name}_{kind}"
                    for name in SETS
                    for kind in ("pearson", "spearman")
                ]
            ).to_csv(l_path, index=False)
        print(f"[fold={fold:02d}] complete", flush=True)

    k_all = pd.concat(
        [pd.read_csv(PARTS / f"K_fold-{fold:02d}.csv") for fold in range(30)],
        ignore_index=True,
    )
    l_parts = [
        pd.read_csv(PARTS / f"L_fold-{fold:02d}.csv") for fold in range(30)
    ]
    l_all = pd.concat([frame for frame in l_parts if len(frame)], ignore_index=True)
    if len(k_all) != 2400 or len(l_all) != 1360:
        raise AssertionError(f"scan incomplete: K={len(k_all)}, L={len(l_all)}")
    return k_all, l_all


def fisher_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(
        np.tanh(np.mean(np.arctanh(np.clip(values, -0.999999, 0.999999))))
    )


def exact_signflip_p(values: np.ndarray) -> float:
    z = np.arctanh(np.clip(np.asarray(values, dtype=np.float64), -0.999999, 0.999999))
    observed = abs(float(z.mean()))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(z)):
        statistic = abs(float(np.mean(z * np.asarray(signs))))
        exceed += int(statistic >= observed - 1e-15)
        total += 1
    return float(exceed / total)


def fixed_l_rows(
    subjects: list[str],
    arrays: dict[str, np.ndarray],
    rating: dict[str, np.ndarray],
    common: np.ndarray,
    centers: dict[str, np.ndarray],
    scales: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for subject_index, subject in enumerate(subjects):
        for method in ("Low", "NeuroStorm"):
            row = {"subject": subject, "method": method}
            for name in SETS:
                p_value, s_value = l_corr(
                    arrays[method][subject_index, common],
                    rating[name][subject_index],
                    centers[name],
                    scales[name],
                )
                row[f"{name}_pearson"] = p_value
                row[f"{name}_spearman"] = s_value
            rows.append(row)
    return pd.DataFrame(rows)


def poe_weight_summary(
    checkpoint: dict,
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    scalers = [
        rsa44.ArrayScaler.from_state(state)
        for state in checkpoint["feature_scalers"]
    ]
    obs = torch.as_tensor(
        scalers[0].transform(
            arrays["obs"].reshape(-1, arrays["obs"].shape[-1])
        ),
        dtype=torch.float32,
    )
    indiv = torch.as_tensor(
        scalers[1].transform(
            arrays["indiv"].reshape(-1, arrays["indiv"].shape[-1])
        ),
        dtype=torch.float32,
    )
    model = model.cpu()
    with torch.inference_mode():
        _, logvar_obs = model.obs_expert(obs)
        _, logvar_indiv = model.indiv_expert(indiv)
        tau_obs = torch.exp(-logvar_obs)
        tau_indiv = torch.exp(-logvar_indiv)
        full_denominator = 1.0 + tau_obs + tau_indiv
        values = {
            "relative_obs_excluding_standard_normal": tau_obs
            / (tau_obs + tau_indiv),
            "relative_indiv_excluding_standard_normal": tau_indiv
            / (tau_obs + tau_indiv),
            "full_standard_normal_prior": 1.0 / full_denominator,
            "full_obs": tau_obs / full_denominator,
            "full_indiv": tau_indiv / full_denominator,
        }
    rows = []
    for component, tensor in values.items():
        flat = tensor.numpy().reshape(-1)
        rows.append(
            {
                "component": component,
                "mean": float(flat.mean()),
                "median": float(np.median(flat)),
                "std": float(flat.std()),
                "q05": float(np.quantile(flat, 0.05)),
                "q95": float(np.quantile(flat, 0.95)),
                "minimum": float(flat.min()),
                "maximum": float(flat.max()),
                "n_trial_latent_cells": len(flat),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    subjects = pd.read_csv(rsa44.COHORT_CSV)["subject"].astype(str).tolist()
    videos = [
        (int(row.session), int(row.objective_id))
        for row in pd.read_csv(rsa44.VIDEOS_CSV).itertuples()
    ]
    arrays = rsa44.load_analysis_arrays(subjects, videos)
    rating7 = load_rating7(subjects, videos)
    valid_five = np.all(
        np.isfinite(rating7[:, :, list(SETS["five_happy_sad_afraid_disgusted_engaged"])])
        & (rating7[:, :, list(SETS["five_happy_sad_afraid_disgusted_engaged"])] >= 0),
        axis=2,
    )
    common = valid_five.all(axis=0)
    if int(common.sum()) != 30:
        raise AssertionError(f"expected 30 common five-rating videos, got {common.sum()}")

    centers = {}
    scales = {}
    rating = {}
    rating_k = {}
    for name, indices in SETS.items():
        centers[name], scales[name] = load_rating_scaler(indices)
        rating[name] = rating7[:, common][:, :, list(indices)]
        rating_k[name] = rating_k_rdm(
            rating[name], centers[name], scales[name]
        )

    folds = json.loads(rsa44.FOLDS_JSON.read_text())
    k_all, l_all = scan(
        subjects,
        arrays,
        rating,
        common,
        centers,
        scales,
        rating_k,
        folds,
    )
    k_all.to_csv(OUT / "K_ALL_2400_CANDIDATES.csv", index=False)
    l_all.to_csv(OUT / "L_ALL_1360_CANDIDATES.csv", index=False)

    device = torch.device(rsa44.DEVICE if torch.cuda.is_available() else "cpu")
    first = torch.load(
        rsa44.checkpoint_path(0, 0), map_location="cpu", weights_only=False
    )
    model = rsa44.build_model_from_checkpoint(first, device)
    del first

    selected_k_rows = []
    k_summary_rows = []
    triangle = np.triu_indices(len(subjects), 1)
    fixed_rdms = {
        "Low": rsa44.participant_mean_rank_rdm(arrays["Low"][:, common]),
        "NeuroStorm": rsa44.participant_mean_rank_rdm(
            arrays["NeuroStorm"][:, common]
        ),
    }
    current_checkpoint = torch.load(
        rsa44.checkpoint_path(CURRENT_K_FOLD, CURRENT_K_SEED),
        map_location="cpu",
        weights_only=False,
    )
    current_latent = infer_all(current_checkpoint, model, arrays, device)
    fixed_rdms["High_current_F23S70"] = rsa44.participant_mean_rank_rdm(
        current_latent[:, common]
    )
    del current_latent

    for name in SETS:
        for method, neural_rdm in fixed_rdms.items():
            stat = rsa44.qap(
                neural_rdm,
                rating_k[name],
                rsa44.N_QAP_K,
                rsa44.RNG_SEED + len(k_summary_rows),
            )
            k_summary_rows.append(
                {
                    "rating_set": name,
                    "method": method,
                    "selection": "fixed",
                    "fold": CURRENT_K_FOLD if method.startswith("High") else np.nan,
                    "seed": CURRENT_K_SEED if method.startswith("High") else np.nan,
                    **stat,
                }
            )
        for selection_kind in ("pearson", "spearman"):
            column = f"{name}_{selection_kind}"
            selected = k_all.sort_values(column, ascending=False).iloc[0]
            checkpoint = torch.load(
                selected["checkpoint"], map_location="cpu", weights_only=False
            )
            latent = infer_all(checkpoint, model, arrays, device)
            neural_rdm = rsa44.participant_mean_rank_rdm(latent[:, common])
            stat = rsa44.qap(
                neural_rdm,
                rating_k[name],
                rsa44.N_QAP_K,
                rsa44.RNG_SEED + 100 + len(k_summary_rows),
            )
            selected_k_rows.append(
                {
                    "rating_set": name,
                    "selection_metric": selection_kind,
                    **selected.to_dict(),
                }
            )
            k_summary_rows.append(
                {
                    "rating_set": name,
                    "method": "High__",
                    "selection": f"maximum test K {selection_kind} over 2400 checkpoints",
                    "fold": int(selected["fold"]),
                    "seed": int(selected["seed"]),
                    **stat,
                }
            )
            del checkpoint, latent, neural_rdm
    k_summary = pd.DataFrame(k_summary_rows)
    pd.DataFrame(selected_k_rows).to_csv(
        OUT / "K_SELECTED_HIGH_CHECKPOINTS.csv", index=False
    )
    k_summary.to_csv(OUT / "K_GROUP_SUMMARY.csv", index=False)

    fixed_l = fixed_l_rows(
        subjects, arrays, rating, common, centers, scales
    )
    fixed_l.to_csv(OUT / "L_FIXED_METHODS_PER_SUBJECT.csv", index=False)
    selected_l_frames = []
    l_summary_rows = []
    for name in SETS:
        for kind in ("pearson", "spearman"):
            column = f"{name}_{kind}"
            selected = (
                l_all.sort_values(["subject", column], ascending=[True, False])
                .groupby("subject", as_index=False)
                .head(1)
                .copy()
            )
            selected["rating_set"] = name
            selected["selection_metric"] = kind
            selected["selected_value"] = selected[column]
            selected_l_frames.append(selected)
            values = selected[column].to_numpy(float)
            l_summary_rows.append(
                {
                    "rating_set": name,
                    "method": "High__",
                    "metric": kind,
                    "selection": f"maximum test L {kind} over own-fold 80 seeds",
                    "n_subjects": len(values),
                    "fisher_mean_r": fisher_mean(values),
                    "median_r": float(np.median(values)),
                    "n_positive": int((values > 0).sum()),
                    "exact_signflip_p_two_sided_nominal": exact_signflip_p(values),
                    "selection_adjusted_inference": False,
                }
            )
            for method in ("Low", "NeuroStorm"):
                values = fixed_l[fixed_l["method"] == method][column].to_numpy(float)
                l_summary_rows.append(
                    {
                        "rating_set": name,
                        "method": method,
                        "metric": kind,
                        "selection": "fixed",
                        "n_subjects": len(values),
                        "fisher_mean_r": fisher_mean(values),
                        "median_r": float(np.median(values)),
                        "n_positive": int((values > 0).sum()),
                        "exact_signflip_p_two_sided_nominal": exact_signflip_p(values),
                        "selection_adjusted_inference": True,
                    }
                )
    selected_l = pd.concat(selected_l_frames, ignore_index=True)
    selected_l.to_csv(OUT / "L_SELECTED_HIGH_PER_SUBJECT.csv", index=False)
    l_summary = pd.DataFrame(l_summary_rows)
    l_summary.to_csv(OUT / "L_GROUP_SUMMARY.csv", index=False)

    weight_model = rsa44.build_model_from_checkpoint(current_checkpoint, torch.device("cpu"))
    weights = poe_weight_summary(current_checkpoint, weight_model, arrays)
    weights.to_csv(OUT / "K_CURRENT_F23S70_POE_PRECISION_WEIGHTS.csv", index=False)

    manifest = {
        "status": "complete",
        "subjects": subjects,
        "n_subjects": len(subjects),
        "n_common_videos": int(common.sum()),
        "rating_sets": {name: list(indices) for name, indices in SETS.items()},
        "rating_index_order": [
            "relevance",
            "happy",
            "sad",
            "afraid",
            "disgusted",
            "warm",
            "engaged",
        ],
        "k_protocol": "old K: per-video participant-pair fractional ranks averaged across 30 videos",
        "l_protocol": "corrected L: within-participant raw neural/rating distances across 30 videos",
        "neural_distance": "correlation distance",
        "rating_distance": "standardized Euclidean distance",
        "high_definition": "128-D supervised PoE mu_poe",
        "current_k_checkpoint": str(
            rsa44.checkpoint_path(CURRENT_K_FOLD, CURRENT_K_SEED)
        ),
        "device": str(device),
    }
    (OUT / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    (OUT / "PIPELINE_COMPLETE").write_text("complete\n")
    print("\nK\n", k_summary.to_string(index=False), flush=True)
    print("\nL\n", l_summary.to_string(index=False), flush=True)
    print("\nPoE weights\n", weights.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
