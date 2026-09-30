







from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, rankdata, spearmanr

from decode_narratives_shortvideo_poe import (
    GaussianPoERegressor,
    narrative_xy_to_semantic,
)


CONDITIONS = ("similarity", "likeability", "mentalizing")
NARRATIVE_RATING_SETS = {
    "feeling_2d": (0, 1),
    "expectation_2d": (2, 3),
    "joint_4d": (0, 1, 2, 3),
}


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Scaler":
        values = np.asarray(values, dtype=np.float32)
        return cls(
            np.nanmean(values, axis=0),
            np.nanstd(values, axis=0) + 1e-6,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def state(self) -> dict:
        return {
            "mean": self.mean.astype(np.float32),
            "std": self.std.astype(np.float32),
        }


class ScalarPrecisionPoERegressor(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.raw_tau_obs = nn.Parameter(torch.zeros(1))
        self.raw_tau_indiv = nn.Parameter(torch.zeros(1))
        self.head = nn.Linear(d_in, d_out)

    def precisions(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.softplus(self.raw_tau_obs) + 1e-4,
            F.softplus(self.raw_tau_indiv) + 1e-4,
        )

    def forward(self, obs: torch.Tensor, indiv: torch.Tensor) -> dict:
        tau_obs, tau_indiv = self.precisions()
        latent = (tau_obs * obs + tau_indiv * indiv) / (tau_obs + tau_indiv)
        return {
            "poe": self.head(latent),
            "mu_poe": latent,
            "tau_obs": tau_obs,
            "tau_indiv": tau_indiv,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=path.stem + "_", suffix=".npz", dir="/tmp"
    ) as handle:
        np.savez_compressed(handle.name, **arrays)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        shutil.copyfile(handle.name, temporary)
    os.replace(temporary, path)


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=path.stem + "_", suffix=".pt", dir="/tmp"
    ) as handle:
        torch.save(payload, handle.name)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        shutil.copyfile(handle.name, temporary)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in values
        ],
        dtype=str,
    )


def finite_corr(left: np.ndarray, right: np.ndarray, kind: str) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    keep = np.isfinite(left) & np.isfinite(right)
    if keep.sum() < 3 or left[keep].std() < 1e-12 or right[keep].std() < 1e-12:
        return math.nan
    if kind == "spearman":
        return float(spearmanr(left[keep], right[keep]).statistic)
    return float(pearsonr(left[keep], right[keep]).statistic)


def fractional_rank(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=np.float64), method="average")
    return (ranks - 1.0) / max(len(ranks) - 1.0, 1.0)


def correlation_distances(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(centered**2, axis=1, keepdims=True))
    norms[norms < 1e-12] = 1.0
    normalized = centered / norms
    matrix = 1.0 - normalized @ normalized.T
    np.fill_diagonal(matrix, 0.0)
    return np.clip(matrix, 0.0, 2.0)


def participant_k_rdm(cube: np.ndarray) -> np.ndarray:

    n_subjects, n_stimuli, _ = cube.shape
    triangle = np.triu_indices(n_subjects, 1)
    edges = []
    for stimulus in range(n_stimuli):
        matrix = correlation_distances(cube[:, stimulus, :])
        edges.append(fractional_rank(matrix[triangle]))
    mean_edge = np.mean(np.asarray(edges), axis=0)
    output = np.zeros((n_subjects, n_subjects), dtype=np.float64)
    output[triangle] = mean_edge
    output[(triangle[1], triangle[0])] = mean_edge
    return output


def subject_l_edges(cube: np.ndarray) -> np.ndarray:

    return np.asarray(
        [pdist(cube[index], metric="correlation") for index in range(len(cube))],
        dtype=np.float64,
    )


def euclidean_k_rdm(cube: np.ndarray) -> np.ndarray:
    n_subjects, n_stimuli, _ = cube.shape
    triangle = np.triu_indices(n_subjects, 1)
    edges = [
        fractional_rank(pdist(cube[:, stimulus, :], metric="euclidean"))
        for stimulus in range(n_stimuli)
    ]
    mean_edge = np.mean(np.asarray(edges), axis=0)
    output = np.zeros((n_subjects, n_subjects), dtype=np.float64)
    output[triangle] = mean_edge
    output[(triangle[1], triangle[0])] = mean_edge
    return output


def euclidean_l_edges(cube: np.ndarray) -> np.ndarray:
    return np.asarray(
        [pdist(cube[index], metric="euclidean") for index in range(len(cube))],
        dtype=np.float64,
    )


def load_data(path: Path, task: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    subjects = data["subject"].astype(str)
    order = list(dict.fromkeys(subjects.tolist()))
    if len(order) != 30 or set(order) != set(np.unique(subjects)):
        raise RuntimeError(f"{path}: expected exactly 30 intact subjects")
    data["subject"] = subjects
    data["subject_order"] = np.asarray(order, dtype=str)
    if task == "narratives":
        labels = narrative_xy_to_semantic(
            np.asarray(data["labels_subjective"], dtype=np.float32)
        )
        if labels.shape[1:] != (4,):
            raise RuntimeError(f"{path}: invalid Narratives labels {labels.shape}")
        data["target"] = labels
        data["condition_code"] = np.full(len(labels), -1, dtype=np.int8)
    else:
        condition = decode_strings(data["condition"])
        mapping = {name: index for index, name in enumerate(CONDITIONS)}
        codes = np.asarray([mapping.get(value, -1) for value in condition], dtype=np.int8)
        if np.any(codes < 0):
            raise RuntimeError(f"{path}: unknown Shortvideo condition")
        data["target"] = np.asarray(
            data["labels_subjective"], dtype=np.float32
        ).reshape(-1)
        data["condition_code"] = codes
        data["condition"] = condition
    return data


def fit_target_scaler(data: dict, mask: np.ndarray, task: str) -> Scaler:
    if task == "narratives":
        return Scaler.fit(data["target"][mask])
    values = data["target"][mask]
    codes = data["condition_code"][mask]
    means, scales = [], []
    for condition in range(3):
        selected = np.log1p(np.maximum(values[codes == condition], 0.0))
        if len(selected) < 2:
            raise RuntimeError(f"condition {condition}: insufficient targets")
        means.append(selected.mean())
        scales.append(selected.std() + 1e-6)
    return Scaler(np.asarray(means, np.float32), np.asarray(scales, np.float32))


def prepare_tensors(
    data: dict,
    mask: np.ndarray,
    obs_scaler: Scaler,
    indiv_scaler: Scaler,
    target_scaler: Scaler,
    task: str,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    obs = obs_scaler.transform(data["obs"][mask])
    indiv = indiv_scaler.transform(data["indiv"][mask])
    condition = data["condition_code"][mask]
    if task == "narratives":
        target = target_scaler.transform(data["target"][mask])
    else:
        raw = data["target"][mask]
        target = np.empty(len(raw), dtype=np.float32)
        transformed = np.log1p(np.maximum(raw, 0.0))
        for code in range(3):
            selected = condition == code
            target[selected] = (
                transformed[selected] - target_scaler.mean[code]
            ) / target_scaler.std[code]
    return (
        torch.as_tensor(obs, dtype=torch.float32, device=device),
        torch.as_tensor(indiv, dtype=torch.float32, device=device),
        torch.as_tensor(target, dtype=torch.float32, device=device),
        torch.as_tensor(condition, dtype=torch.long, device=device),
    )


def new_model(task: str, fusion: str, args, device: torch.device) -> nn.Module:
    d_out = 4 if task == "narratives" else 3
    if fusion == "gaussian_poe128":
        return GaussianPoERegressor(
            512, args.latent_dim, args.hidden, d_out, args.dropout
        ).to(device)
    if fusion == "scalar_poe512":
        return ScalarPrecisionPoERegressor(512, d_out).to(device)
    raise ValueError(fusion)


def output_model(
    model: nn.Module, fusion: str, obs: torch.Tensor, indiv: torch.Tensor
) -> dict:
    if fusion == "gaussian_poe128":
        return model(obs, indiv, sample=False)
    return model(obs, indiv)


def prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    condition: torch.Tensor,
    task: str,
) -> torch.Tensor:
    if task == "narratives":
        valid = torch.isfinite(target)
        return F.smooth_l1_loss(prediction[valid], target[valid])
    selected = prediction.gather(1, condition[:, None]).squeeze(1)
    return F.smooth_l1_loss(selected, target)


def total_loss(
    model: nn.Module,
    fusion: str,
    tensors: tuple[torch.Tensor, ...],
    task: str,
    args,
) -> tuple[torch.Tensor, torch.Tensor]:
    obs, indiv, target, condition = tensors
    output = output_model(model, fusion, obs, indiv)
    main = prediction_loss(output["poe"], target, condition, task)
    if fusion == "gaussian_poe128":
        auxiliary = (
            prediction_loss(output["obs"], target, condition, task)
            + prediction_loss(output["indiv"], target, condition, task)
        )
        mu, var = output["mu_poe"], output["var_poe"]
        kl = 0.5 * (
            mu.square() + var - torch.log(var + 1e-8) - 1.0
        ).mean()
        total = main + args.aux_weight * auxiliary + args.kl_weight * kl
    else:
        total = main
    return total, main


def train_epoch(
    model: nn.Module,
    fusion: str,
    tensors: tuple[torch.Tensor, ...],
    task: str,
    optimizer: torch.optim.Optimizer,
    args,
    rng: np.random.Generator,
) -> float:
    model.train()
    order = rng.permutation(len(tensors[0]))
    total = 0.0
    for start in range(0, len(order), args.batch_size):
        indices = torch.as_tensor(
            order[start:start + args.batch_size],
            dtype=torch.long,
            device=tensors[0].device,
        )
        batch = tuple(value[indices] for value in tensors)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = total_loss(model, fusion, batch, task, args)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total += float(loss.detach()) * len(indices)
    return total / max(len(order), 1)


@torch.inference_mode()
def validation_loss(
    model: nn.Module,
    fusion: str,
    tensors: tuple[torch.Tensor, ...],
    task: str,
    args,
) -> float:
    model.eval()
    _, main = total_loss(model, fusion, tensors, task, args)
    return float(main.detach())


def fit_model(
    data: dict,
    task: str,
    fusion: str,
    seed: int,
    args,
    device: torch.device,
) -> tuple[nn.Module, Scaler, Scaler, Scaler, dict]:
    subjects = data["subject_order"].astype(str).tolist()
    validation_subjects = subjects[::5]
    inner_subjects = [
        subject for subject in subjects if subject not in set(validation_subjects)
    ]
    inner_mask = np.isin(data["subject"], inner_subjects)
    validation_mask = np.isin(data["subject"], validation_subjects)
    all_mask = np.ones(len(data["subject"]), dtype=bool)

    obs_scaler = Scaler.fit(data["obs"][inner_mask])
    indiv_scaler = Scaler.fit(data["indiv"][inner_mask])
    target_scaler = fit_target_scaler(data, inner_mask, task)
    train = prepare_tensors(
        data, inner_mask, obs_scaler, indiv_scaler, target_scaler, task, device
    )
    validation = prepare_tensors(
        data, validation_mask, obs_scaler, indiv_scaler,
        target_scaler, task, device,
    )
    set_seed(seed)
    model = new_model(task, fusion, args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(seed)
    best_loss, best_epoch, bad = math.inf, 1, 0
    selection_history = []
    for epoch in range(1, args.epochs + 1):
        train_value = train_epoch(
            model, fusion, train, task, optimizer, args, rng
        )
        validation_value = validation_loss(
            model, fusion, validation, task, args
        )
        selection_history.append(
            {
                "epoch": epoch,
                "train_loss": train_value,
                "validation_loss": validation_value,
            }
        )
        if validation_value < best_loss - args.min_delta:
            best_loss, best_epoch, bad = validation_value, epoch, 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    obs_scaler = Scaler.fit(data["obs"][all_mask])
    indiv_scaler = Scaler.fit(data["indiv"][all_mask])
    target_scaler = fit_target_scaler(data, all_mask, task)
    all_tensors = prepare_tensors(
        data, all_mask, obs_scaler, indiv_scaler, target_scaler, task, device
    )
    set_seed(seed)
    model = new_model(task, fusion, args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(seed + 7919)
    re_history = []
    for epoch in range(1, best_epoch + 1):
        re_history.append(
            {
                "epoch": epoch,
                "train_loss": train_epoch(
                    model, fusion, all_tensors, task, optimizer, args, rng
                ),
            }
        )
    provenance = {
        "subjects": subjects,
        "inner_train_subjects": inner_subjects,
        "validation_subjects": validation_subjects,
        "best_inner_validation_epoch": best_epoch,
        "best_inner_validation_loss": best_loss,
        "selection_history": selection_history,
        "re_history": re_history,
    }
    return model, obs_scaler, indiv_scaler, target_scaler, provenance


@torch.inference_mode()
def infer_latents(
    model: nn.Module,
    fusion: str,
    data: dict,
    obs_scaler: Scaler,
    indiv_scaler: Scaler,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    obs = torch.as_tensor(
        obs_scaler.transform(data["obs"]), dtype=torch.float32, device=device
    )
    indiv = torch.as_tensor(
        indiv_scaler.transform(data["indiv"]), dtype=torch.float32, device=device
    )
    if fusion == "gaussian_poe128":
        mu_obs, _ = model.obs_expert(obs)
        output = model(obs, indiv, sample=False)
        return {
            "obs_mlp_prepoe128": mu_obs.float().cpu().numpy(),
            "gaussian_poe128": output["mu_poe"].float().cpu().numpy(),
        }
    output = model(obs, indiv)
    return {"scalar_poe512": output["mu_poe"].float().cpu().numpy()}


def geometry_cubes(
    data: dict, latent: np.ndarray, task: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    subjects = data["subject_order"].astype(str).tolist()
    if task == "narratives":
        keys = sorted(
            set(
                zip(
                    data["story_id"].astype(int).tolist(),
                    data["situation_index"].astype(int).tolist(),
                )
            )
        )
        valid_keys = []
        for key in keys:
            rows = [
                np.flatnonzero(
                    (data["subject"] == subject)
                    & (data["story_id"].astype(int) == key[0])
                    & (data["situation_index"].astype(int) == key[1])
                )
                for subject in subjects
            ]
            if any(len(indices) != 1 for indices in rows):
                continue
            labels = np.asarray(
                [data["target"][indices[0]] for indices in rows]
            )
            if np.all(np.isfinite(labels)):
                valid_keys.append(key)
        if len(valid_keys) != 71:
            raise RuntimeError(
                f"Narratives expected 71 common complete stimuli, got {len(valid_keys)}"
            )
        feature_cube, rating_cube = [], []
        for subject in subjects:
            subject_features, subject_ratings = [], []
            for story, situation in valid_keys:
                index = np.flatnonzero(
                    (data["subject"] == subject)
                    & (data["story_id"].astype(int) == story)
                    & (data["situation_index"].astype(int) == situation)
                )
                if len(index) != 1:
                    raise RuntimeError(f"{subject}: missing/duplicate narrative key")
                subject_features.append(latent[index[0]])
                subject_ratings.append(data["target"][index[0]])
            feature_cube.append(subject_features)
            rating_cube.append(subject_ratings)
        features = np.asarray(feature_cube, dtype=np.float32)
        ratings = np.asarray(rating_cube, dtype=np.float64)
        neural = {
            name: features for name in NARRATIVE_RATING_SETS
        }
        behavior = {
            name: ratings[:, :, list(indices)]
            for name, indices in NARRATIVE_RATING_SETS.items()
        }
        return neural, behavior, len(valid_keys)

    videos = sorted(set(data["video_id"].astype(str).tolist()))
    if len(videos) != 21:
        raise RuntimeError(f"Shortvideo expected 21 videos, got {len(videos)}")
    feature_by_condition = []
    rating_cube = np.full((30, 21, 3), np.nan, dtype=np.float64)
    for condition_index, condition in enumerate(CONDITIONS):
        condition_cube = []
        for subject_index, subject in enumerate(subjects):
            subject_features = []
            for video_index, video in enumerate(videos):
                index = np.flatnonzero(
                    (data["subject"] == subject)
                    & (data["video_id"].astype(str) == video)
                    & (data["condition_code"] == condition_index)
                )
                if len(index) != 1:
                    raise RuntimeError(
                        f"{subject}/{video}/{condition}: missing or duplicate"
                    )
                subject_features.append(latent[index[0]])
                rating_cube[subject_index, video_index, condition_index] = (
                    data["target"][index[0]]
                )
            condition_cube.append(subject_features)
        feature_by_condition.append(np.asarray(condition_cube, dtype=np.float32))
    if not np.isfinite(rating_cube).all():
        raise RuntimeError("Shortvideo rating cube is incomplete")
    neural = {
        condition: feature_by_condition[index]
        for index, condition in enumerate(CONDITIONS)
    }
    neural["joint_3d"] = np.asarray(feature_by_condition, dtype=np.float32)
    behavior = {
        condition: rating_cube[:, :, index:index + 1]
        for index, condition in enumerate(CONDITIONS)
    }
    behavior["joint_3d"] = rating_cube
    return neural, behavior, len(videos)


def compute_geometry(
    data: dict, latent: np.ndarray, task: str
) -> tuple[list[dict], dict[str, np.ndarray]]:
    neural_cubes, behavior_cubes, n_stimuli = geometry_cubes(data, latent, task)
    subjects = data["subject_order"].astype(str).tolist()
    rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    for rating_set, behavior in behavior_cubes.items():
        mean = behavior.reshape(-1, behavior.shape[-1]).mean(axis=0)
        scale = behavior.reshape(-1, behavior.shape[-1]).std(axis=0)
        scale[scale < 1e-12] = 1.0
        standardized = (behavior - mean) / scale
        behavior_k = euclidean_k_rdm(standardized)
        behavior_l = euclidean_l_edges(standardized)
        if task == "shortvideo" and rating_set == "joint_3d":
            condition_cubes = neural_cubes[rating_set]
            k_rdms = [
                participant_k_rdm(condition_cubes[index])
                for index in range(3)
            ]
            neural_k = np.mean(np.asarray(k_rdms), axis=0)
            neural_l = np.mean(
                np.asarray(
                    [
                        subject_l_edges(condition_cubes[index])
                        for index in range(3)
                    ]
                ),
                axis=0,
            )
        else:
            cube = neural_cubes[rating_set]
            neural_k = participant_k_rdm(cube)
            neural_l = subject_l_edges(cube)
        triangle = np.triu_indices(len(subjects), 1)
        rows.append(
            {
                "scope": "K",
                "subject": "group",
                "rating_set": rating_set,
                "n_subjects": len(subjects),
                "n_stimuli": n_stimuli,
                "pearson_r": finite_corr(
                    neural_k[triangle], behavior_k[triangle], "pearson"
                ),
                "spearman_r": finite_corr(
                    neural_k[triangle], behavior_k[triangle], "spearman"
                ),
            }
        )
        for subject_index, subject in enumerate(subjects):
            rows.append(
                {
                    "scope": "L",
                    "subject": subject,
                    "rating_set": rating_set,
                    "n_subjects": len(subjects),
                    "n_stimuli": n_stimuli,
                    "pearson_r": finite_corr(
                        neural_l[subject_index],
                        behavior_l[subject_index],
                        "pearson",
                    ),
                    "spearman_r": finite_corr(
                        neural_l[subject_index],
                        behavior_l[subject_index],
                        "spearman",
                    ),
                }
            )
        arrays[f"{rating_set}__neural_k"] = neural_k
        arrays[f"{rating_set}__behavior_k"] = behavior_k
        arrays[f"{rating_set}__neural_l"] = neural_l
        arrays[f"{rating_set}__behavior_l"] = behavior_l
        arrays[f"{rating_set}__rating_mean"] = mean
        arrays[f"{rating_set}__rating_std"] = scale
    return rows, arrays


def candidate(
    data: dict,
    task: str,
    initialization: str,
    mae_epoch: int,
    fusion: str,
    seed: int,
    output_root: Path,
    feature_path: Path,
    args,
    device: torch.device,
) -> list[dict]:
    run_dir = (
        output_root / "runs" / task / initialization
        / f"epoch_{mae_epoch:03d}" / fusion / f"seed_{seed:02d}"
    )
    marker = run_dir / "CANDIDATE_COMPLETE"
    metrics_path = run_dir / "metrics.csv"
    if marker.is_file() and metrics_path.is_file():
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    model, obs_scaler, indiv_scaler, target_scaler, provenance = fit_model(
        data, task, fusion, seed, args, device
    )
    latents = infer_latents(
        model, fusion, data, obs_scaler, indiv_scaler, device
    )
    rows = []
    rdm_arrays = {}
    for representation, latent in latents.items():
        representation_rows, geometry = compute_geometry(data, latent, task)
        for row in representation_rows:
            rows.append(
                {
                    "task": task,
                    "initialization": initialization,
                    "mae_epoch": mae_epoch,
                    "fusion": fusion,
                    "representation": representation,
                    "seed": seed,
                    **row,
                }
            )
        for name, value in geometry.items():
            rdm_arrays[f"{representation}__{name}"] = value

    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        run_dir / "latents_and_rdms.npz",
        subject=data["subject"],
        trial_index=data["trial_index"],
        **latents,
        **rdm_arrays,
    )
    checkpoint = {
        "status": "complete",
        "task": task,
        "initialization": initialization,
        "mae_epoch": mae_epoch,
        "fusion": fusion,
        "seed": seed,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": {
            "input_dim": 512,
            "latent_dim": 128 if fusion == "gaussian_poe128" else 512,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "output_dim": 4 if task == "narratives" else 3,
        },
        "obs_scaler": obs_scaler.state(),
        "indiv_scaler": indiv_scaler.state(),
        "target_scaler": target_scaler.state(),
        "feature_source": str(feature_path),
        **provenance,
    }
    if fusion == "scalar_poe512":
        tau_obs, tau_indiv = model.precisions()
        checkpoint["tau_obs"] = float(tau_obs.detach().cpu())
        checkpoint["tau_indiv"] = float(tau_indiv.detach().cpu())
    atomic_torch_save(checkpoint, run_dir / "checkpoint.pt")
    write_csv(metrics_path, rows)
    atomic_json(
        run_dir / "metrics.json",
        {
            "status": "complete",
            "task": task,
            "initialization": initialization,
            "mae_epoch": mae_epoch,
            "fusion": fusion,
            "seed": seed,
            "rows": rows,
        },
    )
    marker.write_text("complete\n", encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return rows


def raw_obs(
    data: dict,
    task: str,
    initialization: str,
    mae_epoch: int,
    output_root: Path,
) -> list[dict]:
    run_dir = (
        output_root / "runs" / task / initialization
        / f"epoch_{mae_epoch:03d}" / "raw_obs512"
    )
    marker = run_dir / "CANDIDATE_COMPLETE"
    metrics_path = run_dir / "metrics.csv"
    if marker.is_file() and metrics_path.is_file():
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    metrics, geometry = compute_geometry(data, data["obs"], task)
    rows = [
        {
            "task": task,
            "initialization": initialization,
            "mae_epoch": mae_epoch,
            "fusion": "raw_obs512",
            "representation": "raw_obs512",
            "seed": -1,
            **row,
        }
        for row in metrics
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_npz(run_dir / "rdms.npz", **geometry)
    write_csv(metrics_path, rows)
    atomic_json(
        run_dir / "metrics.json",
        {
            "status": "complete",
            "task": task,
            "initialization": initialization,
            "mae_epoch": mae_epoch,
            "fusion": "raw_obs512",
            "seed": None,
            "rows": rows,
        },
    )
    marker.write_text("complete\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("narratives", "shortvideo"), required=True)
    parser.add_argument("--initialization", choices=("crossdataset", "random"), required=True)
    parser.add_argument("--mae-epoch", type=int, required=True)
    parser.add_argument("--feature-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fusions", default="gaussian_poe128,scalar_poe512")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--aux-weight", type=float, default=0.25)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    args = parser.parse_args()

    if args.mae_epoch < 0 or args.mae_epoch > 20:
        raise RuntimeError("MAE epoch must be in 0..20")
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if seeds != [0, 1, 2, 3, 4]:
        raise RuntimeError("the exploratory protocol requires saved seeds 0..4")
    fusions = [value for value in args.fusions.split(",") if value]
    if set(fusions) != {"gaussian_poe128", "scalar_poe512"}:
        raise RuntimeError("both PoE variants are required")
    data = load_data(args.feature_path, args.task)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    all_rows = raw_obs(
        data, args.task, args.initialization, args.mae_epoch, args.output_root
    )
    for fusion in fusions:
        for seed in seeds:
            print(
                f"task={args.task} init={args.initialization} "
                f"mae_epoch={args.mae_epoch:03d} fusion={fusion} seed={seed}",
                flush=True,
            )
            all_rows.extend(
                candidate(
                    data, args.task, args.initialization, args.mae_epoch,
                    fusion, seed, args.output_root, args.feature_path,
                    args, device,
                )
            )
    epoch_dir = (
        args.output_root / "per_epoch" / args.task / args.initialization
        / f"epoch_{args.mae_epoch:03d}"
    )
    write_csv(epoch_dir / "ALL_CANDIDATE_METRICS.csv", all_rows)
    atomic_json(
        epoch_dir / "CANDIDATE_MANIFEST.json",
        {
            "status": "complete",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "task": args.task,
            "initialization": args.initialization,
            "mae_epoch": args.mae_epoch,
            "feature_path": str(args.feature_path),
            "fusions": fusions,
            "seeds": seeds,
            "n_rows": len(all_rows)
        },
    )
    (epoch_dir / "CANDIDATES_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
