


















from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr, wilcoxon

from decode_narratives_shortvideo_poe import GaussianExpert, GaussianPoERegressor


ROOT = Path(str(_LOCAL_SOURCE / ''))
DEFAULT_OURS = ROOT / "lizhuo_exp/lizhuo_exp_data/SKIP/enc_trained_all/features_four_tasks"
DEFAULT_NS = ROOT / "lizhuo_exp/lizhuo_exp_class2/SKIP/result/enc_trained_all/neurostorm_mni_four_tasks"
DEFAULT_MANIFEST = ROOT / "lizhuo_exp/lizhuo_exp_data/SKIP/enc_trained_all/residual_trial_manifests"
DEFAULT_OUT = ROOT / "lizhuo_exp/lizhuo_exp_class2/SKIP/result/cross_subject_residual_5fold"
DEFAULT_OBS_CKPT = ROOT / (
    "lizhuo_exp/lizhuo_exp_data/SKIP/enc_trained_all/"
    "checkpoints_four_tasks_obs/cifti_encoder_skip_continue.pt"
)
DEFAULT_INDIV_CKPT = ROOT / (
    "lizhuo_exp/lizhuo_exp_data/SKIP/enc_trained_all/"
    "checkpoints_four_tasks_indiv/fmri_prior_indiv_skip.pt"
)
DEFAULT_NS_CKPT = ROOT / "weights/NeuroStorm/NeuroSTORM/neurostorm/pt_neurostorm_mae_ratio0.5.ckpt"

TASKS = ("alignvideo", "faces", "narratives", "shortvideo")
TARGETS = {
    "alignvideo": ("relevance", "happy", "sad", "afraid", "disgusted", "warm", "engaged"),
    "faces": ("intensity", "sex", "age"),
    "narratives": (
        "feeling_valence", "feeling_intensity",
        "expectation_valence", "expectation_intensity",
    ),
    "shortvideo": ("similarity", "likeability", "mentalizing"),
}
METHODS = ("poe", "neurostorm")


@dataclass
class SubjectData:
    subject: str
    obs: np.ndarray
    indiv: np.ndarray
    neurostorm: np.ndarray
    target_values: np.ndarray
    stim_file: np.ndarray
    trial_index: np.ndarray


@dataclass
class ResidualReference:
    stim_file: np.ndarray
    mean: np.ndarray
    count: np.ndarray

    def lookup(self) -> dict[str, int]:
        return {str(value): i for i, value in enumerate(self.stim_file)}

    def state(self) -> dict:
        return {
            "stim_file": self.stim_file.astype(str),
            "mean": self.mean.astype(np.float32),
            "count": self.count.astype(np.int16),
        }

    @classmethod
    def from_state(cls, state: dict) -> "ResidualReference":
        return cls(
            np.asarray(state["stim_file"]).astype(str),
            np.asarray(state["mean"], dtype=np.float32),
            np.asarray(state["count"], dtype=np.int16),
        )


@dataclass
class ArrayScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ArrayScaler":
        values = np.asarray(values, dtype=np.float32)
        return cls(values.mean(axis=0), values.std(axis=0) + 1e-6)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std + self.mean).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.astype(np.float32), "std": self.std.astype(np.float32)}

    @classmethod
    def from_state(cls, state: dict) -> "ArrayScaler":
        return cls(np.asarray(state["mean"], dtype=np.float32),
                   np.asarray(state["std"], dtype=np.float32))


class NeuroStormRegressor(nn.Module):
    def __init__(self, d_in: int, d_latent: int, hidden: int, d_out: int, dropout: float):
        super().__init__()
        self.expert = GaussianExpert(d_in, d_latent, hidden, dropout)
        self.head = nn.Linear(d_latent, d_out)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.expert(x)
        return {"pred": self.head(mu), "mu": mu, "var": torch.exp(logvar)}


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=np.str_)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def mean_tokens(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 3:
        return values.mean(axis=1, dtype=np.float32)
    if values.ndim == 2:
        return values
    raise ValueError(f"expected feature rank 2/3, got {values.shape}")


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_sha256(path: Path, supplied: str = "") -> str:







    value = supplied.strip().lower()
    if not value:
        return sha256_file(path)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"invalid supplied SHA-256 for {path}: {supplied!r}")
    return value


def normalized_path(value: str | os.PathLike) -> str:
    return str(Path(str(value)).expanduser().resolve())


def load_task(task: str, ours_root: Path, ns_root: Path, manifest_root: Path,
              expected_obs_ckpt: Path, expected_indiv_ckpt: Path,
              strict_encoder_weights: bool,
              methods: tuple[str, ...] | list[str] = METHODS) -> list[SubjectData]:
    manifest_paths = sorted((manifest_root / task).glob("sub-*.npz"))
    if not manifest_paths:
        raise FileNotFoundError(f"No manifests: {manifest_root / task}")
    subjects = []
    for manifest_path in manifest_paths:
        subject = manifest_path.stem
        ours_path = ours_root / task / f"{subject}.h5"
        ns_path = ns_root / task / f"{subject}.npz"
        need_ours = "poe" in methods
        need_neurostorm = "neurostorm" in methods
        if (need_ours and not ours_path.exists()) or (need_neurostorm and not ns_path.exists()):
            raise FileNotFoundError(
                f"{task}/{subject}: required features missing: "
                f"ours={ours_path.exists()} ns={ns_path.exists()} "
                f"need_ours={need_ours} need_ns={need_neurostorm}"
            )
        manifest = np.load(manifest_path, allow_pickle=False)
        stim_file = decode_strings(manifest["stim_file"])
        trial_index = np.asarray(manifest["trial_index"], dtype=np.int64)
        target_values = np.asarray(manifest["target_values"], dtype=np.float32)
        if task == "alignvideo":
            target_values[target_values < 0] = np.nan
        if need_ours:
            with h5py.File(ours_path, "r") as h5:
                for key in ("bci_obs", "bci_prior_indiv"):
                    if key not in h5:
                        raise KeyError(f"{ours_path}: missing {key}")
                obs = mean_tokens(h5["bci_obs"][...])
                indiv = mean_tokens(h5["bci_prior_indiv"][...])
                if strict_encoder_weights:
                    actual_obs = normalized_path(h5.attrs.get("cifti_ckpt", ""))
                    actual_indiv = normalized_path(h5.attrs.get("indiv_modes_ckpt", ""))
                    if actual_obs != normalized_path(expected_obs_ckpt):
                        raise RuntimeError(f"{ours_path}: unexpected obs checkpoint {actual_obs}")
                    if actual_indiv != normalized_path(expected_indiv_ckpt):
                        raise RuntimeError(f"{ours_path}: unexpected indiv checkpoint {actual_indiv}")
        else:
            obs = np.empty((len(stim_file), 0), dtype=np.float32)
            indiv = np.empty((len(stim_file), 0), dtype=np.float32)
        if need_neurostorm:
            ns = np.load(ns_path, allow_pickle=False)
            neurostorm = np.asarray(ns["feat"], dtype=np.float32)
        else:
            ns = None
            neurostorm = np.empty((len(stim_file), 0), dtype=np.float32)
        lengths = {len(stim_file), len(trial_index), len(target_values), len(obs), len(indiv), len(neurostorm)}
        if len(lengths) != 1:
            raise RuntimeError(f"{task}/{subject}: trial length mismatch {sorted(lengths)}")
        if not np.array_equal(trial_index, np.arange(len(trial_index))):
            raise RuntimeError(f"{manifest_path}: non-canonical trial_index")
        if ns is not None and "trial_index" in ns and not np.array_equal(
                np.asarray(ns["trial_index"], dtype=np.int64), trial_index):
            raise RuntimeError(f"{ns_path}: trial order mismatch")
        if not (np.isfinite(obs).all() and np.isfinite(indiv).all()
                and np.isfinite(neurostorm).all()):
            raise RuntimeError(f"{task}/{subject}: non-finite features")
        if target_values.shape != (len(stim_file), len(TARGETS[task])):
            raise RuntimeError(f"{manifest_path}: target shape {target_values.shape}")
        subjects.append(SubjectData(
            subject, obs, indiv, neurostorm, target_values, stim_file, trial_index
        ))
    return subjects


def load_folds(manifest_root: Path, task: str, subjects: list[SubjectData]) -> list[dict]:
    payload = json.loads((manifest_root / task / "outer_folds.json").read_text(encoding="utf-8"))
    expected = sorted(subject.subject for subject in subjects)
    if sorted(payload["subjects"]) != expected:
        raise RuntimeError(f"{task}: fold subjects differ from available feature subjects")
    observed = [subject for fold in payload["folds"] for subject in fold["test_subjects"]]
    if sorted(observed) != expected or len(observed) != len(set(observed)):
        raise RuntimeError(f"{task}: each subject must be outer-test exactly once")
    return payload["folds"]


def compute_reference(subjects: list[SubjectData], d_out: int) -> ResidualReference:

    by_stim: dict[str, list[list[float]]] = {}
    for subject in subjects:
        for stim in np.unique(subject.stim_file):
            row = by_stim.setdefault(str(stim), [[] for _ in range(d_out)])
            trial_mask = subject.stim_file == stim
            for target in range(d_out):
                values = subject.target_values[trial_mask, target]
                values = values[np.isfinite(values)]
                if len(values):
                    row[target].append(float(values.mean()))
    stim_file = np.asarray(sorted(by_stim), dtype=np.str_)
    mean = np.full((len(stim_file), d_out), np.nan, dtype=np.float32)
    count = np.zeros((len(stim_file), d_out), dtype=np.int16)
    for i, stim in enumerate(stim_file):
        for target, values in enumerate(by_stim[str(stim)]):
            if values:
                mean[i, target] = float(np.mean(values))
                count[i, target] = len(values)
    return ResidualReference(stim_file, mean, count)


def residualize(subject: SubjectData, reference: ResidualReference,
                min_reference_subjects: int) -> dict[str, np.ndarray]:
    d_out = subject.target_values.shape[1]
    ref_mean = np.full((len(subject.trial_index), d_out), np.nan, dtype=np.float32)
    ref_count = np.zeros((len(subject.trial_index), d_out), dtype=np.int16)
    lookup = reference.lookup()
    for trial, stim in enumerate(subject.stim_file):
        index = lookup.get(str(stim))
        if index is not None:
            ref_mean[trial] = reference.mean[index]
            ref_count[trial] = reference.count[index]
    valid = (
        np.isfinite(subject.target_values)
        & np.isfinite(ref_mean)
        & (ref_count >= min_reference_subjects)
    )
    residual = np.full(subject.target_values.shape, np.nan, dtype=np.float32)
    residual[valid] = subject.target_values[valid] - ref_mean[valid]
    return {
        "raw": subject.target_values.copy(),
        "reference_mean": ref_mean,
        "reference_count": ref_count,
        "residual": residual,
        "valid": valid,
    }


def prepare_targets(subject: SubjectData, reference: ResidualReference,
                    min_reference_subjects: int, target_mode: str) -> dict[str, np.ndarray]:








    if target_mode == "residual":
        result = residualize(subject, reference, min_reference_subjects)
        result["target"] = result["residual"]
        return result
    if target_mode != "raw":
        raise ValueError(f"unknown target_mode={target_mode!r}")
    raw = subject.target_values.copy()
    valid = np.isfinite(raw)
    return {
        "raw": raw,
        "reference_mean": np.full(raw.shape, np.nan, dtype=np.float32),
        "reference_count": np.zeros(raw.shape, dtype=np.int16),
        "residual": np.full(raw.shape, np.nan, dtype=np.float32),
        "target": raw,
        "valid": valid,
    }


def select_validation_subject(outer_train: list[SubjectData], d_out: int, seed: int,
                              min_reference_subjects: int,
                              min_target_trials: int,
                              target_mode: str = "residual") -> tuple[int, list[int]]:

    coverage = []
    eligible = []
    for index, candidate in enumerate(outer_train):
        inner_train = [subject for i, subject in enumerate(outer_train) if i != index]
        reference = compute_reference(inner_train, d_out)
        row = prepare_targets(candidate, reference, min_reference_subjects, target_mode)
        counts = row["valid"].sum(axis=0).astype(int).tolist()
        coverage.append(counts)
        if min(counts) >= min_target_trials:
            eligible.append(index)
    if eligible:
        selected = eligible[seed % len(eligible)]
    else:

        selected = max(
            range(len(outer_train)),
            key=lambda index: (min(coverage[index]), sum(coverage[index]), -index),
        )
    return selected, coverage[selected]


def write_reference(reference: ResidualReference, target_names: tuple[str, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, stim in enumerate(reference.stim_file):
        for target, target_name in enumerate(target_names):
            rows.append({
                "stim_file": str(stim),
                "target": target_name,
                "training_subject_mean": float(reference.mean[i, target]),
                "n_training_subjects": int(reference.count[i, target]),
            })
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_target_scaler(prepared_targets: list[dict[str, np.ndarray]], d_out: int) -> ArrayScaler:
    means = np.empty(d_out, dtype=np.float32)
    stds = np.empty(d_out, dtype=np.float32)
    for target in range(d_out):
        values = np.concatenate([
            row["target"][:, target][row["valid"][:, target]]
            for row in prepared_targets
        ])
        if len(values) < 2:
            raise RuntimeError(f"target {target}: fewer than two valid training targets")
        means[target] = float(values.mean())
        stds[target] = float(values.std() + 1e-6)
    return ArrayScaler(means, stds)


def feature_rows(subject: SubjectData, method: str) -> tuple[np.ndarray, ...]:
    if method == "poe":
        return subject.obs, subject.indiv
    if method == "neurostorm":
        return (subject.neurostorm,)
    raise ValueError(method)


def prepare_arrays(subjects: list[SubjectData], reference: ResidualReference,
                   method: str, min_reference_subjects: int,
                   feature_scalers: list[ArrayScaler] | None = None,
                   target_scaler: ArrayScaler | None = None,
                   target_mode: str = "residual"):
    prepared_targets = [
        prepare_targets(subject, reference, min_reference_subjects, target_mode)
        for subject in subjects
    ]
    keep = [row["valid"].any(axis=1) for row in prepared_targets]
    if not all(mask.any() for mask in keep):
        missing = [subject.subject for subject, mask in zip(subjects, keep) if not mask.any()]
        raise RuntimeError(f"{method}: subjects without usable {target_mode} labels: {missing}")
    n_inputs = 2 if method == "poe" else 1
    raw_features = [
        np.concatenate([feature_rows(subject, method)[branch][mask]
                        for subject, mask in zip(subjects, keep)], axis=0)
        for branch in range(n_inputs)
    ]
    if feature_scalers is None:
        feature_scalers = [ArrayScaler.fit(values) for values in raw_features]
    features = [scaler.transform(values) for scaler, values in zip(feature_scalers, raw_features)]
    if target_scaler is None:
        target_scaler = fit_target_scaler(prepared_targets, subjects[0].target_values.shape[1])
    y = np.concatenate([row["target"][mask] for row, mask in zip(prepared_targets, keep)], axis=0)
    valid = np.concatenate([row["valid"][mask] for row, mask in zip(prepared_targets, keep)], axis=0)
    y_scaled = np.zeros_like(y, dtype=np.float32)
    for target in range(y.shape[1]):
        target_valid = valid[:, target]
        y_scaled[target_valid, target] = (
            (y[target_valid, target] - target_scaler.mean[target]) / target_scaler.std[target]
        )
    return features, y_scaled, valid, feature_scalers, target_scaler, prepared_targets


def tensorize(prepared, device: torch.device):
    features, y, valid = prepared[:3]
    return (
        *[torch.as_tensor(value, dtype=torch.float32, device=device) for value in features],
        torch.as_tensor(y, dtype=torch.float32, device=device),
        torch.as_tensor(valid, dtype=torch.bool, device=device),
    )


def new_model(method: str, input_dim: int, d_out: int, args, device: torch.device) -> nn.Module:
    if method == "poe":
        return GaussianPoERegressor(
            input_dim, args.latent_dim, args.hidden, d_out, args.dropout
        ).to(device)
    return NeuroStormRegressor(
        input_dim, args.latent_dim, args.hidden, d_out, args.dropout
    ).to(device)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor,
               valid: torch.Tensor) -> torch.Tensor:
    difference = torch.where(valid, prediction - target, torch.zeros_like(prediction))
    return difference.square().sum() / valid.sum().clamp_min(1)


def model_loss(model: nn.Module, tensors: tuple[torch.Tensor, ...], method: str, args):
    if method == "poe":
        obs, indiv, target, valid = tensors
        output = model(obs, indiv, sample=False)
        main = masked_mse(output["poe"], target, valid)
        aux = masked_mse(output["obs"], target, valid) + masked_mse(output["indiv"], target, valid)
        mu, var = output["mu_poe"], output["var_poe"]
        kl = 0.5 * (mu.square() + var - torch.log(var + 1e-8) - 1.0).mean()
        total = main + args.aux_weight * aux + args.kl_weight * kl
    else:
        x, target, valid = tensors
        output = model(x)
        main = masked_mse(output["pred"], target, valid)
        mu, var = output["mu"], output["var"]
        kl = 0.5 * (mu.square() + var - torch.log(var + 1e-8) - 1.0).mean()
        total = main + args.kl_weight * kl
    return total, main


def train_epoch(model: nn.Module, tensors: tuple[torch.Tensor, ...], method: str,
                args, optimizer, rng: np.random.Generator) -> float:
    model.train()
    n = len(tensors[0])
    order = rng.permutation(n)
    total = 0.0
    for start in range(0, n, args.batch_size):
        index = torch.as_tensor(order[start:start + args.batch_size],
                                dtype=torch.long, device=tensors[0].device)
        batch = tuple(value[index] for value in tensors)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model_loss(model, batch, method, args)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total += float(loss.detach()) * len(index)
    return total / max(n, 1)


@torch.inference_mode()
def validation_loss(model: nn.Module, tensors: tuple[torch.Tensor, ...], method: str, args) -> float:
    model.eval()
    _, main = model_loss(model, tensors, method, args)
    return float(main.detach())


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def choose_epoch(method: str, inner_train: list[SubjectData], inner_val: list[SubjectData],
                 d_out: int, args, device: torch.device, seed: int,
                 target_mode: str = "residual"):
    reference = compute_reference(inner_train, d_out)
    train_prepared = prepare_arrays(
        inner_train, reference, method, args.min_reference_subjects,
        target_mode=target_mode,
    )
    val_prepared = prepare_arrays(
        inner_val, reference, method, args.min_reference_subjects,
        feature_scalers=train_prepared[3], target_scaler=train_prepared[4],
        target_mode=target_mode,
    )
    train_tensors = tensorize(train_prepared, device)
    val_tensors = tensorize(val_prepared, device)
    model = new_model(method, train_tensors[0].shape[1], d_out, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed)
    best_loss = math.inf
    best_epoch = 1
    bad_epochs = 0
    best_state = cpu_state_dict(model)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_tensors, method, args, optimizer, rng)
        val_loss = validation_loss(model, val_tensors, method, args)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss - args.min_delta:
            best_loss = val_loss
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    return best_epoch, best_loss, best_state, history


def re_outer(method: str, outer_train: list[SubjectData], reference: ResidualReference,
                d_out: int, epochs: int, args, device: torch.device, seed: int):
    prepared = prepare_arrays(
        outer_train, reference, method, args.min_reference_subjects,
        target_mode=args.target_mode,
    )
    tensors = tensorize(prepared, device)
    model = new_model(method, tensors[0].shape[1], d_out, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed + 7919)
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, tensors, method, args, optimizer, rng)
        history.append({"epoch": epoch, "train_loss": loss})
    return model, prepared[3], prepared[4], history


def scaler_states(scalers: list[ArrayScaler]) -> list[dict]:
    return [scaler.state() for scaler in scalers]


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> nn.Module:
    args = SimpleNamespace(**checkpoint["model_config"])
    model = new_model(
        checkpoint["method"], int(checkpoint["input_dim"]),
        len(checkpoint["target_names"]), args, device
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.inference_mode()
def predict_subject(model: nn.Module, subject: SubjectData, method: str,
                    feature_scalers: list[ArrayScaler], target_scaler: ArrayScaler,
                    reference: ResidualReference, min_reference_subjects: int,
                    device: torch.device, target_mode: str = "residual") -> dict[str, np.ndarray]:
    model.eval()
    features = feature_rows(subject, method)
    tensors = [
        torch.as_tensor(scaler.transform(value), dtype=torch.float32, device=device)
        for scaler, value in zip(feature_scalers, features)
    ]
    if method == "poe":
        prediction_scaled = model(tensors[0], tensors[1], sample=False)["poe"]
    else:
        prediction_scaled = model(tensors[0])["pred"]
    prediction_target = target_scaler.inverse(prediction_scaled.cpu().numpy())
    truth = prepare_targets(subject, reference, min_reference_subjects, target_mode)
    common = {
        "subject": np.full(len(subject.trial_index), subject.subject, dtype=np.str_),
        "trial_index": subject.trial_index.copy(),
        "stim_file": subject.stim_file.copy(),
        "target_values_raw": truth["raw"],
        "valid_mask": truth["valid"].astype(np.uint8),
    }
    if target_mode == "raw":
        common["prediction_raw_direct"] = prediction_target.astype(np.float32)
        return common
    common.update({
        "target_values_residual": truth["residual"],
        "reference_mean": truth["reference_mean"],
        "reference_count": truth["reference_count"],
        "prediction_residual": prediction_target.astype(np.float32),
        "prediction_raw": (prediction_target + truth["reference_mean"]).astype(np.float32),
    })
    return common


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.stem + "_", suffix=".pt", dir="/tmp") as temp:
        torch.save(payload, temp.name)
        destination = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        shutil.copyfile(temp.name, destination)
        os.replace(destination, path)


def atomic_npz_save(payload: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.stem + "_", suffix=".npz", dir="/tmp") as temp:
        np.savez_compressed(temp.name, **payload)
        destination = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        shutil.copyfile(temp.name, destination)
        os.replace(destination, path)


def prediction_difference(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> float:
    for key in ("trial_index", "stim_file", "valid_mask"):
        if not np.array_equal(left[key], right[key]):
            raise RuntimeError(f"replay mismatch in {key}")
    maximum = 0.0
    numeric_keys = ["prediction_residual", "prediction_raw", "prediction_raw_direct"]
    numeric_keys.extend(
        key for key in (
            "representation_low", "representation_indiv", "representation_high"
        ) if key in left or key in right
    )
    for key in numeric_keys:
        if key not in left and key not in right:
            continue
        if key not in left or key not in right:
            raise RuntimeError(f"replay mismatch: optional output {key} missing on one side")
        difference = np.abs(left[key] - right[key])
        finite = np.isfinite(difference)
        if finite.any():
            maximum = max(maximum, float(difference[finite].max()))
    return maximum


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def replay_checkpoint(checkpoint_path: Path, test_subjects: list[SubjectData],
                      run_dir: Path, device: torch.device,
                      expected_predictions: dict[str, dict[str, np.ndarray]] | None = None,
                      expected_target_mode: str | None = None) -> dict:
    checkpoint = load_checkpoint(checkpoint_path, device)
    checkpoint_target_mode = str(checkpoint.get("target_mode", "residual"))
    if expected_target_mode is not None and checkpoint_target_mode != expected_target_mode:
        raise RuntimeError(
            f"{checkpoint_path}: target_mode={checkpoint_target_mode!r} "
            f"!= requested {expected_target_mode!r}"
        )
    model = build_model_from_checkpoint(checkpoint, device)
    feature_scalers = [ArrayScaler.from_state(state) for state in checkpoint["feature_scalers"]]
    target_scaler = ArrayScaler.from_state(checkpoint["target_scaler"])
    reference = ResidualReference.from_state(checkpoint["outer_training_reference"])
    test_lookup = {subject.subject: subject for subject in test_subjects}
    if sorted(test_lookup) != sorted(checkpoint["outer_test_subjects"]):
        raise RuntimeError(f"{checkpoint_path}: outer-test subject mismatch")
    max_difference = 0.0
    for subject_name in checkpoint["outer_test_subjects"]:
        prediction = predict_subject(
            model, test_lookup[subject_name], checkpoint["method"],
            feature_scalers, target_scaler, reference,
            int(checkpoint["min_reference_subjects"]), device,
            target_mode=checkpoint_target_mode,
        )
        prediction_path = run_dir / f"{subject_name}_predictions.npz"
        if expected_predictions is not None:
            max_difference = max(
                max_difference,
                prediction_difference(expected_predictions[subject_name], prediction),
            )
        if prediction_path.exists():
            max_difference = max(
                max_difference,
                prediction_difference(load_prediction(prediction_path), prediction),
            )
        else:
            atomic_npz_save(prediction, prediction_path)
    if max_difference > 1e-6:
        raise RuntimeError(f"{checkpoint_path}: inference replay max difference {max_difference}")
    replay = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "replay_status": "exact_within_1e-6",
        "max_abs_prediction_difference": max_difference,
        "outer_test_subjects": checkpoint["outer_test_subjects"],
        "target_mode": checkpoint_target_mode,
    }
    (run_dir / "INFERENCE_REPLAY.json").write_text(
        json.dumps(replay, indent=2) + "\n", encoding="utf-8")
    return replay


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    truth, prediction = truth[valid], prediction[valid]
    result: dict[str, float | int] = {"n": int(len(truth))}
    if len(truth) < 2:
        return {**result, "pearson_r": math.nan, "spearman_rho": math.nan,
                "mae": math.nan, "rmse": math.nan, "r2": math.nan}
    error = prediction - truth
    result["mae"] = float(np.mean(np.abs(error)))
    result["rmse"] = float(np.sqrt(np.mean(error * error)))
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    result["r2"] = float(1.0 - np.sum(error * error) / denominator) if denominator > 0 else math.nan
    if len(truth) >= 3 and np.std(truth) > 0 and np.std(prediction) > 0:
        result["pearson_r"] = float(pearsonr(truth, prediction).statistic)
        result["spearman_rho"] = float(spearmanr(truth, prediction).statistic)
    else:
        result["pearson_r"] = math.nan
        result["spearman_rho"] = math.nan
    return result


def metric_rows(task: str, method: str, seed: int | str, subject: str,
                prediction: dict[str, np.ndarray], target_mode: str = "residual") -> list[dict]:
    valid = prediction["valid_mask"].astype(bool)
    rows = []
    for target, target_name in enumerate(TARGETS[task]):
        mask = valid[:, target]
        metric_specs = (
            (("raw_direct", "target_values_raw", "prediction_raw_direct"),)
            if target_mode == "raw" else
            (
                ("residual", "target_values_residual", "prediction_residual"),
                ("raw_reconstructed", "target_values_raw", "prediction_raw"),
            )
        )
        for scale, truth_key, prediction_key in metric_specs:
            metrics = regression_metrics(
                prediction[truth_key][mask, target], prediction[prediction_key][mask, target]
            )
            rows.append({
                "task": task, "method": method, "seed": seed,
                "subject": subject, "target": target_name, "scale": scale,
                **metrics,
            })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fisher_mean(values: list[float]) -> float:
    values = [value for value in values if np.isfinite(value)]
    if not values:
        return math.nan
    clipped = np.clip(np.asarray(values, dtype=float), -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(clipped))))


def fdr_bh(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result.tolist()
    order = finite[np.argsort(values[finite])]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def aggregate_subject_rows(task: str, ensemble_rows: list[dict], methods: list[str],
                           task_out: Path, target_mode: str = "residual") -> None:
    aggregate_rows = []
    for method in methods:
        for target in TARGETS[task]:
            scales = ("raw_direct",) if target_mode == "raw" else ("residual", "raw_reconstructed")
            for scale in scales:
                selected = [
                    row for row in ensemble_rows
                    if row["method"] == method and row["target"] == target and row["scale"] == scale
                ]
                pearson = [float(row["pearson_r"]) for row in selected if np.isfinite(row["pearson_r"])]
                aggregate_rows.append({
                    "task": task, "method": method, "target": target, "scale": scale,
                    "n_subjects": len(selected),
                    "n_subjects_finite_pearson": len(pearson),
                    "pearson_fisher_r": fisher_mean(pearson),
                    "pearson_arithmetic_mean": float(np.mean(pearson)) if pearson else math.nan,
                    "pearson_between_subject_sd": float(np.std(pearson, ddof=1)) if len(pearson) > 1 else math.nan,
                    "spearman_mean": float(np.nanmean([row["spearman_rho"] for row in selected])),
                    "mae_mean": float(np.nanmean([row["mae"] for row in selected])),
                    "rmse_mean": float(np.nanmean([row["rmse"] for row in selected])),
                    "r2_mean": float(np.nanmean([row["r2"] for row in selected])),
                })
    write_csv(aggregate_rows, task_out / "AGGREGATE_SEED_ENSEMBLE_METRICS.csv")

    comparisons = []
    if set(("poe", "neurostorm")).issubset(methods):
        for target in TARGETS[task]:
            by_method = {}
            for method in ("poe", "neurostorm"):
                by_method[method] = {
                    row["subject"]: float(row["pearson_r"])
                    for row in ensemble_rows
                    if row["method"] == method and row["target"] == target
                    and row["scale"] == ("raw_direct" if target_mode == "raw" else "residual")
                    and np.isfinite(row["pearson_r"])
                }
            paired_subjects = sorted(set(by_method["poe"]) & set(by_method["neurostorm"]))
            ours = np.asarray([by_method["poe"][subject] for subject in paired_subjects])
            baseline = np.asarray([by_method["neurostorm"][subject] for subject in paired_subjects])
            if len(ours) >= 2 and np.any(ours != baseline):
                p_value = float(wilcoxon(ours, baseline, alternative="two-sided").pvalue)
            else:
                p_value = math.nan
            comparisons.append({
                "task": task, "target": target,
                "scale": "raw_direct" if target_mode == "raw" else "residual",
                "n_paired_subjects": len(paired_subjects),
                "poe_fisher_r": fisher_mean(ours.tolist()),
                "neurostorm_fisher_r": fisher_mean(baseline.tolist()),
                "mean_paired_difference_raw_r": float(np.mean(ours - baseline)) if len(ours) else math.nan,
                "wilcoxon_two_sided_p": p_value,
            })
        for row, q_value in zip(comparisons, fdr_bh([row["wilcoxon_two_sided_p"] for row in comparisons])):
            row["fdr_bh_q_within_task"] = q_value
        write_csv(comparisons, task_out / "POE_VS_NEUROSTORM_PAIRED.csv")


def aggregate_task(task: str, folds: list[dict], seeds: list[int], methods: list[str],
                   task_out: Path, target_mode: str = "residual") -> None:
    seed_rows = []
    ensemble_rows = []
    for fold in folds:
        fold_index = int(fold["fold"])
        for method in methods:
            for subject in fold["test_subjects"]:
                predictions = []
                for seed in seeds:
                    path = task_out / f"fold-{fold_index:02d}" / f"seed-{seed:02d}" / method / f"{subject}_predictions.npz"
                    if not path.exists():
                        raise FileNotFoundError(path)
                    prediction = load_prediction(path)
                    predictions.append(prediction)
                    seed_rows.extend(metric_rows(
                        task, method, seed, subject, prediction, target_mode=target_mode
                    ))
                ensemble = {key: value.copy() for key, value in predictions[0].items()}
                prediction_keys = (
                    ("prediction_raw_direct",) if target_mode == "raw"
                    else ("prediction_residual", "prediction_raw")
                )
                for key in prediction_keys:
                    ensemble[key] = np.mean(
                        np.stack([prediction[key] for prediction in predictions], axis=0), axis=0
                    ).astype(np.float32)
                ensemble_rows.extend(metric_rows(
                    task, method, "ensemble", subject, ensemble, target_mode=target_mode
                ))
                atomic_npz_save(ensemble, task_out / "seed_ensemble_predictions" / method / f"{subject}.npz")
    write_csv(seed_rows, task_out / "PER_SEED_PER_SUBJECT_METRICS.csv")
    write_csv(ensemble_rows, task_out / "PER_SUBJECT_METRICS_SEED_ENSEMBLE.csv")
    aggregate_subject_rows(task, ensemble_rows, methods, task_out, target_mode=target_mode)


def train_or_replay(task: str, fold: dict, seed: int, method: str,
                    subject_lookup: dict[str, SubjectData], task_out: Path,
                    args, device: torch.device) -> None:
    fold_index = int(fold["fold"])
    run_dir = task_out / f"fold-{fold_index:02d}" / f"seed-{seed:02d}" / method
    checkpoint_path = run_dir / "checkpoint.pt"
    outer_train = [subject_lookup[name] for name in fold["train_subjects"]]
    outer_test = [subject_lookup[name] for name in fold["test_subjects"]]
    if checkpoint_path.exists() and not args.force:
        replay_checkpoint(
            checkpoint_path, outer_test, run_dir, device,
            expected_target_mode=args.target_mode,
        )
        print(f"[{task} fold={fold_index} seed={seed} {method}] replayed existing", flush=True)
        return
    if args.mode == "inference-only":
        raise FileNotFoundError(f"inference-only checkpoint missing: {checkpoint_path}")

    d_out = len(TARGETS[task])
    outer_reference = compute_reference(outer_train, d_out)
    reference_path = task_out / f"fold-{fold_index:02d}" / "outer_training_reference.csv"
    if not reference_path.exists():
        write_reference(outer_reference, TARGETS[task], reference_path)
    val_index, validation_coverage = select_validation_subject(
        outer_train, d_out, seed + fold_index,
        args.min_reference_subjects, args.min_validation_target_trials,
        target_mode=args.target_mode,
    )
    inner_val = [outer_train[val_index]]
    inner_train = [subject for i, subject in enumerate(outer_train) if i != val_index]
    model_seed = seed + fold_index * 1000 + (0 if method == "poe" else 500_000)
    set_seed(model_seed)
    best_epoch, best_val_loss, inner_best_state, inner_history = choose_epoch(
        method, inner_train, inner_val, d_out, args, device, model_seed,
        target_mode=args.target_mode,
    )
    set_seed(model_seed + 17)
    model, feature_scalers, target_scaler, outer_history = re_outer(
        method, outer_train, outer_reference, d_out, best_epoch,
        args, device, model_seed + 17,
    )
    in_memory = {
        subject.subject: predict_subject(
            model, subject, method, feature_scalers, target_scaler,
            outer_reference, args.min_reference_subjects, device,
            target_mode=args.target_mode,
        )
        for subject in outer_test
    }
    model_config = {
        "latent_dim": args.latent_dim, "hidden": args.hidden, "dropout": args.dropout,
        "aux_weight": args.aux_weight, "kl_weight": args.kl_weight,
    }
    checkpoint = {
        "format_version": 1,
        "training_complete": True,
        "task": task,
        "target_mode": args.target_mode,
        "method": method,
        "fold": fold_index,
        "seed": seed,
        "model_seed": model_seed,
        "target_names": list(TARGETS[task]),
        "outer_train_subjects": list(fold["train_subjects"]),
        "outer_test_subjects": list(fold["test_subjects"]),
        "inner_train_subjects": [subject.subject for subject in inner_train],
        "inner_validation_subject": inner_val[0].subject,
        "inner_validation_valid_trials_per_target": validation_coverage,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "input_dim": int(feature_rows(outer_train[0], method)[0].shape[1]),
        "model_config": model_config,
        "training_config": {
            "epochs_max": args.epochs, "patience": args.patience,
            "min_delta": args.min_delta, "batch_size": args.batch_size,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "min_validation_target_trials": args.min_validation_target_trials,
        },
        "min_reference_subjects": args.min_reference_subjects,
        "feature_scalers": scaler_states(feature_scalers),
        "target_scaler": target_scaler.state(),
        "outer_training_reference": outer_reference.state(),
        "model_state": cpu_state_dict(model),
        "inner_best_model_state": inner_best_state,
        "inner_history": inner_history,
        "outer_re_history": outer_history,
        "frozen_feature_sources": {
            "ours_root": str(args.ours_root),
            "neurostorm_root": str(args.ns_root),
            "manifest_root": str(args.manifest_root),
            "obs_encoder_checkpoint": str(args.expected_obs_checkpoint),
            "indiv_encoder_checkpoint": str(args.expected_indiv_checkpoint),
            "neurostorm_encoder_checkpoint": str(args.expected_neurostorm_checkpoint),
            "encoder_checkpoint_sha256": dict(args.encoder_checkpoint_sha256),
            "encoder_training": "frozen; no encoder instantiated by decoder",
        },
        "target_protocol": (
            "raw observed rating; finite labels only; no stimulus reference subtraction"
            if args.target_mode == "raw" else
            "residual = observed rating - exact stim_file x target mean over "
            "outer-training subjects only; within-subject duplicates averaged "
            "before equal-subject mean"
        ),


        "residual_protocol": (
            "unused in target_mode=raw"
            if args.target_mode == "raw" else
            "exact stim_file x target mean over outer-training subjects only; "
            "within-subject duplicates averaged before equal-subject mean"
        ),
    }
    atomic_torch_save(checkpoint, checkpoint_path)
    replay = replay_checkpoint(
        checkpoint_path, outer_test, run_dir, device,
        expected_predictions=in_memory, expected_target_mode=args.target_mode,
    )
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": replay["checkpoint_sha256"],
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "inner_validation_subject": inner_val[0].subject,
        "inner_validation_valid_trials_per_target": validation_coverage,
        "inference_replay": replay["replay_status"],
    }
    (run_dir / "TRAINING_SUMMARY.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"[{task} fold={fold_index} seed={seed} {method}] "
        f"epoch={best_epoch} val={best_val_loss:.6f} replay=OK", flush=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument(
        "--target-mode", choices=("residual", "raw"), default="residual",
        help="residualize by outer-training stimulus means, or train directly on raw ratings",
    )
    parser.add_argument("--seeds", default=",".join(str(i) for i in range(10)))
    parser.add_argument("--ours-root", type=Path, default=DEFAULT_OURS)
    parser.add_argument("--ns-root", type=Path, default=DEFAULT_NS)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--expected-obs-checkpoint", type=Path, default=DEFAULT_OBS_CKPT)
    parser.add_argument("--expected-indiv-checkpoint", type=Path, default=DEFAULT_INDIV_CKPT)
    parser.add_argument("--expected-neurostorm-checkpoint", type=Path, default=DEFAULT_NS_CKPT)
    parser.add_argument("--expected-obs-sha256", default="")
    parser.add_argument("--expected-indiv-sha256", default="")
    parser.add_argument("--expected-neurostorm-sha256", default="")
    parser.add_argument("--no-strict-encoder-weights", action="store_true")
    parser.add_argument(
        "--mode", choices=("train", "inference-only", "aggregate-only"),
        default="train",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda:6")
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
    parser.add_argument("--min-reference-subjects", type=int, default=2)
    parser.add_argument("--min-validation-target-trials", type=int, default=5)
    parser.add_argument("--folds", default="", help="optional comma-separated fold indices")
    parser.add_argument(
        "--skip-aggregate", action="store_true",
        help="train/replay only selected folds; defer task-wide tables to a complete final pass",
    )
    parser.add_argument(
        "--skip-protocol-write", action="store_true",
        help="avoid a shared protocol-file write in parallel shards; final pass must write it",
    )
    args = parser.parse_args()

    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    selected_folds = {int(value) for value in args.folds.split(",") if value.strip()}
    if any(task not in TASKS for task in tasks):
        raise ValueError(f"tasks must be a subset of {TASKS}")
    if any(method not in METHODS for method in methods):
        raise ValueError(f"methods must be a subset of {METHODS}")
    if not seeds:
        raise ValueError("at least one seed is required")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.mode == "aggregate-only":


        args.encoder_checkpoint_sha256 = {
            "obs": None, "indiv": None, "neurostorm": None,
        }
    else:
        args.encoder_checkpoint_sha256 = {
            "obs": (
                checkpoint_sha256(
                    args.expected_obs_checkpoint, args.expected_obs_sha256
                ) if "poe" in methods else None
            ),
            "indiv": (
                checkpoint_sha256(
                    args.expected_indiv_checkpoint, args.expected_indiv_sha256
                ) if "poe" in methods else None
            ),
            "neurostorm": (
                checkpoint_sha256(
                    args.expected_neurostorm_checkpoint,
                    args.expected_neurostorm_sha256,
                )
                if "neurostorm" in methods else None
            ),
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_protocol = {
        "tasks": tasks, "methods": methods, "seeds": seeds,
        "target_mode": args.target_mode,
        "outer_cv": "defined by task manifest; each subject tested exactly once",
        "inner_validation": "one outer-training subject; remaining subjects inner-train",
        "target_definition": (
            "raw observed rating; no reference subtraction"
            if args.target_mode == "raw" else
            "observed rating minus outer-training exact stimulus x target mean"
        ),
        "residual_reference": (
            "unused for labels/predictions in raw mode"
            if args.target_mode == "raw" else
            "training subjects only, exact stimulus x target"
        ),
        "encoder_weights": "frozen feature extraction; decoder-only training",
        "checkpoint_policy": "final outer decoder + inner best weights + all scalers/reference/splits",
        "inference_policy": "every checkpoint immediately reloaded and prediction replay verified",
        "early_stopping_patience": args.patience,
        "minimum_validation_trials_per_target": args.min_validation_target_trials,
        "frozen_encoder_checkpoint_sha256": args.encoder_checkpoint_sha256,
    }
    if not args.skip_protocol_write:
        (args.output_root / "EXPERIMENT_PROTOCOL.json").write_text(
            json.dumps(run_protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for task in tasks:
        if args.mode == "aggregate-only":
            fold_payload = json.loads(
                (args.manifest_root / task / "outer_folds.json").read_text(encoding="utf-8")
            )
            all_folds = fold_payload["folds"]
            folds = all_folds
            if selected_folds:
                folds = [fold for fold in folds if int(fold["fold"]) in selected_folds]
            if len(folds) != len(all_folds):
                raise RuntimeError(
                    f"{task}: aggregation requires all {len(all_folds)} folds; got {len(folds)}"
                )
            task_out = args.output_root / task
            aggregate_task(
                task, folds, seeds, methods, task_out,
                target_mode=args.target_mode,
            )
            print(f"[{task}] aggregate-only complete", flush=True)
            continue
        subjects = load_task(
            task, args.ours_root, args.ns_root, args.manifest_root,
            args.expected_obs_checkpoint, args.expected_indiv_checkpoint,
            not args.no_strict_encoder_weights,
            methods,
        )
        all_folds = load_folds(args.manifest_root, task, subjects)
        folds = all_folds
        if selected_folds:
            folds = [fold for fold in folds if int(fold["fold"]) in selected_folds]
        subject_lookup = {subject.subject: subject for subject in subjects}
        task_out = args.output_root / task
        task_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[{task}] subjects={len(subjects)} folds={len(folds)} seeds={len(seeds)}", flush=True)
        for fold in folds:
            for seed in seeds:
                for method in methods:
                    train_or_replay(
                        task, fold, seed, method, subject_lookup, task_out, args, device
                    )
        if not args.skip_aggregate:
            if len(folds) != len(all_folds):
                raise RuntimeError(
                    f"{task}: aggregation requires all {len(all_folds)} folds; "
                    f"got {len(folds)}. Use --skip-aggregate for shards."
                )
            aggregate_task(
                task, folds, seeds, methods, task_out,
                target_mode=args.target_mode,
            )
    print(
        f"[DONE] all requested {args.target_mode} decoders/checkpoints/replays complete",
        flush=True,
    )


if __name__ == "__main__":
    main()
