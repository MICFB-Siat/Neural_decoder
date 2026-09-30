





















from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr, wilcoxon


NARRATIVE_TARGETS = [
    "feeling_valence", "feeling_intensity",
    "expectation_valence", "expectation_intensity",
]
SHORTVIDEO_TARGETS = ["similarity", "likeability", "mentalizing"]
NARRATIVE_SCALE_X0 = 960.0
NARRATIVE_SCALE_Y0 = 707.0
NARRATIVE_SCALE_RADIUS = 500.0


@dataclass
class SubjectData:
    subject: str
    obs: np.ndarray
    indiv: np.ndarray
    y: np.ndarray
    condition: np.ndarray
    trial_index: np.ndarray
    metadata: dict[str, np.ndarray]


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, x: np.ndarray, axis=0) -> "Standardizer":
        return cls(np.nanmean(x, axis=axis), np.nanstd(x, axis=axis) + 1e-6)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)

    def state(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


class GaussianExpert(nn.Module):
    def __init__(self, d_in: int, d_latent: int, hidden: int, dropout: float):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden, d_latent)
        self.logvar = nn.Linear(hidden, d_latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.mu(h), self.logvar(h).clamp(-5.0, 5.0)


class GaussianPoERegressor(nn.Module):


    def __init__(self, d_in: int, d_latent: int, hidden: int, d_out: int, dropout: float):
        super().__init__()
        self.obs_expert = GaussianExpert(d_in, d_latent, hidden, dropout)
        self.indiv_expert = GaussianExpert(d_in, d_latent, hidden, dropout)
        self.obs_head = nn.Linear(d_latent, d_out)
        self.indiv_head = nn.Linear(d_latent, d_out)
        self.poe_head = nn.Linear(d_latent, d_out)

    def forward(self, obs: torch.Tensor, indiv: torch.Tensor, sample: bool = False) -> dict[str, torch.Tensor]:
        mu_o, lv_o = self.obs_expert(obs)
        mu_i, lv_i = self.indiv_expert(indiv)
        tau_o = torch.exp(-lv_o)
        tau_i = torch.exp(-lv_i)

        var_p = (1.0 + tau_o + tau_i).reciprocal()
        mu_p = var_p * (tau_o * mu_o + tau_i * mu_i)
        z_p = mu_p
        if sample and self.training:
            z_p = mu_p + torch.randn_like(mu_p) * torch.sqrt(var_p)
        alpha_obs = tau_o / (tau_o + tau_i + 1e-8)
        return {
            "obs": self.obs_head(mu_o),
            "indiv": self.indiv_head(mu_i),
            "poe": self.poe_head(z_p),
            "mu_poe": mu_p,
            "var_poe": var_p,
            "alpha_obs": alpha_obs,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_strings(x: np.ndarray) -> np.ndarray:
    return np.asarray([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in x])


def narrative_xy_to_semantic(y: np.ndarray) -> np.ndarray:







    y = np.asarray(y, dtype=np.float32)
    out = np.full_like(y, np.nan, dtype=np.float32)
    for src, dst in ((0, 0), (2, 2)):
        x = y[:, src]
        yy = y[:, src + 1]
        dx = x - NARRATIVE_SCALE_X0
        dy = yy - NARRATIVE_SCALE_Y0
        out[:, dst] = dx / NARRATIVE_SCALE_RADIUS
        out[:, dst + 1] = np.sqrt(dx * dx + dy * dy) / NARRATIVE_SCALE_RADIUS
    return out


def load_task(feature_root: str, task: str) -> list[SubjectData]:
    subjects = []
    for path in sorted(glob.glob(os.path.join(feature_root, task, "sub-*.h5"))):
        with h5py.File(path, "r") as f:
            missing = [k for k in ("bci_obs", "bci_prior_indiv", "labels_subjective") if k not in f]
            if missing:
                raise KeyError(f"{path}: missing {missing}")

            obs = f["bci_obs"][...].astype(np.float32).mean(axis=1)
            indiv = f["bci_prior_indiv"][...].astype(np.float32).mean(axis=1)
            y = f["labels_subjective"][...].astype(np.float32)
            n = len(y)
            meta = {}
            for key in ("story_id", "situation_index", "video_id", "condition", "mentalizing_level", "run", "session"):
                if key in f:
                    a = f[key][...]
                    meta[key] = decode_strings(a) if a.dtype.kind in "SO" else a
            if task == "narratives":
                if y.ndim != 2 or y.shape[1] != 4:
                    raise ValueError(f"{path}: narratives labels must be (N,4), got {y.shape}")
                meta["raw_labels_xy"] = y.copy()
                y = narrative_xy_to_semantic(y)
                condition = np.full(n, -1, dtype=np.int64)
            else:
                if "condition" not in meta:
                    raise KeyError(f"{path}: shortvideo condition metadata was not preserved")
                cmap = {name: j for j, name in enumerate(SHORTVIDEO_TARGETS)}
                condition = np.asarray([cmap.get(str(v), -1) for v in meta["condition"]], dtype=np.int64)
                if np.any(condition < 0):
                    raise ValueError(f"{path}: unknown shortvideo condition")
                y = y.reshape(-1)
            subject = str(f.attrs.get("subject_id", Path(path).stem))
        if obs.shape != indiv.shape or len(obs) != len(y):
            raise ValueError(f"{path}: incompatible shapes obs={obs.shape} indiv={indiv.shape} y={y.shape}")
        trial_index = np.arange(len(y), dtype=np.int64)



        valid = np.any(np.isfinite(y), axis=1) if task == "narratives" else np.isfinite(y)
        if not valid.all():
            obs, indiv, y, condition, trial_index = (
                obs[valid], indiv[valid], y[valid], condition[valid], trial_index[valid]
            )
            meta = {
                key: value[valid] if isinstance(value, np.ndarray) and len(value) == len(valid) else value
                for key, value in meta.items()
            }
        subjects.append(SubjectData(subject, obs, indiv, y, condition, trial_index, meta))
    if len(subjects) < 3:
        raise RuntimeError(f"{task}: need at least 3 subjects, found {len(subjects)} in {feature_root}")
    return subjects


def fit_x_scalers(subjects: list[SubjectData]) -> tuple[Standardizer, Standardizer]:
    return (
        Standardizer.fit(np.concatenate([s.obs for s in subjects], axis=0)),
        Standardizer.fit(np.concatenate([s.indiv for s in subjects], axis=0)),
    )


def fit_y_scaler(subjects: list[SubjectData], task: str) -> Standardizer:
    if task == "narratives":
        y = np.concatenate([s.y for s in subjects], axis=0)
        return Standardizer.fit(y)
    means, stds = [], []
    for c in range(3):
        vals = np.concatenate([np.log1p(s.y[s.condition == c]) for s in subjects])
        means.append(float(vals.mean()))
        stds.append(float(vals.std() + 1e-6))
    return Standardizer(np.asarray(means), np.asarray(stds))


def transform_y(y: np.ndarray, condition: np.ndarray, task: str, scaler: Standardizer) -> np.ndarray:
    if task == "narratives":
        return scaler.transform(y)
    out = np.empty(len(y), dtype=np.float32)
    z = np.log1p(np.maximum(y, 0.0))
    for c in range(3):
        idx = condition == c
        out[idx] = (z[idx] - scaler.mean[c]) / scaler.std[c]
    return out


def inverse_y(pred: np.ndarray, condition: np.ndarray, task: str, scaler: Standardizer) -> np.ndarray:
    if task == "narratives":
        return scaler.inverse(pred)
    out = np.empty(len(condition), dtype=np.float32)
    for c in range(3):
        idx = condition == c
        z = pred[idx, c] * scaler.std[c] + scaler.mean[c]
        out[idx] = np.maximum(np.expm1(z), 0.0)
    return out


def stack_subjects(
    subjects: list[SubjectData], task: str, xo: Standardizer, xi: Standardizer, ys: Standardizer
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = xo.transform(np.concatenate([s.obs for s in subjects], axis=0))
    indiv = xi.transform(np.concatenate([s.indiv for s in subjects], axis=0))
    condition = np.concatenate([s.condition for s in subjects])
    y_raw = np.concatenate([s.y for s in subjects], axis=0)
    y = transform_y(y_raw, condition, task, ys)
    owner = np.concatenate([np.full(len(s.y), s.subject, dtype=object) for s in subjects])
    return obs, indiv, y, condition, owner


def selected_loss(pred: torch.Tensor, y: torch.Tensor, condition: torch.Tensor, task: str) -> torch.Tensor:
    if task == "narratives":
        mask = torch.isfinite(y)
        return F.smooth_l1_loss(pred[mask], y[mask])
    p = pred.gather(1, condition[:, None]).squeeze(1)
    return F.smooth_l1_loss(p, y)


def model_loss(out: dict[str, torch.Tensor], y: torch.Tensor, condition: torch.Tensor, task: str,
               aux_weight: float, kl_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    main = selected_loss(out["poe"], y, condition, task)
    aux = selected_loss(out["obs"], y, condition, task) + selected_loss(out["indiv"], y, condition, task)
    mu, var = out["mu_poe"], out["var_poe"]
    kl = 0.5 * (mu.square() + var - torch.log(var + 1e-8) - 1.0).mean()
    return main + aux_weight * aux + kl_weight * kl, main


def tensorize(arrays, device):
    obs, indiv, y, condition, _ = arrays
    return (
        torch.as_tensor(obs, dtype=torch.float32, device=device),
        torch.as_tensor(indiv, dtype=torch.float32, device=device),
        torch.as_tensor(y, dtype=torch.float32, device=device),
        torch.as_tensor(condition, dtype=torch.long, device=device),
    )


def train_epoch(model, tensors, optimizer, task, args, rng) -> float:
    model.train()
    n = len(tensors[0])
    order = rng.permutation(n)
    total = 0.0
    total_n = 0
    for start in range(0, n, args.batch_size):
        idx = torch.as_tensor(order[start:start + args.batch_size], dtype=torch.long, device=tensors[0].device)
        obs, indiv, y, condition = [x[idx] for x in tensors]
        optimizer.zero_grad(set_to_none=True)
        out = model(obs, indiv, sample=args.sample_latent)
        loss, _ = model_loss(out, y, condition, task, args.aux_weight, args.kl_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total += float(loss.detach()) * len(idx)
        total_n += len(idx)
    return total / max(1, total_n)


@torch.no_grad()
def eval_loss(model, tensors, task, args) -> float:
    model.eval()
    out = model(tensors[0], tensors[1], sample=False)
    _, main = model_loss(out, tensors[2], tensors[3], task, args.aux_weight, args.kl_weight)
    return float(main.detach())


def new_model(task: str, d_in: int, args, device) -> GaussianPoERegressor:
    d_out = 4 if task == "narratives" else 3
    return GaussianPoERegressor(d_in, args.latent_dim, args.hidden, d_out, args.dropout).to(device)


def choose_epoch(train_subjects, val_subjects, task, args, device, seed):
    xo, xi = fit_x_scalers(train_subjects)
    ys = fit_y_scaler(train_subjects, task)
    train = tensorize(stack_subjects(train_subjects, task, xo, xi, ys), device)
    val = tensorize(stack_subjects(val_subjects, task, xo, xi, ys), device)
    model = new_model(task, train[0].shape[1], args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_ep, bad = math.inf, 1, 0
    history = []
    rng = np.random.default_rng(seed)
    for ep in range(1, args.epochs + 1):
        tr = train_epoch(model, train, opt, task, args, rng)
        va = eval_loss(model, val, task, args)
        history.append({"epoch": ep, "train_loss": tr, "val_loss": va})
        if va < best - args.min_delta:
            best, best_ep, bad = va, ep, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    return best_ep, best, history


def re_outer(train_subjects, task, args, device, seed, epochs):
    xo, xi = fit_x_scalers(train_subjects)
    ys = fit_y_scaler(train_subjects, task)
    arrays = stack_subjects(train_subjects, task, xo, xi, ys)
    tensors = tensorize(arrays, device)
    model = new_model(task, tensors[0].shape[1], args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed + 7919)
    for _ in range(epochs):
        train_epoch(model, tensors, opt, task, args, rng)
    return model, xo, xi, ys


@torch.no_grad()
def predict_subject(model, subject, task, xo, xi, ys, device):
    model.eval()
    obs = torch.as_tensor(xo.transform(subject.obs), dtype=torch.float32, device=device)
    indiv = torch.as_tensor(xi.transform(subject.indiv), dtype=torch.float32, device=device)
    out = model(obs, indiv, sample=False)
    predictions = {}
    for method in ("obs", "indiv", "poe"):
        z = out[method].cpu().numpy()
        predictions[method] = inverse_y(z, subject.condition, task, ys)
    alpha = out["alpha_obs"].cpu().numpy()
    return predictions, alpha


def regression_metrics(y, pred) -> dict[str, float]:
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    if len(y) < 3:
        return {"n": len(y), "pearson_r": np.nan, "spearman_rho": np.nan,
                "mae": np.nan, "rmse": np.nan, "r2": np.nan}
    err = pred - y
    denom = np.sum((y - y.mean()) ** 2)
    return {
        "n": int(len(y)),
        "pearson_r": float(pearsonr(y, pred).statistic) if y.std() > 1e-12 and pred.std() > 1e-12 else np.nan,
        "spearman_rho": float(spearmanr(y, pred).statistic) if y.std() > 1e-12 and pred.std() > 1e-12 else np.nan,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1.0 - np.sum(err ** 2) / denom) if denom > 1e-12 else np.nan,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", prefix=path.stem + "_", suffix=".csv", dir="/tmp", delete=False
    ) as f:
        local_path = Path(f.name)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    try:
        last_error = None
        for attempt in range(1, 6):
            destination_tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
            try:
                shutil.copyfile(local_path, destination_tmp)
                os.replace(destination_tmp, path)
                return
            except OSError as exc:
                last_error = exc
                try:
                    destination_tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                if attempt < 5:
                    time.sleep(float(attempt))
        raise RuntimeError(f"unable to write {path} after 5 attempts") from last_error
    finally:
        local_path.unlink(missing_ok=True)


def atomic_torch_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def standardizer_from_state(state: dict) -> Standardizer:
    if not isinstance(state, dict) or set(state) != {"mean", "std"}:
        raise ValueError("invalid saved Standardizer state")
    scaler = Standardizer(np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32))
    if scaler.mean.shape != scaler.std.shape or not np.isfinite(scaler.mean).all() or not np.isfinite(scaler.std).all():
        raise ValueError("non-finite or incompatible saved Standardizer state")
    if np.any(scaler.std <= 0):
        raise ValueError("saved Standardizer has non-positive scale")
    return scaler


def feature_fingerprint(feature_root: str, task: str) -> tuple[str, list[dict]]:
    manifest = []
    for path in sorted(glob.glob(os.path.join(feature_root, task, "sub-*.h5"))):
        stat = os.stat(path)
        with h5py.File(path, "r") as f:
            manifest.append({
                "subject": str(f.attrs.get("subject_id", Path(path).stem)),
                "path": os.path.abspath(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "bci_obs_shape": list(f["bci_obs"].shape),
                "bci_prior_indiv_shape": list(f["bci_prior_indiv"].shape),
                "cifti_ckpt_sha256": str(f.attrs.get("cifti_ckpt_sha256", "")),
                "indiv_modes_ckpt_sha256": str(f.attrs.get("indiv_modes_ckpt_sha256", "")),
                "source_h5": str(f.attrs.get("source_h5", "")),
            })
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), manifest


def decoder_config(args) -> dict:
    keys = (
        "epochs", "patience", "min_delta", "batch_size", "latent_dim", "hidden",
        "dropout", "lr", "weight_decay", "aux_weight", "kl_weight", "grad_clip",
        "sample_latent",
    )
    return {key: getattr(args, key) for key in keys}


def _finite_mean(values) -> float:
    values = np.asarray([float(v) for v in values], dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def fisher_mean(values) -> tuple[float, float, float]:

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan

    z = np.arctanh(np.clip(values, -1.0 + 1e-7, 1.0 - 1e-7))
    mean_z = float(z.mean())
    std_z = float(z.std(ddof=1)) if len(z) > 1 else 0.0
    return float(np.tanh(mean_z)), mean_z, std_z


def ensemble_subject_metrics(
    predictions: list[dict], task: str, methods=("obs", "indiv", "poe")
) -> tuple[list[dict], list[dict]]:






    by_trial = defaultdict(list)
    for row in predictions:
        by_trial[(str(row["test_subject"]), int(row["trial_index"]))].append(row)

    ensemble_rows = []
    for (subject, trial_index), rows in sorted(by_trial.items()):
        out = {
            "task": task,
            "test_subject": subject,
            "trial_index": trial_index,
            "n_seeds": len(rows),
        }
        if task == "narratives":
            for target in NARRATIVE_TARGETS:
                out[f"true_{target}"] = float(rows[0][f"true_{target}"])
                for method in methods:
                    out[f"pred_{method}_{target}"] = _finite_mean(
                        r[f"pred_{method}_{target}"] for r in rows
                    )
        else:
            out["condition"] = str(rows[0]["condition"])
            out["true_response_angle"] = float(rows[0]["true_response_angle"])
            for method in methods:
                out[f"pred_{method}_response_angle"] = _finite_mean(
                    r[f"pred_{method}_response_angle"] for r in rows
                )
        ensemble_rows.append(out)

    subjects = sorted({r["test_subject"] for r in ensemble_rows})
    subject_rows = []
    for subject in subjects:
        rows = [r for r in ensemble_rows if r["test_subject"] == subject]
        target_names = NARRATIVE_TARGETS if task == "narratives" else SHORTVIDEO_TARGETS
        for method in methods:
            for target in target_names:
                if task == "narratives":
                    selected = rows
                    y = [r[f"true_{target}"] for r in selected]
                    pred = [r[f"pred_{method}_{target}"] for r in selected]
                else:
                    selected = [r for r in rows if r["condition"] == target]
                    y = [r["true_response_angle"] for r in selected]
                    pred = [r[f"pred_{method}_response_angle"] for r in selected]
                met = regression_metrics(y, pred)
                subject_rows.append({
                    "method": method,
                    "target": target,
                    "subject": subject,
                    "aggregation": "trialwise_mean_prediction_across_seeds",
                    "n_seeds": int(min((r["n_seeds"] for r in selected), default=0)),
                    **met,
                })
    return subject_rows, ensemble_rows


def summarize_subject_metrics(subject_rows: list[dict]) -> list[dict]:

    out = []
    present = {r["method"] for r in subject_rows}
    preferred = ["obs", "indiv", "poe", "neurostorm"]
    methods = [m for m in preferred if m in present] + sorted(present.difference(preferred))
    for method in methods:
        targets = sorted({r["target"] for r in subject_rows if r["method"] == method})
        for target in targets:
            rows = [r for r in subject_rows if r["method"] == method and r["target"] == target]
            pearson = np.asarray([r["pearson_r"] for r in rows], dtype=float)
            mean_r, mean_z, std_z = fisher_mean(pearson)
            row = {
                "method": method,
                "target": target,
                "n_subjects": int(np.isfinite(pearson).sum()),
                "mean_pearson_r": mean_r,
                "mean_fisher_z": mean_z,
                "std_fisher_z": std_z,


                "std_pearson_r": (
                    float(np.nanstd(pearson, ddof=1))
                    if np.isfinite(pearson).sum() > 1 else 0.0
                ),
            }
            for metric in ("spearman_rho", "mae", "rmse", "r2"):
                vals = np.asarray([r[metric] for r in rows], dtype=float)
                row[f"mean_{metric}"] = float(np.nanmean(vals))
                row[f"std_{metric}"] = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else 0.0
            out.append(row)
    return out


def fdr_bh(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    if not len(ok):
        return q.tolist()
    order = ok[np.argsort(p[ok])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q.tolist()


def paired_statistics(subject_rows: list[dict]) -> list[dict]:
    rows = []
    targets = sorted({r["target"] for r in subject_rows})
    for target in targets:
        poe = {r["subject"]: r for r in subject_rows if r["target"] == target and r["method"] == "poe"}
        for baseline in ("obs", "indiv"):
            base = {r["subject"]: r for r in subject_rows if r["target"] == target and r["method"] == baseline}
            subjects = sorted(set(poe).intersection(base))
            a = np.asarray([poe[s]["pearson_r"] for s in subjects], dtype=float)
            b = np.asarray([base[s]["pearson_r"] for s in subjects], dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            try:
                p = float(wilcoxon(a, b, alternative="greater").pvalue) if len(a) >= 2 else np.nan
            except ValueError:
                p = 1.0
            mean_a, _, _ = fisher_mean(a)
            mean_b, _, _ = fisher_mean(b)
            rows.append({
                "target": target, "comparison": f"poe_vs_{baseline}",
                "metric": "pearson_r", "alternative": "poe_greater",
                "n_subjects": int(len(a)), "mean_poe_fisher": mean_a,
                "mean_baseline_fisher": mean_b,
                "delta_fisher_mean_r": mean_a - mean_b,
                "mean_paired_raw_r_delta": float(np.mean(a - b)) if len(a) else np.nan,
                "p_wilcoxon_one_sided": p,
            })
    q = fdr_bh([r["p_wilcoxon_one_sided"] for r in rows])
    for row, qi in zip(rows, q):
        row["q_fdr_bh"] = qi
    return rows


def run_task(task: str, args, device) -> None:
    subjects = load_task(args.feature_root, task)
    feature_sha256, feature_manifest = feature_fingerprint(args.feature_root, task)
    current_decoder_config = decoder_config(args)
    out_dir = Path(args.out_dir) / task
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "fold_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    print(f"[{task}] subjects={[s.subject for s in subjects]}", flush=True)
    metric_rows, prediction_rows, weight_rows, fold_rows = [], [], [], []

    for oi, test_subject in enumerate(subjects):
        if oi % args.outer_modulus != args.outer_remainder:
            continue
        outer_train = [s for s in subjects if s.subject != test_subject.subject]
        for seed_index in range(args.seeds):
            seed = args.seed + oi * 1009 + seed_index * 97
            set_seed(seed)


            val_pos = (oi + seed_index) % len(outer_train)
            val_subject = outer_train[val_pos]
            inner_train = [s for s in outer_train if s.subject != val_subject.subject]
            checkpoint_path = checkpoint_dir / f"{test_subject.subject}_seed{seed_index:02d}.pt"
            resumed = False
            if args.resume_existing and checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                expected = {
                    "task": task,
                    "test_subject": test_subject.subject,
                    "seed_index": seed_index,
                    "seed": seed,
                    "inner_val_subject": val_subject.subject,
                    "outer_train_subjects": [s.subject for s in outer_train],
                    "feature_fingerprint_sha256": feature_sha256,
                    "decoder_config": current_decoder_config,
                }
                mismatch = {
                    key: (checkpoint.get(key), value)
                    for key, value in expected.items() if checkpoint.get(key) != value
                }
                if mismatch:
                    raise RuntimeError(f"{checkpoint_path}: incompatible saved fold: {mismatch}")
                xo = standardizer_from_state(checkpoint["obs_scaler"])
                xi = standardizer_from_state(checkpoint["indiv_scaler"])
                ys = standardizer_from_state(checkpoint["target_scaler"])
                model = new_model(task, len(xo.mean), args, device)
                model.load_state_dict(checkpoint["model"], strict=True)
                best_ep = int(checkpoint["best_epoch"])
                best_val = float(checkpoint["best_val_loss"])
                history = list(checkpoint.get("inner_history", []))
                predictions, alpha = predict_subject(
                    model, test_subject, task, xo, xi, ys, device
                )
                resumed = True
            else:
                best_ep, best_val, history = choose_epoch(
                    inner_train, [val_subject], task, args, device, seed
                )
                set_seed(seed + 1)
                model, xo, xi, ys = re_outer(
                    outer_train, task, args, device, seed, best_ep
                )
                predictions, alpha = predict_subject(
                    model, test_subject, task, xo, xi, ys, device
                )

            fold_rows.append({
                "task": task, "test_subject": test_subject.subject, "seed_index": seed_index,
                "seed": seed, "inner_val_subject": val_subject.subject,
                "best_epoch": best_ep, "best_val_loss": best_val,
                "resumed_from_checkpoint": int(resumed),
            })
            weight_rows.append({
                "task": task, "test_subject": test_subject.subject, "seed_index": seed_index,
                "alpha_obs_mean": float(alpha.mean()), "alpha_obs_std": float(alpha.std()),
                "alpha_obs_min": float(alpha.min()), "alpha_obs_max": float(alpha.max()),
            })

            target_names = NARRATIVE_TARGETS if task == "narratives" else SHORTVIDEO_TARGETS
            for method, pred in predictions.items():
                if task == "narratives":
                    for j, target in enumerate(target_names):
                        met = regression_metrics(test_subject.y[:, j], pred[:, j])
                        metric_rows.append({"task": task, "test_subject": test_subject.subject,
                                            "seed_index": seed_index, "method": method,
                                            "target": target, **met})
                else:
                    for j, target in enumerate(target_names):
                        idx = test_subject.condition == j
                        met = regression_metrics(test_subject.y[idx], pred[idx])
                        metric_rows.append({"task": task, "test_subject": test_subject.subject,
                                            "seed_index": seed_index, "method": method,
                                            "target": target, **met})

            for k in range(len(test_subject.y)):
                row = {"task": task, "test_subject": test_subject.subject,
                       "seed_index": seed_index, "trial_index": int(test_subject.trial_index[k])}
                if task == "narratives":
                    for j, target in enumerate(target_names):
                        row[f"true_{target}"] = float(test_subject.y[k, j])
                        for method in ("obs", "indiv", "poe"):
                            row[f"pred_{method}_{target}"] = float(predictions[method][k, j])
                else:
                    c = int(test_subject.condition[k])
                    row["condition"] = target_names[c]
                    row["true_response_angle"] = float(test_subject.y[k])
                    for method in ("obs", "indiv", "poe"):
                        row[f"pred_{method}_response_angle"] = float(predictions[method][k])
                prediction_rows.append(row)

            if not resumed:
                atomic_torch_save({
                    "checkpoint_format_version": 3,
                    "model_class": "GaussianPoERegressor",
                    "task": task, "test_subject": test_subject.subject,
                    "seed_index": seed_index, "seed": seed,
                    "outer_train_subjects": [s.subject for s in outer_train],
                    "inner_train_subjects": [s.subject for s in inner_train],
                    "inner_val_subject": val_subject.subject,
                    "best_epoch": best_ep, "best_val_loss": best_val,
                    "inner_history": history,
                    "model": model.state_dict(),
                    "obs_scaler": xo.state(), "indiv_scaler": xi.state(),
                    "target_scaler": ys.state(),
                    "feature_root": os.path.abspath(args.feature_root),
                    "feature_fingerprint_sha256": feature_sha256,
                    "decoder_config": current_decoder_config,
                    "args": vars(args),
                }, checkpoint_path)
            print(f"[{task}] test={test_subject.subject} seed={seed_index} "
                  f"val={val_subject.subject} epoch={best_ep} val_loss={best_val:.4f} "
                  f"alpha_obs={alpha.mean():.3f} resumed={resumed}", flush=True)

    if args.skip_finalize:
        print(
            f"[{task}] shard complete outer_remainder={args.outer_remainder}/"
            f"{args.outer_modulus}; final tables intentionally deferred",
            flush=True,
        )
        return

    subject_rows, ensemble_rows = ensemble_subject_metrics(prediction_rows, task)
    summary_rows = summarize_subject_metrics(subject_rows)
    stats_rows = paired_statistics(subject_rows)
    write_csv(out_dir / "metrics_per_seed.csv", metric_rows)
    write_csv(out_dir / "metrics_per_subject.csv", subject_rows)
    write_csv(out_dir / "predictions_seed_ensemble.csv", ensemble_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "paired_statistics.csv", stats_rows)
    write_csv(out_dir / "predictions.csv", prediction_rows)
    write_csv(out_dir / "poe_weights.csv", weight_rows)
    write_csv(out_dir / "fold_training.csv", fold_rows)
    with (out_dir / "run_summary.json").open("w") as f:
        json.dump({
            "generated": datetime.now().isoformat(timespec="seconds"),
            "task": task, "feature_root": args.feature_root,
            "feature_fingerprint_sha256": feature_sha256,
            "feature_manifest": feature_manifest,
            "decoder_config": current_decoder_config,
            "subjects": [s.subject for s in subjects], "n_seeds": args.seeds,
            "outer_split": "LOSO", "inner_split": "one training subject, rotated across seeds",
            "ours": "learned latent Gaussian PoE(obs, indiv), no concatenation",
            "metric_aggregation": {
                "within_subject": "mean prediction across seeds for each trial, then one metric",
                "across_subjects_pearson": "mean Fisher-z, inverse transformed to r",
                "inference_unit": "held-out subject",
            },
            "targets": NARRATIVE_TARGETS if task == "narratives" else SHORTVIDEO_TARGETS,
            "narrative_coordinate_transform": (
                {"origin_xy": [NARRATIVE_SCALE_X0, NARRATIVE_SCALE_Y0],
                 "radius": NARRATIVE_SCALE_RADIUS,
                 "valence": "(x-x0)/radius; BAD negative, GOOD positive",
                 "intensity": "sqrt((x-x0)^2+(y-y0)^2)/radius"}
                if task == "narratives" else None
            ),
            "summary": summary_rows,
            "paired_statistics": stats_rows,
        }, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", default="narratives,shortvideo")
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260713)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--aux-weight", type=float, default=0.2)
    ap.add_argument("--kl-weight", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--sample-latent", action="store_true")
    ap.add_argument(
        "--outer-modulus", type=int, default=1,
        help="train only outer folds whose zero-based index modulo this value matches --outer-remainder",
    )
    ap.add_argument("--outer-remainder", type=int, default=0)
    ap.add_argument(
        "--skip-finalize", action="store_true",
        help="save fold checkpoints but defer aggregate CSV/JSON generation (for disjoint parallel shards)",
    )
    ap.add_argument(
        "--resume-existing", action="store_true",
        help="load compatible completed fold checkpoints instead of retraining them",
    )
    args = ap.parse_args()
    if args.outer_modulus < 1 or not 0 <= args.outer_remainder < args.outer_modulus:
        ap.error("require outer_modulus >= 1 and 0 <= outer_remainder < outer_modulus")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    for task in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        if task not in ("narratives", "shortvideo"):
            raise ValueError(f"unsupported task {task}")
        run_task(task, args, device)


if __name__ == "__main__":
    main()
