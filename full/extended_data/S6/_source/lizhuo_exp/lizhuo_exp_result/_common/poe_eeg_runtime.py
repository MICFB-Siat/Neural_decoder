








from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


REPO = Path(__file__).resolve().parents[3]
CANONICAL = (
    REPO
    / "data_check_20260507/Exp_tyf_Results/Exp_Classification/code/"
    "classify_baseline_v2.py"
)
PRIOR_CKPT = REPO / "checkpoints/unified_v3/prior_unified_v3_20260514_020311.pt"
REGISTRY_PATH = Path(__file__).with_name("poe_eeg_registry.json")


@dataclass(frozen=True)
class InferenceRequest:
    dataset: str
    checkpoint_dir: Path
    h5_dir: Path
    subject: str | None
    fold: int
    output_dir: Path
    device: str
    batch_size: int
    input_h5: Path | None = None


def _load_canonical():
    if not CANONICAL.is_file():
        raise FileNotFoundError(f"Canonical model code is missing: {CANONICAL}")
    name = "tyf_classify_baseline_v2_runtime"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, CANONICAL)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {CANONICAL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["datasets"]


def resolve_repo_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    marker = "Eigen_brain_decoding/"
    text = str(path).replace("\\", "/")
    if marker in text:
        local = REPO / text.split(marker, 1)[1]
        if local.exists():
            return local.resolve()
    return path.resolve()


def _safe_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise TypeError(f"Expected a state_dict at {path}")
    tensors = {k: v for k, v in state.items() if torch.is_tensor(v)}
    if not tensors:
        raise ValueError(f"No tensors found in checkpoint: {path}")
    return tensors


def _safe_prior_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "encoder" not in checkpoint:
        raise ValueError(f"Invalid prior checkpoint: {path}")
    return checkpoint


def _load_prior(module, path: Path, device: torch.device, n_samples: int):
    checkpoint = _safe_prior_checkpoint(path)
    encoder = module.EEGMEGPriorEigenval4(
        patch_size=checkpoint.get("patch_size", module.PATCH_SIZE),
        d_model=checkpoint["d_model"],
        n_heads=8,
        n_factor_layers=checkpoint.get("n_factor_layers", 2),
        n_time_layers=checkpoint.get("n_time_layers", 4),
        dropout=0.0,
    ).to(device)
    state = checkpoint["encoder"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected prior keys: {unexpected}")
    allowed_missing = {k for k in missing if "eigval" in k or "mode" in k}
    if set(missing) - allowed_missing:
        raise RuntimeError(f"Missing prior keys: {missing}")
    encoder.expected_n_samples = int(n_samples)
    encoder.d_model_dim = int(checkpoint["d_model"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder


def _load_brainomni(module, device: torch.device):
    from brainomni.model import BrainOmni

    cfg_path = Path(module.TINY_CKPT_DIR) / "model_cfg.json"
    weight_path = Path(module.TINY_CKPT_DIR) / "BrainOmni.pt"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model = BrainOmni(**cfg)
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, int(cfg["lm_dim"])


def _checkpoint_args(checkpoint_dir: Path) -> dict[str, Any]:
    for parent in (checkpoint_dir, *checkpoint_dir.parents):
        candidate = parent / "args.json"
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
        if parent == REPO:
            break
    return {}


def _read_split(fold_dir: Path) -> dict[str, Any]:
    path = fold_dir / "split.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing fold split: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _subject_from_fold(fold_dir: Path, split: dict[str, Any]) -> str | None:
    if split.get("subject"):
        return str(split["subject"])
    if fold_dir.parent.name.startswith("sub-"):
        return fold_dir.parent.name
    return None


def _resolve_fold_dir(checkpoint_root: Path, subject: str | None, fold: int) -> Path:
    fold_name = f"fold_{fold:02d}"
    candidates = []
    if checkpoint_root.name == fold_name:
        candidates.append(checkpoint_root)
    if subject:
        candidates.append(checkpoint_root / subject / fold_name)
    candidates.append(checkpoint_root / fold_name)
    for candidate in candidates:
        if (candidate / "obs.pt").is_file() and (candidate / "prior.pt").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find obs.pt/prior.pt. Checked: "
        + ", ".join(str(item) for item in candidates)
    )


def _resolve_eigval(module, h5_path: Path, d_path: Path | None, ratio: float,
                    lam_cortex_path: Path | None, modality: str) -> torch.Tensor:
    if d_path is not None:
        if not d_path.is_file():
            raise FileNotFoundError(f"D matrix is missing: {d_path}")
        D = np.load(d_path)
        lam = np.load(lam_cortex_path) if lam_cortex_path else None
        eigval, _ = module.compute_eigval_from_D(D, lam_cortex=lam, ratio=ratio)
        return torch.from_numpy(np.asarray(eigval, dtype=np.float32))
    key = f"{modality}_mode_eigval"
    with h5py.File(h5_path, "r") as handle:
        if key not in handle:
            raise KeyError(f"{h5_path} has no {key}; the registry must provide d_path")
        return torch.from_numpy(handle[key][:].astype(np.float32))


def _load_single_subject(module, h5_dir: Path, subject: str, h5_path: Path,
                         n_samples: int, modality: str, label_key: str,
                         label_offset: int, n_classes: int,
                         eigval: torch.Tensor) -> dict[str, Any]:


    with h5py.File(h5_path, "r") as handle:
        modes = torch.tensor(handle[f"{modality}_modes"][:], dtype=torch.float32)
        raw = torch.tensor(handle[f"{modality}_raw"][:], dtype=torch.float32)
        labels = torch.tensor(handle[label_key][:], dtype=torch.long) - label_offset
        pos = torch.tensor(handle[f"{modality}_pos"][:], dtype=torch.float32)
        sensor_type = torch.tensor(handle[f"{modality}_sensor_type"][:], dtype=torch.long)
    if modes.shape[1] != eigval.shape[0]:
        raise ValueError(
            f"Mode/eigval mismatch: H5 K={modes.shape[1]}, eigval K={eigval.shape[0]}"
        )
    if int(labels.min()) < 0 or int(labels.max()) >= n_classes:
        raise ValueError(
            f"Labels outside [0,{n_classes - 1}] after offset {label_offset}: "
            f"[{int(labels.min())},{int(labels.max())}]"
        )
    modes = modes[:, :, :n_samples]
    raw = raw[:, :, :n_samples]
    count = len(labels)
    return {
        "modes": modes,
        "raw": raw,
        "labels": labels,
        "pos": pos.unsqueeze(0).expand(count, -1, -1).contiguous(),
        "stype": sensor_type.unsqueeze(0).expand(count, -1).contiguous(),
        "eigval": eigval.float(),
        "subj": subject,
        "K": int(modes.shape[1]),
        "subj_ids": np.asarray([subject] * count),
    }


def _model_dimensions(obs_state: dict[str, torch.Tensor],
                      prior_state: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    d_model = int(prior_state["ghead.net.0.weight"].shape[1])
    lat_dim = int(prior_state["ghead.mu_head.weight"].shape[0])
    class_key = "cls_head.net.7.weight"
    if class_key not in prior_state:
        candidates = [k for k in prior_state if k.startswith("cls_head") and k.endswith("weight")]
        class_key = sorted(candidates)[-1]
    n_classes = int(prior_state[class_key].shape[0])
    obs_d_model = int(obs_state["encoder.adapter.0.weight"].shape[0])
    if obs_d_model != d_model:
        raise ValueError(f"Obs/prior d_model mismatch: {obs_d_model} vs {d_model}")
    return d_model, lat_dim, n_classes


def _build_models(module, fold_dir: Path, prior_encoder, backbone, lm_dim: int,
                  n_samples: int, device: torch.device):
    obs_state = _safe_state(fold_dir / "obs.pt")
    prior_state = _safe_state(fold_dir / "prior.pt")
    d_model, lat_dim, n_classes = _model_dimensions(obs_state, prior_state)
    n_tok = int(n_samples // module.PATCH_SIZE)

    obs_encoder = module.ObsEncoder(
        backbone, lm_dim, n_tok, d_model, backbone_trainable=False
    ).to(device)
    obs_model = module.ExpertModel(
        obs_encoder,
        module.GaussHead(d_model, lat_dim).to(device),
        module.ClsHead(lat_dim, n_classes=n_classes, dropout=0.0).to(device),
        is_obs=True,
        encoder_trainable=False,
    ).to(device)
    prior_model = module.ExpertModel(
        prior_encoder,
        module.GaussHead(d_model, lat_dim).to(device),
        module.ClsHead(lat_dim, n_classes=n_classes, dropout=0.0).to(device),
        is_obs=False,
        encoder_trainable=False,
    ).to(device)
    missing_obs, unexpected_obs = obs_model.load_state_dict(obs_state, strict=False)
    missing_prior, unexpected_prior = prior_model.load_state_dict(prior_state, strict=False)
    bad_obs = [k for k in missing_obs if not k.startswith("encoder.backbone.")]
    bad_prior = [k for k in missing_prior if not k.startswith("encoder.")]
    if bad_obs or unexpected_obs or bad_prior or unexpected_prior:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"obs_missing={bad_obs}, obs_unexpected={unexpected_obs}, "
            f"prior_missing={bad_prior}, prior_unexpected={unexpected_prior}"
        )
    obs_model.eval()
    prior_model.eval()
    return obs_model, prior_model, n_classes, n_tok


def _iter_slices(length: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, length, batch_size):
        yield slice(start, min(length, start + batch_size))


def _run_heads(obs_model, prior_model, obs_features: torch.Tensor,
               prior_features: torch.Tensor, batch_size: int, device: torch.device):
    obs_logits, prior_logits, obs_mu, prior_mu = [], [], [], []
    use_cuda_amp = device.type == "cuda"
    with torch.inference_mode():
        for part in _iter_slices(len(obs_features), batch_size):
            of = obs_features[part].to(device)
            pf = prior_features[part].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_cuda_amp):
                lo, mo, _ = obs_model.cached_forward(of)
                lp, mp, _ = prior_model.cached_forward(pf)
            obs_logits.append(lo.float().cpu())
            prior_logits.append(lp.float().cpu())
            obs_mu.append(mo.mean(1).float().cpu())
            prior_mu.append(mp.mean(1).float().cpu())
    return tuple(torch.cat(items).numpy() for items in (
        obs_logits, prior_logits, obs_mu, prior_mu
    ))


def _fuse(z_obs: np.ndarray, z_prior: np.ndarray) -> dict[str, np.ndarray]:
    obs = torch.from_numpy(z_obs).float()
    prior = torch.from_numpy(z_prior).float()
    log_obs = F.log_softmax(obs, dim=-1)
    log_prior = F.log_softmax(prior, dim=-1)
    p_obs, p_prior = log_obs.exp(), log_prior.exp()
    n_classes = z_obs.shape[1]
    chance = 1.0 / n_classes

    confidence_obs = (p_obs.max(1).values - chance).clamp(min=1e-4)
    confidence_prior = (p_prior.max(1).values - chance).clamp(min=0.0)
    confidence_sum = (confidence_obs + confidence_prior).clamp(min=1e-8)
    log_conf = (
        (confidence_obs / confidence_sum).unsqueeze(1) * log_obs
        + (confidence_prior / confidence_sum).unsqueeze(1) * log_prior
    )

    max_entropy = math.log(n_classes)
    entropy_obs = -(p_obs * log_obs).sum(-1).clamp(min=0.0)
    entropy_prior = -(p_prior * log_prior).sum(-1).clamp(min=0.0)
    entropy_weight_obs = (max_entropy - entropy_obs).clamp(min=1e-4)
    entropy_weight_prior = (max_entropy - entropy_prior).clamp(min=0.0)
    entropy_sum = (entropy_weight_obs + entropy_weight_prior).clamp(min=1e-8)
    log_entropy = (
        (entropy_weight_obs / entropy_sum).unsqueeze(1) * log_obs
        + (entropy_weight_prior / entropy_sum).unsqueeze(1) * log_prior
    )

    arithmetic = 0.5 * (p_obs + p_prior)
    return {
        "prob_obs": p_obs.numpy(),
        "prob_prior": p_prior.numpy(),
        "prob_arith": arithmetic.numpy(),
        "prob_conf": F.softmax(log_conf, dim=-1).numpy(),
        "prob_entropy": F.softmax(log_entropy, dim=-1).numpy(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(y_true: np.ndarray, probabilities: dict[str, np.ndarray]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, prob in probabilities.items():
        method = key.removeprefix("prob_")
        pred = prob.argmax(1)
        output[method] = {
            "accuracy": float((pred == y_true).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "confusion_matrix": confusion_matrix(y_true, pred).astype(int).tolist(),
        }
    return output


def infer(request: InferenceRequest, config: dict[str, Any]) -> Path:
    module = _load_canonical()
    output_dir = request.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_device = request.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    checkpoint_root = request.checkpoint_dir.expanduser().resolve()
    fold_dir = _resolve_fold_dir(checkpoint_root, request.subject, request.fold)
    split = _read_split(fold_dir)
    subject = request.subject or _subject_from_fold(fold_dir, split)
    if request.input_h5 is not None and subject is None:
        subject = request.input_h5.stem
    if subject is None:
        raise ValueError("This runtime currently requires --subject for a within-subject fold")

    args = _checkpoint_args(fold_dir)
    h5_dir = request.h5_dir.expanduser().resolve()
    h5_path = (request.input_h5.expanduser().resolve() if request.input_h5
               else h5_dir / f"{subject}.h5")
    if not h5_path.is_file():
        raise FileNotFoundError(f"Input H5 is missing: {h5_path}")

    modality = str(config.get("modality", args.get("modality", "eeg")))
    label_key = str(config.get("label_key", args.get("label_key", "labels")))
    label_offset = int(config.get("label_offset", args.get("label_offset", 0)))
    n_classes_cfg = int(config.get("n_classes", args.get("n_classes", 0)))
    n_samples = int(config.get("n_samples") or args.get("n_samples") or 2560)
    ratio = float(config.get("ratio", args.get("ratio", 1e-3)))
    d_path = resolve_repo_path(config.get("d_path") or args.get("d_path"))
    lam_path = resolve_repo_path(config.get("lam_cortex_path") or args.get("lam_cortex_path"))
    prior_path = resolve_repo_path(config.get("prior_checkpoint")) or PRIOR_CKPT
    if not prior_path.is_file():
        raise FileNotFoundError(f"Prior backbone checkpoint is missing: {prior_path}")

    eigval = _resolve_eigval(module, h5_path, d_path, ratio, lam_path, modality)
    bundle = _load_single_subject(
        module, h5_dir, subject, h5_path, n_samples, modality, label_key,
        label_offset, n_classes_cfg, eigval,
    )
    all_indices = np.arange(len(bundle["labels"]), dtype=np.int64)
    if request.input_h5 is not None:
        inference_indices = all_indices
    else:
        inference_indices = np.asarray(split.get("test_idx", []), dtype=np.int64)
        if not len(inference_indices):
            raise ValueError(f"No test_idx in {fold_dir / 'split.json'}")
    if inference_indices.min(initial=0) < 0 or inference_indices.max(initial=-1) >= len(all_indices):
        raise IndexError("Inference indices are outside the input H5 trial range")

    prior_encoder = _load_prior(module, prior_path, device, n_samples)
    backbone, lm_dim = _load_brainomni(module, device)
    obs_model, prior_model, n_classes, n_tok = _build_models(
        module, fold_dir, prior_encoder, backbone, lm_dim, n_samples, device
    )
    if n_classes_cfg and n_classes_cfg != n_classes:
        raise ValueError(f"Registry/checkpoint n_classes mismatch: {n_classes_cfg} vs {n_classes}")
    prior_model.set_eigval(eigval.to(device))

    obs_features, prior_features, labels = module.precompute_features(
        bundle, inference_indices, backbone, prior_encoder, n_tok,
        eigval.to(device), device, request.batch_size,
    )
    z_obs, z_prior, mu_obs, mu_prior = _run_heads(
        obs_model, prior_model, obs_features, prior_features,
        request.batch_size, device,
    )
    probabilities = _fuse(z_obs, z_prior)
    y_true = labels.numpy().astype(np.int64)
    subject_ids = bundle["subj_ids"][inference_indices]

    arrays = {
        "sample_index": inference_indices,
        "subject_id": subject_ids.astype("U"),
        "y_true": y_true,
        "z_obs": z_obs,
        "z_prior": z_prior,
        "mu_obs": mu_obs,
        "mu_prior": mu_prior,
        **probabilities,
    }

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    (output_dir / "predictions.npz").write_bytes(buffer.getvalue())

    fieldnames = ["sample_index", "subject", "y_true"]
    methods = [key.removeprefix("prob_") for key in probabilities]
    for method in methods:
        fieldnames.extend((f"pred_{method}", f"confidence_{method}"))
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, sample_index in enumerate(inference_indices):
            row: dict[str, Any] = {
                "sample_index": int(sample_index),
                "subject": str(subject_ids[row_index]),
                "y_true": int(y_true[row_index]),
            }
            for key, probability in probabilities.items():
                method = key.removeprefix("prob_")
                row[f"pred_{method}"] = int(probability[row_index].argmax())
                row[f"confidence_{method}"] = float(probability[row_index].max())
            writer.writerow(row)

    metrics = _metrics(y_true, probabilities)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    checkpoint_manifest = {
        "fold_checkpoint_dir": str(fold_dir),
        "obs": {
            "path": str(fold_dir / "obs.pt"),
            "bytes": (fold_dir / "obs.pt").stat().st_size,
            "sha256": _sha256(fold_dir / "obs.pt"),
        },
        "prior_head": {
            "path": str(fold_dir / "prior.pt"),
            "bytes": (fold_dir / "prior.pt").stat().st_size,
            "sha256": _sha256(fold_dir / "prior.pt"),
        },
        "prior_backbone": {
            "path": str(prior_path),
            "bytes": prior_path.stat().st_size,
        },
        "brainomni_backbone": {
            "path": str(Path(module.TINY_CKPT_DIR) / "BrainOmni.pt"),
            "bytes": (Path(module.TINY_CKPT_DIR) / "BrainOmni.pt").stat().st_size,
        },
    }
    (output_dir / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    run_manifest = {
        "mode": "fresh_weight_loaded_inference",
        "dataset": request.dataset,
        "subject": subject,
        "fold": request.fold,
        "device": str(device),
        "input_h5": str(h5_path),
        "n_input_trials": int(len(bundle["labels"])),
        "n_inference_trials": int(len(inference_indices)),
        "uses_saved_test_split": request.input_h5 is None,
        "split_path": str(fold_dir / "split.json"),
        "model_source": str(CANONICAL),
        "modality": modality,
        "label_key": label_key,
        "n_classes": n_classes,
        "n_samples": n_samples,
        "d_path": str(d_path) if d_path else None,
        "eigval_k": int(eigval.numel()),
        "outputs": ["predictions.csv", "predictions.npz", "metrics.json", "checkpoint_manifest.json"],
    }
    (output_dir / "fresh_inference_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_dir


def registry_config(dataset: str, registry_path: Path = REGISTRY_PATH) -> dict[str, Any]:
    registry = load_registry(registry_path)
    if dataset not in registry:
        raise KeyError(f"Unknown dataset {dataset!r}; choose one of {sorted(registry)}")
    return registry[dataset]
