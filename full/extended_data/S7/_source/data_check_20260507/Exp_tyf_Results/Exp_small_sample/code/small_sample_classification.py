












from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import train_test_split


LOGGER = logging.getLogger("small_sample")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def safe_npz(path: Path, **arrays: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(buffer.getvalue())
    os.replace(tmp, path)


def canonical_digest(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("small_sample_reference_classifier", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reference classifier: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rounded_count(total: int, fraction: float) -> int:
    return int(math.floor(total * fraction + 0.5))


def class_counts(labels: np.ndarray, n_classes: int) -> dict[str, int]:
    counts = np.bincount(labels.astype(np.int64), minlength=n_classes)
    return {str(i): int(counts[i]) for i in range(n_classes)}


def nested_stratified_order(indices: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:







    rng = np.random.default_rng(seed)
    classes = np.unique(labels[indices]).astype(int)
    queues: dict[int, list[int]] = {}
    totals: dict[int, int] = {}
    tie_break = {int(c): float(rng.random()) for c in classes}
    for cls in classes:
        values = indices[labels[indices] == cls].copy()
        rng.shuffle(values)
        queues[int(cls)] = values.tolist()
        totals[int(cls)] = len(values)
    proportions = {cls: totals[cls] / len(indices) for cls in totals}
    selected = {cls: 0 for cls in totals}
    order: list[int] = []
    for step in range(len(indices)):
        candidates = [cls for cls in totals if selected[cls] < totals[cls]]
        cls = max(
            candidates,
            key=lambda value: (
                proportions[value] * (step + 1) - selected[value],
                tie_break[value],
            ),
        )
        order.append(queues[cls][selected[cls]])
        selected[cls] += 1
    return np.asarray(order, dtype=np.int64)


def make_split(
    labels: np.ndarray,
    n_classes: int,
    test_fraction: float,
    val_fraction: float,
    train_fractions: list[float],
    seed: int,
    dataset: str,
    subject: str,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    total = len(labels)
    indices = np.arange(total, dtype=np.int64)
    n_test = rounded_count(total, test_fraction)
    n_val = rounded_count(total, val_fraction)
    if n_test < n_classes or n_val < n_classes:
        raise ValueError(
            f"{dataset}/{subject}: test={n_test} and val={n_val} must each contain "
            f"at least n_classes={n_classes} samples"
        )
    dev_idx, test_idx = train_test_split(
        indices,
        test_size=n_test,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    candidate_idx, val_idx = train_test_split(
        np.asarray(dev_idx),
        test_size=n_val,
        random_state=seed + 1,
        shuffle=True,
        stratify=labels[np.asarray(dev_idx)],
    )
    order = nested_stratified_order(np.asarray(candidate_idx), labels, seed + 2)

    train_sets: dict[str, dict] = {}
    previous: set[int] = set()
    for fraction in sorted(train_fractions):
        n_train = rounded_count(total, fraction)
        if n_train < n_classes:
            raise ValueError(f"{dataset}/{subject}: train fraction {fraction} gives only {n_train} trials")
        if n_train > len(order):
            raise ValueError(
                f"{dataset}/{subject}: fraction {fraction} requests {n_train} trials, "
                f"but only {len(order)} remain after test/validation"
            )
        train_idx = order[:n_train]
        current = set(int(x) for x in train_idx)
        if not previous.issubset(current):
            raise AssertionError("nested training-set invariant failed")
        previous = current
        key = f"{fraction:.2f}"
        train_sets[key] = {
            "fraction_requested": float(fraction),
            "n_train": n_train,
            "actual_fraction": n_train / total,
            "indices": train_idx.tolist(),
            "class_counts": class_counts(labels[train_idx], n_classes),
        }

    test_set = set(int(x) for x in test_idx)
    val_set = set(int(x) for x in val_idx)
    max_train_set = set(train_sets[f"{max(train_fractions):.2f}"]["indices"])
    if test_set & val_set or test_set & max_train_set or val_set & max_train_set:
        raise AssertionError("train/validation/test disjointness invariant failed")
    expected_labels = list(range(n_classes))
    for name, idx in (("test", test_idx), ("validation", val_idx)):
        if np.unique(labels[np.asarray(idx)]).tolist() != expected_labels:
            raise AssertionError(f"{dataset}/{subject}: {name} split is missing a class")
    for key, entry in train_sets.items():
        if np.unique(labels[np.asarray(entry["indices"])]).tolist() != expected_labels:
            raise AssertionError(f"{dataset}/{subject}: training split {key} is missing a class")

    split = {
        "dataset": dataset,
        "subject": subject,
        "seed": seed,
        "n_total": total,
        "label_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "all_class_counts": class_counts(labels, n_classes),
        "test": {
            "fraction_requested": test_fraction,
            "n_test": len(test_idx),
            "actual_fraction": len(test_idx) / total,
            "indices": sorted(int(x) for x in test_idx),
            "class_counts": class_counts(labels[test_idx], n_classes),
        },
        "validation": {
            "fraction_requested": val_fraction,
            "n_validation": len(val_idx),
            "actual_fraction": len(val_idx) / total,
            "indices": sorted(int(x) for x in val_idx),
            "class_counts": class_counts(labels[val_idx], n_classes),
        },
        "training_sets": train_sets,
        "unused_after_max_training": sorted(
            set(int(x) for x in indices) - test_set - val_set - max_train_set
        ),
        "invariants": {
            "fixed_test_across_training_fractions": True,
            "fixed_validation_across_training_fractions": True,
            "training_sets_are_nested": True,
            "train_validation_test_are_disjoint": True,
            "all_splits_contain_all_classes": True,
        },
    }
    split["split_sha256"] = canonical_digest(split)
    return split


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_subject_seed(seed: int, dataset: str, subject: str) -> int:
    raw = hashlib.sha256(f"{seed}:{dataset}:{subject}".encode("utf-8")).digest()
    return int.from_bytes(raw[:4], "little") & 0x7FFFFFFF


def train_cached(
    model,
    train_ds,
    val_ds,
    branch: str,
    epochs: int,
    batch_size: int,
    lr_adapter: float,
    lr_head: float,
    patience: int,
    kl_weight: float,
    label: str,
) -> tuple[float, int, int, dict[str, torch.Tensor]]:
    trainable_groups = []
    if branch == "obs":
        trainable_groups.append(
            {
                "params": list(model.encoder.adapter.parameters()),
                "lr": lr_adapter,
                "weight_decay": 1e-3,
            }
        )
    trainable_groups.append(
        {
            "params": list(model.ghead.parameters()) + list(model.cls_head.parameters()),
            "lr": lr_head,
            "weight_decay": 1e-3,
        }
    )
    optimizer = torch.optim.AdamW(trainable_groups)
    steps_per_epoch = max(1, math.ceil(train_ds.N / batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group["lr"] for group in trainable_groups],
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda")
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    best_acc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    no_improve = 0
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        model.train()
        correct = seen = 0
        for obs_feat, prior_tok, labels in train_ds.iter_batches(batch_size):
            cached = obs_feat if branch == "obs" else prior_tok
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, mu, logvar = model.cached_forward(cached)
                kl = 0.5 * (logvar.exp() + mu.pow(2) - 1 - logvar).mean()
                loss = criterion(logits, labels) + kl_weight * kl
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [param for group in trainable_groups for param in group["params"]], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            correct += int((logits.argmax(1) == labels).sum().item())
            seen += len(labels)

        model.eval()
        val_correct = val_seen = 0
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for obs_feat, prior_tok, labels in val_ds.iter_batches(batch_size, shuffle=False):
                cached = obs_feat if branch == "obs" else prior_tok
                logits, _, _ = model.cached_forward(cached)
                val_correct += int((logits.argmax(1) == labels).sum().item())
                val_seen += len(labels)
        train_acc = correct / max(seen, 1)
        val_acc = val_correct / max(val_seen, 1)
        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            best_epoch = epoch
            all_state = model.state_dict()
            best_state = {
                name: all_state[name].detach().cpu().clone()
                for name in trainable_names
            }
            no_improve = 0
        else:
            no_improve += 1
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs or no_improve >= patience:
            LOGGER.info(
                "%s epoch=%d/%d train=%.4f val=%.4f best=%.4f@%d",
                label,
                epoch,
                epochs,
                train_acc,
                val_acc,
                best_acc,
                best_epoch,
            )
        if no_improve >= patience:
            break

    current = model.state_dict()
    current.update({name: value.to(next(model.parameters()).device) for name, value in best_state.items()})
    model.load_state_dict(current)
    return best_acc, best_epoch, epochs_run, best_state


def collect_logits(obs_model, prior_model, dataset, batch_size: int):
    obs_model.eval()
    prior_model.eval()
    obs_logits = []
    prior_logits = []
    labels = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for obs_feat, prior_tok, batch_labels in dataset.iter_batches(batch_size, shuffle=False):
            obs, _, _ = obs_model.cached_forward(obs_feat)
            prior, _, _ = prior_model.cached_forward(prior_tok)
            obs_logits.append(obs.float().cpu())
            prior_logits.append(prior.float().cpu())
            labels.append(batch_labels.cpu())
    return torch.cat(obs_logits), torch.cat(prior_logits), torch.cat(labels)


def branch_metrics(logits: torch.Tensor, labels: torch.Tensor, n_classes: int) -> dict:
    true = labels.numpy()
    pred = logits.argmax(1).numpy()
    recalls = recall_score(
        true, pred, labels=list(range(n_classes)), average=None, zero_division=0
    ).tolist()
    return {
        "accuracy": float((pred == true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "per_class_recall": [float(x) for x in recalls],
    }


def label_free_fusions(base, obs_logits, prior_logits, labels, n_classes: int) -> dict:
    return base._ensemble_from_logs(
        F.log_softmax(obs_logits, dim=-1),
        F.log_softmax(prior_logits, dim=-1),
        labels,
        n_classes,
    )


def validation_temperature_fusion(
    base,
    val_obs: torch.Tensor,
    val_prior: torch.Tensor,
    val_labels: torch.Tensor,
    test_obs: torch.Tensor,
    test_prior: torch.Tensor,
    test_labels: torch.Tensor,
    n_classes: int,
) -> dict:
    grid = np.linspace(0.05, 10.0, 200)
    t_obs = base._fit_temperature_nll(val_obs.numpy(), val_labels.numpy(), grid)
    t_prior = base._fit_temperature_nll(val_prior.numpy(), val_labels.numpy(), grid)
    fused = F.log_softmax(test_obs / t_obs, dim=-1) + F.log_softmax(test_prior / t_prior, dim=-1)
    pred = fused.argmax(1).numpy()
    true = test_labels.numpy()
    recalls = recall_score(
        true, pred, labels=list(range(n_classes)), average=None, zero_division=0
    ).tolist()
    return {
        "poe_valtemp": float((pred == true).mean()),
        "poe_valtemp_balacc": float(balanced_accuracy_score(true, pred)),
        "poe_valtemp_recall": [float(x) for x in recalls],
        "temperature_obs_from_validation": float(t_obs),
        "temperature_prior_from_validation": float(t_prior),
    }


def build_models(base, backbone, prior, lm_dim: int, args, device):
    n_tok = prior.expected_n_samples // base.PATCH_SIZE
    d_model = prior.d_model_dim
    obs_encoder = base.ObsEncoder(backbone, lm_dim, n_tok, d_model, backbone_trainable=False).to(device)
    obs_model = base.ExpertModel(
        obs_encoder,
        base.GaussHead(d_model, args.lat_dim).to(device),
        base.ClsHead(args.lat_dim, n_classes=args.n_classes, dropout=args.dropout).to(device),
        is_obs=True,
        encoder_trainable=False,
    ).to(device)
    prior_model = base.ExpertModel(
        prior,
        base.GaussHead(d_model, args.lat_dim).to(device),
        base.ClsHead(args.lat_dim, n_classes=args.n_classes, dropout=args.dropout).to(device),
        is_obs=False,
        encoder_trainable=False,
    ).to(device)
    return obs_model, prior_model


def load_or_create_split(args, labels: np.ndarray, subject: str, subject_dir: Path) -> dict:
    split_path = subject_dir / "split.json"
    expected = make_split(
        labels,
        args.n_classes,
        args.test_fraction,
        args.val_fraction,
        args.train_fractions,
        args.seed,
        args.dataset,
        subject,
    )
    if split_path.is_file():
        existing = json.loads(split_path.read_text(encoding="utf-8"))
        if existing.get("split_sha256") != expected["split_sha256"]:
            raise RuntimeError(f"{split_path}: existing split differs from current protocol/data")
        return existing
    atomic_json(split_path, expected)
    return expected


def audit_subject(args, cfg: dict, subject: str, subject_dir: Path, eigval_length: int) -> dict:
    h5_path = Path(cfg["h5_dir"]) / f"{subject}.h5"
    with h5py.File(h5_path, "r") as handle:
        labels = np.asarray(handle[cfg["label_key"]]).reshape(-1).astype(np.int64) - args.label_offset
        mode_shape = tuple(int(x) for x in handle["eeg_modes"].shape)
    if mode_shape[1] != eigval_length:
        raise ValueError(
            f"{args.dataset}/{subject}: H5 has K={mode_shape[1]} modes but D/ratio gives K={eigval_length}"
        )
    split = load_or_create_split(args, labels, subject, subject_dir)
    return {
        "subject": subject,
        "n_total": len(labels),
        "mode_shape": mode_shape,
        "split_sha256": split["split_sha256"],
    }


def run_subject(args, cfg: dict, base, subject: str, subject_dir: Path, eigval, prior, backbone, lm_dim, device):
    labels_path = Path(cfg["h5_dir"]) / f"{subject}.h5"
    with h5py.File(labels_path, "r") as handle:
        labels_np = np.asarray(handle[cfg["label_key"]]).reshape(-1).astype(np.int64) - args.label_offset
    split = load_or_create_split(args, labels_np, subject, subject_dir)
    expected_ratio_dirs = [subject_dir / f"train_{int(round(f * 100)):03d}" for f in args.train_fractions]
    if all((path / "completed.flag").is_file() for path in expected_ratio_dirs):
        LOGGER.info("%s/%s already complete", args.dataset, subject)
        return

    item = base.load_subject(
        cfg["h5_dir"],
        subject,
        prior.expected_n_samples,
        "eeg",
        cfg["label_key"],
        args.label_offset,
        args.n_classes,
        override_eigval=eigval,
    )
    bundle = {
        "modes": item["modes"],
        "raw": item["raw"],
        "labels": item["labels"],
        "pos": item["pos"],
        "stype": item["stype"],
        "eigval": item["eigval"],
        "K": item["K"],
    }
    n_tok = prior.expected_n_samples // base.PATCH_SIZE
    all_indices = np.arange(len(item["labels"]), dtype=np.int64)
    LOGGER.info("%s/%s precomputing frozen features for N=%d", args.dataset, subject, len(all_indices))
    obs_all, prior_all, labels_all = base.precompute_features(
        bundle,
        all_indices,
        backbone,
        prior,
        n_tok,
        eigval.to(device),
        device,
        args.batch_size,
    )
    model_seed = stable_subject_seed(args.seed, args.dataset, subject)
    val_idx = np.asarray(split["validation"]["indices"], dtype=np.int64)
    test_idx = np.asarray(split["test"]["indices"], dtype=np.int64)

    for fraction in args.train_fractions:
        key = f"{fraction:.2f}"
        ratio_dir = subject_dir / f"train_{int(round(fraction * 100)):03d}"
        metrics_path = ratio_dir / "metrics.json"
        completed_path = ratio_dir / "completed.flag"
        if metrics_path.is_file() and completed_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("split_sha256") != split["split_sha256"]:
                raise RuntimeError(f"{metrics_path}: split provenance mismatch")
            LOGGER.info("%s/%s fraction=%.2f resumed", args.dataset, subject, fraction)
            continue
        ratio_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        train_idx = np.asarray(split["training_sets"][key]["indices"], dtype=np.int64)
        train_ds = base.CachedGPUDataset(
            obs_all[train_idx], prior_all[train_idx], labels_all[train_idx], device
        )
        val_ds = base.CachedGPUDataset(obs_all[val_idx], prior_all[val_idx], labels_all[val_idx], device)
        test_ds = base.CachedGPUDataset(
            obs_all[test_idx], prior_all[test_idx], labels_all[test_idx], device
        )

        configure_determinism(model_seed * 2 % 2147483647)
        obs_model, _unused_prior_model = build_models(base, backbone, prior, lm_dim, args, device)
        del _unused_prior_model
        obs_best_val, obs_best_epoch, obs_epochs_run, obs_state = train_cached(
            obs_model,
            train_ds,
            val_ds,
            "obs",
            args.epochs,
            args.batch_size,
            args.lr_adapter,
            args.lr_head,
            args.patience,
            args.kl_weight,
            f"{args.dataset}/{subject}/{key}/obs",
        )

        configure_determinism((model_seed * 2 + 1) % 2147483647)
        _unused_obs_model, prior_model = build_models(base, backbone, prior, lm_dim, args, device)
        del _unused_obs_model
        prior_best_val, prior_best_epoch, prior_epochs_run, prior_state = train_cached(
            prior_model,
            train_ds,
            val_ds,
            "prior",
            args.epochs,
            args.batch_size,
            args.lr_adapter,
            args.lr_head,
            args.patience,
            args.kl_weight,
            f"{args.dataset}/{subject}/{key}/prior",
        )

        val_obs, val_prior, val_labels = collect_logits(obs_model, prior_model, val_ds, args.batch_size)
        test_obs, test_prior, test_labels = collect_logits(obs_model, prior_model, test_ds, args.batch_size)
        validation = {
            "obs": branch_metrics(val_obs, val_labels, args.n_classes),
            "prior": branch_metrics(val_prior, val_labels, args.n_classes),
            "fusion": label_free_fusions(base, val_obs, val_prior, val_labels, args.n_classes),
        }
        test = {
            "obs": branch_metrics(test_obs, test_labels, args.n_classes),
            "prior": branch_metrics(test_prior, test_labels, args.n_classes),
            "fusion": label_free_fusions(base, test_obs, test_prior, test_labels, args.n_classes),
        }
        test["fusion"].update(
            validation_temperature_fusion(
                base,
                val_obs,
                val_prior,
                val_labels,
                test_obs,
                test_prior,
                test_labels,
                args.n_classes,
            )
        )
        metrics = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset,
            "subject": subject,
            "train_fraction_requested": fraction,
            "n_total": len(labels_all),
            "n_train": len(train_idx),
            "actual_train_fraction": len(train_idx) / len(labels_all),
            "n_validation": len(val_idx),
            "n_test": len(test_idx),
            "seed": args.seed,
            "model_seed": model_seed,
            "split_sha256": split["split_sha256"],
            "train_class_counts": split["training_sets"][key]["class_counts"],
            "validation_class_counts": split["validation"]["class_counts"],
            "test_class_counts": split["test"]["class_counts"],
            "training": {
                "epochs_limit": args.epochs,
                "patience": args.patience,
                "obs_best_validation_accuracy": obs_best_val,
                "obs_best_epoch": obs_best_epoch,
                "obs_epochs_run": obs_epochs_run,
                "prior_best_validation_accuracy": prior_best_val,
                "prior_best_epoch": prior_best_epoch,
                "prior_epochs_run": prior_epochs_run,
            },
            "validation": validation,
            "test": test,
            "primary_metric_name": "test.fusion.poe_conf",
            "primary_accuracy": test["fusion"]["poe_conf"],
            "elapsed_seconds": time.time() - started,
        }
        safe_npz(
            ratio_dir / "test_predictions.npz",
            test_idx=test_idx,
            labels=test_labels.numpy(),
            obs_logits=test_obs.numpy(),
            prior_logits=test_prior.numpy(),
        )
        if args.save_models:
            torch.save(obs_state, ratio_dir / "obs_trainable.pt", _use_new_zipfile_serialization=False)
            torch.save(prior_state, ratio_dir / "prior_trainable.pt", _use_new_zipfile_serialization=False)
        atomic_json(metrics_path, metrics)
        completed_path.write_text(metrics["completed_at_utc"] + "\n", encoding="utf-8")
        LOGGER.info(
            "%s/%s fraction=%.2f complete primary_acc=%.4f elapsed=%.1fs",
            args.dataset,
            subject,
            fraction,
            metrics["primary_accuracy"],
            metrics["elapsed_seconds"],
        )
        del train_ds, val_ds, test_ds, obs_model, prior_model, obs_state, prior_state
        torch.cuda.empty_cache()

    atomic_json(
        subject_dir / "subject_completed.json",
        {
            "dataset": args.dataset,
            "subject": subject,
            "split_sha256": split["split_sha256"],
            "training_fractions": args.train_fractions,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    del bundle, item, obs_all, prior_all, labels_all
    torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=["BCIC", "FACED", "MOTOR", "SEEDV"])
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-adapter", type=float, default=1e-3)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lat-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cfg = manifest["datasets"][args.dataset]
    protocol = manifest["protocol"]
    args.seed = int(protocol["seed"])
    args.test_fraction = float(protocol["test_fraction_of_total"])
    args.val_fraction = float(protocol["validation_fraction_of_total"])
    args.train_fractions = [float(x) for x in protocol["train_fractions_of_total"]]
    args.n_classes = int(cfg["n_classes"])
    args.label_offset = int(cfg["label_offset"])
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= shard-index < num-shards")

    base_path = Path(manifest["inputs"]["base_classifier_path"])
    expected_hash = manifest["inputs"]["base_classifier_sha256"]
    if file_sha256(base_path) != expected_hash:
        raise RuntimeError("reference classifier changed after experiment manifest was created")
    base = load_reference(base_path)
    eigval, source = base.resolve_shared_eigval(
        cfg["d_path"], cfg["lam_cortex_path"], float(cfg["ratio"]), None, "eeg"
    )
    if source != "D" or eigval is None:
        raise RuntimeError(f"{args.dataset}: expected D-derived eigenvalues, got {source}")
    subjects = args.subjects or [entry["subject"] for entry in cfg["subjects"]]
    subjects = sorted(subjects)[args.shard_index :: args.num_shards]
    dataset_dir = args.manifest.parent / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "dataset=%s shard=%d/%d subjects=%d device=%s audit_only=%s",
        args.dataset,
        args.shard_index,
        args.num_shards,
        len(subjects),
        args.device,
        args.audit_only,
    )

    audit_rows = []
    for subject in subjects:
        subject_dir = dataset_dir / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        audit_rows.append(audit_subject(args, cfg, subject, subject_dir, len(eigval)))
    if args.audit_only:
        atomic_json(
            dataset_dir / f"split_audit_shard_{args.shard_index:02d}.json",
            {
                "dataset": args.dataset,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "subjects": audit_rows,
            },
        )
        LOGGER.info("dataset=%s split audit complete", args.dataset)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the BrainOmni/BT-ND experiment")
    device = torch.device(args.device)
    configure_determinism(args.seed)
    encoder_ckpt = manifest["inputs"]["encoder_ckpt_path"]
    prior = base.load_prior_from_ckpt(encoder_ckpt, device, trainable=False)
    backbone, lm_dim = base.load_brainomni_tiny(device, finetune=False)
    for subject in subjects:
        subject_dir = dataset_dir / subject
        try:
            run_subject(
                args,
                cfg,
                base,
                subject,
                subject_dir,
                eigval,
                prior,
                backbone,
                lm_dim,
                device,
            )
        except Exception as exc:
            atomic_json(
                subject_dir / "failure.json",
                {
                    "dataset": args.dataset,
                    "subject": subject,
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
    LOGGER.info("dataset=%s shard=%d training complete", args.dataset, args.shard_index)


if __name__ == "__main__":
    main()
