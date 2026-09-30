

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold


REPO = Path(__file__).resolve().parents[4]
CANONICAL_PATH = REPO / "data_check_20260507/full_run_5fold/classify_poe_v5_kfold_fp16.py"
DEFAULT_H5 = REPO / "data_check_20260507/full_run_5fold/stage2_new/INNER_class8"
DEFAULT_MAE = REPO / "data_check_20260507/full_run_5fold/ckpts/eeg_mae_INNER_20260508_025549.pt"
DEFAULT_BRAINOMNI = REPO / "weights/BrainOmni/BrainOmni/tiny"
DEFAULT_SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-05")


def canonical_module():
    spec = importlib.util.spec_from_file_location("inner_poe_canonical", CANONICAL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(CANONICAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def torch_load_safe(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError as exc:
        raise RuntimeError("PyTorch with weights_only=True is required") from exc


def atomic_torch_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pt", dir="/tmp", delete=False) as handle:
        local = Path(handle.name)
    staged = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, local)
        shutil.copyfile(local, staged)
        os.replace(staged, destination)
    finally:
        local.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)


def atomic_npz(destination: Path, **arrays) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    memory = io.BytesIO()
    np.savez_compressed(memory, **arrays)
    staged = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    staged.write_bytes(memory.getvalue())
    os.replace(staged, destination)


def load_backbones(module, mae_path: Path, brainomni_root: Path, device: torch.device):
    mae_checkpoint = torch_load_safe(mae_path, torch.device("cpu"))
    mae = module.EEGMAEEncoder(
        n_modes=int(mae_checkpoint["n_modes"]),
        patch_size=int(mae_checkpoint.get("patch_size", module.PATCH_SIZE)),
        n_samples=int(mae_checkpoint.get("n_samples", 1024)),
        d_model=int(mae_checkpoint["d_model"]),
        n_heads=int(mae_checkpoint.get("n_heads", 8)),
        n_layers=int(mae_checkpoint.get("n_layers", 6)),
        dropout=0.0,
    ).to(device).eval()
    state = mae_checkpoint["encoder"]
    if any(key.startswith("_orig_mod.") for key in state):
        state = {key.replace("_orig_mod.", "", 1): value for key, value in state.items()}
    mae.load_state_dict(state, strict=True)
    for parameter in mae.parameters():
        parameter.requires_grad_(False)
    mae.expected_n_modes = int(mae_checkpoint["n_modes"])
    mae.expected_n_samples = int(mae_checkpoint.get("n_samples", 1024))
    mae.d_model_dim = int(mae_checkpoint["d_model"])

    sys.path.insert(0, str(REPO / "weights/BrainOmni/BrainOmni-main"))
    from brainomni.model import BrainOmni

    config = json.loads((brainomni_root / "model_cfg.json").read_text())
    brainomni = BrainOmni(**config)
    brainomni.load_state_dict(
        torch_load_safe(brainomni_root / "BrainOmni.pt", torch.device("cpu")),
        strict=False,
    )
    brainomni = brainomni.to(device).eval()
    for parameter in brainomni.parameters():
        parameter.requires_grad_(False)
    return mae, brainomni, int(config["lm_dim"])


def build_models(module, mae, brainomni, lm_dim: int, config: dict, device):
    n_tokens = int(config["n_samples"]) // int(config["patch_size"])
    obs_encoder = module.ObsEncoder(
        brainomni, lm_dim, n_tokens, int(config["d_model"])
    ).to(device)
    obs_model = module.ExpertModel(
        obs_encoder,
        module.GaussHead(int(config["d_model"]), int(config["lat_dim"])).to(device),
        module.ClsHead(
            int(config["lat_dim"]), n_classes=int(config["n_classes"]),
            dropout=float(config["dropout"]),
        ).to(device),
        is_obs=True,
    ).to(device)
    prior_model = module.ExpertModel(
        mae,
        module.GaussHead(int(config["d_model"]), int(config["lat_dim"])).to(device),
        module.ClsHead(
            int(config["lat_dim"]), n_classes=int(config["n_classes"]),
            dropout=float(config["dropout"]),
        ).to(device),
        is_obs=False,
    ).to(device)
    return obs_model, prior_model


def trainable_state(model, frozen_prefix: str) -> dict:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith(frozen_prefix)
    }


@torch.inference_mode()
def predict(module, obs_model, prior_model, dataset, n_classes: int, batch_size: int):
    obs_model.eval()
    prior_model.eval()
    log_obs_all, log_prior_all, labels_all = [], [], []
    autocast_device = "cuda" if next(obs_model.parameters()).is_cuda else "cpu"
    autocast_enabled = autocast_device == "cuda"
    for group, _indiv, raw, position, sensor_type, labels in dataset.iter_batches(
        batch_size, shuffle=False
    ):
        with torch.amp.autocast(
            autocast_device, dtype=torch.float16, enabled=autocast_enabled
        ):
            obs_logits, _, _ = obs_model(group, raw, position, sensor_type)
            prior_logits, _, _ = prior_model(group, raw, position, sensor_type)
        log_obs_all.append(F.log_softmax(obs_logits.float(), dim=-1).cpu())
        log_prior_all.append(F.log_softmax(prior_logits.float(), dim=-1).cpu())
        labels_all.append(labels.cpu())
    log_obs = torch.cat(log_obs_all)
    log_prior = torch.cat(log_prior_all)
    labels = torch.cat(labels_all)
    prob_obs = log_obs.exp()
    prob_prior = log_prior.exp()
    chance = 1.0 / n_classes
    weight_obs = (prob_obs.max(1).values - chance).clamp(min=1e-4)
    weight_prior = (prob_prior.max(1).values - chance).clamp(min=0.0)
    total = (weight_obs + weight_prior).clamp(min=1e-8)
    log_poe = (
        weight_obs[:, None] / total[:, None] * log_obs
        + weight_prior[:, None] / total[:, None] * log_prior
    )
    prob_arithmetic = (prob_obs + prob_prior) / 2
    return {
        "labels": labels.numpy(),
        "prob_obs": prob_obs.numpy(),
        "prob_prior": prob_prior.numpy(),
        "prob_arithmetic": prob_arithmetic.numpy(),
        "prob_poe": F.softmax(log_poe, dim=-1).numpy(),
    }


def accuracies(prediction: dict) -> dict:
    labels = prediction["labels"]
    return {
        name: float((values.argmax(1) == labels).mean())
        for name, values in (
            ("obs_only", prediction["prob_obs"]),
            ("prior_only", prediction["prob_prior"]),
            ("poe_arith", prediction["prob_arithmetic"]),
            ("poe_full", prediction["prob_poe"]),
        )
    }


def load_data(module, h5_dir: Path, subjects: list[str], mae, device):
    return module.load_all_subjects(
        str(h5_dir), subjects, mae.expected_n_modes, mae.expected_n_samples, "eeg"
    )


def dataset_from_indices(module, arrays, indices, device):
    modes, indiv, raw, position, sensor_type, labels = arrays
    index = torch.as_tensor(indices, dtype=torch.long)
    return module.GPUDataset(
        modes[index], indiv[index], raw[index], position[index],
        sensor_type[index], labels[index], device=device,
    )


def train(args) -> None:
    module = canonical_module()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mae, brainomni, lm_dim = load_backbones(
        module, args.encoder_checkpoint, args.brainomni_root, device
    )
    arrays = load_data(module, args.h5_dir, args.subjects, mae, device)
    labels = arrays[-1].numpy()
    folds = list(
        StratifiedKFold(
            n_splits=args.n_folds, shuffle=True, random_state=args.seed
        ).split(np.zeros(len(labels)), labels)
    )
    selected = list(range(args.n_folds)) if args.folds == "all" else [
        int(value) for value in args.folds.split(",") if value.strip()
    ]
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold_index in selected:
        train_index, test_index = folds[fold_index]
        train_dataset = dataset_from_indices(module, arrays, train_index, device)
        test_dataset = dataset_from_indices(module, arrays, test_index, device)
        config = {
            "n_modes": mae.expected_n_modes,
            "n_samples": mae.expected_n_samples,
            "patch_size": module.PATCH_SIZE,
            "d_model": mae.d_model_dim,
            "lm_dim": lm_dim,
            "lat_dim": args.lat_dim,
            "dropout": args.dropout,
            "n_classes": 8,
        }
        obs_model, prior_model = build_models(
            module, mae, brainomni, lm_dim, config, device
        )
        train_config = SimpleNamespace(
            epochs=args.epochs, batch_size=args.batch_size,
            lr_adapter=args.lr_adapter, lr_head=args.lr_head,
            patience=args.patience, kl_weight=args.kl_weight,
        )
        obs_best = module.train_expert(
            obs_model, train_dataset, test_dataset,
            train_config.epochs, train_config.batch_size,
            train_config.lr_adapter, train_config.lr_head,
            train_config.patience, train_config.kl_weight,
            f"obs.fold{fold_index}",
        )
        prior_best = module.train_expert(
            prior_model, train_dataset, test_dataset,
            train_config.epochs, train_config.batch_size,
            train_config.lr_adapter, train_config.lr_head,
            train_config.patience, train_config.kl_weight,
            f"prior.fold{fold_index}",
        )
        prediction = predict(
            module, obs_model, prior_model, test_dataset, 8, args.batch_size
        )
        metric = accuracies(prediction)
        checkpoint_path = output / f"fold_{fold_index:02d}.pt"
        checkpoint = {
            "format_version": 1,
            "dataset": "Bimodal Inner Speech",
            "modality": "EEG",
            "task": "8-class pooled-subject classification",
            "seed": args.seed,
            "fold": fold_index,
            "n_folds": args.n_folds,
            "subjects": args.subjects,
            "train_indices": torch.as_tensor(train_index, dtype=torch.long),
            "test_indices": torch.as_tensor(test_index, dtype=torch.long),
            "config": config,
            "training": vars(train_config),
            "obs_best_validation_accuracy": obs_best,
            "prior_best_validation_accuracy": prior_best,
            "metrics_after_reload_target": metric,
            "obs_trainable_state": trainable_state(obs_model, "encoder.backbone."),
            "prior_trainable_state": trainable_state(prior_model, "encoder."),
            "base_weights": {
                "mae": str(args.encoder_checkpoint.resolve()),
                "mae_sha256": sha256(args.encoder_checkpoint),
                "brainomni": str((args.brainomni_root / "BrainOmni.pt").resolve()),
                "brainomni_sha256": sha256(args.brainomni_root / "BrainOmni.pt"),
            },
            "selection_boundary": "outer test fold was also used for historical early stopping",
        }
        atomic_torch_save(checkpoint, checkpoint_path)
        rows.append({"fold": fold_index, "checkpoint": str(checkpoint_path), **metric})
        print(f"saved {checkpoint_path} metrics={metric}", flush=True)
        del obs_model, prior_model, train_dataset, test_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with (output / "training_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def restore_trainable(model, state: dict, frozen_prefix: str) -> None:
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    nonfrozen_missing = [
        key for key in result.missing_keys if not key.startswith(frozen_prefix)
    ]
    if unexpected or nonfrozen_missing:
        raise RuntimeError(
            f"checkpoint state mismatch: unexpected={unexpected}, "
            f"nonfrozen_missing={nonfrozen_missing}"
        )


def inference(args) -> None:
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch_load_safe(args.checkpoint, torch.device("cpu"))
    module = canonical_module()
    base = checkpoint["base_weights"]
    mae_path = Path(args.encoder_checkpoint or base["mae"])
    brainomni_root = Path(args.brainomni_root or Path(base["brainomni"]).parent)
    if sha256(mae_path) != base["mae_sha256"]:
        raise RuntimeError("MAE base checkpoint hash mismatch")
    if sha256(brainomni_root / "BrainOmni.pt") != base["brainomni_sha256"]:
        raise RuntimeError("BrainOmni base checkpoint hash mismatch")
    mae, brainomni, lm_dim = load_backbones(module, mae_path, brainomni_root, device)
    subjects = args.subjects or checkpoint["subjects"]
    arrays = load_data(module, args.h5_dir, subjects, mae, device)
    if args.split == "test":
        indices = np.asarray(checkpoint["test_indices"], dtype=np.int64)
    else:
        indices = np.arange(len(arrays[-1]))
    dataset = dataset_from_indices(module, arrays, indices, device)
    obs_model, prior_model = build_models(
        module, mae, brainomni, lm_dim, checkpoint["config"], device
    )
    restore_trainable(obs_model, checkpoint["obs_trainable_state"], "encoder.backbone.")
    restore_trainable(prior_model, checkpoint["prior_trainable_state"], "encoder.")
    prediction = predict(
        module, obs_model, prior_model, dataset,
        int(checkpoint["config"]["n_classes"]), args.batch_size,
    )
    metric = accuracies(prediction)
    atomic_npz(output / "predictions.npz", indices=indices, **prediction)
    rows = []
    labels = prediction["labels"]
    for position, source_index in enumerate(indices):
        row = {"source_index": int(source_index), "label": int(labels[position])}
        for name, key in (
            ("obs", "prob_obs"), ("prior", "prob_prior"),
            ("arith", "prob_arithmetic"), ("poe", "prob_poe"),
        ):
            probability = prediction[key][position]
            row[f"pred_{name}"] = int(probability.argmax())
            row[f"confidence_{name}"] = float(probability.max())
        rows.append(row)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": "fresh_checkpoint_inference_complete",
        "training_steps": 0,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "fold": int(checkpoint["fold"]),
        "seed": int(checkpoint["seed"]),
        "split": args.split,
        "n_samples": len(indices),
        "metrics": metric,
        "base_weights": base,
        "selection_boundary": checkpoint["selection_boundary"],
    }
    (output / "inference_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--h5-dir", type=Path, default=DEFAULT_H5)
    train_parser.add_argument("--encoder-checkpoint", type=Path, default=DEFAULT_MAE)
    train_parser.add_argument("--brainomni-root", type=Path, default=DEFAULT_BRAINOMNI)
    train_parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    train_parser.add_argument("--folds", default="0")
    train_parser.add_argument("--n-folds", type=int, default=5)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr-adapter", type=float, default=5e-4)
    train_parser.add_argument("--lr-head", type=float, default=1e-3)
    train_parser.add_argument("--lat-dim", type=int, default=256)
    train_parser.add_argument("--dropout", type=float, default=0.3)
    train_parser.add_argument("--patience", type=int, default=80)
    train_parser.add_argument("--kl-weight", type=float, default=1e-3)
    train_parser.add_argument("--device", default="cuda:0")
    inference_parser = subparsers.add_parser("inference")
    inference_parser.add_argument("--checkpoint", type=Path, required=True)
    inference_parser.add_argument("--output-dir", type=Path, required=True)
    inference_parser.add_argument("--h5-dir", type=Path, default=DEFAULT_H5)
    inference_parser.add_argument("--encoder-checkpoint", type=Path)
    inference_parser.add_argument("--brainomni-root", type=Path)
    inference_parser.add_argument("--subjects", nargs="+")
    inference_parser.add_argument("--split", choices=("test", "all"), default="test")
    inference_parser.add_argument("--batch-size", type=int, default=64)
    inference_parser.add_argument("--device", default="cuda:0")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "train":
        train(args)
    else:
        inference(args)


if __name__ == "__main__":
    main()
