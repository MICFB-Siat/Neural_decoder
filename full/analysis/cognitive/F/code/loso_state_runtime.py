


from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from modeling import GaussianPoE, GaussianSingleExpert, Standardizer
CHECKPOINT_ROOT = HERE.parent / 'weights'
FEATURE_ROOTS = {}



@dataclass
class Input:
    subject: str
    obs: np.ndarray | None
    indiv: np.ndarray | None
    single: np.ndarray | None
    y: np.ndarray | None
    keys: np.ndarray


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([x.decode("utf-8", errors="replace") if isinstance(x, (bytes, np.bytes_)) else str(x)
                       for x in values])


def load_input(path: Path, dataset: str, method: str) -> Input:
    with h5py.File(path, "r") as f:
        subject = str(f.attrs.get("subject_id", path.stem))
        if method == "Ours":
            obs = f["bci_obs"][:].astype(np.float32)
            indiv = f["bci_prior_indiv"][:].astype(np.float32)
            obs = obs.mean(1) if obs.ndim == 3 else obs
            indiv = indiv.mean(1) if indiv.ndim == 3 else indiv
            single = None
            n = len(obs)
        else:
            single = f["neurostorm_feature"][:].astype(np.float32)
            obs = indiv = None
            n = len(single)
        if "trial_run" in f and "trial_source_event_index" in f:
            runs = decode(f["trial_run"][:])
            events = f["trial_source_event_index"][:].astype(np.int64)
            keys = np.asarray([f"{r}|{int(e)}" for r, e in zip(runs, events)])
        else:
            keys = np.asarray([str(i) for i in range(n)])
        y = None
        if dataset == "ds000210" and "trial_condition" in f:
            mapping = {"Past": 0, "Future": 1, "Other": 2}
            values = decode(f["trial_condition"][:])
            y = np.asarray([mapping[x] for x in values], dtype=np.int64)
        elif dataset == "ds002835" and "trial_context" in f:
            values = decode(f["trial_context"][:])
            y = np.asarray([1 if "farfuture=1" in x else 0 for x in values], dtype=np.int64)
    return Input(subject, obs, indiv, single, y, keys)


def checkpoint_for(dataset: str, method: str, subject: str,
                   checkpoint_root: Path) -> Path:
    records = json.loads((HERE.parent / 'fixed_checkpoints.json').read_text())['rows']
    matches = [r for r in records if (r['dataset'], r['method'], r['subject']) ==
               (dataset, method.lower(), subject)]
    if len(matches) != 1:
        raise ValueError(f'Expected one fixed checkpoint for {dataset}/{method}/{subject}')
    return checkpoint_root / matches[0]['checkpoint']


def safe_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def make_model(payload: dict, method: str, device: torch.device) -> nn.Module:
    d_in = int(payload["input_dim"])
    n_classes = int(payload["n_classes"])
    factory = lambda dimension: nn.Linear(dimension, n_classes)
    if method == "Ours":
        model = GaussianPoE(d_in, factory, latent=128, hidden=256, dropout=0.2)
    else:
        model = GaussianSingleExpert(d_in, factory, latent=128, hidden=256, dropout=0.2)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def forward(payload: dict, item: Input, method: str, device: torch.device,
            batch_size: int) -> np.ndarray:
    model = make_model(payload, method, device)
    x_scaler = Standardizer(payload["x_scaler"]["mean"], payload["x_scaler"]["std"])
    if method == "Ours":
        assert item.obs is not None and item.indiv is not None
        second = Standardizer(payload["indiv_scaler"]["mean"], payload["indiv_scaler"]["std"])
        x1, x2 = x_scaler.transform(item.obs), second.transform(item.indiv)
    else:
        assert item.single is not None
        x1, x2 = x_scaler.transform(item.single), None
    chunks = []
    for start in range(0, len(x1), batch_size):
        first = torch.as_tensor(x1[start:start + batch_size], device=device)
        if method == "Ours":
            other = torch.as_tensor(x2[start:start + batch_size], device=device)
            logits = model(first, other)["poe"]
        else:
            logits = model(first)["pred"]
        chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    path.write_bytes(buffer.getvalue())


def run(dataset: str, method: str, output_dir: Path, device_name: str,
        batch_size: int = 256, subject: str | None = None,
        input_h5: Path | None = None,
        feature_root: Path | None = None, checkpoint_root: Path = CHECKPOINT_ROOT) -> dict:
    if method == "NeuroSTORM":
        method = "NeuroStorm"
    paper_method = "NeuroSTORM" if method == "NeuroStorm" else method
    if method not in ("Ours", "NeuroStorm"):
        raise ValueError(method)
    feature_root = FEATURE_ROOTS[(dataset, method)] if feature_root is None else feature_root
    if input_h5 is not None and subject is None:
        subject = input_h5.stem
    if input_h5 is not None:
        paths = [input_h5]
    elif subject is not None:
        paths = [feature_root / f"{subject}.h5"]
    else:
        paths = sorted(feature_root.glob("sub-*.h5"))
    if not paths:
        raise FileNotFoundError(f"no feature H5 under {feature_root}")

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, fold_rows, checkpoint_rows = [], [], []
    for path in paths:
        item = load_input(path, dataset, method)
        model_subject = subject if input_h5 is not None and subject is not None else item.subject
        checkpoint = checkpoint_for(dataset, method, model_subject, checkpoint_root)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = safe_load(checkpoint)
        probability = forward(payload, item, method, device, batch_size)
        predicted = probability.argmax(1)
        names = tuple(payload["condition_names"])
        for i in range(len(predicted)):
            row = {"dataset": dataset, "method": paper_method, "source_alias": method, "subject": item.subject,
                   "checkpoint": str(checkpoint), "trial_index": i, "trial_key": item.keys[i],
                   "true_index": "" if item.y is None else int(item.y[i]),
                   "predicted_index": int(predicted[i]), "predicted_condition": names[int(predicted[i])]}
            for c, name in enumerate(names):
                row[f"prob_{name}"] = float(probability[i, c])
            all_rows.append(row)
        fold = {"subject": item.subject, "n_samples": len(predicted)}
        if item.y is not None:
            fold.update({"accuracy": float(accuracy_score(item.y, predicted)),
                         "balanced_accuracy": float(balanced_accuracy_score(item.y, predicted)),
                         "macro_f1": float(f1_score(item.y, predicted, average="macro"))})
        fold_rows.append(fold)
        checkpoint_rows.append({"subject": model_subject, "path": str(checkpoint),
                                "sha256": file_sha256(checkpoint),
                                "training_seed": int(payload["training_seed"])})

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    with (output_dir / "per_subject_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = []
        for row in fold_rows:
            for key in row:
                if key not in fields: fields.append(key)
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(fold_rows)
    metrics = {"dataset": dataset, "method": paper_method,
               "source_alias": method,
               "n_subjects": len(fold_rows), "n_trials": len(all_rows),
               "mean_subject_accuracy": None if "accuracy" not in fold_rows[0]
               else float(np.mean([x["accuracy"] for x in fold_rows])),
               "per_subject": fold_rows}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "checkpoint_manifest.json").write_text(json.dumps({
        "weights_are_external_pointers": False, "checkpoint_root": str(checkpoint_root),
        "checkpoints": checkpoint_rows,
    }, indent=2), encoding="utf-8")
    (output_dir / "inference_manifest.json").write_text(json.dumps({
        "fresh_model_forward": True, "dataset": dataset, "method": paper_method,
        "source_alias": method,
        "feature_root": str(feature_root), "input_h5": None if input_h5 is None else str(input_h5),
        "output_dir": str(output_dir), "device": str(device)
    }, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics
