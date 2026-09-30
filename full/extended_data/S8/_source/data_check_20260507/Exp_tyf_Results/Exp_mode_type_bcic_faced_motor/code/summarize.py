

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiment_config import (
    ADAPT_ARGS,
    DATASETS,
    DATASET_ORDER,
    DATA_ROOT,
    MODE_LABELS,
    MODE_TYPES,
    N_FOLDS,
    PRETRAIN_ROOT,
    RESULT_ROOT,
    ROOT,
    SEEDS,
    STATE_ROOT,
    checkpoint_path,
    run_root,
)
from run_one import task_complete


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, payload: dict):
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True))


def atomic_csv(path: Path, header: list[str], rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp, path)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    shifted = x - x.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    grid = np.linspace(0.05, 10.0, 200)
    best_t, best_nll = 1.0, float("inf")
    row = np.arange(len(labels))
    for temperature in grid:
        lp = log_softmax(logits / temperature)
        nll = float(-lp[row, labels].mean())
        if nll < best_nll:
            best_t, best_nll = float(temperature), nll
    return best_t


def crossfit_temperature_poe(z_obs: np.ndarray, z_pri: np.ndarray, labels: np.ndarray, n_classes: int):
    rng = np.random.default_rng(0)
    half_a, half_b = [], []
    for class_id in range(n_classes):
        indices = np.where(labels == class_id)[0]
        rng.shuffle(indices)
        middle = len(indices) // 2
        half_a.extend(indices[:middle].tolist())
        half_b.extend(indices[middle:].tolist())
    half_a = np.asarray(sorted(half_a), dtype=np.int64)
    half_b = np.asarray(sorted(half_b), dtype=np.int64)
    if not len(half_a) or not len(half_b):
        raise ValueError("Temperature cross-fit produced an empty half")
    temperatures = {
        "obs_a": fit_temperature(z_obs[half_a], labels[half_a]),
        "pri_a": fit_temperature(z_pri[half_a], labels[half_a]),
        "obs_b": fit_temperature(z_obs[half_b], labels[half_b]),
        "pri_b": fit_temperature(z_pri[half_b], labels[half_b]),
    }
    prediction = np.empty(len(labels), dtype=np.int64)
    for use, obs_t, pri_t in (
        (half_b, temperatures["obs_a"], temperatures["pri_a"]),
        (half_a, temperatures["obs_b"], temperatures["pri_b"]),
    ):
        fused = log_softmax(z_obs[use] / obs_t) + log_softmax(z_pri[use] / pri_t)
        prediction[use] = fused.argmax(axis=-1)
    return prediction, temperatures


def build_status():
    datasets = {}
    completed_tasks = 0
    total_tasks = len(DATASETS) * len(MODE_TYPES) * 2 * len(SEEDS)
    for dataset in DATASET_ORDER:
        cfg = DATASETS[dataset]
        manifest_path = DATA_ROOT / dataset / "manifest.json"
        manifest_ok = False
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_ok = (
                    manifest["n_subjects"] == cfg["expected_subjects"]
                    and manifest["n_modes"] == cfg["n_modes"]
                    and manifest["n_samples"] == cfg["n_samples"]
                )
            except Exception:
                pass
        dataset_status = {"manifest_complete": manifest_ok, "modes": {}}
        for mode_type in MODE_TYPES:
            h5_count = len(list((DATA_ROOT / dataset / mode_type).glob("sub-*.h5")))
            meta_path = checkpoint_path(dataset, mode_type).with_suffix(".json")
            checkpoint_ok = False
            if checkpoint_path(dataset, mode_type).is_file() and meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    checkpoint_ok = (
                        meta["dataset"] == dataset
                        and meta["mode_type"] == mode_type
                        and meta["epochs"] == ADAPT_ARGS["epochs"]
                        and meta["n_subjects_loaded"] == cfg["expected_subjects"]
                        and meta["subject_limit"] is None
                    )
                except Exception:
                    pass
            paradigms = {}
            for paradigm in ("within", "pool"):
                done = [seed for seed in SEEDS if task_complete(dataset, mode_type, paradigm, seed)]
                completed_tasks += len(done)
                paradigms[paradigm] = {"completed_seeds": done, "n_complete": len(done), "n_total": len(SEEDS)}
            dataset_status["modes"][mode_type] = {
                "h5_subjects": h5_count,
                "h5_complete": manifest_ok and h5_count == cfg["expected_subjects"],
                "checkpoint_complete": checkpoint_ok,
                "paradigms": paradigms,
            }
        datasets[dataset] = dataset_status
    return {
        "experiment_root": str(ROOT),
        "primary_metric": "split-half cross-fit temperature-scaled PoE accuracy",
        "completed_classification_tasks": completed_tasks,
        "total_classification_tasks": total_tasks,
        "all_complete": completed_tasks == total_tasks,
        "datasets": datasets,
    }


def collect_rows(status: dict):
    rows = []
    fold_rows = []
    for dataset in DATASET_ORDER:
        cfg = DATASETS[dataset]
        for mode_type in MODE_TYPES:
            for seed in SEEDS:
                if task_complete(dataset, mode_type, "within", seed):
                    root = run_root(dataset, mode_type, "within", seed)
                    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
                    for subject in sorted(summary["per_subject"]):
                        metric = summary["per_subject"][subject]["poe_temp"]
                        rows.append([
                            dataset, mode_type, "within", seed, subject,
                            float(metric["mean"]), N_FOLDS, "fold_mean",
                        ])
                        for fold, value in enumerate(metric["values"]):
                            fold_rows.append([dataset, mode_type, "within", seed, subject, fold, float(value)])
                if task_complete(dataset, mode_type, "pool", seed):
                    root = run_root(dataset, mode_type, "pool", seed)
                    for fold_dir in sorted(root.glob("fold_*")):
                        fold = int(fold_dir.name.split("_")[-1])
                        with np.load(fold_dir / "holdout_preds.npz") as pred_file:
                            z_obs = pred_file["z_obs"]
                            z_pri = pred_file["z_pri"]
                            labels = pred_file["y"].astype(np.int64)
                        subject_ids = np.load(fold_dir / "holdout_subj_ids.npy").astype(str)
                        prediction, _ = crossfit_temperature_poe(z_obs, z_pri, labels, cfg["n_classes"])
                        fold_acc = float(np.mean(prediction == labels))
                        metrics = json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))
                        if not np.isclose(fold_acc, metrics["poe_temp"], atol=1e-12):
                            raise RuntimeError(
                                f"Pool reconstruction mismatch {fold_dir}: {fold_acc} != {metrics['poe_temp']}"
                            )
                        for subject in sorted(set(subject_ids.tolist())):
                            mask = subject_ids == subject
                            accuracy = float(np.mean(prediction[mask] == labels[mask]))
                            rows.append([
                                dataset, mode_type, "pool", seed, subject,
                                accuracy, int(mask.sum()), "heldout_trials",
                            ])
                            fold_rows.append([dataset, mode_type, "pool", seed, subject, fold, accuracy])
    return rows, fold_rows


def make_summary(rows: list[list]):
    seed_groups = defaultdict(list)
    for dataset, mode_type, paradigm, seed, subject, accuracy, n, unit in rows:
        seed_groups[(dataset, mode_type, paradigm, seed)].append(float(accuracy))
    seed_rows = []
    for key in sorted(seed_groups):
        values = np.asarray(seed_groups[key], dtype=float)
        seed_rows.append([*key, len(values), float(values.mean()), float(values.std())])

    aggregate_groups = defaultdict(list)
    for dataset, mode_type, paradigm, seed, n_subjects, mean, std in seed_rows:
        aggregate_groups[(dataset, mode_type, paradigm)].append(float(mean))
    aggregate_rows = []
    for key in sorted(aggregate_groups):
        values = np.asarray(aggregate_groups[key], dtype=float)
        aggregate_rows.append([
            *key, len(values), float(values.mean()), float(values.std()),
            float(np.median(values)), float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75)),
        ])
    return seed_rows, aggregate_rows


def write_protocol():
    lines = [
        "# BCIC / FACED / MOTOR eigenmode-type ablation protocol",
        "",
        "## Frozen design",
        "",
        "- Conditions: DCT-II Fourier, sampled real spherical harmonics + QR, and 6-NN normalized graph Laplacian.",
        "- The only condition-specific input is the orthonormal sensor-space basis and its native spectrum.",
        "- Projection: `eeg_modes = basis.T @ eeg_raw`.",
        "- Mode count rule: `K=min(30, n_sensors)`; BCIC K=22, FACED K=30, MOTOR K=30.",
        "- Position eigenvalues: native spectrum plus its first non-zero gap, preserving ordering while avoiding `log(0)`.",
        "- Prior: unified_v3 initialization followed independently per dataset and basis by 50-epoch target-data MAE adaptation (90/10 per-subject split, seed 0, mask 0.5, batch 64, lr 3e-4).",
        "- Classifier: frozen adapted prior + frozen BrainOmni; cached features; adapter/Gaussian/classification heads only; 200 epochs, batch 32, lr 1e-3, latent 256, dropout 0.3, patience 30, KL 0.001.",
        "- Within: trial-level stratified 5-fold separately for every subject.",
        "- Pool: shuffled subject-level KFold(5) with no subject overlap between train and held-out fold.",
        "- Seeds: 0-9. Primary metric: split-half cross-fit temperature-scaled PoE accuracy.",
        "",
        "## Dataset-specific settings",
        "",
        "|Dataset|Subjects|Sensors|K|Samples/trial|Label|Classes|",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for dataset in ("BCIC", "FACED", "MOTOR"):
        cfg = DATASETS[dataset]
        lines.append(
            f"|{dataset}|{cfg['expected_subjects']}|{cfg['n_sensors']}|{cfg['n_modes']}|"
            f"{cfg['n_samples']}|`{cfg['label_key']}`|{cfg['n_classes']}|"
        )
    lines += [
        "",
        "## Protocol boundary",
        "",
        "To match the existing SEED-V experiment exactly, the label-free MAE adaptation uses all target subjects before downstream CV; the held-out CV fold is also used for early stopping, and temperature calibration is cross-fitted inside held-out labels. These are preserved design properties, not newly introduced choices.",
    ]
    atomic_text(ROOT / "PROTOCOL.md", "\n".join(lines) + "\n")


def write_human_summary(status: dict, aggregate_rows: list[list]):
    lookup = {(r[0], r[1], r[2]): r for r in aggregate_rows}
    lines = [
        "# BCIC / FACED / MOTOR eigenmode-type ablation status and results",
        "",
        f"Completed formal classification tasks: {status['completed_classification_tasks']}/{status['total_classification_tasks']}.",
        "",
        "Primary metric: split-half cross-fit temperature-scaled PoE accuracy. Values below summarize completed seed-level subject means.",
        "",
        "|Dataset|Basis|Paradigm|Seeds complete|Mean|SD across seeds|Median [IQR]|",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASET_ORDER:
        for mode_type in MODE_TYPES:
            for paradigm in ("within", "pool"):
                row = lookup.get((dataset, mode_type, paradigm))
                if row is None:
                    values = ["0", "—", "—", "—"]
                else:
                    _, _, _, n, mean, std, median, q1, q3 = row
                    values = [str(n), f"{100*mean:.2f}%", f"{100*std:.2f}%", f"{100*median:.2f}% [{100*q1:.2f}, {100*q3:.2f}]%"]
                lines.append(
                    f"|{dataset}|{MODE_LABELS[mode_type]}|{paradigm}|{values[0]}|{values[1]}|{values[2]}|{values[3]}|"
                )
    atomic_text(RESULT_ROOT / "实验状态与结果汇总.md", "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status-only", action="store_true")
    args = ap.parse_args()
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    status = build_status()
    atomic_json(STATE_ROOT / "status.json", status)
    if args.status_only:
        print(json.dumps(status, indent=2))
        return

    rows, fold_rows = collect_rows(status)
    seed_rows, aggregate_rows = make_summary(rows)
    atomic_csv(
        RESULT_ROOT / "seed_subject_accuracy.csv",
        ["dataset", "mode_type", "paradigm", "seed", "subject", "accuracy", "n_folds_or_trials", "unit"],
        rows,
    )
    atomic_csv(
        RESULT_ROOT / "fold_subject_accuracy.csv",
        ["dataset", "mode_type", "paradigm", "seed", "subject", "fold", "accuracy"],
        fold_rows,
    )
    atomic_csv(
        RESULT_ROOT / "seed_level_summary.csv",
        ["dataset", "mode_type", "paradigm", "seed", "n_subjects", "mean_accuracy", "subject_sd"],
        seed_rows,
    )
    atomic_csv(
        RESULT_ROOT / "aggregate_summary.csv",
        ["dataset", "mode_type", "paradigm", "n_seeds", "mean_accuracy", "seed_sd", "median", "q1", "q3"],
        aggregate_rows,
    )
    write_protocol()
    write_human_summary(status, aggregate_rows)
    print(
        f"status={status['completed_classification_tasks']}/{status['total_classification_tasks']} "
        f"seed_subject_rows={len(rows)}"
    )


if __name__ == "__main__":
    main()
