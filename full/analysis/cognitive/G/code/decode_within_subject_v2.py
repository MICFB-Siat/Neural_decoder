

















from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import torch

import decode_subjective as base
from common import experiment_paths, load_config, set_seed, sha256, write_json


METRICS = ("pearson_r", "spearman_rho", "mae", "rmse", "r2")


def safe_torch_save(payload: dict, destination: Path) -> None:

    stage_root = Path("/var/tmp/eigen_brain_decoding_zizhuan/torch_staging")
    stage_root.mkdir(parents=True, exist_ok=True)
    local = stage_root / f"{os.getpid()}_{destination.name}.local"
    incoming = destination.with_name(f"{destination.name}.incoming-{os.getpid()}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    local.unlink(missing_ok=True)
    incoming.unlink(missing_ok=True)
    torch.save(payload, local)
    local_hash = sha256(local)
    shutil.copy2(local, incoming)
    if incoming.stat().st_size != local.stat().st_size or sha256(incoming) != local_hash:
        raise RuntimeError(f"checkpoint staging verification failed: {incoming}")
    try:
        os.replace(incoming, destination)
    except OSError:
        destination.unlink(missing_ok=True)
        shutil.copy2(local, destination)
        incoming.unlink(missing_ok=True)
    if destination.stat().st_size != local.stat().st_size or sha256(destination) != local_hash:
        raise RuntimeError(f"checkpoint publish verification failed: {destination}")
    local.unlink(missing_ok=True)


def read_subject_subset(
    config: dict, dataset: str, method: str, feature_set: str,
    expected_subjects: int, shard_index: int | None, shard_count: int | None,
) -> tuple[list[base.Subject], list[str]]:

    root = (experiment_paths("decoding")["features"] / feature_set /
            ("poe" if method == "poe" else "neurostorm") / dataset)
    if not root.is_dir():
        raise FileNotFoundError(f"missing {method}/{dataset} features: {root}")
    by_stem: dict[str, Path] = {}
    for path in sorted(root.glob("sub-*.h5")):
        previous = by_stem.get(path.stem)
        if previous is not None and (
                previous != path or sha256(previous) != sha256(path)):
            raise RuntimeError(
                f"{dataset}: conflicting feature files for {path.stem}: {previous}, {path}"
            )
        by_stem[path.stem] = path
    all_names = sorted(by_stem)
    if len(all_names) != expected_subjects:
        raise RuntimeError(
            f"{dataset}: expected {expected_subjects} unique subjects, "
            f"found {len(all_names)}: {all_names}"
        )
    selected_names = all_names
    if shard_count is not None:
        selected_names = [
            name for position, name in enumerate(all_names)
            if position % shard_count == shard_index
        ]
    if not selected_names:
        raise RuntimeError(f"empty shard {shard_index}/{shard_count}")

    ordinal = config["datasets"][dataset]["subjective_target"] == "clarity_ordinal"
    result: list[base.Subject] = []
    for name in selected_names:
        path = by_stem[name]
        with h5py.File(path, "r") as handle:
            labels = handle["labels_subjective"][:].astype(np.float32)
            y = ((4.0 - labels[:, 0]).astype(np.float32)
                 if ordinal else labels[:, :2].astype(np.float32))
            objective = handle["labels_objective"][:].astype(np.int64)
            if method == "poe":
                obs = handle["bci_obs"][:].astype(np.float32).mean(axis=1)
                indiv = handle["bci_prior_indiv"][:].astype(np.float32).mean(axis=1)
                if obs.shape != indiv.shape:
                    raise ValueError(f"{path}: obs/indiv mismatch")
                x = None
            else:
                obs = indiv = None
                x = handle["neurostorm_feature"][:].astype(np.float32)
            subject_id = str(handle.attrs.get("subject_id", name))
        if subject_id != name:
            raise RuntimeError(f"feature subject mismatch: {path}: {subject_id} != {name}")
        result.append(base.Subject(
            dataset, subject_id, obs, indiv, x, y, ordinal, objective,
            np.arange(len(y), dtype=np.int64), str(path),
        ))
    return result, all_names


def centered_subject(subject: base.Subject, mean: np.ndarray) -> base.Subject:

    y = np.asarray(subject.y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    return replace(subject, y=(y - mean).astype(np.float32), ordinal=False)


def train_mean(subject: base.Subject, index: np.ndarray) -> np.ndarray:
    y = np.asarray(subject.y[index], dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    mean = y.mean(axis=0).astype(np.float32)
    if not np.isfinite(mean).all():
        raise ValueError(f"{subject.dataset}/{subject.subject}: non-finite train mean")
    return mean


def restore_scaler(state):
    return base.state_to_standardizer(state)


def add_rows(
    rows: list[dict], dataset: str, method: str, target_mode: str,
    subject: base.Subject, test_index: np.ndarray, seed_index: int,
    prediction: dict[str, np.ndarray], label_names: list[str],
    outer_mean: np.ndarray | None,
) -> None:
    original = np.asarray(subject.y, dtype=np.float32)
    if original.ndim == 1:
        original = original[:, None]
    for local, position in enumerate(test_index):
        common = {
            "dataset": dataset,
            "experiment": "within",
            "target_mode": target_mode,
            "method": method,
            "test_subject": subject.subject,
            "seed_index": seed_index,
            "trial_index": int(position),
        }
        if target_mode == "direct":
            if subject.ordinal:
                for branch, value in prediction.items():
                    rows.append(common | {
                        "scale": "rating", "branch": branch, "target": "clarity",
                        "true": float(subject.y[position]),
                        "prediction": float(value[local]),
                        "reference_mean": math.nan,
                    })
            else:
                for dim, target in enumerate(label_names):
                    for branch, value in prediction.items():
                        rows.append(common | {
                            "scale": "rating", "branch": branch, "target": target,
                            "true": float(original[position, dim]),
                            "prediction": float(value[local, dim]),
                            "reference_mean": math.nan,
                        })
            continue

        if outer_mean is None:
            raise RuntimeError("residual prediction requires outer-train mean")
        for dim, target in enumerate(label_names):
            reference = float(outer_mean[dim])
            true_raw = float(original[position, dim])
            true_residual = true_raw - reference
            for branch, value in prediction.items():
                pred_residual = float(value[local, dim])
                rows.append(common | {
                    "scale": "residual", "branch": branch, "target": target,
                    "true": true_residual, "prediction": pred_residual,
                    "reference_mean": reference,
                })
                rows.append(common | {
                    "scale": "rating_reconstructed", "branch": branch, "target": target,
                    "true": true_raw, "prediction": pred_residual + reference,
                    "reference_mean": reference,
                })


def group_values(rows: list[dict], fields: tuple[str, ...]):
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in fields), []).append(row)
    return grouped


def summarize(rows: list[dict], dataset: str, method: str, target_mode: str):
    per_seed = []
    fields = ("test_subject", "seed_index", "scale", "branch", "target")
    for key, values in sorted(group_values(rows, fields).items()):
        per_seed.append({
            "dataset": dataset, "experiment": "within", "target_mode": target_mode,
            "method": method, "subject": key[0], "seed_index": int(key[1]),
            "scale": key[2], "branch": key[3], "target": key[4],
            "aggregation": "single_seed",
            **base.regression_metrics(
                [value["true"] for value in values],
                [value["prediction"] for value in values],
            ),
        })

    ensemble_rows = []
    ensemble_fields = ("test_subject", "trial_index", "scale", "branch", "target")
    for key, values in sorted(group_values(rows, ensemble_fields).items()):
        first = values[0]
        ensemble_rows.append({
            "dataset": dataset, "experiment": "within", "target_mode": target_mode,
            "method": method, "test_subject": key[0], "trial_index": int(key[1]),
            "scale": key[2], "branch": key[3], "target": key[4],
            "true": float(first["true"]),
            "prediction": float(np.mean([value["prediction"] for value in values])),
            "n_seeds": len(values),
        })

    ensemble_metrics = []
    for key, values in sorted(group_values(
        ensemble_rows, ("test_subject", "scale", "branch", "target")
    ).items()):
        ensemble_metrics.append({
            "dataset": dataset, "experiment": "within", "target_mode": target_mode,
            "method": method, "subject": key[0], "scale": key[1],
            "branch": key[2], "target": key[3],
            "aggregation": "trialwise_mean_prediction_across_seeds",
            **base.regression_metrics(
                [value["true"] for value in values],
                [value["prediction"] for value in values],
            ),
        })

    seed_stats = []
    for key, values in sorted(group_values(
        per_seed, ("subject", "scale", "branch", "target")
    ).items()):
        row = {
            "dataset": dataset, "experiment": "within", "target_mode": target_mode,
            "method": method, "subject": key[0], "scale": key[1],
            "branch": key[2], "target": key[3], "n_seeds": len(values),
            "aggregation": "mean_and_sample_sd_of_seed_metrics",
        }
        for metric in METRICS:
            vector = np.asarray([float(value[metric]) for value in values], dtype=float)
            finite = vector[np.isfinite(vector)]
            row[f"{metric}_mean"] = float(finite.mean()) if finite.size else math.nan
            row[f"{metric}_sd"] = float(finite.std(ddof=1)) if finite.size > 1 else math.nan
        seed_stats.append(row)

    return per_seed, ensemble_rows, ensemble_metrics, seed_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ds000210", "ds002835"), required=True)
    parser.add_argument("--method", choices=("poe", "neurostorm"), required=True)
    parser.add_argument("--target-mode", choices=("direct", "subject_centered_residual"), required=True)
    parser.add_argument("--n-seeds", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--expected-subjects", type=int, required=True)
    parser.add_argument("--subject-shard-index", type=int)
    parser.add_argument("--subject-shard-count", type=int)
    parser.add_argument("--overwrite", action="store_true")
    cli = parser.parse_args()
    if cli.n_seeds < 1:
        parser.error("--n-seeds must be positive")
    if (cli.subject_shard_index is None) != (cli.subject_shard_count is None):
        parser.error("--subject-shard-index and --subject-shard-count must be used together")
    if cli.subject_shard_count is not None and not (
            cli.subject_shard_count > 0 and
            0 <= cli.subject_shard_index < cli.subject_shard_count):
        parser.error("invalid subject shard index/count")

    config = load_config()
    protocol = config["decoder_protocol"]
    args = argparse.Namespace(
        epochs=int(protocol["epochs"]), patience=int(protocol["early_stopping_patience"]),
        lr=float(protocol["learning_rate"]), weight_decay=float(protocol["weight_decay"]),
        batch_size=int(protocol["batch_size"]), hidden=int(protocol["hidden_dim"]),
        latent_dim=int(protocol["latent_dim"]), dropout=float(protocol["dropout"]),
        aux_weight=float(protocol["aux_weight"]), kl_weight=float(protocol["kl_weight"]),
        grad_clip=float(protocol["gradient_clip"]), min_delta=1e-4,
    )
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    subjects, all_subject_names = read_subject_subset(
        config, cli.dataset, cli.method, cli.feature_set,
        cli.expected_subjects, cli.subject_shard_index, cli.subject_shard_count,
    )
    label_names = list(config["datasets"][cli.dataset]["label_names"])
    ordinal_model = subjects[0].ordinal and cli.target_mode == "direct"
    d_out = 1 if ordinal_model else len(label_names)
    regression = not ordinal_model

    paths = experiment_paths("decoding")
    split_path = paths["splits"] / cli.run_tag / cli.dataset / "within_splits_50.json"
    if split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        if cli.subject_shard_count is not None:
            raise RuntimeError(
                "initialize the full split with an unsharded run before launching shards"
            )
        split = base.make_splits(
            subjects, "within", int(protocol["split_seed"]), max(50, cli.n_seeds),
        )
        write_json(split_path, split)
    available_seed_ids = {
        int(seed) for subject_spec in split["subjects"].values() for seed in subject_spec
    }
    if not set(range(cli.n_seeds)).issubset(available_seed_ids):
        raise RuntimeError(f"split file lacks seeds 0..{cli.n_seeds - 1}: {split_path}")

    shard_label = (None if cli.subject_shard_count is None else
                   f"shard-{cli.subject_shard_index:02d}-of-{cli.subject_shard_count:02d}")
    lookup = {subject.subject: subject for subject in subjects}

    result_root = (paths["results"] / cli.run_tag / "within_subject_internal" /
                   cli.target_mode / cli.dataset / cli.method)
    if shard_label is not None:
        result_root = result_root / "shards" / shard_label
    checkpoint_root = (paths["checkpoints"] / "decoding" / cli.run_tag /
                       "within_subject_internal" / cli.target_mode / cli.dataset / cli.method)
    result_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict] = []
    fold_rows: list[dict] = []

    for subject_name in sorted(lookup):
        original = lookup[subject_name]
        for seed_index in range(cli.n_seeds):
            spec = split["subjects"][subject_name][str(seed_index)]
            train_index = np.asarray(spec["train"], dtype=np.int64)
            val_index = np.asarray(spec["val"], dtype=np.int64)
            outer_train = np.asarray(spec["re_train"], dtype=np.int64)
            test_index = np.asarray(spec["test"], dtype=np.int64)
            checkpoint_path = checkpoint_root / f"{subject_name}_seed{seed_index:02d}_re.pt"
            selection_path = checkpoint_root / f"{subject_name}_seed{seed_index:02d}_selection.pt"
            training_seed = int(protocol["split_seed"]) + seed_index * 100003 + sum(map(ord, subject_name))

            if checkpoint_path.is_file() and not cli.overwrite:
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                expected = (cli.dataset, cli.method, cli.target_mode, subject_name, seed_index,
                            cli.feature_set, cli.run_tag)
                actual = (payload["dataset"], payload["method"], payload["target_mode"],
                          payload["test_subject"], payload["seed_index"],
                          payload["feature_set"], payload["run_tag"])
                if actual != expected:
                    raise RuntimeError(f"checkpoint provenance mismatch: {checkpoint_path}")
                xo = restore_scaler(payload["x_scaler"])
                xi = restore_scaler(payload.get("indiv_scaler"))
                ys = restore_scaler(payload.get("y_scaler"))
                outer_mean = (None if payload.get("outer_train_rating_mean") is None else
                              np.asarray(payload["outer_train_rating_mean"], dtype=np.float32))
                model = base.build_model(
                    cli.method, int(payload["input_dim"]), bool(payload["ordinal_model"]),
                    int(payload["output_dim"]), args, device,
                )
                model.load_state_dict(payload["model_state"], strict=True)
                selected_epoch = int(payload["selected_epoch"])
                selected_val = float(payload["selected_val_loss"])
                reused = True
            else:
                if cli.target_mode == "subject_centered_residual":
                    inner_mean = train_mean(original, train_index)
                    selection_subject = centered_subject(original, inner_mean)
                else:
                    inner_mean = None
                    selection_subject = original
                train_map = {subject_name: train_index}
                val_map = {subject_name: val_index}
                set_seed(training_seed)
                x_train, y_train, xo, xi, ys = base.train_arrays(
                    [selection_subject], train_map, cli.method, regression,
                )
                x_val, y_val = base.transform_arrays(
                    [selection_subject], val_map, cli.method, xo, xi, ys, regression,
                )
                train_tensors = base.tensors(x_train, y_train, device)
                val_tensors = base.tensors(x_val, y_val, device)
                model = base.build_model(cli.method, x_train[0].shape[1], ordinal_model, d_out, args, device)
                selected_epoch, selected_val, selected_state = base.choose_epoch(
                    model, train_tensors, val_tensors, cli.method, ordinal_model, args, training_seed,
                )
                safe_torch_save({
                    "format": "within_subject_decoder_selection_v2",
                    "dataset": cli.dataset, "method": cli.method, "target_mode": cli.target_mode,
                    "feature_set": cli.feature_set, "run_tag": cli.run_tag,
                    "test_subject": subject_name, "seed_index": seed_index,
                    "selected_epoch": selected_epoch, "selected_val_loss": selected_val,
                    "input_dim": int(x_train[0].shape[1]), "output_dim": d_out,
                    "ordinal_model": ordinal_model, "split": spec,
                    "inner_train_rating_mean": None if inner_mean is None else inner_mean.tolist(),
                    "x_scaler": xo.state(), "indiv_scaler": None if xi is None else xi.state(),
                    "y_scaler": None if ys is None else ys.state(),
                    "model_state": selected_state, "protocol": vars(args),
                    "selection_training_seed": training_seed,
                }, selection_path)

                if cli.target_mode == "subject_centered_residual":
                    outer_mean = train_mean(original, outer_train)
                    re_subject = centered_subject(original, outer_mean)
                else:
                    outer_mean = None
                    re_subject = original
                re_map = {subject_name: outer_train}
                set_seed(training_seed)
                x_re, y_re, xo, xi, ys = base.train_arrays(
                    [re_subject], re_map, cli.method, regression,
                )
                re_tensors = base.tensors(x_re, y_re, device)
                model = base.build_model(cli.method, x_re[0].shape[1], ordinal_model, d_out, args, device)
                base.re(
                    model, re_tensors, cli.method, ordinal_model, args,
                    training_seed, selected_epoch,
                )
                payload = {
                    "format": "within_subject_decoder_re_v2",
                    "dataset": cli.dataset, "method": cli.method, "target_mode": cli.target_mode,
                    "feature_set": cli.feature_set, "run_tag": cli.run_tag,
                    "test_subject": subject_name, "seed_index": seed_index,
                    "selected_epoch": selected_epoch, "selected_val_loss": selected_val,
                    "input_dim": int(x_re[0].shape[1]), "output_dim": d_out,
                    "ordinal_model": ordinal_model, "split": spec,
                    "test_trial_index": test_index.tolist(),
                    "outer_train_rating_mean": None if outer_mean is None else outer_mean.tolist(),
                    "x_scaler": xo.state(), "indiv_scaler": None if xi is None else xi.state(),
                    "y_scaler": None if ys is None else ys.state(),
                    "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "protocol": vars(args), "selection_checkpoint": str(selection_path),
                    "re_training_seed": training_seed,
                    "residual_contract": (
                        "inner validation centered by inner-train mean; outer test centered by outer-train mean"
                        if cli.target_mode == "subject_centered_residual" else None
                    ),
                }
                safe_torch_save(payload, checkpoint_path)
                reused = False

            prediction_subject = (
                centered_subject(original, outer_mean)
                if cli.target_mode == "subject_centered_residual" else original
            )
            prediction = base.predict(
                model, prediction_subject, test_index, cli.method, xo, xi, ys, device,
            )
            add_rows(
                prediction_rows, cli.dataset, cli.method, cli.target_mode, original,
                test_index, seed_index, prediction, label_names, outer_mean,
            )
            fold_rows.append({
                "dataset": cli.dataset, "experiment": "within", "target_mode": cli.target_mode,
                "method": cli.method, "test_subject": subject_name, "seed_index": seed_index,
                "reused_checkpoint": reused, "selected_epoch": selected_epoch,
                "selected_val_loss": selected_val, "n_test_trials": len(test_index),
                "outer_train_rating_mean": (
                    "" if outer_mean is None else json.dumps(outer_mean.tolist())
                ),
                "checkpoint": str(checkpoint_path),
            })
            print(
                f"[{cli.dataset}/{cli.method}/{cli.target_mode}] {subject_name} "
                f"seed={seed_index:02d} epoch={selected_epoch} val={selected_val:.6f} "
                f"{'reused' if reused else 'trained'}",
                flush=True,
            )

    per_seed, ensemble, ensemble_metrics, seed_stats = summarize(
        prediction_rows, cli.dataset, cli.method, cli.target_mode,
    )
    base.write_csv(result_root / "predictions_per_seed.csv", prediction_rows)
    base.write_csv(result_root / "predictions_ensemble.csv", ensemble)
    base.write_csv(result_root / "metrics_per_subject_per_seed.csv", per_seed)
    base.write_csv(result_root / "metrics_per_subject_ensemble.csv", ensemble_metrics)
    base.write_csv(result_root / "metrics_per_subject_seed_mean_sd.csv", seed_stats)
    base.write_csv(result_root / "folds.csv", fold_rows)
    write_json(result_root / "run_manifest.json", {
        "format": "within_subject_decoding_v2",
        "dataset": cli.dataset, "method": cli.method, "target_mode": cli.target_mode,
        "feature_set": cli.feature_set, "run_tag": cli.run_tag,
        "n_subjects": len(subjects), "n_total_subjects": len(all_subject_names),
        "n_seeds": cli.n_seeds, "subject_shard": shard_label,
        "label_head": (
            "coral_ordinal_3class" if ordinal_model else
            f"smoothl1_regression_{d_out}output"
        ),
        "mae_features_frozen": True,
        "split_file": str(split_path), "checkpoint_root": str(checkpoint_root),
        "features": {
            subject.subject: {"path": subject.feature_path, "sha256": sha256(subject.feature_path)}
            for subject in subjects
        },
        "protocol": vars(args),
        "subject_centered_residual": (
            "train/validation uses inner-train mean; re/test uses outer-train mean; no test-label centering"
            if cli.target_mode == "subject_centered_residual" else None
        ),
        "requested_primary_aggregation": "mean and sample SD across all evaluated seeds",
    })


if __name__ == "__main__":
    main()
