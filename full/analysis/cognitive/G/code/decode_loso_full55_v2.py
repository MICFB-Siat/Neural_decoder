












from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch

import decode_subjective as base
import decode_within_subject_v2 as within
from common import experiment_paths, load_config, set_seed, sha256, write_json


PROTOCOL_VERSION = "full55_loso_direct_and_stimulus_residual_frozen_mae_v2"


def safe_torch_save(payload: dict, destination: Path) -> None:






    stage_root = Path("/var/tmp/eigen_brain_decoding_zizhuan/torch_staging_loso")
    stage_root.mkdir(parents=True, exist_ok=True)
    local = stage_root / f"{os.getpid()}_{destination.name}.local"
    incoming = destination.with_name(f"{destination.name}.incoming-{os.getpid()}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for path in (local, incoming):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    torch.save(payload, local)
    local_size, local_hash = local.stat().st_size, sha256(local)

    def valid_target() -> bool:
        try:
            return (destination.is_file() and destination.stat().st_size == local_size
                    and sha256(destination) == local_hash)
        except OSError:
            return False

    try:
        last_error: Exception | None = None
        for attempt in range(8):
            try:
                incoming.unlink(missing_ok=True)
            except OSError:
                pass
            try:


                with local.open("rb") as source, incoming.open("wb") as target:
                    shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                if (
                    incoming.stat().st_size != local_size
                    or sha256(incoming) != local_hash
                ):
                    raise RuntimeError(
                        f"checkpoint staging verification failed: {incoming}"
                    )
                try:
                    os.replace(incoming, destination)
                except OSError:
                    if not valid_target():
                        with local.open("rb") as source, destination.open("wb") as target:
                            shutil.copyfileobj(
                                source, target, length=4 * 1024 * 1024
                            )
                            target.flush()
                            os.fsync(target.fileno())
                if valid_target():
                    break
                raise RuntimeError(
                    f"checkpoint publish verification failed: {destination}"
                )
            except (OSError, RuntimeError) as error:
                last_error = error
                if attempt == 7:
                    raise
                time.sleep(min(8.0, 0.5 * (2 ** attempt)))
        else:
            raise RuntimeError(
                f"checkpoint publish exhausted retries: {destination}: "
                f"{last_error}"
            )
    finally:
        for path in (incoming, local):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


@dataclass
class SubjectData:
    model: base.Subject
    keys: np.ndarray
    metadata_path: str


@dataclass
class Reference:
    means: dict[str, np.ndarray]
    counts: dict[str, np.ndarray]

    def serializable(self) -> dict:
        return {
            "means": {key: value.tolist() for key, value in self.means.items()},
            "counts": {key: value.tolist() for key, value in self.counts.items()},
        }

    @classmethod
    def from_state(cls, state: dict) -> "Reference":
        return cls(
            means={key: np.asarray(value, dtype=np.float32) for key, value in state["means"].items()},
            counts={key: np.asarray(value, dtype=np.int64) for key, value in state["counts"].items()},
        )


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ])


def make_keys(dataset: str, stimulus: np.ndarray, condition: np.ndarray,
              context: np.ndarray) -> np.ndarray:
    stimulus_s = decode_strings(stimulus)
    condition_s = decode_strings(condition)
    context_s = decode_strings(context)
    if dataset == "ds000210":
        return np.asarray([
            f"IAPS={stim}|condition={cond}"
            for stim, cond in zip(stimulus_s, condition_s)
        ])
    result = []
    for stim, text in zip(stimulus_s, context_s):
        match = re.search(r"(?:^|;)farfuture=([01])(?:;|$)", text)
        if match is None:
            raise ValueError(f"ds002835 trial_context lacks farfuture: {text}")
        result.append(f"event={stim}|farfuture={match.group(1)}")
    return np.asarray(result)


def unique_feature_paths(root: Path, expected_subjects: int) -> dict[str, Path]:
    by_stem: dict[str, Path] = {}
    for path in sorted(root.glob("sub-*.h5")):
        previous = by_stem.get(path.stem)
        if previous is not None and previous.resolve() != path.resolve():
            raise RuntimeError(f"duplicate subject feature: {previous} / {path}")
        by_stem[path.stem] = path
    if len(by_stem) != expected_subjects:
        raise RuntimeError(
            f"{root}: expected {expected_subjects} unique subjects, found {len(by_stem)}"
        )
    return by_stem


def assert_paired(meta: h5py.File, feature: h5py.File, meta_path: Path,
                  feature_path: Path) -> None:
    for key in ("labels_subjective", "labels_objective", "trial_condition",
                "trial_context", "trial_run", "trial_source_event_index"):
        if key not in meta or key not in feature:
            raise KeyError(f"missing paired alignment field {key}: {meta_path} / {feature_path}")
        left, right = meta[key][...], feature[key][...]
        if left.dtype.kind in "OSU" or right.dtype.kind in "OSU":
            equal = np.array_equal(decode_strings(left), decode_strings(right))
        else:
            equal = np.array_equal(left, right, equal_nan=True)
        if not equal:
            raise RuntimeError(f"paired feature misalignment for {key}: {meta_path} / {feature_path}")


def read_subjects(config: dict, dataset: str, method: str, feature_set: str,
                  expected_subjects: int) -> tuple[dict[str, SubjectData], dict]:
    paths = experiment_paths("decoding")
    feature_root = paths["features"] / feature_set
    poe_paths = unique_feature_paths(feature_root / "poe" / dataset, expected_subjects)
    method_paths = unique_feature_paths(
        feature_root / ("poe" if method == "poe" else "neurostorm") / dataset,
        expected_subjects,
    )
    if set(poe_paths) != set(method_paths):
        raise RuntimeError(f"{dataset}/{method}: PoE/target-method subject sets differ")

    ordinal = config["datasets"][dataset]["subjective_target"] == "clarity_ordinal"
    result: dict[str, SubjectData] = {}
    records = {}
    for name in sorted(poe_paths):
        meta_path, feature_path = poe_paths[name], method_paths[name]
        with h5py.File(meta_path, "r") as meta, h5py.File(feature_path, "r") as feature:
            if method == "neurostorm":
                assert_paired(meta, feature, meta_path, feature_path)
            subject_id = str(feature.attrs.get("subject_id", name))
            if subject_id != name:
                raise RuntimeError(f"subject attribute mismatch: {feature_path}: {subject_id} != {name}")
            labels = meta["labels_subjective"][:].astype(np.float32)
            y = ((4.0 - labels[:, 0]).astype(np.float32)
                 if ordinal else labels[:, :2].astype(np.float32))
            objective = meta["labels_objective"][:].astype(np.int64)
            keys = make_keys(dataset, meta["stimulus_id"][:],
                             meta["trial_condition"][:], meta["trial_context"][:])
            if method == "poe":
                obs = feature["bci_obs"][:].astype(np.float32).mean(axis=1)
                indiv = feature["bci_prior_indiv"][:].astype(np.float32).mean(axis=1)
                x = None
                if obs.shape != indiv.shape:
                    raise RuntimeError(f"obs/indiv shape mismatch: {feature_path}")
            else:
                obs = indiv = None
                x = feature["neurostorm_feature"][:].astype(np.float32)
        arrays = [value for value in (obs, indiv, x, y) if value is not None]
        if any(len(value) != len(y) for value in arrays) or len(keys) != len(y):
            raise RuntimeError(f"trial count mismatch: {feature_path}")
        if not all(np.isfinite(value).all() for value in arrays):
            raise RuntimeError(f"non-finite feature/label: {feature_path}")
        model = base.Subject(
            dataset, name, obs, indiv, x, y, ordinal, objective,
            np.arange(len(y), dtype=np.int64), str(feature_path),
        )
        result[name] = SubjectData(model=model, keys=keys, metadata_path=str(meta_path))
        stat = feature_path.stat()
        records[name] = {
            "feature_path": str(feature_path), "metadata_path": str(meta_path),
            "n_trials": len(y), "feature_size_bytes": stat.st_size,
            "feature_mtime_ns": stat.st_mtime_ns,
        }
    manifest_candidates = [
        feature_root / "poe" / "manifest.json",
        feature_root / "neurostorm" / "validated_reuse_manifest.json",
    ]
    manifests = [
        {"path": str(path), "sha256": sha256(path)}
        for path in manifest_candidates if path.is_file()
    ]
    return result, {"subjects": records, "feature_manifests": manifests}


def compute_reference(subjects: dict[str, SubjectData], names: Iterable[str],
                      d_out: int) -> Reference:
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for name in names:
        subject = subjects[name]
        y_values = np.asarray(subject.model.y, dtype=np.float32)
        if y_values.ndim == 1:
            y_values = y_values[:, None]
        for key, y in zip(subject.keys, y_values):
            if key not in sums:
                sums[key] = np.zeros(d_out, dtype=np.float64)
                counts[key] = np.zeros(d_out, dtype=np.int64)
            finite = np.isfinite(y)
            sums[key][finite] += y[finite]
            counts[key][finite] += 1
    means = {}
    for key, total in sums.items():
        value = np.full(d_out, np.nan, dtype=np.float32)
        valid = counts[key] > 0
        value[valid] = (total[valid] / counts[key][valid]).astype(np.float32)
        means[key] = value
    return Reference(means=means, counts=counts)


def reference_equal(left: Reference, right: Reference) -> bool:
    return set(left.means) == set(right.means) and all(
        np.array_equal(left.counts[key], right.counts[key]) and
        np.allclose(left.means[key], right.means[key], equal_nan=True)
        for key in left.means
    )


def collect_residual(
    subjects: dict[str, SubjectData], names: Iterable[str], reference: Reference,
    method: str,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, list[dict]]:
    obs_values, indiv_values, ns_values, y_values, records = [], [], [], [], []
    for name in names:
        subject = subjects[name]
        raw_values = np.asarray(subject.model.y, dtype=np.float32)
        if raw_values.ndim == 1:
            raw_values = raw_values[:, None]
        for trial_index, (key, raw_y) in enumerate(zip(subject.keys, raw_values)):
            mean, count = reference.means.get(key), reference.counts.get(key)
            if mean is None or count is None or not np.isfinite(mean).all() or np.any(count <= 0):
                records.append({
                    "subject": name, "trial_index": trial_index,
                    "stimulus_key": key, "included": False,
                    "reason": "zero_training_reference",
                })
                continue
            residual = (raw_y - mean).astype(np.float32)
            y_values.append(residual)
            if method == "poe":
                obs_values.append(subject.model.obs[trial_index])
                indiv_values.append(subject.model.indiv[trial_index])
            else:
                ns_values.append(subject.model.x[trial_index])
            records.append({
                "subject": name, "trial_index": trial_index,
                "stimulus_key": key, "included": True, "reason": "",
                "raw_y": raw_y.copy(), "reference_mean": mean.copy(),
                "reference_n": count.copy(), "residual": residual.copy(),
            })
    if not y_values:
        raise RuntimeError("no trials remain after stimulus-reference matching")
    x = ((np.stack(obs_values).astype(np.float32), np.stack(indiv_values).astype(np.float32))
         if method == "poe" else (np.stack(ns_values).astype(np.float32),))
    return x, np.stack(y_values).astype(np.float32), records


def fit_raw(x: tuple[np.ndarray, ...], y: np.ndarray, method: str):
    xo = base.Standardizer.fit(x[0])
    if method == "poe":
        xi = base.Standardizer.fit(x[1])
        transformed = (xo.transform(x[0]), xi.transform(x[1]))
    else:
        xi = None
        transformed = (xo.transform(x[0]),)
    ys = base.Standardizer.fit(y)
    return transformed, ys.transform(y), xo, xi, ys


def transform_raw(x: tuple[np.ndarray, ...], y: np.ndarray, method: str,
                  xo, xi, ys):
    transformed = ((xo.transform(x[0]), xi.transform(x[1]))
                   if method == "poe" else (xo.transform(x[0]),))
    return transformed, ys.transform(y)


def predict_raw(model, x: tuple[np.ndarray, ...], method: str, xo, xi, ys,
                device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        if method == "poe":
            obs = torch.as_tensor(xo.transform(x[0]), dtype=torch.float32, device=device)
            indiv = torch.as_tensor(xi.transform(x[1]), dtype=torch.float32, device=device)
            output = model(obs, indiv)
            prediction = {branch: output[branch].cpu().numpy()
                          for branch in ("obs", "indiv", "poe")}
        else:
            value = torch.as_tensor(xo.transform(x[0]), dtype=torch.float32, device=device)
            prediction = {"neurostorm": model(value)["pred"].cpu().numpy()}
    return {branch: ys.inverse(value) for branch, value in prediction.items()}


def make_or_load_split(path: Path, names: list[str], n_seeds: int) -> dict:
    if path.is_file():
        split = json.loads(path.read_text(encoding="utf-8"))
        expected = (PROTOCOL_VERSION, names, n_seeds)
        actual = (split.get("protocol_version"), split.get("subjects"),
                  int(split.get("n_seeds", -1)))
        if actual != expected:
            raise RuntimeError(f"incompatible LOSO split: {path}")
        return split
    folds = {}
    for outer_position, test_subject in enumerate(names):
        outer_train = [name for name in names if name != test_subject]
        folds[test_subject] = {}
        for seed_index in range(n_seeds):
            val_subject = outer_train[(outer_position + seed_index) % len(outer_train)]
            folds[test_subject][str(seed_index)] = {
                "train_subjects": [name for name in outer_train if name != val_subject],
                "val_subject": val_subject,
                "re_train_subjects": outer_train,
                "test_subject": test_subject,
            }
    split = {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "nested_subject_loso_one_inner_validation_subject",
        "subjects": names, "n_seeds": n_seeds,
        "uses_subjective_labels_for_split": False,
        "folds": folds,
    }
    write_json(path, split)
    return split


def reference_rows(dataset: str, test_subject: str, seed_index: int | str,
                   phase: str, reference: Reference,
                   label_names: list[str]) -> list[dict]:
    rows = []
    for key in sorted(reference.means):
        for dimension, target in enumerate(label_names):
            rows.append({
                "dataset": dataset, "test_subject": test_subject,
                "seed_index": seed_index, "phase": phase,
                "stimulus_key": key, "target": target,
                "reference_mean": float(reference.means[key][dimension]),
                "reference_n": int(reference.counts[key][dimension]),
            })
    return rows


def add_direct_rows(rows: list[dict], dataset: str, method: str,
                    subject: base.Subject, seed_index: int,
                    prediction: dict[str, np.ndarray], label_names: list[str]) -> None:
    y = np.asarray(subject.y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    for trial_index in range(len(y)):
        common = {
            "dataset": dataset, "experiment": "loso_transductive",
            "target_mode": "direct", "method": method,
            "test_subject": subject.subject, "seed_index": seed_index,
            "trial_index": trial_index, "scale": "rating",
            "reference_mean": math.nan, "reference_n": "",
            "stimulus_key": "",
        }
        targets = ["clarity"] if subject.ordinal else label_names
        for dimension, target in enumerate(targets):
            true = float(y[trial_index, dimension])
            for branch, value in prediction.items():
                pred = float(value[trial_index] if subject.ordinal else value[trial_index, dimension])
                rows.append(common | {"branch": branch, "target": target,
                                      "true": true, "prediction": pred})


def add_residual_rows(rows: list[dict], dataset: str, method: str,
                      test_subject: str, seed_index: int, records: list[dict],
                      prediction: dict[str, np.ndarray],
                      label_names: list[str]) -> None:
    included = [record for record in records if record["included"]]
    if not prediction or len(next(iter(prediction.values()))) != len(included):
        raise RuntimeError("residual prediction/record count mismatch")
    for local, record in enumerate(included):
        for dimension, target in enumerate(label_names):
            mean = float(record["reference_mean"][dimension])
            raw_true = float(record["raw_y"][dimension])
            true_residual = float(record["residual"][dimension])
            common = {
                "dataset": dataset, "experiment": "loso_transductive",
                "target_mode": "stimulus_residual", "method": method,
                "test_subject": test_subject, "seed_index": seed_index,
                "trial_index": int(record["trial_index"]),
                "stimulus_key": record["stimulus_key"],
                "reference_mean": mean,
                "reference_n": int(record["reference_n"][dimension]),
                "target": target,
            }
            for branch, value in prediction.items():
                pred_residual = float(value[local, dimension])
                rows.append(common | {
                    "scale": "residual", "branch": branch,
                    "true": true_residual, "prediction": pred_residual,
                })
                rows.append(common | {
                    "scale": "rating_reconstructed", "branch": branch,
                    "true": raw_true, "prediction": mean + pred_residual,
                })


def checkpoint_provenance(cli, test_subject: str, seed_index: int, spec: dict) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION, "dataset": cli.dataset,
        "method": cli.method, "target_mode": cli.target_mode,
        "feature_set": cli.feature_set, "run_tag": cli.run_tag,
        "test_subject": test_subject, "seed_index": seed_index,
        "split": spec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ds000210", "ds002835"), required=True)
    parser.add_argument("--method", choices=("poe", "neurostorm"), required=True)
    parser.add_argument("--target-mode", choices=("direct", "stimulus_residual"), required=True)
    parser.add_argument("--n-seeds", type=int, required=True)
    parser.add_argument(
        "--seed-start", type=int, default=0,
        help="First seed index to evaluate; --n-seeds remains the exclusive upper bound.",
    )
    parser.add_argument("--split-n-seeds", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--expected-subjects", type=int, required=True)
    parser.add_argument("--subject-shard-index", type=int)
    parser.add_argument("--subject-shard-count", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument(
        "--result-root", type=Path,
        help="Optional isolated result directory; checkpoint locations are unchanged.",
    )
    parser.add_argument(
        "--checkpoint-root", type=Path,
        help="Optional explicit checkpoint directory, useful for a seed-extension run.",
    )
    parser.add_argument("--overwrite", action="store_true")
    cli = parser.parse_args()
    if not (0 <= cli.seed_start < cli.n_seeds <= cli.split_n_seeds):
        parser.error("seed range must satisfy 0 <= seed-start < n-seeds <= split-n-seeds")
    if (cli.subject_shard_index is None) != (cli.subject_shard_count is None):
        parser.error("subject shard index/count must be used together")
    if cli.subject_shard_count is not None and not (
        cli.subject_shard_count > 0 and
        0 <= cli.subject_shard_index < cli.subject_shard_count
    ):
        parser.error("invalid subject shard index/count")

    config = load_config()
    protocol = config["decoder_protocol"]
    args = argparse.Namespace(
        epochs=int(cli.max_epochs or protocol["epochs"]),
        patience=int(cli.patience or protocol["early_stopping_patience"]),
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
        batch_size=int(protocol["batch_size"]), hidden=int(protocol["hidden_dim"]),
        latent_dim=int(protocol["latent_dim"]), dropout=float(protocol["dropout"]),
        aux_weight=float(protocol["aux_weight"]), kl_weight=float(protocol["kl_weight"]),
        grad_clip=float(protocol["gradient_clip"]), min_delta=1e-4,
    )
    torch.set_num_threads(max(1, cli.num_threads))
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    subjects, feature_provenance = read_subjects(
        config, cli.dataset, cli.method, cli.feature_set, cli.expected_subjects,
    )
    names = sorted(subjects)
    label_names = list(config["datasets"][cli.dataset]["label_names"])
    ordinal_model = (cli.dataset == "ds000210" and cli.target_mode == "direct")
    regression = not ordinal_model
    d_out = 1 if cli.dataset == "ds000210" else 2
    paths = experiment_paths("decoding")
    split_path = (
        paths["splits"] / cli.run_tag / cli.dataset
        / f"loso_splits_{cli.split_n_seeds}.json"
    )
    split = make_or_load_split(split_path, names, cli.split_n_seeds)
    if cli.prepare_only:
        print(f"[prepared] {split_path}", flush=True)
        return

    test_names = names
    shard_label = None
    if cli.subject_shard_count is not None:
        test_names = [name for position, name in enumerate(names)
                      if position % cli.subject_shard_count == cli.subject_shard_index]
        shard_label = f"shard-{cli.subject_shard_index:02d}-of-{cli.subject_shard_count:02d}"
    if not test_names:
        raise RuntimeError(f"empty test-subject shard: {shard_label}")

    result_root = (
        cli.result_root.resolve()
        if cli.result_root is not None
        else (
            paths["results"] / cli.run_tag / "loso_transductive"
            / cli.target_mode / cli.dataset / cli.method
        )
    )
    if shard_label:
        result_root = result_root / "shards" / shard_label
    checkpoint_root = (
        cli.checkpoint_root.resolve()
        if cli.checkpoint_root is not None
        else (
            paths["checkpoints"] / "decoding" / cli.run_tag
            / "loso_transductive" / cli.target_mode / cli.dataset / cli.method
        )
    )
    result_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    prediction_rows, fold_rows, mean_rows, coverage_rows = [], [], [], []

    for outer_position, test_name in enumerate(test_names):
        test_subject = subjects[test_name].model
        for seed_index in range(cli.seed_start, cli.n_seeds):
            spec = split["folds"][test_name][str(seed_index)]
            provenance = checkpoint_provenance(cli, test_name, seed_index, spec)
            selection_path = checkpoint_root / f"{test_name}_seed{seed_index:02d}_selection.pt"
            re_path = checkpoint_root / f"{test_name}_seed{seed_index:02d}_re.pt"
            training_seed = (int(protocol["split_seed"]) +
                             names.index(test_name) * 1_000_003 + seed_index * 100_003 +
                             (50_000_017 if cli.target_mode == "stimulus_residual" else 0))

            selection_reference = re_reference = None
            if cli.target_mode == "stimulus_residual":
                selection_reference = compute_reference(subjects, spec["train_subjects"], d_out)
                re_reference = compute_reference(subjects, spec["re_train_subjects"], d_out)
                mean_rows.extend(reference_rows(
                    cli.dataset, test_name, seed_index, "selection",
                    selection_reference, label_names,
                ))
                if seed_index == 0:
                    mean_rows.extend(reference_rows(
                        cli.dataset, test_name, "", "re",
                        re_reference, label_names,
                    ))

            payload = None
            if re_path.is_file() and not cli.overwrite:
                try:
                    payload = torch.load(re_path, map_location="cpu", weights_only=False)
                except Exception as error:
                    print(f"[warning] unreadable re checkpoint will be rebuilt: {re_path}: {error}",
                          flush=True)
            if payload is not None:
                if payload.get("provenance") != provenance:
                    raise RuntimeError(f"checkpoint provenance mismatch: {re_path}")
                if cli.target_mode == "stimulus_residual":
                    saved = Reference.from_state(payload["outer_reference"])
                    if not reference_equal(saved, re_reference):
                        raise RuntimeError(f"saved/current residual reference mismatch: {re_path}")
                xo = base.state_to_standardizer(payload["x_scaler"])
                xi = base.state_to_standardizer(payload.get("indiv_scaler"))
                ys = base.state_to_standardizer(payload.get("y_scaler"))
                model = base.build_model(
                    cli.method, int(payload["input_dim"]), bool(payload["ordinal_model"]),
                    int(payload["output_dim"]), args, device,
                )
                model.load_state_dict(payload["model_state"], strict=True)
                selected_epoch = int(payload["selected_epoch"])
                selected_val = float(payload["selected_val_loss"])
                reused = selection_reused = True
            else:
                selection_payload = None
                if selection_path.is_file() and not cli.overwrite:
                    try:
                        selection_payload = torch.load(
                            selection_path, map_location="cpu", weights_only=False,
                        )
                    except Exception as error:
                        print(
                            f"[warning] unreadable selection checkpoint will be rebuilt: "
                            f"{selection_path}: {error}", flush=True,
                        )
                if selection_payload is not None:
                    if selection_payload.get("provenance") != provenance:
                        raise RuntimeError(f"checkpoint provenance mismatch: {selection_path}")
                    selected_epoch = int(selection_payload["selected_epoch"])
                    selected_val = float(selection_payload["selected_val_loss"])
                    selection_reused = True
                else:
                    set_seed(training_seed)
                    if cli.target_mode == "direct":
                        train_map = {
                            name: np.arange(len(subjects[name].model.y), dtype=np.int64)
                            for name in spec["train_subjects"]
                        }
                        val_name = spec["val_subject"]
                        val_map = {val_name: np.arange(len(subjects[val_name].model.y), dtype=np.int64)}
                        train_models = [subjects[name].model for name in spec["train_subjects"]]
                        val_models = [subjects[val_name].model]
                        x_train, y_train, xo_sel, xi_sel, ys_sel = base.train_arrays(
                            train_models, train_map, cli.method, regression,
                        )
                        x_val, y_val = base.transform_arrays(
                            val_models, val_map, cli.method, xo_sel, xi_sel, ys_sel, regression,
                        )
                    else:
                        x_train_raw, y_train_raw, _ = collect_residual(
                            subjects, spec["train_subjects"], selection_reference, cli.method,
                        )
                        x_val_raw, y_val_raw, _ = collect_residual(
                            subjects, [spec["val_subject"]], selection_reference, cli.method,
                        )
                        x_train, y_train, xo_sel, xi_sel, ys_sel = fit_raw(
                            x_train_raw, y_train_raw, cli.method,
                        )
                        x_val, y_val = transform_raw(
                            x_val_raw, y_val_raw, cli.method, xo_sel, xi_sel, ys_sel,
                        )
                    model = base.build_model(cli.method, x_train[0].shape[1],
                                             ordinal_model, d_out, args, device)
                    selected_epoch, selected_val, selected_state = base.choose_epoch(
                        model, base.tensors(x_train, y_train, device),
                        base.tensors(x_val, y_val, device), cli.method,
                        ordinal_model, args, training_seed,
                    )
                    safe_torch_save({
                        "format": "loso_decoder_selection_v2", "provenance": provenance,
                        "selected_epoch": selected_epoch, "selected_val_loss": selected_val,
                        "input_dim": int(x_train[0].shape[1]), "output_dim": d_out,
                        "ordinal_model": ordinal_model,
                        "inner_reference": (None if selection_reference is None
                                            else selection_reference.serializable()),
                        "x_scaler": xo_sel.state(),
                        "indiv_scaler": None if xi_sel is None else xi_sel.state(),
                        "y_scaler": None if ys_sel is None else ys_sel.state(),
                        "model_state": selected_state, "protocol": vars(args),
                        "training_seed": training_seed,
                    }, selection_path)
                    selection_reused = False

                set_seed(training_seed)
                if cli.target_mode == "direct":
                    re_map = {
                        name: np.arange(len(subjects[name].model.y), dtype=np.int64)
                        for name in spec["re_train_subjects"]
                    }
                    re_models = [subjects[name].model for name in spec["re_train_subjects"]]
                    x_re, y_re, xo, xi, ys = base.train_arrays(
                        re_models, re_map, cli.method, regression,
                    )
                else:
                    x_re_raw, y_re_raw, _ = collect_residual(
                        subjects, spec["re_train_subjects"], re_reference, cli.method,
                    )
                    x_re, y_re, xo, xi, ys = fit_raw(
                        x_re_raw, y_re_raw, cli.method,
                    )
                model = base.build_model(cli.method, x_re[0].shape[1],
                                         ordinal_model, d_out, args, device)
                base.re(model, base.tensors(x_re, y_re, device), cli.method,
                           ordinal_model, args, training_seed, selected_epoch)
                payload = {
                    "format": "loso_decoder_re_v2", "provenance": provenance,
                    "selected_epoch": selected_epoch, "selected_val_loss": selected_val,
                    "input_dim": int(x_re[0].shape[1]), "output_dim": d_out,
                    "ordinal_model": ordinal_model,
                    "outer_reference": (None if re_reference is None
                                        else re_reference.serializable()),
                    "x_scaler": xo.state(),
                    "indiv_scaler": None if xi is None else xi.state(),
                    "y_scaler": None if ys is None else ys.state(),
                    "model_state": {key: value.detach().cpu()
                                    for key, value in model.state_dict().items()},
                    "selection_checkpoint": str(selection_path),
                    "protocol": vars(args), "training_seed": training_seed,
                    "mae_features_frozen": True,
                }
                safe_torch_save(payload, re_path)
                reused = False

            if cli.target_mode == "direct":
                prediction = base.predict(
                    model, test_subject, np.arange(len(test_subject.y), dtype=np.int64),
                    cli.method, xo, xi, ys, device,
                )
                add_direct_rows(prediction_rows, cli.dataset, cli.method,
                                test_subject, seed_index, prediction, label_names)
                n_test, n_zero = len(test_subject.y), 0
            else:
                x_test, _, test_records = collect_residual(
                    subjects, [test_name], re_reference, cli.method,
                )
                prediction = predict_raw(model, x_test, cli.method, xo, xi, ys, device)
                add_residual_rows(prediction_rows, cli.dataset, cli.method,
                                  test_name, seed_index, test_records,
                                  prediction, label_names)
                excluded = [record for record in test_records if not record["included"]]
                n_test, n_zero = len(test_records) - len(excluded), len(excluded)
                if seed_index == 0:
                    coverage_rows.extend({
                        "dataset": cli.dataset, "test_subject": test_name,
                        "trial_index": int(record["trial_index"]),
                        "stimulus_key": record["stimulus_key"],
                        "reason": record["reason"],
                    } for record in excluded)
            fold_rows.append({
                "dataset": cli.dataset, "experiment": "loso_transductive",
                "target_mode": cli.target_mode, "method": cli.method,
                "test_subject": test_name, "seed_index": seed_index,
                "train_subjects": ";".join(spec["train_subjects"]),
                "val_subject": spec["val_subject"],
                "re_train_subjects": ";".join(spec["re_train_subjects"]),
                "n_test_trials_included": n_test, "n_test_trials_zero_support": n_zero,
                "selected_epoch": selected_epoch, "selected_val_loss": selected_val,
                "reused_checkpoint": reused, "selection_reused": selection_reused,
                "selection_checkpoint": str(selection_path),
                "checkpoint": str(re_path),
            })
            print(
                f"[{cli.dataset}/{cli.method}/{cli.target_mode}] {test_name} "
                f"seed={seed_index:02d} epoch={selected_epoch} val={selected_val:.6f} "
                f"{'reused' if reused else 'trained'}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    per_seed, ensemble, ensemble_metrics, seed_stats = within.summarize(
        prediction_rows, cli.dataset, cli.method, cli.target_mode,
    )
    for collection in (per_seed, ensemble, ensemble_metrics, seed_stats):
        for row in collection:
            row["experiment"] = "loso_transductive"
    base.write_csv(result_root / "predictions_per_seed.csv", prediction_rows)
    base.write_csv(result_root / "predictions_ensemble.csv", ensemble)
    base.write_csv(result_root / "metrics_per_subject_per_seed.csv", per_seed)
    base.write_csv(result_root / "metrics_per_subject_ensemble.csv", ensemble_metrics)
    base.write_csv(result_root / "metrics_per_subject_seed_mean_sd.csv", seed_stats)
    base.write_csv(result_root / "folds.csv", fold_rows)
    if cli.target_mode == "stimulus_residual":
        base.write_csv(result_root / "reference_means.csv", mean_rows)
        if coverage_rows:
            base.write_csv(result_root / "zero_support_trials.csv", coverage_rows)
        else:
            (result_root / "zero_support_trials.csv").write_text(
                "dataset,test_subject,trial_index,stimulus_key,reason\n", encoding="utf-8"
            )
    write_json(result_root / "run_manifest.json", {
        "format": "full55_loso_decoder_v2", "protocol_version": PROTOCOL_VERSION,
        "dataset": cli.dataset, "method": cli.method, "target_mode": cli.target_mode,
        "feature_set": cli.feature_set, "run_tag": cli.run_tag,
        "n_test_subjects": len(test_names), "n_total_subjects": len(names),
        "test_subjects": test_names,
        "seed_start": cli.seed_start,
        "seed_end_exclusive": cli.n_seeds,
        "n_seeds": cli.n_seeds - cli.seed_start,
        "subject_shard": shard_label, "split_file": str(split_path),
        "split_sha256": sha256(split_path), "checkpoint_root": str(checkpoint_root),
        "feature_provenance": feature_provenance,
        "mae_features_frozen": True, "mae_pretraining_performed": False,
        "residual_contract": (
            "stimulus-configuration mean from training subjects only; no test-label "
            "centering; unseen test configurations excluded without fallback"
            if cli.target_mode == "stimulus_residual" else None
        ),
        "requested_primary_aggregation": "mean and sample SD across all evaluated seeds",
        "protocol": vars(args),
    })


if __name__ == "__main__":
    main()
