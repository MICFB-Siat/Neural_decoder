

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit

from common import experiment_paths, load_config, set_seed, sha256, write_json
from modeling import (GaussianPoE, GaussianSingleExpert, Standardizer, fisher_mean,
                      ordinal_expected, poe_loss, regression_metrics, single_loss)


@dataclass
class Subject:
    dataset: str
    subject: str
    obs: np.ndarray | None
    indiv: np.ndarray | None
    x: np.ndarray | None
    y: np.ndarray
    ordinal: bool
    objective: np.ndarray
    trial_index: np.ndarray
    feature_path: str


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def read_subjects(config: dict, dataset: str, method: str, feature_set: str, expected_subjects: int | None) -> list[Subject]:
    root = experiment_paths("decoding")["features"] / feature_set / ("poe" if method == "poe" else "neurostorm") / dataset
    if not root.is_dir():
        raise FileNotFoundError(f"未找到 {method}/{dataset} 特征: {root}")
    ordinal = config["datasets"][dataset]["subjective_target"] == "clarity_ordinal"
    out = []
    for path in sorted(root.glob("sub-*.h5")):
        with h5py.File(path, "r") as handle:
            labels = handle["labels_subjective"][:].astype(np.float32)
            y = (4.0 - labels[:, 0]).astype(np.float32) if ordinal else labels[:, :2].astype(np.float32)
            objective = handle["labels_objective"][:].astype(np.int64)
            if method == "poe":
                obs = handle["bci_obs"][:].astype(np.float32).mean(axis=1)
                indiv = handle["bci_prior_indiv"][:].astype(np.float32).mean(axis=1)
                if obs.shape != indiv.shape:
                    raise ValueError(f"{path}: obs/indiv 不匹配")
                x = None
            else:
                obs = indiv = None
                x = handle["neurostorm_feature"][:].astype(np.float32)
            subject = str(handle.attrs.get("subject_id", path.stem))
        out.append(Subject(dataset, subject, obs, indiv, x, y, ordinal, objective,
                           np.arange(len(y), dtype=np.int64), str(path)))
    if len(out) < 3:
        raise RuntimeError(f"{dataset}: 被试数不足 ({len(out)})")
    if expected_subjects is not None and len(out) != expected_subjects:
        raise RuntimeError(f"{dataset}: 期望 {expected_subjects} 名被试，实际发现 {len(out)}: {[x.subject for x in out]}")
    return out


def stratified_split(indices: np.ndarray, strata: np.ndarray, test_fraction: float, seed: int):

    _, counts = np.unique(strata, return_counts=True)
    if counts.min() >= 2:
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        train, test = next(split.split(indices, strata))
        return np.sort(indices[train]), np.sort(indices[test])
    rng = np.random.default_rng(seed); shuffled = rng.permutation(indices)
    n_test = max(1, round(test_fraction * len(indices)))
    return np.sort(shuffled[n_test:]), np.sort(shuffled[:n_test])


def make_splits(subjects: list[Subject], experiment: str, seed: int, n_seeds: int) -> dict:
    if experiment == "loso":
        result = {"kind": "nested_loso_transductive", "subjects": {}}
        names = [s.subject for s in subjects]
        for outer, name in enumerate(names):
            outer_train = [x for x in names if x != name]
            result["subjects"][name] = {
                str(k): {"train_subjects": [x for x in outer_train if x != outer_train[(outer + k) % len(outer_train)]],
                         "val_subject": outer_train[(outer + k) % len(outer_train)],
                         "re_train_subjects": outer_train, "test_subject": name}
                for k in range(n_seeds)
            }
        return result
    result = {"kind": "within_subject_fixed_stratified_80_20", "seed": seed, "subjects": {}}
    for position, subject in enumerate(subjects):
        strata = subject.y.astype(int) if subject.ordinal else subject.objective
        outer_train, test = stratified_split(subject.trial_index, strata, 0.2, seed + position * 1009)
        seed_splits = {}
        for k in range(n_seeds):
            inner_strata = subject.y[outer_train].astype(int) if subject.ordinal else subject.objective[outer_train]
            inner_train, val = stratified_split(outer_train, inner_strata, 0.2, seed + position * 5003 + k)
            seed_splits[str(k)] = {"train": inner_train.tolist(), "val": val.tolist(),
                                   "re_train": outer_train.tolist(), "test": test.tolist()}
        result["subjects"][subject.subject] = seed_splits
    return result


def positions(subject: Subject, trial_ids: list[int]) -> np.ndarray:
    return np.asarray(trial_ids, dtype=np.int64)


def train_arrays(subjects: list[Subject], selection: dict[str, np.ndarray], method: str, regression: bool):
    pairs = [(subject, selection[subject.subject]) for subject in subjects]
    if method == "poe":
        xo = Standardizer.fit(np.concatenate([subject.obs[index] for subject, index in pairs]))
        xi = Standardizer.fit(np.concatenate([subject.indiv[index] for subject, index in pairs]))
        x = (np.concatenate([xo.transform(subject.obs[index]) for subject, index in pairs]),
             np.concatenate([xi.transform(subject.indiv[index]) for subject, index in pairs]))
    else:
        xo = Standardizer.fit(np.concatenate([subject.x[index] for subject, index in pairs])); xi = None
        x = (np.concatenate([xo.transform(subject.x[index]) for subject, index in pairs]),)
    y_raw = np.concatenate([subject.y[index] for subject, index in pairs])
    ys = Standardizer.fit(y_raw) if regression else None
    return x, (ys.transform(y_raw) if regression else y_raw.astype(np.float32)), xo, xi, ys


def transform_arrays(subjects: list[Subject], selection: dict[str, np.ndarray], method: str, xo, xi, ys, regression: bool):
    pairs = [(subject, selection[subject.subject]) for subject in subjects]
    if method == "poe":
        x = (np.concatenate([xo.transform(subject.obs[index]) for subject, index in pairs]),
             np.concatenate([xi.transform(subject.indiv[index]) for subject, index in pairs]))
    else:
        x = (np.concatenate([xo.transform(subject.x[index]) for subject, index in pairs]),)
    y = np.concatenate([subject.y[index] for subject, index in pairs])
    return x, (ys.transform(y) if regression else y.astype(np.float32))


def build_model(method: str, d_in: int, ordinal: bool, d_out: int, args, device):
    factory = (lambda d: nn.Linear(d, 2)) if ordinal else (lambda d: nn.Linear(d, d_out))
    if method == "poe":
        return GaussianPoE(d_in, factory, args.latent_dim, args.hidden, args.dropout).to(device)
    return GaussianSingleExpert(d_in, factory, args.latent_dim, args.hidden, args.dropout).to(device)


def tensors(x, y, device):
    return tuple(torch.as_tensor(value, dtype=torch.float32, device=device) for value in x) + (torch.as_tensor(y, dtype=torch.float32, device=device),)


def loss_for(model, arrays, method: str, ordinal: bool, args):
    if method == "poe":
        output = model(arrays[0], arrays[1])
        return poe_loss(output, arrays[-1], "ordinal" if ordinal else "regression", args.aux_weight, args.kl_weight)
    output = model(arrays[0])
    return single_loss(output, arrays[-1], "ordinal" if ordinal else "regression", args.kl_weight)


def choose_epoch(model, train, val, method: str, ordinal: bool, args, seed: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed); best, best_epoch, bad, best_state = math.inf, 1, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train(); order = rng.permutation(len(train[-1]))
        for start in range(0, len(order), args.batch_size):
            index = torch.as_tensor(order[start:start + args.batch_size], dtype=torch.long, device=train[-1].device)
            batch = tuple(value[index] for value in train)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_for(model, batch, method, ordinal, args)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); optimizer.step()
        model.eval()
        with torch.no_grad(): _, val_loss = loss_for(model, val, method, ordinal, args)
        value = float(val_loss)
        if value < best - args.min_delta:
            best, best_epoch, bad = value, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience: break
    return best_epoch, best, best_state


def re(model, train, method: str, ordinal: bool, args, seed: int, epochs: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed + 7919)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(train[-1]))
        for start in range(0, len(order), args.batch_size):
            index = torch.as_tensor(order[start:start + args.batch_size], dtype=torch.long, device=train[-1].device)
            batch = tuple(value[index] for value in train)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_for(model, batch, method, ordinal, args)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); optimizer.step()


def predict(model, subject: Subject, index: np.ndarray, method: str, xo, xi, ys, device):
    model.eval()
    with torch.no_grad():
        if method == "poe":
            obs = torch.as_tensor(xo.transform(subject.obs[index]), dtype=torch.float32, device=device)
            indiv = torch.as_tensor(xi.transform(subject.indiv[index]), dtype=torch.float32, device=device)
            output = model(obs, indiv)
            pred = {"obs": output["obs"].cpu().numpy(), "indiv": output["indiv"].cpu().numpy(), "poe": output["poe"].cpu().numpy()}
        else:
            x = torch.as_tensor(xo.transform(subject.x[index]), dtype=torch.float32, device=device)
            pred = {"neurostorm": model(x)["pred"].cpu().numpy()}
    if subject.ordinal:
        return {name: ordinal_expected(torch.as_tensor(value)).numpy() for name, value in pred.items()}
    return {name: ys.inverse(value) for name, value in pred.items()}


def get_fold(subjects: list[Subject], split: dict, experiment: str, test_subject: str, seed_index: int):
    lookup = {subject.subject: subject for subject in subjects}
    spec = split["subjects"][test_subject][str(seed_index)]
    if experiment == "within":
        subject = lookup[test_subject]
        train = {test_subject: positions(subject, spec["train"])}
        val = {test_subject: positions(subject, spec["val"])}
        re = {test_subject: positions(subject, spec["re_train"])}
        return train, val, re, positions(subject, spec["test"]), spec
    train = {name: np.arange(len(lookup[name].y)) for name in spec["train_subjects"]}
    val = {spec["val_subject"]: np.arange(len(lookup[spec["val_subject"]].y))}
    re = {name: np.arange(len(lookup[name].y)) for name in spec["re_train_subjects"]}
    return train, val, re, np.arange(len(lookup[test_subject].y)), spec


def add_predictions(rows, dataset: str, experiment: str, method: str, subject: Subject, index, seed_index, prediction, label_names):
    for local, position in enumerate(index):
        base = {"dataset": dataset, "experiment": experiment, "method": method,
                "test_subject": subject.subject, "seed_index": seed_index, "trial_index": int(position)}
        if subject.ordinal:
            for branch, value in prediction.items():
                rows.append(base | {"branch": branch, "target": "clarity", "true": float(subject.y[position]), "prediction": float(value[local])})
        else:
            for dimension, target in enumerate(label_names):
                for branch, value in prediction.items():
                    rows.append(base | {"branch": branch, "target": target,
                                        "true": float(subject.y[position, dimension]), "prediction": float(value[local, dimension])})


def summarize(rows: list[dict], dataset: str, experiment: str, method: str):
    per_seed_metrics = []
    for seed in sorted({int(row["seed_index"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["seed_index"]) == seed]
        for subject in sorted({row["test_subject"] for row in seed_rows}):
            for branch in sorted({row["branch"] for row in seed_rows}):
                for target in sorted({row["target"] for row in seed_rows}):
                    values = [
                        row for row in seed_rows
                        if row["test_subject"] == subject and row["branch"] == branch and row["target"] == target
                    ]
                    if values:
                        per_seed_metrics.append({
                            "dataset": dataset, "experiment": experiment, "method": method,
                            "subject": subject, "branch": branch, "target": target,
                            "seed_index": seed, "aggregation": "single_seed",
                            **regression_metrics(
                                [value["true"] for value in values],
                                [value["prediction"] for value in values],
                            ),
                        })

    grouped = {}
    for row in rows:
        grouped.setdefault((row["test_subject"], row["trial_index"], row["branch"], row["target"]), []).append(row)
    ensemble = []
    for key, values in grouped.items():
        first = values[0]
        ensemble.append({"dataset": dataset, "experiment": experiment, "method": method,
                         "test_subject": key[0], "trial_index": key[1], "branch": key[2], "target": key[3],
                         "true": first["true"], "prediction": float(np.mean([x["prediction"] for x in values])),
                         "n_seeds": len(values)})
    metrics = []
    for subject in sorted({x["test_subject"] for x in ensemble}):
        for branch in sorted({x["branch"] for x in ensemble}):
            for target in sorted({x["target"] for x in ensemble}):
                values = [x for x in ensemble if x["test_subject"] == subject and x["branch"] == branch and x["target"] == target]
                metrics.append({"dataset": dataset, "experiment": experiment, "method": method, "subject": subject,
                                "branch": branch, "target": target, "aggregation": "trialwise_mean_prediction_across_seeds",
                                **regression_metrics([x["true"] for x in values], [x["prediction"] for x in values])})
    summary = []
    for branch in sorted({x["branch"] for x in metrics}):
        for target in sorted({x["target"] for x in metrics}):
            values = [x for x in metrics if x["branch"] == branch and x["target"] == target]
            summary.append({"dataset": dataset, "experiment": experiment, "method": method,
                            "branch": branch, "target": target, "n_subjects": len(values),
                            "pearson_fisher_mean": fisher_mean([x["pearson_r"] for x in values]),
                            "spearman_mean": float(np.nanmean([x["spearman_rho"] for x in values])),
                            "mae_mean": float(np.nanmean([x["mae"] for x in values])),
                            "rmse_mean": float(np.nanmean([x["rmse"] for x in values])),
                            "r2_mean": float(np.nanmean([x["r2"] for x in values]))})
    best_seed_metrics = []
    best_groups = {}
    for row in per_seed_metrics:
        best_groups.setdefault((row["subject"], row["branch"], row["target"]), []).append(row)

    def seed_score(row):
        pearson = float(row["pearson_r"])
        spearman = float(row["spearman_rho"])
        rmse = float(row["rmse"])
        return (
            pearson if math.isfinite(pearson) else -math.inf,
            spearman if math.isfinite(spearman) else -math.inf,
            -rmse if math.isfinite(rmse) else -math.inf,
        )

    for _, values in sorted(best_groups.items()):
        best = max(values, key=seed_score).copy()
        best["aggregation"] = "best_single_seed_selected_on_test_exploratory"
        best["selection_policy"] = "max pearson_r; tie max spearman_rho; tie min rmse"
        best["selection_scope"] = "independent per subject x branch x target x method"
        pass
        best_seed_metrics.append(best)
    return ensemble, per_seed_metrics, metrics, summary, best_seed_metrics


def state_to_standardizer(state: dict | None) -> Standardizer | None:
    if state is None:
        return None
    return Standardizer(np.asarray(state["mean"], dtype=np.float32), np.asarray(state["std"], dtype=np.float32))


def atomic_torch_save(payload: dict, path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ds000210", "ds002835"), required=True)
    parser.add_argument("--method", choices=("poe", "neurostorm"), required=True)
    parser.add_argument("--experiment", choices=("within", "loso"), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--overwrite", action="store_true", help="重新训练已有的 seed checkpoint")
    parser.add_argument("--feature-set", required=True, help="冻结特征的不可变命名空间。")
    parser.add_argument("--run-tag", required=True, help="结果、划分和解码器检查点的不可变命名空间。")
    parser.add_argument("--expected-subjects", type=int, default=None, help="若指定，拒绝在被试数不匹配时运行。")
    cli = parser.parse_args()

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
    subjects = read_subjects(config, cli.dataset, cli.method, cli.feature_set, cli.expected_subjects)
    lookup = {subject.subject: subject for subject in subjects}
    ordinal = subjects[0].ordinal
    if any(subject.ordinal != ordinal for subject in subjects):
        raise RuntimeError("同一数据集的标签头类型不一致")
    d_out = 1 if ordinal else len(config["datasets"][cli.dataset]["label_names"])
    n_seeds = int(protocol["seeds"])
    paths = experiment_paths("decoding")
    split_path = paths["splits"] / cli.run_tag / cli.dataset / f"{cli.experiment}_splits.json"
    if split_path.exists():
        with split_path.open(encoding="utf-8") as handle:
            split = json.load(handle)
    else:
        split = make_splits(subjects, cli.experiment, int(protocol["split_seed"]), n_seeds)
        write_json(split_path, split)

    run_name = f"{cli.experiment}_{'transductive' if cli.experiment == 'loso' else 'subject_internal'}"
    result_root = paths["results"] / cli.run_tag / run_name / cli.dataset / cli.method
    checkpoint_root = paths["checkpoints"] / "decoding" / cli.run_tag / run_name / cli.dataset / cli.method
    result_root.mkdir(parents=True, exist_ok=True); checkpoint_root.mkdir(parents=True, exist_ok=True)
    label_names = config["datasets"][cli.dataset]["label_names"]
    seed_rows: list[dict] = []
    fold_rows: list[dict] = []

    for test_subject in sorted(lookup):
        test = lookup[test_subject]
        for seed_index in range(n_seeds):
            checkpoint_path = checkpoint_root / f"{test_subject}_seed{seed_index:02d}_re.pt"
            if checkpoint_path.exists() and not cli.overwrite:
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if (payload["dataset"], payload["method"], payload["experiment"], payload["test_subject"], payload["seed_index"], payload.get("feature_set"), payload.get("run_tag")) != (cli.dataset, cli.method, cli.experiment, test_subject, seed_index, cli.feature_set, cli.run_tag):
                    raise RuntimeError(f"checkpoint provenance does not match: {checkpoint_path}")
                xo = state_to_standardizer(payload["x_scaler"])
                xi = state_to_standardizer(payload.get("indiv_scaler"))
                ys = state_to_standardizer(payload.get("y_scaler"))
                model = build_model(cli.method, int(payload["input_dim"]), ordinal, d_out, args, device)
                model.load_state_dict(payload["model_state"], strict=True)
                test_index = np.asarray(payload["test_trial_index"], dtype=np.int64)
                best_epoch, best_value = int(payload["selected_epoch"]), float(payload["selected_val_loss"])
                reused = True
            else:
                train_map, val_map, re_map, test_index, split_spec = get_fold(subjects, split, cli.experiment, test_subject, seed_index)
                train_subjects = [lookup[name] for name in train_map]
                val_subjects = [lookup[name] for name in val_map]
                re_subjects = [lookup[name] for name in re_map]
                regression = not ordinal
                seed = int(protocol["split_seed"]) + seed_index * 100003 + sum(map(ord, test_subject))
                set_seed(seed)
                x_train, y_train, xo, xi, ys = train_arrays(train_subjects, train_map, cli.method, regression)
                x_val, y_val = transform_arrays(val_subjects, val_map, cli.method, xo, xi, ys, regression)
                train_tensor = tensors(x_train, y_train, device); val_tensor = tensors(x_val, y_val, device)
                model = build_model(cli.method, x_train[0].shape[1], ordinal, d_out, args, device)
                best_epoch, best_value, selected_state = choose_epoch(model, train_tensor, val_tensor, cli.method, ordinal, args, seed)
                selection_path = checkpoint_root / f"{test_subject}_seed{seed_index:02d}_selection.pt"
                atomic_torch_save({
                    "format": "subjective_decoder_selection_v1", "dataset": cli.dataset, "method": cli.method,
                    "experiment": cli.experiment, "feature_set": cli.feature_set, "run_tag": cli.run_tag,
                    "test_subject": test_subject, "seed_index": seed_index,
                    "selected_epoch": best_epoch, "selected_val_loss": best_value, "input_dim": int(x_train[0].shape[1]),
                    "split": split_spec, "x_scaler": xo.state(), "indiv_scaler": None if xi is None else xi.state(),
                    "y_scaler": None if ys is None else ys.state(), "model_state": selected_state,
                    "protocol": vars(args), "selection_training_seed": seed,
                }, selection_path)


                set_seed(seed)
                x_re, y_re, xo, xi, ys = train_arrays(re_subjects, re_map, cli.method, regression)
                re_tensor = tensors(x_re, y_re, device)
                model = build_model(cli.method, x_re[0].shape[1], ordinal, d_out, args, device)
                re(model, re_tensor, cli.method, ordinal, args, seed, best_epoch)
                payload = {
                    "format": "subjective_decoder_re_v1", "dataset": cli.dataset, "method": cli.method,
                    "experiment": cli.experiment, "feature_set": cli.feature_set, "run_tag": cli.run_tag,
                    "test_subject": test_subject, "seed_index": seed_index,
                    "transductive": cli.experiment == "loso", "selected_epoch": best_epoch,
                    "selected_val_loss": best_value, "input_dim": int(x_re[0].shape[1]), "split": split_spec,
                    "test_trial_index": test_index.tolist(), "x_scaler": xo.state(),
                    "indiv_scaler": None if xi is None else xi.state(), "y_scaler": None if ys is None else ys.state(),
                    "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "protocol": vars(args), "selection_checkpoint": str(selection_path), "re_training_seed": seed,
                }
                atomic_torch_save(payload, checkpoint_path)
                reused = False
            prediction = predict(model, test, test_index, cli.method, xo, xi, ys, device)
            add_predictions(seed_rows, cli.dataset, cli.experiment, cli.method, test, test_index, seed_index, prediction, label_names)
            fold_rows.append({"dataset": cli.dataset, "experiment": cli.experiment, "method": cli.method,
                              "test_subject": test_subject, "seed_index": seed_index, "reused_checkpoint": reused,
                              "selected_epoch": best_epoch, "selected_val_loss": best_value,
                              "n_test_trials": len(test_index), "checkpoint": str(checkpoint_path)})
            print(f"[{cli.dataset}/{cli.method}/{cli.experiment}] {test_subject} seed={seed_index:02d} "
                  f"epoch={best_epoch} val={best_value:.6f} {'reused' if reused else 'trained'}", flush=True)

    ensemble, per_seed_metrics, metrics, summary, best_seed_metrics = summarize(
        seed_rows, cli.dataset, cli.experiment, cli.method,
    )
    write_csv(result_root / "predictions_per_seed.csv", seed_rows)
    write_csv(result_root / "predictions_ensemble.csv", ensemble)
    write_csv(result_root / "metrics_per_subject_per_seed.csv", per_seed_metrics)
    write_csv(result_root / "metrics_best_seed_test_selected_exploratory.csv", best_seed_metrics)
    write_csv(result_root / "metrics_per_subject.csv", metrics)
    write_csv(result_root / "metrics_summary.csv", summary)
    write_csv(result_root / "folds.csv", fold_rows)
    write_json(result_root / "run_manifest.json", {
        "dataset": cli.dataset, "method": cli.method, "experiment": cli.experiment,
        "feature_set": cli.feature_set, "run_tag": cli.run_tag,
        "transductive": cli.experiment == "loso", "n_subjects": len(subjects), "n_seeds": n_seeds,
        "label_head": "coral_ordinal_3class" if ordinal else "two_output_smoothl1_regression",
        "features": {subject.subject: {"path": subject.feature_path, "sha256": sha256(subject.feature_path)} for subject in subjects},
        "checkpoints": str(checkpoint_root), "split_file": str(split_path), "protocol": vars(args),
        "best_seed_summary": {
            "file": "metrics_best_seed_test_selected_exploratory.csv"
        },
    })


if __name__ == "__main__":
    main()
