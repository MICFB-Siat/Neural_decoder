

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from experiment_config import (
    CLASSIFIER,
    CLASSIFIER_ARGS,
    DATASETS,
    MODE_TYPES,
    N_FOLDS,
    SEEDS,
    assert_classifier_frozen,
    checkpoint_path,
    data_dir,
    run_root,
    run_tag,
    seed_out_dir,
)


def task_complete(dataset: str, mode_type: str, paradigm: str, seed: int, verbose: bool = False) -> bool:
    root = run_root(dataset, mode_type, paradigm, seed)
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        meta = summary["meta"]
        expected_cv = "within" if paradigm == "within" else "cross"
        if (
            meta["seed"] != seed
            or meta["cv_mode"] != expected_cv
            or meta["n_folds"] != N_FOLDS
            or meta["label_key"] != DATASETS[dataset]["label_key"]
            or meta["n_classes"] != DATASETS[dataset]["n_classes"]
            or meta.get("eigval_source") != "h5"
            or meta.get("d_path") is not None
        ):
            return False
        subjects = meta["subjects"]
        if len(subjects) != DATASETS[dataset]["expected_subjects"]:
            return False
        if paradigm == "within":
            metrics = list(root.glob("sub-*/fold_*/metrics.json"))
            if len(metrics) != len(subjects) * N_FOLDS:
                return False
            if set(summary["per_subject"]) != set(subjects):
                return False
            if any(len(summary["per_subject"][s]["poe_temp"]["values"]) != N_FOLDS for s in subjects):
                return False
        else:
            metrics = list(root.glob("fold_*/metrics.json"))
            preds = list(root.glob("fold_*/holdout_preds.npz"))
            subject_ids = list(root.glob("fold_*/holdout_subj_ids.npy"))
            if not (len(metrics) == len(preds) == len(subject_ids) == N_FOLDS):
                return False
            if len(summary["poe_temp"]["values"]) != N_FOLDS:
                return False
        for path in metrics:
            row = json.loads(path.read_text(encoding="utf-8"))
            if "poe_temp" not in row or not (0.0 <= float(row["poe_temp"]) <= 1.0):
                return False
        return True
    except Exception as exc:
        if verbose:
            print(f"Incomplete {root}: {exc}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--mode-type", choices=MODE_TYPES, required=True)
    ap.add_argument("--paradigm", choices=("within", "pool"), required=True)
    ap.add_argument("--seed", type=int, choices=SEEDS, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=CLASSIFIER_ARGS["epochs"], help="Formal value is 200")
    ap.add_argument("--patience", type=int, default=CLASSIFIER_ARGS["patience"], help="Formal value is 30")
    ap.add_argument("--subjects", nargs="+", default=None, help="Smoke-test only")
    ap.add_argument("--fold", type=int, default=None, help="Smoke-test only")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    assert_classifier_frozen()
    formal = (
        args.epochs == CLASSIFIER_ARGS["epochs"]
        and args.patience == CLASSIFIER_ARGS["patience"]
        and args.subjects is None
        and args.fold is None
        and args.data_dir is None
        and args.checkpoint is None
        and args.out_dir is None
        and args.tag is None
    )
    if formal and task_complete(args.dataset, args.mode_type, args.paradigm, args.seed, verbose=True):
        print(f"SKIP complete {args.dataset}/{args.mode_type}/{args.paradigm}/seed_{args.seed:02d}", flush=True)
        return

    cfg = DATASETS[args.dataset]
    ckpt = (args.checkpoint or checkpoint_path(args.dataset, args.mode_type)).resolve()
    h5_dir = (args.data_dir or data_dir(args.dataset, args.mode_type)).resolve()
    out = (args.out_dir or seed_out_dir(args.dataset, args.mode_type, args.paradigm, args.seed)).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    if not (h5_dir / "sub-001.h5").is_file() and not list(h5_dir.glob("sub-*.h5")):
        raise FileNotFoundError(f"No subject H5 files in {h5_dir}")
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or run_tag(args.dataset, args.mode_type, args.paradigm, args.seed)
    command = [
        sys.executable,
        str(CLASSIFIER),
        "--h5_dir", str(h5_dir),
        "--encoder_ckpt", str(ckpt),
        "--train_mode", CLASSIFIER_ARGS["train_mode"],
        "--cache_features",
        "--cv_mode", "within" if args.paradigm == "within" else "cross",
        "--modality", "eeg",
        "--label_key", cfg["label_key"],
        "--n_folds", str(CLASSIFIER_ARGS["n_folds"]),
        "--epochs", str(args.epochs),
        "--batch_size", str(CLASSIFIER_ARGS["batch_size"]),
        "--lr_adapter", str(CLASSIFIER_ARGS["lr_adapter"]),
        "--lr_head", str(CLASSIFIER_ARGS["lr_head"]),
        "--lat_dim", str(CLASSIFIER_ARGS["lat_dim"]),
        "--dropout", str(CLASSIFIER_ARGS["dropout"]),
        "--patience", str(args.patience),
        "--kl_weight", str(CLASSIFIER_ARGS["kl_weight"]),
        "--n_classes", str(cfg["n_classes"]),
        "--n_samples", str(cfg["n_samples"]),
        "--device", args.device,
        "--seed", str(args.seed),
        "--tag", tag,
        "--out_dir", str(out),
        "--run_id", "run",
    ]
    if args.subjects:
        command += ["--subjects", *args.subjects]
    if args.fold is not None:
        command += ["--fold", str(args.fold)]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("XDG_CACHE_HOME", "/tmp/mode_type_bcic_faced_motor_cache/xdg")
    env.setdefault("TRITON_CACHE_DIR", "/tmp/mode_type_bcic_faced_motor_cache/triton")
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/mode_type_bcic_faced_motor_cache/torchinductor")
    print("COMMAND " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)
    if formal and not task_complete(args.dataset, args.mode_type, args.paradigm, args.seed, verbose=True):
        raise RuntimeError("Classifier returned successfully but formal task audit is incomplete")


if __name__ == "__main__":
    main()
