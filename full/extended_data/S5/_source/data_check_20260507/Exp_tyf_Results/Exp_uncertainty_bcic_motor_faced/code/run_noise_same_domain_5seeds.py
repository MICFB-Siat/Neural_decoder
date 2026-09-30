


from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from run_noise_same_domain_rerun import (
    BASE_CLASSIFIER,
    BUILDER,
    CACHE_ROOT,
    CLASSIFIER,
    CONFIG_PATH,
    DATASETS,
    METRICS,
    PRIOR_CKPT,
    RUNTIME_PYTHON,
    TAGS,
    copy_file_sequential_atomic,
    load_configs,
    now,
    sha256,
    sync_tree,
    validate_cache,
    write_json_atomic,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = SCRIPT_DIR.parent
SEED0_ROOT = SOURCE_ROOT / "rerun_noise_same_domain_20260723"
ROOT = SOURCE_ROOT / "rerun_noise_same_domain_5seeds_20260723"
WORK_ROOT = Path("/tmp/eigen_uncertainty_noise_same_domain_5seeds_20260723")
SUMMARIZER = SCRIPT_DIR / "summarize_noise_same_domain_5seeds.py"
RUN_ID_BASE = "same_domain_5seeds_20260723"
NEW_SEEDS = [1, 2, 3, 4]


def run_id(seed: int) -> str:
    return f"{RUN_ID_BASE}_seed{seed:02d}"


def result_dir(root: Path, seed: int, dataset: str, tag: str) -> Path:
    return (
        root / "result" / f"seed_{seed:02d}" / dataset / "eigen"
        / f"{dataset}_{tag}_eeg_within_frozen_cache_{run_id(seed)}"
    )


def job_complete(
    root: Path,
    seed: int,
    dataset: str,
    tag: str,
    expected: int,
) -> bool:
    run_root = result_dir(root, seed, dataset, tag)
    summary_path = run_root / "summary.json"
    determinism_path = run_root / "determinism.json"
    if not summary_path.is_file() or not determinism_path.is_file():
        return False
    try:
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
        with determinism_path.open(encoding="utf-8") as f:
            determinism = json.load(f)
        meta = summary.get("meta", {})
        per_subject = summary.get("per_subject", {})
        return (
            len(per_subject) == expected
            and meta.get("cv_mode") == "within"
            and meta.get("n_folds") == 5
            and meta.get("seed") == seed
            and meta.get("train_mode") == "frozen"
            and meta.get("cache_features") is True
            and meta.get("RUN_ID") == run_id(seed)
            and Path(str(meta.get("h5_dir", ""))).resolve()
            == (CACHE_ROOT / dataset / tag).resolve()
            and determinism.get("optimization_seed") == seed
            and determinism.get("fold_seed") == seed
            and all(
                len(metrics.get(metric, {}).get("values", [])) == 5
                and all(math.isfinite(float(value)) for value in metrics[metric]["values"])
                for metrics in per_subject.values()
                for metric in METRICS
            )
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def make_command(
    seed: int,
    dataset: str,
    tag: str,
    gpu: int,
    cfg: dict[str, Any],
) -> list[str]:
    return [
        str(RUNTIME_PYTHON), "-u", str(CLASSIFIER),
        "--h5_dir", str(CACHE_ROOT / dataset / tag),
        "--encoder_ckpt", str(PRIOR_CKPT),
        "--d_path", str(cfg["d_path"]),
        "--ratio", str(cfg["ratio"]),
        "--n_samples", str(cfg["n_samples"]),
        "--cv_mode", "within",
        "--train_mode", "frozen",
        "--cache_features",
        "--label_key", "labels",
        "--n_classes", str(cfg["n_classes"]),
        "--modality", "eeg",
        "--n_folds", "5",
        "--seed", str(seed),
        "--epochs", "50",
        "--batch_size", "32",
        "--patience", "10",
        "--tag", f"{dataset}_{tag}",
        "--run_id", run_id(seed),
        "--out_dir", str(
            WORK_ROOT / "result" / f"seed_{seed:02d}" / dataset / "eigen"
        ),
        "--device", f"cuda:{gpu}",
    ]


def validate_seed0() -> dict[str, Any]:
    verification_path = SEED0_ROOT / "verification.json"
    independent_path = SEED0_ROOT / "independent_validation.json"
    if not verification_path.is_file() or not independent_path.is_file():
        raise FileNotFoundError("seed-0 validation artifacts are missing")
    with verification_path.open(encoding="utf-8") as f:
        verification = json.load(f)
    with independent_path.open(encoding="utf-8") as f:
        independent = json.load(f)
    if verification.get("status") != "COMPLETE" or independent.get("status") != "COMPLETE":
        raise RuntimeError("seed-0 source is not independently validated COMPLETE")
    return {
        "source_root": str(SEED0_ROOT),
        "verification": str(verification_path),
        "verification_status": verification.get("status"),
        "independent_validation": str(independent_path),
        "independent_validation_status": independent.get("status"),
        "reused_jobs": 9,
    }


def write_protocol(
    configs: dict[str, dict[str, Any]],
    cache_audits: list[dict[str, Any]],
    seed0_audit: dict[str, Any],
    gpus: list[int],
) -> None:
    protocol = {
        "schema_version": 1,
        "created_at": now(),
        "scope": {
            "datasets": DATASETS,
            "noise_levels": ["noise_03", "noise_05", "noise_10"],
            "seeds": [0, 1, 2, 3, 4],
            "expected_seed_conditions": 45,
            "reused_seed0_conditions": 9,
            "new_conditions": 36,
        },
        "seed_contract": {
            "split_seed_equals_seed": True,
            "optimization_seed_equals_seed": True,
            "per_fold_seed": (
                "stable uint32 derived from SHA256(optimization seed, subject/fold label)"
            ),
            "seed0": seed0_audit,
        },
        "selection_contract": {
            "selection_unit": "dataset x noise condition",
            "ours": (
                "For each seed, first select one PoE variant by unrounded "
                "across-subject mean; then choose the seed with the highest mean."
            ),
            "brainomni": (
                "Choose the seed with the lowest unrounded across-subject obs_acc mean."
            ),
            "per_subject_selection": False,
            "tie_break": "smallest seed; deterministic PoE metric order"
        },
        "training": {
            "independent_model_per_dataset_noise_seed": True,
            "train_domain": "the condition's noisy H5 files",
            "test_domain": "held-out fold from the same condition's noisy H5 files",
            "cv": "within-subject stratified 5-fold",
            "train_mode": "frozen backbones; train adapter/Gaussian/classification heads",
            "epochs": 50,
            "batch_size": 32,
            "patience": 10,
            "outer_fold_early_stopping": True,
            "heldout_temperature_calibration": True,
        },
        "data": {
            "cache_root": str(CACHE_ROOT),
            "regenerated": False,
            "builder": str(BUILDER),
            "builder_sha256": sha256(BUILDER),
            "cache_audits": cache_audits,
        },
        "runtime": {
            "python": str(RUNTIME_PYTHON),
            "gpus": gpus,
            "work_root": str(WORK_ROOT),
            "durable_root": str(ROOT),
            "local_staging": True,
            "resume_from_fold_metrics": True,
        },
        "artifacts": {
            "prior_checkpoint": str(PRIOR_CKPT),
            "prior_checkpoint_sha256": sha256(PRIOR_CKPT),
            "base_classifier": str(BASE_CLASSIFIER),
            "base_classifier_sha256": sha256(BASE_CLASSIFIER),
            "classifier": str(CLASSIFIER),
            "classifier_sha256": sha256(CLASSIFIER),
            "summarizer": str(SUMMARIZER),
            "summarizer_sha256": sha256(SUMMARIZER),
            "orchestrator": str(Path(__file__).resolve()),
            "orchestrator_sha256": sha256(Path(__file__).resolve()),
        },
        "datasets": {dataset: configs[dataset] for dataset in DATASETS},
    }
    write_json_atomic(protocol, WORK_ROOT / "protocol.json")
    copy_file_sequential_atomic(WORK_ROOT / "protocol.json", ROOT / "protocol.json")
    write_json_atomic(seed0_audit, WORK_ROOT / "seed0_reuse.json")
    copy_file_sequential_atomic(WORK_ROOT / "seed0_reuse.json", ROOT / "seed0_reuse.json")


def run(gpus: list[int], dry_run: bool) -> int:
    configs = load_configs()
    cache_audits = validate_cache(configs)
    seed0_audit = validate_seed0()
    for required in (
        RUNTIME_PYTHON, PRIOR_CKPT, BASE_CLASSIFIER, CLASSIFIER, SUMMARIZER,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    write_protocol(configs, cache_audits, seed0_audit, gpus)

    ordered_jobs = [
        *((seed, "FACED", tag) for seed in NEW_SEEDS for tag in TAGS),
        *((seed, "BCIC", tag) for seed in NEW_SEEDS for tag in TAGS),
        *((seed, "MOTOR", tag) for seed in NEW_SEEDS for tag in TAGS),
    ]
    pending: list[tuple[int, str, str]] = []
    already_complete: list[dict[str, Any]] = []
    for seed, dataset, tag in ordered_jobs:
        expected = int(configs[dataset]["expected_subjects"])
        local = result_dir(WORK_ROOT, seed, dataset, tag)
        durable = result_dir(ROOT, seed, dataset, tag)
        if not local.is_dir() and durable.is_dir():
            sync_tree(durable, local)
        if job_complete(WORK_ROOT, seed, dataset, tag, expected):
            if not job_complete(ROOT, seed, dataset, tag, expected):
                sync_tree(local, durable, completion_last=True)
            already_complete.append({
                "job": f"seed{seed}/{dataset}/{tag}",
                "status": "verified_resume",
            })
            print(f"[skip] seed{seed}/{dataset}/{tag}: verified complete", flush=True)
        else:
            pending.append((seed, dataset, tag))

    if dry_run:
        for index, (seed, dataset, tag) in enumerate(pending):
            print(
                " ".join(make_command(
                    seed, dataset, tag, gpus[index % len(gpus)], configs[dataset]
                ))
            )
        return 0

    state: dict[str, Any] = {
        "schema_version": 1,
        "state": "running",
        "started_at": now(),
        "updated_at": now(),
        "pid": os.getpid(),
        "gpus": gpus,
        "expected_seed_conditions": 45,
        "reused_seed0_conditions": 9,
        "requested_new_jobs": 36,
        "pending": [f"seed{seed}/{dataset}/{tag}" for seed, dataset, tag in pending],
        "running": {},
        "completed": already_complete,
        "failed": [],
        "work_root": str(WORK_ROOT),
        "durable_root": str(ROOT),
    }
    local_status = WORK_ROOT / "status.json"
    durable_status = ROOT / "status.json"
    lock = threading.Lock()

    def save_state() -> None:
        state["updated_at"] = now()
        write_json_atomic(state, local_status)
        copy_file_sequential_atomic(local_status, durable_status)

    save_state()
    work: queue.Queue[tuple[int, str, str]] = queue.Queue()
    for item in pending:
        work.put(item)

    def worker(gpu: int) -> None:
        while True:
            try:
                seed, dataset, tag = work.get_nowait()
            except queue.Empty:
                return
            key = f"seed{seed}/{dataset}/{tag}"
            expected = int(configs[dataset]["expected_subjects"])
            command = make_command(seed, dataset, tag, gpu, configs[dataset])
            log_path = WORK_ROOT / "logs" / f"seed{seed}_{dataset}_{tag}.log"
            durable_log = ROOT / "logs" / f"seed{seed}_{dataset}_{tag}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with lock:
                state["pending"] = [item for item in state["pending"] if item != key]
                state["running"][str(gpu)] = {
                    "job": key,
                    "started_at": now(),
                    "local_log": str(log_path),
                }
                save_state()
            env = os.environ.copy()
            env.update(
                PYTHONUNBUFFERED="1",
                OPTIMIZATION_SEED=str(seed),
                PYTHONHASHSEED=str(seed),
                CUBLAS_WORKSPACE_CONFIG=":4096:8",
                OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
                CUDA_HOME="/home/guoyi/anaconda3/envs/BrainOmni",
                HF_HOME=str(WORK_ROOT / "hf"),
                TORCH_HOME=str(WORK_ROOT / "torch"),
            )
            started = time.time()
            attempts: list[dict[str, Any]] = []
            verified = False
            returncode = -1
            for attempt in range(1, 4):
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(
                        f"\n[{now()}] GPU {gpu} ATTEMPT {attempt}/3 "
                        f"CMD: {' '.join(command)}\n"
                    )
                    log.flush()
                    process = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, env=env
                    )
                returncode = process.returncode
                local_ok = job_complete(WORK_ROOT, seed, dataset, tag, expected)
                sync_error = None
                if returncode == 0 and local_ok:
                    try:
                        sync_tree(
                            result_dir(WORK_ROOT, seed, dataset, tag),
                            result_dir(ROOT, seed, dataset, tag),
                            completion_last=True,
                        )
                        verified = job_complete(ROOT, seed, dataset, tag, expected)
                    except OSError as exc:
                        sync_error = repr(exc)
                attempts.append({
                    "attempt": attempt,
                    "returncode": returncode,
                    "local_verified": local_ok,
                    "durable_verified": verified,
                    "sync_error": sync_error,
                    "finished_at": now(),
                })
                if returncode == 0 and verified:
                    break
                if attempt < 3:
                    time.sleep(10)
            log_sync_error = None
            try:
                copy_file_sequential_atomic(log_path, durable_log)
            except OSError as exc:
                log_sync_error = repr(exc)
            record = {
                "job": key,
                "gpu": gpu,
                "returncode": returncode,
                "verified": verified,
                "elapsed_minutes": round((time.time() - started) / 60.0, 2),
                "finished_at": now(),
                "attempts": attempts,
                "local_log": str(log_path),
                "durable_log": str(durable_log),
                "log_sync_error": log_sync_error,
            }
            with lock:
                state["running"].pop(str(gpu), None)
                (state["completed"] if verified else state["failed"]).append(record)
                save_state()
            print(
                f"[{'done' if verified else 'fail'}] {key} gpu={gpu} "
                f"{record['elapsed_minutes']:.2f} min",
                flush=True,
            )
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    with lock:
        state["state"] = "aggregating" if not state["failed"] else "partial_failure"
        save_state()
    aggregation = subprocess.run([str(RUNTIME_PYTHON), "-u", str(SUMMARIZER)])
    verification_status = None
    verification_path = ROOT / "verification.json"
    if verification_path.is_file():
        with verification_path.open(encoding="utf-8") as f:
            verification_status = json.load(f).get("status")
    with lock:
        state["aggregation"] = {
            "returncode": aggregation.returncode,
            "verification": str(verification_path),
            "verification_status": verification_status,
            "finished_at": now(),
        }
        state["state"] = (
            "complete"
            if not state["failed"]
            and aggregation.returncode == 0
            and verification_status == "COMPLETE"
            else "partial_failure"
        )
        state["finished_at"] = now()
        save_state()
    return 0 if state["state"] == "complete" else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="3,4,7")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [int(part) for part in args.gpus.split(",") if part.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    raise SystemExit(run(gpus, args.dry_run))


if __name__ == "__main__":
    main()
