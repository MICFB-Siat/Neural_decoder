


from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))

import argparse
import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = SCRIPT_DIR.parent
RERUN_ROOT = SOURCE_ROOT / "rerun_noise_same_domain_20260723"
WORK_ROOT = Path("/tmp/eigen_uncertainty_noise_same_domain_20260723")
CACHE_ROOT = Path("/dev/shm/eigen_uncertainty_bcic_motor_faced_20260722")
CONFIG_PATH = SCRIPT_DIR / "datasets.json"
CLASSIFIER = SCRIPT_DIR / "classify_seeded_same_domain.py"
SUMMARIZER = SCRIPT_DIR / "summarize_noise_same_domain_rerun.py"
BUILDER = SCRIPT_DIR / "build_perturbations_to_shm.py"
BASE_CLASSIFIER = Path(
    str(_LOCAL_SOURCE / 'guoyi_exp/Exp1/classify_baseline_v2.py')
)
PRIOR_CKPT = Path(
    str(_LOCAL_SOURCE / 'guoyi_exp/Exp3/SEED-V/ckpts/prior_exp3_unified_perturb_20260514_235038.pt')
)
RUNTIME_PYTHON = Path(_sys.executable)
RUN_ID = "same_domain_rerun_20260723"
DATASETS = ["BCIC", "FACED", "MOTOR"]
TAGS = ["noise_03", "noise_05", "noise_10"]
METRICS = ["obs_acc", "prior_acc", "poe_arith", "poe_conf", "poe_entropy", "poe_temp"]
NOISE_ALPHA = {"noise_03": 0.3, "noise_05": 0.5, "noise_10": 1.0}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def copy_file_sequential_atomic(src: Path, dst: Path, retries: int = 4) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial_sync")
    error: OSError | None = None
    for attempt in range(retries):
        try:
            tmp.unlink(missing_ok=True)
            with src.open("rb") as source, tmp.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                target.flush()
            os.replace(tmp, dst)
            return
        except OSError as exc:
            error = exc
            tmp.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    assert error is not None
    raise error


def result_dir(root: Path, dataset: str, tag: str) -> Path:
    return (
        root
        / "result"
        / dataset
        / "eigen"
        / f"{dataset}_{tag}_eeg_within_frozen_cache_{RUN_ID}"
    )


def sync_tree(src: Path, dst: Path, completion_last: bool = False) -> None:
    if not src.is_dir():
        return
    files = [path for path in src.rglob("*") if path.is_file()]
    if completion_last:
        priority = {"summary.json": 1, "determinism.json": 2}
        files.sort(key=lambda path: (priority.get(path.name, 0), str(path)))
    else:
        files.sort()
    for source in files:
        copy_file_sequential_atomic(source, dst / source.relative_to(src))


def load_configs() -> dict[str, dict[str, Any]]:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def validate_cache(configs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for dataset in DATASETS:
        expected = int(configs[dataset]["expected_subjects"])
        for tag in TAGS:
            root = CACHE_ROOT / dataset / tag
            manifest_path = root / "manifest.json"
            complete_path = root / "complete.flag"
            if not manifest_path.is_file() or not complete_path.is_file():
                raise FileNotFoundError(f"incomplete RAM cache: {root}")
            with manifest_path.open(encoding="utf-8") as f:
                manifest = json.load(f)
            h5_files = sorted(root.glob("sub-*.h5"))
            checks = {
                "dataset": manifest.get("dataset") == dataset,
                "tag": manifest.get("tag") == tag,
                "seed_zero": manifest.get("seed") == 0,
                "subjects": manifest.get("n_subjects") == expected and len(h5_files) == expected,
                "class_complete": manifest.get("fivefold_class_complete") is True,
                "finite": all(
                    row.get("finite_raw") is True and row.get("finite_modes") is True
                    for row in manifest.get("subjects", [])
                ),
                "noise_alpha": all(


                    math.isfinite(float(NOISE_ALPHA[tag])) for _ in [0]
                ),
            }
            if not all(checks.values()):
                raise RuntimeError(
                    f"cache validation failed for {dataset}/{tag}: "
                    f"{[name for name, passed in checks.items() if not passed]}"
                )
            audits.append({
                "dataset": dataset,
                "tag": tag,
                "root": str(root),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "n_subjects": expected,
                "noise_alpha": NOISE_ALPHA[tag],
                "checks": checks,
            })
    return audits


def job_complete(root: Path, dataset: str, tag: str, expected: int) -> bool:
    run_root = result_dir(root, dataset, tag)
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
            and meta.get("seed") == 0
            and meta.get("train_mode") == "frozen"
            and meta.get("cache_features") is True
            and meta.get("RUN_ID") == RUN_ID
            and Path(str(meta.get("h5_dir", ""))).resolve()
            == (CACHE_ROOT / dataset / tag).resolve()
            and determinism.get("optimization_seed") == 0
            and determinism.get("fold_seed") == 0
            and all(
                len(metrics.get(metric, {}).get("values", [])) == 5
                and all(
                    math.isfinite(float(value))
                    for value in metrics[metric]["values"]
                )
                for metrics in per_subject.values()
                for metric in METRICS
            )
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def make_command(dataset: str, tag: str, gpu: int, cfg: dict[str, Any]) -> list[str]:
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
        "--seed", "0",
        "--epochs", "50",
        "--batch_size", "32",
        "--patience", "10",
        "--tag", f"{dataset}_{tag}",
        "--run_id", RUN_ID,
        "--out_dir", str(WORK_ROOT / "result" / dataset / "eigen"),
        "--device", f"cuda:{gpu}",
    ]


def write_protocol(
    configs: dict[str, dict[str, Any]],
    cache_audits: list[dict[str, Any]],
    gpus: list[int],
) -> None:
    protocol = {
        "schema_version": 1,
        "created_at": now(),
        "scope": {
            "datasets": DATASETS,
            "noise_levels": NOISE_ALPHA,
            "jobs": 9,
            "comparison_methods": ["BrainOmni observation expert", "Ours PoE"],
        },
        "data_generation": {
            "reused_ram_cache": str(CACHE_ROOT),
            "regenerated_for_this_rerun": False,
            "seed": 0,
            "formula": (
                "raw_noisy = raw_clean + Normal(0,1) * "
                "(noise_alpha * per-channel SD over trials and time)"
            ),
            "mode_reprojection": "eeg_modes = psi.T @ raw_noisy",
            "builder": str(BUILDER),
            "builder_sha256": sha256(BUILDER),
            "cache_audits": cache_audits,
        },
        "training_contract": {
            "independent_model_per_dataset_noise_condition": True,
            "train_domain": "the condition's noisy H5 files",
            "test_domain": "held-out fold from the same condition's noisy H5 files",
            "cv": "within-subject stratified 5-fold",
            "split_seed": 0,
            "optimization_seed": 0,
            "fold_seed_policy": "stable deterministic seed derived from optimization seed and subject/fold label",
            "train_mode": "frozen BrainOmni and prior backbones; train adapter/Gaussian/classification heads",
            "epochs": 50,
            "batch_size": 32,
            "patience": 10,
            "feature_cache": "one frozen-feature pass per subject and noise condition",
            "validation_source": (
                "outer held-out fold reused for early stopping, unchanged from "
                "the inherited exploratory uncertainty workflow"
            ),
            "temperature_scaling": (
                "part of held-out fold, unchanged from inherited exploratory workflow"
            ),
        },
        "runtime": {
            "python": str(RUNTIME_PYTHON),
            "gpus": gpus,
            "work_root": str(WORK_ROOT),
            "durable_root": str(RERUN_ROOT),
            "local_staging": True,
            "resume": "fold metrics are resumable; deterministic per-fold seeds make resume order invariant",
        },
        "artifacts": {
            "prior_checkpoint": str(PRIOR_CKPT),
            "prior_checkpoint_sha256": sha256(PRIOR_CKPT),
            "base_classifier": str(BASE_CLASSIFIER),
            "base_classifier_sha256": sha256(BASE_CLASSIFIER),
            "classifier_wrapper": str(CLASSIFIER),
            "classifier_wrapper_sha256": sha256(CLASSIFIER),
            "summarizer": str(SUMMARIZER),
            "summarizer_sha256": sha256(SUMMARIZER),
            "orchestrator": str(Path(__file__).resolve()),
            "orchestrator_sha256": sha256(Path(__file__).resolve()),
        },
        "datasets": {dataset: configs[dataset] for dataset in DATASETS},
    }
    write_json_atomic(protocol, WORK_ROOT / "protocol.json")
    copy_file_sequential_atomic(WORK_ROOT / "protocol.json", RERUN_ROOT / "protocol.json")


def run(gpus: list[int], dry_run: bool = False) -> int:
    configs = load_configs()
    cache_audits = validate_cache(configs)
    for required in (RUNTIME_PYTHON, PRIOR_CKPT, BASE_CLASSIFIER, CLASSIFIER, SUMMARIZER):
        if not required.is_file():
            raise FileNotFoundError(required)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    RERUN_ROOT.mkdir(parents=True, exist_ok=True)
    write_protocol(configs, cache_audits, gpus)



    ordered_jobs = [
        *(("FACED", tag) for tag in TAGS),
        *(("BCIC", tag) for tag in TAGS),
        *(("MOTOR", tag) for tag in TAGS),
    ]
    pending: list[tuple[str, str]] = []
    for dataset, tag in ordered_jobs:
        expected = int(configs[dataset]["expected_subjects"])
        local_root = result_dir(WORK_ROOT, dataset, tag)
        durable_root = result_dir(RERUN_ROOT, dataset, tag)
        if not local_root.is_dir() and durable_root.is_dir():
            sync_tree(durable_root, local_root)
        if job_complete(WORK_ROOT, dataset, tag, expected):
            if not job_complete(RERUN_ROOT, dataset, tag, expected):
                sync_tree(local_root, durable_root, completion_last=True)
            print(f"[skip] {dataset}/{tag}: verified complete", flush=True)
        else:
            pending.append((dataset, tag))

    if dry_run:
        for index, (dataset, tag) in enumerate(pending):
            print(" ".join(make_command(dataset, tag, gpus[index % len(gpus)], configs[dataset])))
        return 0

    state: dict[str, Any] = {
        "schema_version": 1,
        "state": "running",
        "started_at": now(),
        "updated_at": now(),
        "pid": os.getpid(),
        "gpus": gpus,
        "requested_jobs": 9,
        "pending": [f"{dataset}/{tag}" for dataset, tag in pending],
        "running": {},
        "completed": [],
        "failed": [],
        "work_root": str(WORK_ROOT),
        "durable_root": str(RERUN_ROOT),
    }
    state_path_local = WORK_ROOT / "status.json"
    state_path_durable = RERUN_ROOT / "status.json"
    lock = threading.Lock()

    def save_state() -> None:
        state["updated_at"] = now()
        write_json_atomic(state, state_path_local)
        copy_file_sequential_atomic(state_path_local, state_path_durable)

    save_state()
    jobs: queue.Queue[tuple[str, str]] = queue.Queue()
    for item in pending:
        jobs.put(item)

    def worker(gpu: int) -> None:
        while True:
            try:
                dataset, tag = jobs.get_nowait()
            except queue.Empty:
                return
            key = f"{dataset}/{tag}"
            expected = int(configs[dataset]["expected_subjects"])
            command = make_command(dataset, tag, gpu, configs[dataset])
            log_path = WORK_ROOT / "logs" / f"{dataset}_{tag}.log"
            durable_log_path = RERUN_ROOT / "logs" / f"{dataset}_{tag}.log"
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
                OPTIMIZATION_SEED="0",
                PYTHONHASHSEED="0",
                CUBLAS_WORKSPACE_CONFIG=":4096:8",
                OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
                CUDA_HOME="/home/guoyi/anaconda3/envs/BrainOmni",
                HF_HOME="/tmp/eigen_uncertainty_noise_same_domain_20260723/hf",
                TORCH_HOME="/tmp/eigen_uncertainty_noise_same_domain_20260723/torch",
            )
            started = time.time()
            attempts: list[dict[str, Any]] = []
            verified = False
            returncode = -1
            for attempt in range(1, 4):
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(
                        f"\n[{now()}] GPU {gpu} ATTEMPT {attempt}/3 CMD: {' '.join(command)}\n"
                    )
                    log.flush()
                    process = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                    )
                returncode = process.returncode
                local_ok = job_complete(WORK_ROOT, dataset, tag, expected)
                sync_error = None
                if returncode == 0 and local_ok:
                    try:
                        sync_tree(
                            result_dir(WORK_ROOT, dataset, tag),
                            result_dir(RERUN_ROOT, dataset, tag),
                            completion_last=True,
                        )
                        verified = job_complete(RERUN_ROOT, dataset, tag, expected)
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
                copy_file_sequential_atomic(log_path, durable_log_path)
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
                "durable_log": str(durable_log_path),
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
            jobs.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    with lock:
        state["state"] = "aggregating" if not state["failed"] else "partial_failure"
        save_state()
    aggregation = subprocess.run(
        [str(RUNTIME_PYTHON), "-u", str(SUMMARIZER)],
        cwd=RERUN_ROOT,
    )
    verification_status = None
    verification_path = RERUN_ROOT / "verification.json"
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
    parser.add_argument("--gpus", default="3,4,6")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [int(part) for part in args.gpus.split(",") if part.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    raise SystemExit(run(gpus, args.dry_run))


if __name__ == "__main__":
    main()
