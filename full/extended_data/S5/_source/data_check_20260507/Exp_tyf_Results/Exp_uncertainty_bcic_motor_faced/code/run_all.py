


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
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
CONFIG = SCRIPT_DIR / "datasets.json"
BUILDER = SCRIPT_DIR / "build_perturbations_to_shm.py"
CLASSIFIER = SCRIPT_DIR / "classify_once_per_subject_cache.py"
LABRAM_LAUNCHER = SCRIPT_DIR / "labram_dataset_launcher.py"
CBRAMOD_LAUNCHER = SCRIPT_DIR / "cbramod_dataset_launcher.py"
SUMMARIZER = SCRIPT_DIR / "summarize_results.py"
EDGE_HELPER = SCRIPT_DIR / "edge_case_splits.py"
BASE_CLASSIFIER = Path(
    str(_LOCAL_SOURCE / 'guoyi_exp/Exp1/classify_baseline_v2.py')
)
PRIOR_CKPT = Path(
    str(_LOCAL_SOURCE / 'guoyi_exp/Exp3/SEED-V/ckpts/prior_exp3_unified_perturb_20260514_235038.pt')
)
LABRAM_CKPT = EXP_ROOT.parent / "Models" / "LaBraM" / "checkpoints" / "labram-base.pth"
CBRAMOD_CKPT = (
    EXP_ROOT.parent / "Models" / "CBraMod" / "pretrained_weights" / "pretrained_weights.pth"
)
DEFAULT_CACHE = Path("/dev/shm/eigen_uncertainty_bcic_motor_faced_20260722")
WORK_ROOT = Path("/tmp/eigen_uncertainty_bmf_20260722_work")
RUNTIME_PYTHON = Path(_sys.executable)
BASELINE_PYTHON = Path(_sys.executable)
ALL_DATASETS = ["BCIC", "MOTOR", "FACED"]
ALL_METHODS = ["eigen", "labram", "cbramod"]
ALL_TAGS = [
    "noise_03", "noise_05", "noise_10",
    "drop_01", "drop_03", "drop_05",
    "red_50", "red_25", "red_10",
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def copy_file_sequential_atomic(src: Path, dst: Path, retries: int = 4) -> None:

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial_sync")
    last_error: OSError | None = None
    for attempt in range(1, retries + 1):
        try:
            tmp.unlink(missing_ok=True)
            with src.open("rb") as source, tmp.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                target.flush()
            os.replace(tmp, dst)
            return
        except OSError as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    assert last_error is not None
    raise last_error


def sync_result_tree(method: str, dataset: str, tag: str) -> None:

    source_root = result_dir(method, dataset, tag, WORK_ROOT)
    destination_root = result_dir(method, dataset, tag, EXP_ROOT)
    files = [path for path in source_root.rglob("*") if path.is_file()]
    files.sort(key=lambda path: (path.name == "summary.json", str(path)))
    for source in files:
        destination = destination_root / source.relative_to(source_root)
        copy_file_sequential_atomic(source, destination)


def load_configs() -> dict[str, dict[str, Any]]:
    with CONFIG.open(encoding="utf-8") as f:
        return json.load(f)


def result_dir(
    method: str,
    dataset: str,
    tag: str,
    artifact_root: Path = EXP_ROOT,
) -> Path:
    if method == "eigen":
        run_id = f"uncertainty_20260722_{tag}"
        return (
            artifact_root / "result" / dataset / "eigen" /
            f"{dataset}_{tag}_eeg_within_frozen_cache_{run_id}"
        )
    return artifact_root / "result" / dataset / method / f"{dataset}_{tag}_within5fold"


def job_is_complete(
    method: str,
    dataset: str,
    tag: str,
    expected_subjects: int,
    artifact_root: Path = EXP_ROOT,
) -> bool:
    root = result_dir(method, dataset, tag, artifact_root)
    summary = root / "summary.json"
    if not summary.is_file():
        return False
    try:
        with summary.open(encoding="utf-8") as f:
            data = json.load(f)
        if len(data.get("per_subject", {})) != expected_subjects:
            return False
        if method == "eigen":
            for metrics in data["per_subject"].values():
                for key in ("obs_acc", "prior_acc", "poe_arith", "poe_conf", "poe_entropy", "poe_temp"):
                    if len(metrics.get(key, {}).get("values", [])) != 5:
                        return False
        else:
            for metrics in data["per_subject"].values():
                n_folds = metrics.get("n_folds")
                if n_folds != 5:
                    return False
    except (OSError, ValueError, TypeError):
        return False
    return True


def baseline_feasibility(dataset: str, tag: str, n_classes: int) -> tuple[bool, str | None]:

    manifest_path = EXP_ROOT / "data_manifests" / dataset / f"{tag}.json"
    if not manifest_path.is_file():
        return True, None
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    for row in manifest.get("subjects", []):
        n = int(row["n_output"])
        outer_train_min = n - math.ceil(n / 5)
        inner_val_n = math.ceil(0.1 * outer_train_min)
        min_class_outer_train = min(
            int(count) - math.ceil(int(count) / 5)
            for count in row["output_class_counts"].values()
        )
        if inner_val_n < n_classes or min_class_outer_train < 2:
            return False, (
                f"{row['subject']}: outer-train minimum n={outer_train_min}, "
                f"10% inner-val n={inner_val_n}, classes={n_classes}, "
                f"minimum class count in outer train={min_class_outer_train}"
            )
    return True, None


def make_command(
    method: str,
    dataset: str,
    tag: str,
    gpu: int,
    cfg: dict[str, Any],
    cache_root: Path,
    artifact_root: Path = WORK_ROOT,
) -> list[str]:
    if method == "eigen":
        return [
            str(RUNTIME_PYTHON),
            "-u",
            str(CLASSIFIER),
        "--h5_dir", str(cache_root / dataset / tag),
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
        "--run_id", f"uncertainty_20260722_{tag}",
        "--out_dir", str(artifact_root / "result" / dataset / "eigen"),
            "--device", f"cuda:{gpu}",
        ]
    if method == "labram":
        return [
            str(BASELINE_PYTHON), "-u", str(LABRAM_LAUNCHER),
            "--dataset", dataset,
            "--h5_dir", str(cache_root / dataset / tag),
            "--tag", f"{dataset}_{tag}",
            "--out_dir", str(result_dir(method, dataset, tag, artifact_root)),
            "--pretrained_ckpt", str(LABRAM_CKPT),
            "--cuda", str(gpu),
            "--seed", "3407",
            "--epochs", "30",
            "--batch_size", "64",
            "--patience", "8",
            "--n_folds", "5",
            "--frozen",
        ]
    if method == "cbramod":
        return [
            str(BASELINE_PYTHON), "-u", str(CBRAMOD_LAUNCHER),
            "--dataset", dataset,
            "--h5_dir", str(cache_root / dataset / tag),
            "--tag", f"{dataset}_{tag}",
            "--out_dir", str(result_dir(method, dataset, tag, artifact_root)),
            "--backbone_ckpt", str(CBRAMOD_CKPT),
            "--cuda", str(gpu),
            "--seed", "3407",
            "--epochs", "30",
            "--batch_size", "64",
            "--patience", "8",
            "--n_folds", "5",
            "--frozen",
        ]
    raise ValueError(method)


def prepare_cache(datasets: list[str], tags: list[str], cache_root: Path) -> None:
    command = [
        str(RUNTIME_PYTHON),
        "-u",
        str(BUILDER),
        "--config", str(CONFIG),
        "--cache-root", str(cache_root),
        "--manifest-mirror", str(EXP_ROOT / "data_manifests"),
        "--datasets", *datasets,
        "--tags", *tags,
        "--seed", "0",
    ]
    subprocess.run(command, check=True)


def write_protocol(
    configs: dict[str, dict[str, Any]],
    cache_root: Path,
    edge_case_fallback: bool,
) -> None:
    protocol = {
        "schema_version": 1,
        "created_at": now(),
        "perturbations": ALL_TAGS,
        "methods": {
            "eigen": "BrainOmni observation + K-free prior + PoE variants",
            "labram": "LaBraM-base frozen backbone",
            "cbramod": "CBraMod frozen backbone",
        },
        "prior_checkpoint": str(PRIOR_CKPT),
        "prior_checkpoint_sha256": sha256(PRIOR_CKPT),
        "base_classifier": str(BASE_CLASSIFIER),
        "base_classifier_sha256": sha256(BASE_CLASSIFIER),
        "cache_launcher": str(CLASSIFIER),
        "cache_launcher_sha256": sha256(CLASSIFIER),
        "labram_launcher": str(LABRAM_LAUNCHER),
        "labram_launcher_sha256": sha256(LABRAM_LAUNCHER),
        "cbramod_launcher": str(CBRAMOD_LAUNCHER),
        "cbramod_launcher_sha256": sha256(CBRAMOD_LAUNCHER),
        "summarizer": str(SUMMARIZER),
        "summarizer_sha256": sha256(SUMMARIZER),
        "orchestrator": str(Path(__file__).resolve()),
        "orchestrator_sha256": sha256(Path(__file__).resolve()),
        "builder": str(BUILDER),
        "builder_sha256": sha256(BUILDER),
        "edge_case_helper": str(EDGE_HELPER),
        "edge_case_helper_sha256": sha256(EDGE_HELPER),
        "labram_checkpoint": str(LABRAM_CKPT),
        "labram_checkpoint_sha256": sha256(LABRAM_CKPT),
        "cbramod_checkpoint": str(CBRAMOD_CKPT),
        "cbramod_checkpoint_sha256": sha256(CBRAMOD_CKPT),
        "cache_root": str(cache_root),
        "work_root": str(WORK_ROOT),
        "runtime_python": str(RUNTIME_PYTHON),
        "baseline_python": str(BASELINE_PYTHON),
        "cache_policy": (
            "perturbed H5 in /dev/shm; frozen features once per subject per condition; "
            "runtime outputs/logs staged in local /tmp and sequentially synchronized "
            "to the shared experiment tree with summary.json copied last"
        ),
        "training": {
            "cv_mode": "within",
            "n_folds": 5,
            "split_seed": 0,
            "train_mode": "frozen",
            "epochs": 50,
            "batch_size": 32,
            "patience": 10,
            "validation_source": "outer held-out fold reused for early stopping, unchanged from SEED-V exploratory protocol",
            "temperature_scaling": "held-out fold split internally, unchanged from SEED-V exploratory protocol",
            "baseline_seed": 3407,
            "baseline_epochs": 30,
            "baseline_patience": 8,
            "baseline_inner_validation": "10% stratified from the train fold",
            "edge_case_fallback_enabled": edge_case_fallback,
            "edge_case_fallback": (
                "Only when stratification raises: deterministic unstratified split "
                "with unchanged fold count/validation size/seed; when temperature "
                "cross-fit has an empty half: neutral T=1. Every use is recorded in "
                "edge_case_fallback.json."
            ),
        },
        "datasets": configs,
    }
    write_json_atomic(protocol, EXP_ROOT / "protocol.json")


def run_jobs(
    datasets: list[str],
    tags: list[str],
    methods: list[str],
    gpus: list[int],
    cache_root: Path,
    configs: dict[str, dict[str, Any]],
    dry_run: bool,
    edge_case_fallback: bool,
) -> int:
    jobs = []
    blocked = []
    for dataset in datasets:
        for tag in tags:
            for method in methods:
                if job_is_complete(method, dataset, tag, int(configs[dataset]["expected_subjects"])):
                    print(f"[skip] {method}/{dataset}/{tag}: verified summary exists", flush=True)
                elif method in ("labram", "cbramod") and not edge_case_fallback:
                    feasible, reason = baseline_feasibility(
                        dataset, tag, int(configs[dataset]["n_classes"])
                    )
                    if not feasible:
                        record = {
                            "job": f"{method}/{dataset}/{tag}",
                            "reason": reason,
                            "status": "structurally_blocked_by_unchanged_inner_validation",
                        }
                        blocked.append(record)
                        print(f"[blocked] {record['job']}: {reason}", flush=True)
                    else:
                        jobs.append((method, dataset, tag))
                else:
                    jobs.append((method, dataset, tag))
    if dry_run:
        for index, (method, dataset, tag) in enumerate(jobs):
            gpu = gpus[index % len(gpus)]
            print(" ".join(make_command(method, dataset, tag, gpu, configs[dataset], cache_root)))
        return 0

    status_path = EXP_ROOT / "status.json"
    state: dict[str, Any] = {
        "schema_version": 1,
        "state": "running",
        "started_at": now(),
        "pid": os.getpid(),
        "cache_root": str(cache_root),
        "work_root": str(WORK_ROOT),
        "gpus": gpus,
        "methods": methods,
        "edge_case_fallback": edge_case_fallback,
        "requested_jobs": len(datasets) * len(tags) * len(methods),
        "pending": [f"{m}/{d}/{t}" for m, d, t in jobs],
        "running": {},
        "completed": [],
        "failed": [],
        "structurally_blocked": blocked,
    }
    lock = threading.Lock()
    write_json_atomic(state, status_path)
    work: queue.Queue[tuple[str, str, str]] = queue.Queue()
    for item in jobs:
        work.put(item)

    def save_state() -> None:
        state["updated_at"] = now()
        write_json_atomic(state, status_path)

    def worker(gpu: int) -> None:
        while True:
            try:
                method, dataset, tag = work.get_nowait()
            except queue.Empty:
                return
            key = f"{method}/{dataset}/{tag}"
            log_path = WORK_ROOT / "logs" / f"{method}_{dataset}_{tag}.log"
            durable_log_path = EXP_ROOT / "logs" / f"{method}_{dataset}_{tag}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = make_command(
                method, dataset, tag, gpu, configs[dataset], cache_root, WORK_ROOT
            )
            with lock:
                state["pending"] = [x for x in state["pending"] if x != key]
                state["running"][str(gpu)] = {
                    "job": key,
                    "started_at": now(),
                    "local_log": str(log_path),
                    "durable_log": str(durable_log_path),
                }
                save_state()
            t0 = time.time()
            env = os.environ.copy()
            env.update(
                PYTHONUNBUFFERED="1",
                OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
                CUDA_HOME="/home/guoyi/anaconda3/envs/BrainOmni",
            )
            attempts = []
            proc = None
            verified = False
            verified_local = False
            sync_error = None
            for attempt in range(1, 4):
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(
                        f"\n[{now()}] GPU {gpu} ATTEMPT {attempt}/3 CMD: {' '.join(command)}\n"
                    )
                    log.flush()
                    proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                verified_local = job_is_complete(
                    method,
                    dataset,
                    tag,
                    int(configs[dataset]["expected_subjects"]),
                    WORK_ROOT,
                )
                verified = False
                sync_error = None
                if proc.returncode == 0 and verified_local:
                    try:
                        sync_result_tree(method, dataset, tag)
                        verified = job_is_complete(
                            method,
                            dataset,
                            tag,
                            int(configs[dataset]["expected_subjects"]),
                            EXP_ROOT,
                        )
                    except OSError as exc:
                        sync_error = repr(exc)
                attempts.append({
                    "attempt": attempt,
                    "returncode": proc.returncode,
                    "verified_local_summary": verified_local,
                    "verified_durable_summary": verified,
                    "sync_error": sync_error,
                    "finished_at": now(),
                })
                if proc.returncode == 0 and verified:
                    break
                if attempt < 3:
                    print(
                        f"[retry] {key} gpu={gpu} attempt={attempt} "
                        f"rc={proc.returncode} local={verified_local} durable={verified} "
                        f"sync_error={sync_error}",
                        flush=True,
                    )
                    time.sleep(10)
            assert proc is not None
            log_sync_error = None
            try:
                copy_file_sequential_atomic(log_path, durable_log_path)
            except OSError as exc:
                log_sync_error = repr(exc)
            elapsed = round((time.time() - t0) / 60.0, 2)
            record = {
                "job": key,
                "gpu": gpu,
                "returncode": proc.returncode,
                "verified_local_summary": verified_local,
                "verified_durable_summary": verified,
                "elapsed_minutes": elapsed,
                "finished_at": now(),
                "local_log": str(log_path),
                "durable_log": str(durable_log_path),
                "log_sync_error": log_sync_error,
                "attempts": attempts,
            }
            with lock:
                state["running"].pop(str(gpu), None)
                (state["completed"] if proc.returncode == 0 and verified else state["failed"]).append(record)
                save_state()
            print(
                f"[{'done' if proc.returncode == 0 and verified else 'fail'}] {key} "
                f"gpu={gpu} {elapsed:.1f} min",
                flush=True,
            )
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with lock:
        state["state"] = (
            "partial_failure" if state["failed"] else
            "complete_with_structural_blocks" if blocked else
            "complete"
        )
        state["finished_at"] = now()
        save_state()
    aggregation = subprocess.run(
        [str(RUNTIME_PYTHON), "-u", str(SUMMARIZER)],
        cwd=EXP_ROOT,
    )
    with lock:
        state["aggregation"] = {
            "returncode": aggregation.returncode,
            "finished_at": now(),
            "verification": str(EXP_ROOT / "verification.json"),
        }
        save_state()
    return 0 if not state["failed"] else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "run", "all"], default="all")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--tags", nargs="+", default=ALL_TAGS)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS)
    parser.add_argument("--gpus", default="5,7")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--edge-case-fallback",
        action="store_true",
        help="Run mathematically impossible tiny-sample strata with audited deterministic fallbacks.",
    )
    args = parser.parse_args()

    configs = load_configs()
    unknown_datasets = sorted(set(args.datasets) - set(configs))
    unknown_tags = sorted(set(args.tags) - set(ALL_TAGS))
    unknown_methods = sorted(set(args.methods) - set(ALL_METHODS))
    if unknown_datasets or unknown_tags or unknown_methods:
        raise ValueError(
            f"unknown datasets={unknown_datasets}, tags={unknown_tags}, methods={unknown_methods}"
        )
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    for required in (
        PRIOR_CKPT, LABRAM_CKPT, CBRAMOD_CKPT, BASE_CLASSIFIER, CLASSIFIER, LABRAM_LAUNCHER,
        CBRAMOD_LAUNCHER, SUMMARIZER, EDGE_HELPER, BUILDER, CONFIG,
        RUNTIME_PYTHON, BASELINE_PYTHON,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    write_protocol(configs, args.cache_root, args.edge_case_fallback)
    if args.phase in ("prepare", "all") and not args.dry_run:
        prepare_cache(args.datasets, args.tags, args.cache_root)
    if args.phase in ("run", "all"):
        raise SystemExit(
            run_jobs(
                args.datasets, args.tags, args.methods, gpus,
                args.cache_root, configs, args.dry_run, args.edge_case_fallback,
            )
        )


if __name__ == "__main__":
    main()
