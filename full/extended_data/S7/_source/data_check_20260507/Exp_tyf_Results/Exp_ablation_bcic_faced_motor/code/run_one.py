

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from experiment_config import (
    ALL_KS,
    CHECKPOINT,
    CLASSIFIER,
    CONFIRMED_KS,
    DATASETS,
    DATA_ROOT,
    RESULT_STAGE_ROOT,
    ROOT,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_file_verified(src: Path, dst: Path, attempts: int = 8) -> str:

    dst.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        partial = dst.with_name(f".{dst.name}.sync-{os.getpid()}.partial")
        try:
            src_size = src.stat().st_size
            src_hash = sha256(src)
            if dst.is_file() and dst.stat().st_size == src_size:
                if sha256(dst) == src_hash:
                    return "reused"
            partial.unlink(missing_ok=True)
            with src.open("rb") as r, partial.open("wb") as w:
                shutil.copyfileobj(r, w, length=8 * 1024 * 1024)
                w.flush()
                os.fsync(w.fileno())
            if partial.stat().st_size != src_size:
                raise OSError(
                    f"size mismatch: {partial.stat().st_size} != {src_size}"
                )
            if sha256(partial) != src_hash:
                raise OSError(f"sha256 mismatch while copying {src}")
            os.replace(partial, dst)
            return "copied"
        except (OSError, RuntimeError) as exc:
            last_error = exc
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt < attempts:
                time.sleep(min(2 * attempt, 15))
    raise RuntimeError(f"failed to copy {src} -> {dst}: {last_error}")


def sync_tree(src_root: Path, dst_root: Path) -> dict:

    if not src_root.is_dir():
        return {"copied": 0, "reused": 0}
    files = [p for p in src_root.rglob("*") if p.is_file()]
    files = [p for p in files if ".partial" not in p.name]
    files.sort(key=lambda p: (p.name == "summary.json", str(p)))
    stats = {"copied": 0, "reused": 0}
    for src in files:
        status = copy_file_verified(src, dst_root / src.relative_to(src_root))
        stats[status] += 1
    return stats


def completed_bundle(seed_out: Path, expected_folds: int,
                     expected_subjects: int) -> Path | None:
    summaries = list(seed_out.glob("*/summary.json"))
    if not summaries:
        return None
    if len(summaries) != 1:
        raise RuntimeError(f"expected one summary under {seed_out}, got {summaries}")
    summary = summaries[0]
    doc = json.loads(summary.read_text())
    if len(doc.get("per_subject", {})) != expected_subjects:
        raise RuntimeError(
            f"{summary}: subjects={len(doc.get('per_subject', {}))}, "
            f"expected={expected_subjects}"
        )
    metrics = list(summary.parent.glob("sub-*/fold_*/metrics.json"))
    if len(metrics) != expected_folds:
        raise RuntimeError(
            f"{summary.parent}: metrics={len(metrics)}, expected={expected_folds}"
        )
    for path in metrics:
        json.loads(path.read_text())
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    ap.add_argument("--k", required=True, type=int, choices=ALL_KS)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    if args.k not in CONFIRMED_KS[args.dataset]:
        ap.error(
            f"K={args.k} is outside the confirmed grid for {args.dataset}: "
            f"{CONFIRMED_KS[args.dataset]}"
        )
    data = DATA_ROOT / args.dataset / f"K{args.k}"
    shared_out = (
        ROOT / "result" / args.dataset / f"K{args.k}" / f"seed_{args.seed:02d}"
    )
    shared_summaries = list(shared_out.glob("*/summary.json"))
    if shared_summaries:
        print(f"skip completed {shared_summaries[0]}", flush=True)
        return

    template = np.load(cfg["template"])
    lam = DATA_ROOT / args.dataset / f"fsLR_32k_group_eval_{template.shape[1]}.npy"
    d_rank = data / f"D_rank{args.k}.npy"
    subjects = sorted(data.glob("sub-*.h5"))
    if not subjects or not d_rank.is_file() or not lam.is_file():
        raise FileNotFoundError(f"incomplete data bundle: {data}")
    with h5py.File(subjects[0], "r") as f:
        if f["eeg_modes"].shape[1] != args.k:
            raise ValueError(f"{subjects[0]} does not contain K={args.k}")
        if cfg["label_key"] not in f:
            raise KeyError(f"{subjects[0]} missing {cfg['label_key']}")

    expected_folds = len(subjects) * 5
    local_out = (
        RESULT_STAGE_ROOT
        / args.dataset / f"K{args.k}" / f"seed_{args.seed:02d}"
    )
    local_out.mkdir(parents=True, exist_ok=True)
    if completed_bundle(local_out, expected_folds, len(subjects)) is None:
        staged = sync_tree(shared_out, local_out)
        print(
            f"staged existing results {shared_out} -> {local_out}: {staged}",
            flush=True,
        )

    tag = f"{args.dataset}_ablation_K{args.k}_seed{args.seed:02d}"
    cmd = [
        sys.executable, "-u", str(CLASSIFIER),
        "--h5_dir", str(data),
        "--encoder_ckpt", str(CHECKPOINT),
        "--train_mode", "frozen", "--cache_features",
        "--cv_mode", "within", "--modality", "eeg",
        "--label_key", cfg["label_key"], "--n_folds", "5",
        "--epochs", "200", "--batch_size", "32",
        "--lr_adapter", "0.001", "--lr_head", "0.001",
        "--lat_dim", "256", "--dropout", "0.3", "--patience", "30",
        "--kl_weight", "0.001", "--n_classes", str(cfg["n_classes"]),
        "--device", args.device, "--seed", str(args.seed),
        "--tag", tag, "--out_dir", str(local_out), "--run_id", "run",
        "--d_path", str(d_rank), "--ratio", "1e-12",
        "--lam_cortex_path", str(lam),
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    cache_root = Path("/tmp") / "eigenmode_ablation_bfm_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    env.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    env.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_root / "torchinductor"))
    if completed_bundle(local_out, expected_folds, len(subjects)) is None:
        subprocess.run(cmd, check=True, env=env)

    local_summary = completed_bundle(local_out, expected_folds, len(subjects))
    if local_summary is None:
        raise RuntimeError(f"local run did not produce a complete bundle: {local_out}")
    synced = sync_tree(local_out, shared_out)
    shared_summary = completed_bundle(shared_out, expected_folds, len(subjects))
    if shared_summary is None:
        raise RuntimeError(f"sync did not produce a complete bundle: {shared_out}")
    print(
        f"verified sync {local_summary} -> {shared_summary}: {synced}",
        flush=True,
    )


if __name__ == "__main__":
    main()
