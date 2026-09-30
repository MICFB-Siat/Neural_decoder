








from __future__ import annotations

import json
import hashlib
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import classify_once_per_subject_cache as cached_launcher


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def option(name: str, default: str) -> str:
    if name not in sys.argv:
        return default
    return sys.argv[sys.argv.index(name) + 1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    optimization_seed = int(os.environ.get("OPTIMIZATION_SEED", option("--seed", "0")))
    os.environ.setdefault("PYTHONHASHSEED", str(optimization_seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(optimization_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




    original_load_base = cached_launcher.load_base

    def load_seeded_base(path: Path):
        base = original_load_base(path)
        original_run_one_fold = base.run_one_fold

        def run_one_fold_seeded(*args, **kwargs):
            label = str(kwargs.get("label", args[11] if len(args) > 11 else "fold"))
            digest = hashlib.sha256(
                f"{optimization_seed}:{label}".encode("utf-8")
            ).digest()
            fold_seed = int.from_bytes(digest[:4], "little")
            seed_everything(fold_seed)
            return original_run_one_fold(*args, **kwargs)

        base.run_one_fold = run_one_fold_seeded
        return base

    cached_launcher.load_base = load_seeded_base

    out_dir = Path(option("--out_dir", "."))
    tag = option("--tag", "data")
    modality = option("--modality", "eeg")
    cv_mode = option("--cv_mode", "within")
    train_mode = option("--train_mode", "frozen")
    cache_label = "cache" if "--cache_features" in sys.argv else "nocache"
    run_id = option("--run_id", "same_domain_rerun")
    result_root = (
        out_dir / f"{tag}_{modality}_{cv_mode}_{train_mode}_{cache_label}_{run_id}"
    )

    cached_launcher.main()

    payload = {
        "schema_version": 1,
        "created_at": now(),
        "optimization_seed": optimization_seed,
        "fold_seed": int(option("--seed", "0")),
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "per_fold_seed": "uint32(sha256(f'{optimization_seed}:{subject_fold_label}')[:4])",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__
    }
    result_root.mkdir(parents=True, exist_ok=True)
    tmp = result_root / "determinism.json.partial"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, result_root / "determinism.json")


if __name__ == "__main__":
    main()
