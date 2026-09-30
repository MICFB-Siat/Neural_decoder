








from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

from edge_case_splits import FallbackAudit, make_safe_splitters


DEFAULT_BASE = Path(
    str(_LOCAL_SOURCE / 'guoyi_exp/Exp1/classify_baseline_v2.py')
)


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("audited_uncertainty_classifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import classifier: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cli_value(name: str, default: str | None = None) -> str | None:
    if name not in sys.argv:
        return default
    index = sys.argv.index(name)
    return sys.argv[index + 1]


def main() -> None:
    os.environ.setdefault("CUDA_HOME", "/home/guoyi/anaconda3/envs/BrainOmni")
    base_path = Path(os.environ.get("UNCERTAINTY_BASE_CLASSIFIER", str(DEFAULT_BASE)))
    base = load_base(base_path)
    out_dir = Path(cli_value("--out_dir", ".") or ".")
    tag = cli_value("--tag", "data") or "data"
    run_id = cli_value("--run_id", base.RUN_ID) or base.RUN_ID
    modality = cli_value("--modality", "eeg") or "eeg"
    cv_mode = cli_value("--cv_mode", "within") or "within"
    train_mode = cli_value("--train_mode", "frozen") or "frozen"
    cache_str = "cache" if "--cache_features" in sys.argv else "nocache"
    result_root = out_dir / f"{tag}_{modality}_{cv_mode}_{train_mode}_{cache_str}_{run_id}"
    audit = FallbackAudit(result_root / "edge_case_fallback.json", "eigen")
    SafeStratifiedKFold, _ = make_safe_splitters(audit)
    base.StratifiedKFold = SafeStratifiedKFold

    original_temp_fusion = base._temp_scaled_fusion

    def safe_temp_fusion(z_obs, z_pri, y, n_classes, seed=0):
        A, B = base._stratified_half_split(y, n_classes, seed=seed)
        A = np.asarray(A, dtype=np.int64)
        B = np.asarray(B, dtype=np.int64)
        if len(A) > 0 and len(B) > 0:
            return original_temp_fusion(z_obs, z_pri, y, n_classes, seed=seed)

        audit.record(
            "temperature_neutral_t1",
            reason="split-half cross-fitting produced an empty calibration half",
            n_samples=int(len(y)),
            label_counts={str(v): int((np.asarray(y) == v).sum()) for v in np.unique(y)},
            half_a_size=int(len(A)),
            half_b_size=int(len(B)),
            seed=int(seed),
        )
        zo = torch.from_numpy(np.asarray(z_obs)).float()
        zp = torch.from_numpy(np.asarray(z_pri)).float()
        fused = base.F.log_softmax(zo, dim=-1) + base.F.log_softmax(zp, dim=-1)
        preds = fused.argmax(-1).numpy()
        y_np = np.asarray(y, dtype=np.int64)
        acc = float((preds == y_np).mean())
        balacc = float(base.balanced_accuracy_score(y_np, preds))
        recall, _ = base._per_class_recall(y_np, preds, n_classes)
        return {
            "poe_temp": acc,
            "poe_temp_balacc": balacc,
            "poe_temp_recall": recall,
            "temp_T_obs_A": 1.0,
            "temp_T_pri_A": 1.0,
            "temp_T_obs_B": 1.0,
            "temp_T_pri_B": 1.0,
            "temp_T_obs_mean": 1.0,
            "temp_T_pri_mean": 1.0,
            "temp_fallback": "neutral_T1_empty_crossfit_half",
        }

    base._temp_scaled_fusion = safe_temp_fusion
    original = base.precompute_features
    cache = {"bundle": None, "obs": None, "prior": None, "labels": None}

    def precompute_once(bundle, idx, backbone, prior_enc, n_tok, eigval, device, batch_size=64):
        if cache["bundle"] is not bundle:
            full_idx = np.arange(len(bundle["labels"]), dtype=np.int64)
            base.log.info(
                "  [subject-cache] computing frozen features once for all %d trials",
                len(full_idx),
            )
            obs, prior, labels = original(
                bundle,
                full_idx,
                backbone,
                prior_enc,
                n_tok,
                eigval,
                device,
                batch_size,
            )
            cache.update(bundle=bundle, obs=obs, prior=prior, labels=labels)
        take = torch.as_tensor(idx, dtype=torch.long)
        return cache["obs"][take], cache["prior"][take], cache["labels"][take]

    base.precompute_features = precompute_once
    try:
        base.main()
    finally:
        audit.write()


if __name__ == "__main__":
    main()
