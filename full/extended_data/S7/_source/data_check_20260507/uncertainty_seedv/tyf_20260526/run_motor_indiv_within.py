


















from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))

import os
import sys

PROJECT_ROOT = str(_LOCAL_SOURCE / '')
EXP1_DIR     = os.path.join(PROJECT_ROOT, "guoyi_exp/Exp1")
BRAINOMNI    = os.path.join(PROJECT_ROOT, "weights/BrainOmni/BrainOmni-main")
FULLRUN_DIR  = os.path.join(PROJECT_ROOT, "data_check_20260507/full_run_5fold")
for p in (PROJECT_ROOT, BRAINOMNI, EXP1_DIR, FULLRUN_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import h5py
import torch

import classify_baseline_v2 as base


REQUIRED_H5_KEYS = (
    "eeg_modes_indiv", "eigval_indiv",
    "eeg_raw", "eeg_pos", "eeg_sensor_type",
)


def load_subject_indiv(h5_dir: str, subj: str, n_samples: int,
                       modality: str = "eeg",
                       label_key: str = "labels",
                       override_eigval: torch.Tensor | None = None):






    if modality != "eeg":
        raise ValueError(f"indiv driver only supports modality=eeg, got {modality}")

    h5_path = os.path.join(h5_dir, f"{subj}.h5")
    if not os.path.isfile(h5_path):
        base.log.warning(f"  {subj}: not found — skip")
        return None

    with h5py.File(h5_path, "r") as f:
        for k in REQUIRED_H5_KEYS + (label_key,):
            if k not in f:
                raise KeyError(f"{h5_path}: missing required key '{k}'")
        modes  = torch.tensor(f["eeg_modes_indiv"][:],    dtype=torch.float32)
        raw    = torch.tensor(f["eeg_raw"][:],            dtype=torch.float32)
        labels = torch.tensor(f[label_key][:],            dtype=torch.long)
        pos    = torch.tensor(f["eeg_pos"][:],            dtype=torch.float32)
        stype  = torch.tensor(f["eeg_sensor_type"][:],    dtype=torch.long)
        eigval = torch.tensor(f["eigval_indiv"][:],       dtype=torch.float32)

    if override_eigval is not None:
        base.log.warning(
            f"  {subj}: --d_path / override_eigval is ignored under indiv mode."
        )

    if eigval.shape[0] != modes.shape[1]:
        raise ValueError(
            f"{subj}: eigval_indiv K={eigval.shape[0]} ≠ eeg_modes_indiv K={modes.shape[1]}"
        )

    modes = modes[:, :, :n_samples]
    raw   = raw[:,   :, :n_samples]
    N = len(labels)
    base.log.info(
        f"  {subj}: indiv modes{tuple(modes.shape)}  "
        f"eigval_indiv[{eigval.shape[0]}] "
        f"range=[{eigval.min().item():.3g}..{eigval.max().item():.3g}]"
    )
    return {
        "modes":  modes,
        "raw":    raw,
        "labels": labels,
        "pos":    pos.unsqueeze(0).expand(N, -1, -1).contiguous(),
        "stype":  stype.unsqueeze(0).expand(N, -1).contiguous(),
        "eigval": eigval,
        "subj":   subj,
        "K":      modes.shape[1],
    }


def resolve_shared_eigval_indiv(d_path, lam_cortex_path, ratio,
                                h5_sample_path, modality: str = "eeg"):


    if d_path is not None:
        base.log.warning(
            f"  [indiv mode] --d_path={d_path} ignored "
            f"(per-subject eigval_indiv is loaded directly from h5)."
        )
    base.log.info("  [indiv mode] shared eigval resolution skipped → per-subject indiv eigval.")
    return None, "deferred"


base.load_subject          = load_subject_indiv
base.resolve_shared_eigval = resolve_shared_eigval_indiv


if __name__ == "__main__":
    base.main()
