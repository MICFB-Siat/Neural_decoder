

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
STAGE2 = PROJECT / "data_check_20260507/full_run_5fold/stage2_new"
CLASSIFIER = ROOT.parent / "Exp_Classification/code/classify_baseline_v2.py"
INIT_CHECKPOINT = PROJECT / "checkpoints/unified_v3/prior_unified_v3_20260514_020311.pt"



EXPECTED_CLASSIFIER_SHA256 = "61330ea2f91c1d1e6b1af97a128273fb3c2da7c5e4b08496461f8fc9b193530a"

MODE_TYPES = ("fourier_dct", "spherical_harmonic", "laplacian")
MODE_LABELS = {
    "fourier_dct": "DCT-II Fourier basis",
    "spherical_harmonic": "real spherical-harmonic basis",
    "laplacian": "6-NN normalized graph-Laplacian basis",
}
DATASET_ORDER = ("BCIC", "MOTOR", "FACED")
SEEDS = tuple(range(10))
N_FOLDS = 5



DATASETS = {
    "BCIC": {
        "source": STAGE2 / "BCIC",
        "label_key": "labels",
        "n_classes": 4,
        "n_sensors": 22,
        "n_modes": 22,
        "n_samples": 1024,
        "expected_subjects": 9,
    },
    "FACED": {
        "source": STAGE2 / "FACED_new",
        "label_key": "label2",
        "n_classes": 3,
        "n_sensors": 32,
        "n_modes": 30,
        "n_samples": 2560,
        "expected_subjects": 123,
    },
    "MOTOR": {
        "source": STAGE2 / "MOTOR",
        "label_key": "labels",
        "n_classes": 2,
        "n_sensors": 63,
        "n_modes": 30,
        "n_samples": 2560,
        "expected_subjects": 10,
    },
}

DATA_ROOT = ROOT / "data"
PRETRAIN_ROOT = ROOT / "pretrain"
RESULT_ROOT = ROOT / "result"
STATE_ROOT = ROOT / "state"

CLASSIFIER_ARGS = {
    "train_mode": "frozen",
    "cache_features": True,
    "n_folds": N_FOLDS,
    "epochs": 200,
    "batch_size": 32,
    "lr_adapter": 1e-3,
    "lr_head": 1e-3,
    "lat_dim": 256,
    "dropout": 0.3,
    "patience": 30,
    "kl_weight": 1e-3,
}

ADAPT_ARGS = {
    "epochs": 50,
    "batch_size": 64,
    "lr": 3e-4,
    "seed": 0,
    "mask_ratio": 0.5,
    "val_ratio": 0.1,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assert_classifier_frozen() -> str:
    actual = sha256_file(CLASSIFIER)
    if actual != EXPECTED_CLASSIFIER_SHA256:
        raise RuntimeError(
            "Shared classifier changed after protocol audit: "
            f"expected {EXPECTED_CLASSIFIER_SHA256}, got {actual} ({CLASSIFIER})"
        )
    return actual


def data_dir(dataset: str, mode_type: str) -> Path:
    return DATA_ROOT / dataset / mode_type


def checkpoint_path(dataset: str, mode_type: str) -> Path:
    return PRETRAIN_ROOT / dataset / mode_type / f"prior_{dataset}_{mode_type}_mae50.pt"


def seed_out_dir(dataset: str, mode_type: str, paradigm: str, seed: int) -> Path:
    return RESULT_ROOT / dataset / mode_type / paradigm / f"seed_{seed:02d}"


def run_tag(dataset: str, mode_type: str, paradigm: str, seed: int) -> str:
    return f"{dataset}_{mode_type}_matched_{paradigm}_seed{seed:02d}"


def run_root(dataset: str, mode_type: str, paradigm: str, seed: int) -> Path:
    cv_mode = "within" if paradigm == "within" else "cross"
    tag = run_tag(dataset, mode_type, paradigm, seed)
    return seed_out_dir(dataset, mode_type, paradigm, seed) / f"{tag}_eeg_{cv_mode}_frozen_cache_run"
