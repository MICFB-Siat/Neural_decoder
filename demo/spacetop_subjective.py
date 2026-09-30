









from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import pearsonr



import sys
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
if "numpy._core.multiarray" not in sys.modules:
    sys.modules["numpy._core.multiarray"] = np.core.multiarray

from models import GaussianPoERegressor


ROOT = Path(__file__).resolve().parent
SUBJECT = "sub-0018"
TASK = "alignvideo"
FOLD = 4
SEED = 46
METHOD = "poe"
TARGET_MODE = "raw"
TARGET_NAMES = (
    "relevance", "happy", "sad", "afraid", "disgusted", "warm", "engaged"
)
EXPECTED_MEAN_PEARSON = 0.421994691865
EXPECTED_PARAMETER_COUNT = 398_997

FEATURES = ROOT / "assets/spacetop_sub0018/sub-0018_features.h5"
LABELS = ROOT / "assets/spacetop_sub0018/sub-0018_manifest.npz"
REFERENCE_PREDICTION = (
    ROOT / "assets/spacetop_sub0018/sub-0018_reference_prediction.npz"
)
CHECKPOINT = ROOT / "weights/spacetop_sub0018/fold-04_checkpoint.pt"
BUNDLE_MANIFEST = ROOT / "assets/SPACETOP_SUB0018_MANIFEST.json"
DEFAULT_OUTPUT = ROOT / "outputs/spacetop_sub0018"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="auto",
        help="auto, cpu, cuda, or a concrete device such as cuda:0",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def validate_bundle() -> dict:
    if not BUNDLE_MANIFEST.is_file():
        raise FileNotFoundError(BUNDLE_MANIFEST)
    manifest = json.loads(BUNDLE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "spacetop-sub0018-fixed-replay-v1":
        raise RuntimeError("unsupported SpaceTop bundle manifest")
    fixed = manifest.get("fixed_test", {})
    expected = {"subject": SUBJECT, "task": TASK, "fold": FOLD, "seed": SEED}
    if fixed != expected:
        raise RuntimeError(f"fixed-test manifest mismatch: {fixed!r}")
    for relative, record in manifest.get("files", {}).items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["size_bytes"]):
            raise RuntimeError(f"size mismatch: {relative}")
        observed = sha256_file(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {relative}")
    return manifest


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def require_equal(name: str, actual, expected) -> None:
    if actual != expected:
        raise RuntimeError(f"{name}: expected {expected!r}, got {actual!r}")


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=np.str_)


def mean_tokens(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 3:
        return values.mean(axis=1, dtype=np.float32)
    if values.ndim == 2:
        return values
    raise RuntimeError(f"expected rank-2/rank-3 features, got {values.shape}")


def validate_checkpoint(checkpoint: dict) -> None:
    require_equal("training_complete", bool(checkpoint["training_complete"]), True)
    require_equal("task", checkpoint["task"], TASK)
    require_equal("target_mode", checkpoint["target_mode"], TARGET_MODE)
    require_equal("method", checkpoint["method"], METHOD)
    require_equal("fold", int(checkpoint["fold"]), FOLD)
    require_equal("seed", int(checkpoint["seed"]), SEED)
    require_equal("outer_test_subjects", checkpoint["outer_test_subjects"], [SUBJECT])
    require_equal("target_names", checkpoint["target_names"], list(TARGET_NAMES))
    parameters = sum(value.numel() for value in checkpoint["model_state"].values())
    require_equal("model parameter count", parameters, EXPECTED_PARAMETER_COUNT)


def load_subject(checkpoint: dict) -> dict[str, np.ndarray]:
    with np.load(LABELS, allow_pickle=False) as archive:
        for key in ("stim_file", "trial_index", "target_values"):
            if key not in archive.files:
                raise KeyError(f"{LABELS}: missing {key}")
        stim_file = decode_strings(archive["stim_file"])
        trial_index = np.asarray(archive["trial_index"], dtype=np.int64)
        target_values = np.asarray(archive["target_values"], dtype=np.float32)
    target_values[target_values < 0] = np.nan

    with h5py.File(FEATURES, "r") as h5:
        for key in ("bci_obs", "bci_prior_indiv"):
            if key not in h5:
                raise KeyError(f"{FEATURES}: missing {key}")
        require_equal("feature subject", str(h5.attrs.get("subject_id", "")), SUBJECT)
        require_equal("feature task", str(h5.attrs.get("task", "")), TASK)
        obs = mean_tokens(h5["bci_obs"][...])
        indiv = mean_tokens(h5["bci_prior_indiv"][...])

        source_hashes = checkpoint["frozen_feature_sources"]["encoder_checkpoint_sha256"]
        require_equal(
            "observation encoder hash",
            str(h5.attrs.get("cifti_ckpt_sha256", "")),
            str(source_hashes["obs"]),
        )
        require_equal(
            "individual encoder hash",
            str(h5.attrs.get("indiv_modes_ckpt_sha256", "")),
            str(source_hashes["indiv"]),
        )
        if "labels_subjective" in h5:
            duplicate_targets = np.asarray(h5["labels_subjective"][...], dtype=np.float32)
            duplicate_targets[duplicate_targets < 0] = np.nan
            if not np.allclose(duplicate_targets, target_values, equal_nan=True):
                raise RuntimeError("H5 ratings and manifest ratings differ")

    n_trials = len(trial_index)
    if not np.array_equal(trial_index, np.arange(n_trials)):
        raise RuntimeError("trial_index must be exactly 0..n-1")
    if target_values.shape != (n_trials, len(TARGET_NAMES)):
        raise RuntimeError(f"unexpected target shape: {target_values.shape}")
    expected_feature_shape = (n_trials, int(checkpoint["input_dim"]))
    if obs.shape != expected_feature_shape or indiv.shape != expected_feature_shape:
        raise RuntimeError(
            f"feature shape mismatch: obs={obs.shape}, indiv={indiv.shape}, "
            f"expected={expected_feature_shape}"
        )
    if len(stim_file) != n_trials:
        raise RuntimeError("stimulus and trial counts differ")
    if not (np.isfinite(obs).all() and np.isfinite(indiv).all()):
        raise RuntimeError("input features contain non-finite values")
    return {
        "obs": obs,
        "indiv": indiv,
        "target_values": target_values,
        "stim_file": stim_file,
        "trial_index": trial_index,
    }


@torch.inference_mode()
def predict(checkpoint: dict, subject: dict[str, np.ndarray],
            device: torch.device) -> dict[str, np.ndarray]:
    config = checkpoint["model_config"]
    model = GaussianPoERegressor(
        d_in=int(checkpoint["input_dim"]),
        d_latent=int(config["latent_dim"]),
        hidden=int(config["hidden"]),
        d_out=len(TARGET_NAMES),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    feature_scalers = checkpoint["feature_scalers"]
    if len(feature_scalers) != 2:
        raise RuntimeError("PoE checkpoint must contain two feature scalers")
    inputs = []
    for values, state in zip((subject["obs"], subject["indiv"]), feature_scalers):
        mean = np.asarray(state["mean"], dtype=np.float32)
        std = np.asarray(state["std"], dtype=np.float32)
        standardized = ((values - mean) / std).astype(np.float32)
        inputs.append(torch.as_tensor(standardized, dtype=torch.float32, device=device))

    scaled_prediction = model(inputs[0], inputs[1], sample=False)["poe"]
    target_scaler = checkpoint["target_scaler"]
    target_mean = np.asarray(target_scaler["mean"], dtype=np.float32)
    target_std = np.asarray(target_scaler["std"], dtype=np.float32)
    prediction = scaled_prediction.cpu().numpy() * target_std + target_mean
    valid = np.isfinite(subject["target_values"])
    return {
        "subject": np.full(len(subject["trial_index"]), SUBJECT, dtype=np.str_),
        "trial_index": subject["trial_index"].copy(),
        "stim_file": subject["stim_file"].copy(),
        "target_values_raw": subject["target_values"].copy(),
        "valid_mask": valid.astype(np.uint8),
        "prediction_raw_direct": prediction.astype(np.float32),
    }


def compare_reference(prediction: dict[str, np.ndarray],
                      tolerance: float) -> float:
    with np.load(REFERENCE_PREDICTION, allow_pickle=False) as archive:
        for key in ("trial_index", "stim_file", "valid_mask"):
            if not np.array_equal(archive[key], prediction[key]):
                raise RuntimeError(f"reference prediction mismatch: {key}")
        difference = np.abs(
            np.asarray(archive["prediction_raw_direct"], dtype=np.float32)
            - prediction["prediction_raw_direct"]
        )
    maximum = float(difference.max())
    if maximum > tolerance:
        raise RuntimeError(
            f"reference prediction difference {maximum} exceeds {tolerance}"
        )
    return maximum


def compute_metrics(prediction: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    truth = prediction["target_values_raw"]
    predicted = prediction["prediction_raw_direct"]
    valid = prediction["valid_mask"].astype(bool)
    for index, target in enumerate(TARGET_NAMES):
        mask = valid[:, index] & np.isfinite(predicted[:, index])
        observed = truth[mask, index].astype(float)
        estimate = predicted[mask, index].astype(float)
        if len(observed) < 3 or np.std(observed) == 0 or np.std(estimate) == 0:
            correlation = math.nan
        else:
            correlation = float(pearsonr(observed, estimate).statistic)
        rows.append({
            "subject": SUBJECT,
            "fold": f"fold-{FOLD:02d}",
        "target": target,
            "n": int(mask.sum()),
            "pearson_r": correlation,
        })
    return rows


def save_outputs(output_dir: Path, prediction: dict[str, np.ndarray],
                 rows: list[dict], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    prediction_path = output_dir / f"{SUBJECT}_predictions.npz"
    with tempfile.NamedTemporaryFile(
        prefix="spacetop_prediction_", suffix=".npz", dir="/tmp"
    ) as temporary:
        np.savez_compressed(temporary.name, **prediction)
        staged = prediction_path.with_name(f".{prediction_path.name}.tmp.{os.getpid()}")
        shutil.copyfile(temporary.name, staged)
        os.replace(staged, prediction_path)
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    tolerance = 1e-6 if device.type == "cuda" else 2e-5

    validate_bundle()
    checkpoint = load_checkpoint(CHECKPOINT, device)
    validate_checkpoint(checkpoint)
    subject = load_subject(checkpoint)
    prediction = predict(checkpoint, subject, device)
    reference_difference = compare_reference(prediction, tolerance)
    rows = compute_metrics(prediction)

    pearson_values = np.asarray([row["pearson_r"] for row in rows], dtype=float)
    if not np.isfinite(pearson_values).all():
        raise RuntimeError("all seven Pearson correlations must be finite")
    mean_pearson = float(pearson_values.mean())
    if abs(mean_pearson - EXPECTED_MEAN_PEARSON) > 1e-6:
        raise RuntimeError(f"unexpected mean Pearson: {mean_pearson:.12f}")

    summary = {
        "status": "PASS",
        "inference_only": True,
        "subject": SUBJECT,
        "task": TASK,
        "fold": f"fold-{FOLD:02d}",
        "device": str(device),
        "reference_max_abs_prediction_difference": reference_difference,
        "reference_tolerance": tolerance,
        "seven_target_arithmetic_mean_pearson_r": mean_pearson,
        "metrics": rows,
    }
    save_outputs(args.output_dir, prediction, rows, summary)

    print(
        f"PASS inference-only subject={SUBJECT} fold=fold-{FOLD:02d} "
        f"device={device}"
    )
    print(f"{'target':<12} {'n':>3} {'Pearson r':>12}")
    for row in rows:
        print(f"{row['target']:<12} {row['n']:>3d} {row['pearson_r']:>12.9f}")
    print(f"{'mean(7)':<12} {'':>3} {mean_pearson:>12.9f}")
    print(f"reference_max_abs_prediction_difference={reference_difference:.3g}")
    print(f"outputs={args.output_dir}")


if __name__ == "__main__":
    main()
