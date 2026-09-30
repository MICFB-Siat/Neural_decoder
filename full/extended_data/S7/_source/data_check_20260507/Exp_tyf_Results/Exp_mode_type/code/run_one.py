

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER = ROOT.parent / "Exp_Classification" / "code" / "classify_baseline_v2.py"
CKPT = ROOT.parents[2] / "checkpoints" / "unified_v3" / "prior_unified_v3_20260514_020311.pt"
MODE_TYPES = ("laplacian", "spherical_harmonic", "fourier_dct")
PARADIGMS = ("within", "pool")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode-type", required=True, choices=MODE_TYPES)
    ap.add_argument("--paradigm", required=True, choices=PARADIGMS)
    ap.add_argument("--seed", type=int, required=True, choices=range(5))
    ap.add_argument("--device", required=True)
    args = ap.parse_args()

    if not CLASSIFIER.is_file():
        raise FileNotFoundError(f"Classifier not found: {CLASSIFIER}")
    if not CKPT.is_file():
        raise FileNotFoundError(f"Encoder checkpoint not found: {CKPT}")

    data = ROOT / "data" / args.mode_type
    out = ROOT / "result" / args.paradigm / args.mode_type / f"seed_{args.seed:02d}"
    if list(out.glob("*/summary.json")):
        print(f"skip completed {args.paradigm}/{args.mode_type}/seed_{args.seed:02d}")
        return
    out.mkdir(parents=True, exist_ok=True)
    cv_mode = "within" if args.paradigm == "within" else "cross"
    cmd = [
        "python", str(CLASSIFIER), "--h5_dir", str(data),
        "--encoder_ckpt", str(CKPT), "--train_mode", "frozen",
        "--cache_features", "--cv_mode", cv_mode, "--modality", "eeg",
        "--label_key", "labels", "--n_folds", "5", "--epochs", "200",
        "--batch_size", "32", "--lr_adapter", "0.001", "--lr_head", "0.001",
        "--lat_dim", "256", "--dropout", "0.3", "--patience", "30",
        "--kl_weight", "0.001", "--n_classes", "5", "--device", args.device,
        "--seed", str(args.seed),
        "--tag", f"SEEDV_mode_type_{args.mode_type}_{args.paradigm}_seed{args.seed:02d}",
        "--out_dir", str(out), "--run_id", "run", "--d_path",
        str(data / "D_K30.npy"), "--ratio", "1e-12",
        "--lam_cortex_path", str(ROOT / "data" / "lam_cortex_K48.npy"),
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
