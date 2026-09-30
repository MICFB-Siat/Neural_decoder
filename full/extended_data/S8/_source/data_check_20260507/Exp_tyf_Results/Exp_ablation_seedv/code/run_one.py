

from pathlib import Path
import argparse
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, choices=(1, 3, 5, 10, 20, 40, 48))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", required=True)
    args = ap.parse_args()
    data = ROOT / "data" / f"K{args.k}"
    out = ROOT / "result" / f"K{args.k}" / f"seed_{args.seed:02d}"
    completed = list(out.glob("*/summary.json"))
    if completed:
        print(f"skip completed {completed[0]}")
        return
    out.mkdir(parents=True, exist_ok=True)
    script = ROOT.parent / "Exp_Classification" / "code" / "classify_baseline_v2.py"
    ckpt = PROJECT / "checkpoints/unified_v3/prior_unified_v3_20260514_020311.pt"
    cmd = [
        "python", str(script), "--h5_dir", str(data),
        "--encoder_ckpt", str(ckpt), "--train_mode", "frozen",
        "--cache_features", "--cv_mode", "within", "--modality", "eeg",
        "--label_key", "labels", "--n_folds", "5", "--epochs", "200",
        "--batch_size", "32", "--lr_adapter", "0.001", "--lr_head", "0.001",
        "--lat_dim", "256", "--dropout", "0.3", "--patience", "30",
        "--kl_weight", "0.001", "--n_classes", "5", "--device", args.device,
        "--seed", str(args.seed), "--tag", f"SEEDV_ablation_K{args.k}_seed{args.seed:02d}",
        "--out_dir", str(out), "--run_id", "run", "--d_path",
        str(data / f"D_rank{args.k}.npy"), "--ratio", "1e-12",
        "--lam_cortex_path", str(ROOT / "data" / "cortex_eigval_48.npy"),
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
