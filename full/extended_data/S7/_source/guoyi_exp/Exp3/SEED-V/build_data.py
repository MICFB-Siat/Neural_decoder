

















from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))
import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = str(_LOCAL_SOURCE / '')
sys.path.insert(0, PROJECT_ROOT)
from utils.eigenmode_coef import build_psi

SRC_DIR  = "/media/wsqlab/data/tyf/ugreen/NoLanguageData/SEED-V/preprocess/stage2"
PHI_PATH = str(_LOCAL_SOURCE / 'guoyi_exp/data/eigenmode_test/SEED-V/sensor_geom_eigenmode_template_60x48.npy')
LH_EVAL  = str(_LOCAL_SOURCE / 'guoyi_exp/data/eigenmode_test/fs32k/fsLR_32k_lh_eval_1024.npy')
RH_EVAL  = str(_LOCAL_SOURCE / 'guoyi_exp/data/eigenmode_test/fs32k/fsLR_32k_rh_eval_1024.npy')
RATIO    = 1e-3
SEED     = 0
N_FOLDS  = 5


def compute_E_lambda(phi_path, lh_eval, rh_eval, ratio):

    Phi = np.load(phi_path).astype(np.float64)
    n_sensor, K_cortex = Phi.shape
    half_lh = (K_cortex + 1) // 2
    half_rh = K_cortex // 2
    lh = np.load(lh_eval)[:half_lh].astype(np.float64)
    rh = np.load(rh_eval)[:half_rh].astype(np.float64)
    lam_cortex = np.concatenate([lh, rh])
    _, S, Vt = np.linalg.svd(Phi, full_matrices=False)
    K_eff = int((S / S[0] >= ratio).sum())
    V = Vt.T
    E_lambda = (V[:, :K_eff] ** 2 * lam_cortex[:, None]).sum(axis=0)
    return E_lambda.astype(np.float32), S[:K_eff].astype(np.float32), K_eff

NOISE_ALPHAS = {"noise_03": 0.3, "noise_05": 0.5, "noise_10": 1.0}
DROP_FRACS   = {"drop_01": 0.1, "drop_03": 0.3, "drop_05": 0.5}
RED_FRACS    = {"red_50": 0.5, "red_25": 0.25, "red_10": 0.1}


def stratified_kfold_template(labels: np.ndarray, n_folds: int, seed: int) -> list[dict]:


    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fid, (tr, te) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        folds.append({
            "fold": fid,
            "train_idx": [int(i) for i in tr.tolist()],
            "test_idx":  [int(i) for i in te.tolist()],
            "n_train": int(len(tr)),
            "n_test":  int(len(te)),
        })
    return folds


def save_splits(dst_dir: str, labels: np.ndarray, seed: int):
    folds = stratified_kfold_template(labels, N_FOLDS, seed)
    meta = {
        "n_trials":  int(len(labels)),
        "n_folds":   N_FOLDS,
        "seed":      int(seed),
        "label_dist": {str(int(c)): int((labels==c).sum()) for c in np.unique(labels)},
        "folds":     folds,
    }
    with open(os.path.join(dst_dir, "splits.json"), "w") as f:
        json.dump(meta, f, indent=2)


def copy_h5(fs, fd, raw_new, modes_new, eigval, attrs=None):
    fd.create_dataset("eeg_raw",         data=raw_new.astype(np.float32))
    fd.create_dataset("eeg_modes",       data=modes_new.astype(np.float32))
    fd.create_dataset("eeg_mode_eigval", data=eigval.astype(np.float32))
    fd.create_dataset("eeg_pos",         data=fs["eeg_pos"][:])
    fd.create_dataset("eeg_sensor_type", data=fs["eeg_sensor_type"][:])
    fd.create_dataset("labels",          data=fs["labels"][:])
    if attrs:
        for k, v in attrs.items():
            fd.attrs[k] = v


def write_subset_h5(fs, fd, keep_idx, raw_new, modes_new, eigval, attrs=None):
    fd.create_dataset("eeg_raw",         data=raw_new[keep_idx].astype(np.float32))
    fd.create_dataset("eeg_modes",       data=modes_new[keep_idx].astype(np.float32))
    fd.create_dataset("eeg_mode_eigval", data=eigval.astype(np.float32))
    fd.create_dataset("eeg_pos",         data=fs["eeg_pos"][:])
    fd.create_dataset("eeg_sensor_type", data=fs["eeg_sensor_type"][:])
    fd.create_dataset("labels",          data=fs["labels"][:][keep_idx])
    if attrs:
        for k, v in attrs.items():
            fd.attrs[k] = v


def build_noise(src_files, dst_dir, alpha, Psi, sigma_eig, seed):
    os.makedirs(dst_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    K = Psi.shape[1]
    eigval = sigma_eig.astype(np.float32)
    for fp in src_files:
        sub = os.path.basename(fp)
        with h5py.File(fp, "r") as fs, h5py.File(os.path.join(dst_dir, sub), "w") as fd:
            raw = fs["eeg_raw"][:].astype(np.float32)
            sigma_per_ch = raw.std(axis=(0, 2), keepdims=True)
            noise = rng.standard_normal(raw.shape).astype(np.float32) * (alpha * sigma_per_ch)
            raw_noisy = raw + noise
            modes_new = np.matmul(Psi.T[None, :, :], raw_noisy).astype(np.float32)
            copy_h5(fs, fd, raw_noisy, modes_new, eigval,
                    attrs={"noise_alpha": float(alpha), "eeg_modes_K": int(K),
                           "perturbation": "noise_per_channel_sigma"})

    with h5py.File(src_files[0], "r") as f0:
        save_splits(dst_dir, f0["labels"][:], seed)
    print(f"  [noise α={alpha}] → {dst_dir}  K={K}  splits.json saved")


def build_dropchan(src_files, dst_dir, drop_frac, Psi, sigma_eig, seed, k_neighbors=4):
    os.makedirs(dst_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    K = Psi.shape[1]
    eigval = sigma_eig.astype(np.float32)
    for fp in src_files:
        sub = os.path.basename(fp)
        with h5py.File(fp, "r") as fs, h5py.File(os.path.join(dst_dir, sub), "w") as fd:
            raw = fs["eeg_raw"][:].astype(np.float32)
            pos = fs["eeg_pos"][:]
            xyz = pos[:, :3].astype(np.float64)
            C = xyz.shape[0]
            n_drop = int(round(drop_frac * C))
            dropped = np.sort(rng.choice(C, size=n_drop, replace=False))
            kept = np.setdiff1d(np.arange(C), dropped)
            raw_int = raw.copy()
            for d in dropped:
                dists = np.linalg.norm(xyz[kept] - xyz[d][None, :], axis=1)
                topk = np.argsort(dists)[:k_neighbors]
                w = 1.0 / (dists[topk] + 1e-6); w = w / w.sum()
                raw_int[:, d, :] = (w[None, :, None] * raw[:, kept[topk], :]).sum(axis=1)
            modes_new = np.matmul(Psi.T[None, :, :], raw_int).astype(np.float32)
            copy_h5(fs, fd, raw_int, modes_new, eigval,
                    attrs={"drop_frac": float(drop_frac), "n_drop": int(n_drop),
                           "eeg_modes_K": int(K),
                           "perturbation": "dropchan_invdist_nn_k4"})
    with h5py.File(src_files[0], "r") as f0:
        save_splits(dst_dir, f0["labels"][:], seed)
    print(f"  [drop {drop_frac:.2f}] → {dst_dir}  K={K}  splits.json saved")


def build_reduction(src_files, dst_dir, frac, Psi, sigma_eig, seed):




    os.makedirs(dst_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    K = Psi.shape[1]
    eigval = sigma_eig.astype(np.float32)


    with h5py.File(src_files[0], "r") as f0:
        labels = f0["labels"][:]
    keep_idx = []
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        n_keep = max(1, int(round(frac * len(idx_c))))
        n_keep = min(n_keep, len(idx_c))
        chosen = rng.choice(idx_c, size=n_keep, replace=False)
        keep_idx.append(chosen)
    keep_idx = np.sort(np.concatenate(keep_idx))

    for fp in src_files:
        sub = os.path.basename(fp)
        with h5py.File(fp, "r") as fs, h5py.File(os.path.join(dst_dir, sub), "w") as fd:
            raw = fs["eeg_raw"][:].astype(np.float32)
            modes_new = np.matmul(Psi.T[None, :, :], raw).astype(np.float32)
            write_subset_h5(fs, fd, keep_idx, raw, modes_new, eigval,
                            attrs={"reduction_frac": float(frac),
                                   "n_kept": int(len(keep_idx)),
                                   "eeg_modes_K": int(K),
                                   "perturbation": "stratified_template_by_labels"})

    save_splits(dst_dir, labels[keep_idx], seed)
    print(f"  [reduce frac={frac}] → {dst_dir}  K={K}  kept={len(keep_idx)} trials  splits.json saved")


def build_clean(src_files, dst_dir, Psi, sigma_eig, seed):


    os.makedirs(dst_dir, exist_ok=True)
    K = Psi.shape[1]
    eigval = sigma_eig.astype(np.float32)
    for fp in src_files:
        sub = os.path.basename(fp)
        with h5py.File(fp, "r") as fs, h5py.File(os.path.join(dst_dir, sub), "w") as fd:
            raw = fs["eeg_raw"][:].astype(np.float32)
            modes_new = np.matmul(Psi.T[None, :, :], raw).astype(np.float32)
            copy_h5(fs, fd, raw, modes_new, eigval,
                    attrs={"perturbation": "clean_reprojected_K41",
                           "eeg_modes_K": int(K)})
    with h5py.File(src_files[0], "r") as f0:
        save_splits(dst_dir, f0["labels"][:], seed)
    print(f"  [clean] → {dst_dir}  K={K}  splits.json saved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst_root", required=True,
                    help="parent dir; subdirs noise_*, drop_*, red_* (and clean) will be created.")
    ap.add_argument("--src_dir",  default=SRC_DIR)
    ap.add_argument("--phi_path", default=PHI_PATH)
    ap.add_argument("--lh_eval",  default=LH_EVAL)
    ap.add_argument("--rh_eval",  default=RH_EVAL)
    ap.add_argument("--ratio",    type=float, default=RATIO)
    ap.add_argument("--seed",     type=int,   default=SEED)
    ap.add_argument("--no_clean", action="store_true",
                    help="Skip the clean/ baseline dataset (default: build it).")
    ap.add_argument("--only",     nargs="*",  default=None,
                    help="Only build these subdirs (e.g. noise_03 drop_05).")
    args = ap.parse_args()

    src_files = sorted(glob.glob(os.path.join(args.src_dir, "sub-*.h5")))
    print(f"[src] {len(src_files)} SEED-V subjects in {args.src_dir}")

    Psi, K_eff, sigma = build_psi(args.phi_path, ratio=args.ratio)
    print(f"[Psi] D={args.phi_path}")
    print(f"      ratio={args.ratio}  K_eff={K_eff}  sigma_range=[{sigma[-1]:.3g}..{sigma[0]:.3g}]")


    E_lambda, _, K_check = compute_E_lambda(args.phi_path, args.lh_eval, args.rh_eval, args.ratio)
    assert K_check == K_eff, f"K mismatch: {K_check} vs {K_eff}"
    print(f"[E[λ_cortex]] range=[{E_lambda.min():.4g}..{E_lambda.max():.4g}]  first 5: {E_lambda[:5]}")
    sigma = E_lambda

    plan = list(NOISE_ALPHAS) + list(DROP_FRACS) + list(RED_FRACS)
    if not args.no_clean:
        plan = ["clean"] + plan
    if args.only:
        plan = [n for n in plan if n in args.only]

    os.makedirs(args.dst_root, exist_ok=True)
    for name in plan:
        dst = os.path.join(args.dst_root, name)
        existing = len(glob.glob(os.path.join(dst, "sub-*.h5")))
        if existing == len(src_files) and os.path.isfile(os.path.join(dst, "splits.json")):
            print(f"  [skip] {dst} already populated ({existing} h5 + splits.json)")
            continue
        if name == "clean":
            build_clean(src_files, dst, Psi, sigma, args.seed)
        elif name in NOISE_ALPHAS:
            build_noise(src_files, dst, NOISE_ALPHAS[name], Psi, sigma, args.seed)
        elif name in DROP_FRACS:
            build_dropchan(src_files, dst, DROP_FRACS[name], Psi, sigma, args.seed)
        elif name in RED_FRACS:
            build_reduction(src_files, dst, RED_FRACS[name], Psi, sigma, args.seed)
        else:
            print(f"  [unknown] {name} — skipped")


if __name__ == "__main__":
    main()
