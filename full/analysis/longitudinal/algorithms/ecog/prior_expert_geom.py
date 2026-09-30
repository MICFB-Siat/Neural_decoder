
















from __future__ import annotations
import os, sys, json, glob, re, argparse
import numpy as np, scipy.io as sio
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from build_ecog_geom32k_time_h5 import build_geom_D, ROOT, FS_IN, N_NEURAL, LINE_FREQ, DAYS, CLASSES

WINDOW_SEC = 2.0
N_IN = int(WINDOW_SEC * FS_IN)
CLS2I = {c: i for i, c in enumerate(CLASSES)}
BANDS = [("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30),
         ("gamma", 30, 70), ("highgamma", 70, 150)]
_band_sos = {nm: butter(4, [lo, hi], btype="band", fs=FS_IN, output="sos") for nm, lo, hi in BANDS}
_bp = butter(4, [0.1, 150.0], btype="band", fs=FS_IN, output="sos")


def parse_lab(p):
    out = []
    for ln in open(p):
        ln = ln.strip()
        if not ln: continue
        a = ln.split("\t") if "\t" in ln else ln.split()
        out.append((float(a[0]), float(a[1]), a[2]))
    return out


def find_runs(day):
    pat = os.path.join(ROOT, "KeywordReading", "*", day, "*_trials.lab")
    return sorted(p.replace("_trials.lab", ".mat") for p in glob.glob(pat))


def _clean(mat_path):
    sig = sio.loadmat(mat_path, variable_names=["signal"])["signal"][:, :N_NEURAL].astype(np.float64)
    sig = sig - np.median(sig, axis=1, keepdims=True)
    for h in (1, 2, 3):
        f0 = LINE_FREQ * h
        if f0 < FS_IN / 2 - 5:
            b, a = iirnotch(f0, 30, FS_IN); sig = filtfilt(b, a, sig, axis=0)
    return sig


def _bandpow(sig):

    return {nm: sosfiltfilt(_band_sos[nm], sig, axis=0) ** 2 for nm, _, _ in BANDS}


def features_run(mat_path, mode, P):

    sig = _clean(mat_path)
    bp_raw = _bandpow(sig)
    if mode == "time":
        sig_g = sosfiltfilt(_bp, sig, axis=0) @ P.T
        bp_g = _bandpow(sig_g)
    else:
        bp_g = None
    out = []
    for s, e, label in parse_lab(mat_path.replace(".mat", "_trials.lab")):
        if label not in CLS2I: continue
        i0 = int(round(s * FS_IN)); i1 = i0 + N_IN
        F = np.stack([np.log(bp_raw[nm][i0:i1].mean(0) + 1e-6) for nm, _, _ in BANDS], 1)
        if mode == "time":
            Fg = np.stack([np.log(bp_g[nm][i0:i1].mean(0) + 1e-6) for nm, _, _ in BANDS], 1)
        else:
            Fg = (P @ F)
        out.append((np.concatenate([F, Fg], 1).astype(np.float32), CLS2I[label]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--mode", choices=["time", "bandpower"], default="time")
    ap.add_argument("--k_modes", type=int, default=64, help="几何本征模数 (空间基 Ψ 的列数)")
    ap.add_argument("--shrinkage", default="auto")
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run_dir if os.path.isabs(args.run_dir) else os.path.join(HERE, args.run_dir))
    out_dir = os.path.join(run_dir, f"prior_expert_geom_{args.mode}"); os.makedirs(out_dir, exist_ok=True)
    split = json.load(open(os.path.join(run_dir, "split.json")))["split"]


    D, eigval, _ = build_geom_D(args.k_modes)
    U, S, _ = np.linalg.svd(D, full_matrices=False)
    Psi = U[:, :args.k_modes]
    P = Psi @ Psi.T
    print(f"[geom-prior mode={args.mode}] Ψ_geom k={args.k_modes}  rank(D)~{(S>1e-9*S[0]).sum()}")

    F, y, sess = [], [], []
    for day, dt in DAYS.items():
        for mp in find_runs(day):
            for feat, lab in features_run(mp, args.mode, P):
                F.append(feat); y.append(lab); sess.append(dt)
    F = np.stack(F, 0).reshape(len(y), -1)
    y = np.asarray(y, np.int64); sess = np.asarray(sess, np.int64)
    print(f"features {F.shape}  N={len(y)}")
    tr = np.array(split["train"])

    feat_z = F.copy()
    for d in np.unique(sess):
        m = sess == d; mu = F[m].mean(0); sd = F[m].std(0) + 1e-6; feat_z[m] = (F[m] - mu) / sd

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=args.shrinkage).fit(feat_z[tr], y[tr])
    proba = lda.predict_proba(feat_z)
    np.save(os.path.join(out_dir, "p_prior.npy"), proba.astype(np.float32))
    np.save(os.path.join(out_dir, "labels.npy"), y); np.save(os.path.join(out_dir, "session.npy"), sess)

    test_segs = sorted(((k, int(re.search(r"dt(\d+)", k).group(1))) for k in split if k.startswith("test")),
                       key=lambda x: x[1])
    rows = {}; print(f"\n  Δt   n   prior_acc  (chance=0.167)")
    for seg_key, dt in test_segs:
        idx = np.array(split[seg_key]); acc = float((proba[idx].argmax(1) == y[idx]).mean())
        rows[dt] = {"acc": acc, "n": int(len(idx)), "seg": seg_key}
        print(f"  {dt:3d} {len(idx):3d}   {acc*100:5.1f}")
    json.dump({"run_dir": run_dir, "mode": args.mode, "k_modes": args.k_modes,
               "eigenmode_source": "fsLR_32k_lh", "chance": 1/6, "per_dt": rows},
              open(os.path.join(out_dir, "prior_summary.json"), "w"), indent=2)
    print(f"\n[saved] {out_dir}/p_prior.npy")


if __name__ == "__main__":
    main()
