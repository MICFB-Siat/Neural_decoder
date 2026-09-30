















from __future__ import annotations
import os, sys, json, glob, re, argparse
from datetime import date
import numpy as np, scipy.io as sio
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ROOT = "/home/guoyi/a800/share/code/Eigen_brain_decoding/ECoG_speech"
FS_IN = 1000
WINDOW_SEC = 2.0
N_IN = int(WINDOW_SEC * FS_IN)
N_NEURAL = 128
LINE_FREQ = 60.0
BANDS = [("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30),
         ("gamma", 30, 70), ("highgamma", 70, 150)]
CLASSES = ["Back", "Down", "Enter", "Left", "Right", "Up"]
CLS2I = {c: i for i, c in enumerate(CLASSES)}
ANCHOR = date(2022, 9, 22)
DAYS = {"2022_09_22": 0, "2022_09_23": 1, "2022_09_28": 6, "2022_09_30": 8,
        "2022_10_05": 13, "2022_10_06": 14, "2022_10_10": 18, "2022_10_27": 35,
        "2022_11_03": 42, "2022_11_04": 43,
        "2023_04_14": 204, "2023_04_18": 208, "2023_04_21": 211}

_band_sos = {nm: butter(4, [lo, hi], btype="band", fs=FS_IN, output="sos") for nm, lo, hi in BANDS}


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


def multiband_features_run(mat_path):

    sig = sio.loadmat(mat_path, variable_names=["signal"])["signal"][:, :N_NEURAL].astype(np.float64)
    sig = sig - np.median(sig, axis=1, keepdims=True)
    for h in (1, 2, 3):
        f0 = LINE_FREQ * h
        if f0 < FS_IN / 2 - 5:
            b, a = iirnotch(f0, 30, FS_IN); sig = filtfilt(b, a, sig, axis=0)

    band_pow = {nm: sosfiltfilt(_band_sos[nm], sig, axis=0) ** 2 for nm, _, _ in BANDS}
    out = []
    for s, e, label in parse_lab(mat_path.replace(".mat", "_trials.lab")):
        if label not in CLS2I: continue
        i0 = int(round(s * FS_IN)); i1 = i0 + N_IN
        feat = np.stack([np.log(band_pow[nm][i0:i1].mean(0) + 1e-6) for nm, _, _ in BANDS], 1)
        out.append((feat.astype(np.float32), label))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="all-days band-power run dir (含 split.json)")
    ap.add_argument("--k_modes", type=int, default=64, help="SVD-U 保留的空间本征模数 (去噪)")
    ap.add_argument("--shrinkage", default="auto")
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run_dir if os.path.isabs(args.run_dir)
                              else os.path.join(os.path.dirname(__file__), args.run_dir))
    out_dir = os.path.join(run_dir, "prior_expert"); os.makedirs(out_dir, exist_ok=True)
    split = json.load(open(os.path.join(run_dir, "split.json")))["split"]


    F_cache = os.path.join(out_dir, "F_multiband.npy")
    yz_cache = os.path.join(out_dir, "F_meta.npz")
    if os.path.exists(F_cache) and os.path.exists(yz_cache):
        F = np.load(F_cache); mz = np.load(yz_cache)
        y, sess = mz["y"], mz["sess"]
        print(f"loaded cached F {F.shape}")
    else:
        F, y, sess = [], [], []
        for day, dt in DAYS.items():
            for mp in find_runs(day):
                for feat, label in multiband_features_run(mp):
                    F.append(feat); y.append(CLS2I[label]); sess.append(dt)
        F = np.stack(F, 0)
        y = np.asarray(y, np.int64); sess = np.asarray(sess, np.int64)
        np.save(F_cache, F); np.savez(yz_cache, y=y, sess=sess)
    N, C, B = F.shape
    print(f"features F={F.shape}  (N trials, 128 ch, {B} bands)")

    tr = np.array(split["train"])
    assert F.shape[0] == 1860, f"trial count mismatch {F.shape[0]} (split assumes 1860)"


    Ftr = F[tr]
    M = Ftr.transpose(1, 0, 2).reshape(C, -1)
    M = M - M.mean(1, keepdims=True)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    Psi = U[:, :args.k_modes]
    P = Psi @ Psi.T
    print(f"Ψ SVD-U: keep k={args.k_modes}/128  (top mode energy {S[0]**2/np.sum(S**2):.2%})")


    F_rec = np.einsum("cd,ndb->ncb", P, F)
    feat = np.concatenate([F, F_rec], axis=2).reshape(N, -1)


    feat_z = feat.copy()
    for d in np.unique(sess):
        m = sess == d
        mu = feat[m].mean(0); sd = feat[m].std(0) + 1e-6
        feat_z[m] = (feat[m] - mu) / sd


    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=args.shrinkage)
    lda.fit(feat_z[tr], y[tr])
    proba_all = lda.predict_proba(feat_z)
    np.save(os.path.join(out_dir, "p_prior.npy"), proba_all.astype(np.float32))
    np.save(os.path.join(out_dir, "feat_z.npy"), feat_z.astype(np.float32))
    np.save(os.path.join(out_dir, "labels.npy"), y)
    np.save(os.path.join(out_dir, "session.npy"), sess)

    test_segs = sorted(((k, int(re.search(r"dt(\d+)", k).group(1)))
                        for k in split if k.startswith("test")), key=lambda x: x[1])
    rows = {}
    print(f"\n  Δt   n   prior_acc  (chance=0.167)")
    for seg_key, dt in test_segs:
        idx = np.array(split[seg_key])
        acc = float((proba_all[idx].argmax(1) == y[idx]).mean())
        rows[dt] = {"acc": acc, "n": int(len(idx)), "seg": seg_key}
        print(f"  {dt:3d} {len(idx):3d}   {acc*100:5.1f}")
    json.dump({"run_dir": run_dir, "k_modes": args.k_modes, "bands": [b[0] for b in BANDS],
               "chance": 1/6, "per_dt": rows},
              open(os.path.join(out_dir, "prior_summary.json"), "w"), indent=2)
    print(f"\n[saved] {out_dir}/p_prior.npy + prior_summary.json")


if __name__ == "__main__":
    main()
