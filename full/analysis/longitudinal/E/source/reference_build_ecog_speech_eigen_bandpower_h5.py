

















from __future__ import annotations
import os, json, glob
from datetime import date, datetime
import numpy as np, scipy.io as sio
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, resample_poly
from lapy import Solver, TriaMesh

ROOT = "/home/guoyi/a800/share/code/Eigen_brain_decoding/ECoG_speech"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(ROOT, "zhf_work", "code_my", "data_ecog_eigen_bp", f"run_{RUN_ID}")

FS_IN, FS_OUT = 1000, 256
WINDOW_SEC = 2.0
N_SAMP = int(WINDOW_SEC * FS_OUT)
N_IN = int(WINDOW_SEC * FS_IN)
N_NEURAL = 128
GRID_R, GRID_C = 8, 16
K_MODES = round(N_NEURAL * 0.8)
RIDGE_ALPHA = 1e-2
LINE_FREQ = 60.0
BANDPASS = (0.1, 150.0)
ANCHOR = date(2022, 9, 22)
SEED = 0
CLASSES = ["Back", "Down", "Enter", "Left", "Right", "Up"]
CLS2I = {c: i for i, c in enumerate(CLASSES)}
PER_CLASS_TEST = 10
DAYS = {"2022_09_22": 0, "2022_09_23": 1, "2022_09_28": 6, "2022_09_30": 8,
        "2022_10_05": 13, "2022_10_06": 14, "2022_10_10": 18, "2022_10_27": 35,
        "2022_11_03": 42, "2022_11_04": 43,
        "2023_04_14": 204, "2023_04_18": 208, "2023_04_21": 211}
TRAIN_DAYS = {"2022_09_22", "2022_09_23"}


def grid_lb(rows, cols, k):
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    coords = np.stack([xs.ravel().astype(float), ys.ravel().astype(float),
                       np.zeros(rows * cols)], 1)
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            v = r * cols + c
            faces.append([v, v + 1, v + cols]); faces.append([v + 1, v + cols + 1, v + cols])
    faces = np.array(faces)
    solver = Solver(TriaMesh(coords, faces))
    evals, evecs = solver.eigs(k=k + 1)
    phi = np.asarray(evecs[:, 1:k + 1], np.float64)
    ev = np.asarray(evals[1:k + 1], np.float64)
    return phi, ev, coords


def parse_lab(p):
    rows = []
    for ln in open(p):
        ln = ln.strip()
        if not ln: continue
        a = ln.split("\t") if "\t" in ln else ln.split()
        rows.append((float(a[0]), float(a[1]), a[2]))
    return rows


def find_runs(day):
    pat = os.path.join(ROOT, "KeywordReading", "*", day, "*_trials.lab")
    return sorted(p.replace("_trials.lab", ".mat") for p in glob.glob(pat))


from scipy.signal import hilbert
HG_BAND = (70, 150)
_hg = butter(4, list(HG_BAND), btype="band", fs=FS_IN, output="sos")
def preprocess_run(sig):


    x = sig - np.median(sig, axis=1, keepdims=True)
    for h in (1, 2, 3):
        f0 = LINE_FREQ * h
        if f0 < FS_IN / 2 - 5:
            b, a = iirnotch(f0, 30, FS_IN)
            x = filtfilt(b, a, x, axis=0)
    env = np.abs(hilbert(sosfiltfilt(_hg, x, axis=0), axis=0))
    env = resample_poly(env, FS_OUT, FS_IN, axis=0)
    x = np.log(np.maximum(env, 1e-6))
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    return x.astype(np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.RandomState(SEED)
    phi, eigval, gcoords = grid_lb(GRID_R, GRID_C, K_MODES)
    D = phi.astype(np.float64)
    DtD = D.T @ D
    sigma_max = np.linalg.svd(D, compute_uv=False)[0]
    lam = (RIDGE_ALPHA * sigma_max) ** 2
    Rmat = np.linalg.solve(DtD + lam * np.eye(K_MODES), D.T)
    print(f"grid LB: D{D.shape} K={K_MODES} cond={np.linalg.cond(D):.2e} lam={lam:.3e}")

    X, y, sess, day_arr = [], [], [], []
    for day, dt in DAYS.items():
        for mp in find_runs(day):
            sig = sio.loadmat(mp, variable_names=["signal"])["signal"][:, :N_NEURAL].astype(np.float64)
            xpp = preprocess_run(sig)
            for s, e, label in parse_lab(mp.replace(".mat", "_trials.lab")):
                if label not in CLS2I: continue
                i0 = int(round(s * FS_OUT))
                seg = xpp[i0:i0 + N_SAMP]
                if seg.shape[0] < N_SAMP:
                    seg = np.pad(seg, ((0, N_SAMP - seg.shape[0]), (0, 0)), mode="edge")
                X.append(seg.T.astype(np.float32)); y.append(CLS2I[label])
                sess.append(dt); day_arr.append(day)
    X = np.stack(X, 0); y = np.asarray(y, np.int64)
    sess = np.asarray(sess, np.int64); day_arr = np.asarray(day_arr)
    print(f"loaded {len(y)} trials  eeg_raw={X.shape} (preprocessed signal)")
    for day, dt in DAYS.items():
        m = day_arr == day
        print(f"  {day} Δt={dt:2d}: {m.sum()} per-class={np.bincount(y[m], minlength=6)}")


    modes = np.einsum("kc,nct->nkt", Rmat.astype(np.float32), X).astype(np.float32)
    print(f"eeg_modes={modes.shape}  std={modes.std():.3f}")

    def balanced_pick(mask, k):
        idx, pool = [], np.where(mask)[0]
        for c in range(6):
            cand = pool[y[pool] == c]; rng.shuffle(cand); idx.extend(cand[:k].tolist())
        return sorted(int(i) for i in idx)
    tdm = np.isin(day_arr, list(TRAIN_DAYS))
    test_sameday = balanced_pick(tdm, PER_CLASS_TEST); tset = set(test_sameday)
    split = {"train": sorted(int(i) for i in np.where(tdm)[0] if i not in tset),
             "test_dt0_sameday": test_sameday}
    for day, dt in DAYS.items():
        if day in TRAIN_DAYS: continue
        split[f"test_dt{dt}_{day.replace('_', '')[4:]}"] = balanced_pick(day_arr == day, PER_CLASS_TEST)
    for k, v in split.items():
        print(f"  split[{k}]: n={len(v)} per-class={np.bincount(y[v], minlength=6)}")

    gxyz = gcoords.astype(np.float32); gxyz = (gxyz - gxyz.mean(0))
    eeg_pos = np.concatenate([gxyz, np.tile([0, 0, 1.], (N_NEURAL, 1))], 1).astype(np.float32)

    import h5py
    h5_path = os.path.join(OUT_DIR, "sub-01.h5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("eeg_raw", data=X, compression="gzip", compression_opts=4)
        f.create_dataset("eeg_modes", data=modes, compression="gzip", compression_opts=4)
        f.create_dataset("eeg_mode_eigval", data=eigval.astype(np.float32))
        f.create_dataset("eeg_pos", data=eeg_pos)
        f.create_dataset("eeg_sensor_type", data=np.zeros(N_NEURAL, np.int64))
        f.create_dataset("labels", data=y)
        f.create_dataset("session", data=sess)
        f.attrs.update({"subject_id": "sub-01", "modalities": "eeg", "source_modality": "ecog",
                        "n_samples": N_SAMP, "stage": 2, "window_sec": WINDOW_SEC, "indiv": 0,
                        "num_classes": 6, "eeg_sr": FS_OUT, "meg_sr": -1, "fmri_tr": -1.0,
                        "bandpass": np.array(BANDPASS), "line_freq": LINE_FREQ,
                        "modes_method": "ridge", "ridge_alpha": RIDGE_ALPHA, "ridge_lambda": lam,
                        "orig_sr": float(FS_IN), "grid": f"{GRID_R}x{GRID_C}",
                        "class_names": ",".join(CLASSES),
                        "repr_variant": "log_high_gamma_power"})
    cfg = {"run_id": RUN_ID, "fs_in": FS_IN, "fs_out": FS_OUT, "window_sec": WINDOW_SEC,
           "bandpass": BANDPASS, "line_freq": LINE_FREQ, "grid": [GRID_R, GRID_C],
           "k_modes": K_MODES, "ridge_alpha": RIDGE_ALPHA, "ridge_lambda": float(lam),
           "seed": SEED, "classes": CLASSES, "days_dt": DAYS, "train_days": sorted(TRAIN_DAYS),
           "repr": "log_high_gamma_power", "hg_band": HG_BAND, "per_class_test": PER_CLASS_TEST}
    json.dump({"meta": cfg, "split": split}, open(os.path.join(OUT_DIR, "split.json"), "w"),
              indent=2, default=int)
    json.dump(cfg, open(os.path.join(OUT_DIR, "build_config.json"), "w"), indent=2, default=int)
    print(f"\nwrote {h5_path}\nOUT_DIR = {OUT_DIR}")
    with h5py.File(h5_path, "r") as f:
        for k in f: print(f"  {k:18s} {f[k].shape} {f[k].dtype}")


if __name__ == "__main__":
    main()
