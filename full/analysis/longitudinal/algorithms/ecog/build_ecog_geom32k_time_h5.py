


















from __future__ import annotations
import os, json, glob
from datetime import date, datetime
import numpy as np, scipy.io as sio
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, resample_poly
from scipy.spatial import cKDTree
import nibabel as nib

ROOT = "/home/guoyi/a800/share/code/Eigen_brain_decoding/ECoG_speech"
EMODE_DIR = "/home/guoyi/a800/share/code/Eigen_brain_decoding/guoyi_exp/data/eigenmode_test/fs32k"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(ROOT, "zhf_work", "code_my", "data_ecog_geom32k_time", f"run_{RUN_ID}")

FS_IN, FS_OUT = 1000, 256
WINDOW_SEC = 2.0
N_SAMP = int(WINDOW_SEC * FS_OUT)
N_NEURAL = 128
K_MODES = 102
RIDGE_ALPHA = 1e-2
LINE_FREQ = 60.0
BANDPASS = (0.1, 150.0)
SEED = 0
CLASSES = ["Back", "Down", "Enter", "Left", "Right", "Up"]
CLS2I = {c: i for i, c in enumerate(CLASSES)}
PER_CLASS_TEST = 10
DAYS = {"2022_09_22": 0, "2022_09_23": 1, "2022_09_28": 6, "2022_09_30": 8,
        "2022_10_05": 13, "2022_10_06": 14, "2022_10_10": 18, "2022_10_27": 35,
        "2022_11_03": 42, "2022_11_04": 43,
        "2023_04_14": 204, "2023_04_18": 208, "2023_04_21": 211}
TRAIN_DAYS = {"2022_09_22", "2022_09_23"}


GRID_R, GRID_C, PITCH = 8, 8, 4.0
GRID_SEEDS = {"hand": (-38., -18., 52.), "face": (-58., -8., 28.)}



def _vertex_normals(coords, faces):
    n = np.zeros_like(coords)
    tris = coords[faces]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    nrm = np.linalg.norm(n, axis=1, keepdims=True); nrm[nrm == 0] = 1
    return n / nrm


def place_grid(coords, tree, seed_xyz, rows, cols, pitch):


    seed_xyz = np.asarray(seed_xyz, float)
    _, s0 = tree.query(seed_xyz)
    center = coords[s0]

    nb = tree.query_ball_point(center, r=20.0)
    P = coords[nb] - center
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    u, v = Vt[0], Vt[1]
    idx = []
    for i in range(rows):
        for j in range(cols):
            off = (i - (rows - 1) / 2) * pitch * u + (j - (cols - 1) / 2) * pitch * v
            _, vi = tree.query(center + off)
            idx.append(int(vi))
    return np.array(idx, np.int64)


def build_geom_D(k_modes):


    g = nib.load(os.path.join(EMODE_DIR, "S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii"))
    coords = np.asarray(g.darrays[0].data, np.float64)
    faces = np.asarray(g.darrays[1].data, np.int64)
    emode = np.load(os.path.join(EMODE_DIR, "fsLR_32k_lh_emode_1024.npy"))
    eval_ = np.load(os.path.join(EMODE_DIR, "fsLR_32k_lh_eval_1024.npy"))
    vnorm = _vertex_normals(coords, faces)
    tree = cKDTree(coords)
    vtx = np.concatenate([place_grid(coords, tree, GRID_SEEDS["hand"], GRID_R, GRID_C, PITCH),
                          place_grid(coords, tree, GRID_SEEDS["face"], GRID_R, GRID_C, PITCH)])
    assert len(vtx) == N_NEURAL, len(vtx)
    D = emode[vtx, 1:1 + k_modes].astype(np.float64)
    eigval = eval_[1:1 + k_modes].astype(np.float64)
    xyz = coords[vtx]; xyz = xyz - xyz.mean(0)
    pos6 = np.concatenate([xyz, vnorm[vtx]], 1).astype(np.float32)
    uniq = len(set(vtx.tolist()))
    print(f"[geom] grids placed: {len(vtx)} electrodes, {uniq} unique vertices "
          f"(dup={len(vtx)-uniq}); D{D.shape} eigval[:3]={np.round(eigval[:3],5)}")
    return D, eigval, pos6



_bp = butter(4, list(BANDPASS), btype="band", fs=FS_IN, output="sos")
def preprocess_run(sig):

    x = sig - np.median(sig, axis=1, keepdims=True)
    for h in (1, 2, 3):
        f0 = LINE_FREQ * h
        if f0 < FS_IN / 2 - 5:
            b, a = iirnotch(f0, 30, FS_IN); x = filtfilt(b, a, x, axis=0)
    x = sosfiltfilt(_bp, x, axis=0)
    x = resample_poly(x, FS_OUT, FS_IN, axis=0)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    return x.astype(np.float32)


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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.RandomState(SEED)
    D, eigval, pos6 = build_geom_D(K_MODES)
    sigma_max = np.linalg.svd(D, compute_uv=False)[0]
    lam = (RIDGE_ALPHA * sigma_max) ** 2
    Rmat = np.linalg.solve(D.T @ D + lam * np.eye(K_MODES), D.T)
    print(f"ridge: lam={lam:.3e}  cond(D)={np.linalg.cond(D):.2e}")

    X, y, sess, day_arr = [], [], [], []
    for day, dt in DAYS.items():
        for mp in find_runs(day):
            sig = sio.loadmat(mp, variable_names=["signal"])["signal"][:, :N_NEURAL].astype(np.float64)
            xpp = preprocess_run(sig)
            for s, e, label in parse_lab(mp.replace(".mat", "_trials.lab")):
                if label not in CLS2I: continue
                i0 = int(round(s * FS_OUT)); seg = xpp[i0:i0 + N_SAMP]
                if seg.shape[0] < N_SAMP:
                    seg = np.pad(seg, ((0, N_SAMP - seg.shape[0]), (0, 0)), mode="edge")
                X.append(seg.T.astype(np.float32)); y.append(CLS2I[label])
                sess.append(dt); day_arr.append(day)
    X = np.stack(X, 0); y = np.asarray(y, np.int64)
    sess = np.asarray(sess, np.int64); day_arr = np.asarray(day_arr)
    print(f"loaded {len(y)} trials  eeg_raw={X.shape} (TIME-DOMAIN broadband)")
    for day, dt in DAYS.items():
        m = day_arr == day
        print(f"  {day} dt={dt:3d}: {int(m.sum())} per-class={np.bincount(y[m], minlength=6)}")

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

    import h5py
    h5_path = os.path.join(OUT_DIR, "sub-01.h5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("eeg_raw", data=X, compression="gzip", compression_opts=4)
        f.create_dataset("eeg_modes", data=modes, compression="gzip", compression_opts=4)
        f.create_dataset("eeg_mode_eigval", data=eigval.astype(np.float32))
        f.create_dataset("eeg_pos", data=pos6)
        f.create_dataset("eeg_sensor_type", data=np.zeros(N_NEURAL, np.int64))
        f.create_dataset("labels", data=y)
        f.create_dataset("session", data=sess)
        f.attrs.update({"subject_id": "sub-01", "modalities": "eeg", "source_modality": "ecog",
                        "n_samples": N_SAMP, "stage": 2, "window_sec": WINDOW_SEC, "indiv": 0,
                        "num_classes": 6, "eeg_sr": FS_OUT, "meg_sr": -1, "fmri_tr": -1.0,
                        "bandpass": np.array(BANDPASS), "line_freq": LINE_FREQ,
                        "modes_method": "ridge", "ridge_alpha": RIDGE_ALPHA, "ridge_lambda": lam,
                        "orig_sr": float(FS_IN), "grid": f"2x{GRID_R}x{GRID_C}", "grid_pitch_mm": PITCH,
                        "class_names": ",".join(CLASSES),
                        "repr_variant": "time_domain_broadband",
                        "eigenmode_source": "fsLR_32k_lh_LBO (sampled at estimated electrode vertices)"})
    cfg = {"run_id": RUN_ID, "fs_in": FS_IN, "fs_out": FS_OUT, "window_sec": WINDOW_SEC,
           "bandpass": BANDPASS, "line_freq": LINE_FREQ, "grid": [2, GRID_R, GRID_C],
           "grid_pitch_mm": PITCH, "grid_seeds_mni": GRID_SEEDS, "k_modes": K_MODES,
           "ridge_alpha": RIDGE_ALPHA, "ridge_lambda": float(lam), "seed": SEED, "classes": CLASSES,
           "days_dt": DAYS, "train_days": sorted(TRAIN_DAYS), "repr": "time_domain_broadband",
           "eigenmode_source": "fsLR_32k_lh", "per_class_test": PER_CLASS_TEST}
    json.dump({"meta": cfg, "split": split}, open(os.path.join(OUT_DIR, "split.json"), "w"),
              indent=2, default=int)
    json.dump(cfg, open(os.path.join(OUT_DIR, "build_config.json"), "w"), indent=2, default=int)
    print(f"\nwrote {h5_path}\nOUT_DIR = {OUT_DIR}")
    with h5py.File(h5_path, "r") as f:
        for k in f: print(f"  {k:18s} {f[k].shape} {f[k].dtype}")


if __name__ == "__main__":
    main()
