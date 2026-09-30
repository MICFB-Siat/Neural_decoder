







from __future__ import annotations
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from prior_expert_geom import build_geom_D, find_runs, features_run
from build_ecog_geom32k_time_h5 import DAYS

RUN = "data_ecog_eigen_bp/run_20260529_215039"
OUT = os.path.join(HERE, RUN, "feat_similarity"); os.makedirs(OUT, exist_ok=True)
K_MODES = 64
MODE = "time"
N_CLS = 6

DATE = {0:"2022-09-22",1:"2022-09-23",6:"2022-09-28",8:"2022-09-30",13:"2022-10-05",
        14:"2022-10-06",18:"2022-10-10",35:"2022-10-27",42:"2022-11-03",43:"2022-11-04",
        204:"2023-04-14",208:"2023-04-18",211:"2023-04-21"}


def main():
    D, _, _ = build_geom_D(K_MODES)
    U, S, _ = np.linalg.svd(D, full_matrices=False)
    Psi = U[:, :K_MODES]; P = Psi @ Psi.T
    F, y, sess = [], [], []
    for day, dt in DAYS.items():
        for mp in find_runs(day):
            for feat, lab in features_run(mp, MODE, P):
                F.append(feat); y.append(lab); sess.append(dt)
    F = np.stack(F, 0).reshape(len(y), -1); y = np.asarray(y); sess = np.asarray(sess)
    print("features", F.shape)

    Fz = F.copy()
    for d in np.unique(sess):
        m = sess == d; mu = F[m].mean(0); sd = F[m].std(0) + 1e-6; Fz[m] = (F[m] - mu) / sd

    days = sorted(np.unique(sess).tolist())

    cent = np.full((len(days), N_CLS, Fz.shape[1]), np.nan)
    for di, d in enumerate(days):
        for c in range(N_CLS):
            m = (sess == d) & (y == c)
            if m.sum() > 0: cent[di, c] = Fz[m].mean(0)

    def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    Sim = np.zeros((len(days), len(days)))
    for i in range(len(days)):
        for j in range(len(days)):
            cs = [cos(cent[i, c], cent[j, c]) for c in range(N_CLS)
                  if not (np.isnan(cent[i, c]).any() or np.isnan(cent[j, c]).any())]
            Sim[i, j] = np.mean(cs)
    np.save(os.path.join(OUT, "day_similarity.npy"), Sim)
    np.save(os.path.join(OUT, "feat_z.npy"), Fz.astype(np.float32))
    np.save(os.path.join(OUT, "session.npy"), sess); np.save(os.path.join(OUT, "labels.npy"), y)
    json.dump({"days_dt": days, "dates": [DATE[d] for d in days], "mode": MODE,
               "k_modes": K_MODES, "metric": "mean per-class cosine of feat_z centroids"},
              open(os.path.join(OUT, "day_labels.json"), "w"), indent=2)
    print("similarity matrix\n", np.round(Sim, 2))
    print(f"[saved] {OUT}/day_similarity.npy")


if __name__ == "__main__":
    main()
