



import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
d = dict(np.load(f"{HERE}/sub06_layers_repr.npz"))
ses, y = d["session"], d["y"]
DAYMAP = {3: 0, 4: 0, 5: 1, 6: 2, 7: 3}
SESSIONS = [3, 4, 5, 6, 7]; DAYS = np.array([DAYMAP[s] for s in SESSIONS]); LAB = [str(x) for x in DAYS]
REPS = [("Ours (Eigen ea_combo)", d["ours"].astype(np.float64)),
        ("SPaRCNet (pre-pool)",    d["sparc_prepool"].astype(np.float64))]


def transfer_matrix(X):
    M = np.zeros((5, 5))
    for i, si in enumerate(SESSIONS):
        tr = ses == si; sc = StandardScaler().fit(X[tr])
        clf = LDA(solver="lsqr", shrinkage="auto").fit(sc.transform(X[tr]), y[tr])
        for j, sj in enumerate(SESSIONS):
            te = ses == sj
            if i == j:
                cv = StratifiedKFold(3, shuffle=True, random_state=0)
                M[i, j] = cross_val_score(LDA(solver="lsqr", shrinkage="auto"),
                                          sc.transform(X[te]), y[te], cv=cv).mean()
            else:
                M[i, j] = (clf.predict(sc.transform(X[te])) == y[te]).mean()
    return M


def dt_curve(M):
    out = {}
    for i in range(5):
        for j in range(5):
            out.setdefault(abs(DAYS[i] - DAYS[j]), []).append(M[i, j])
    dts = sorted(out); return dts, [np.mean(out[t]) for t in dts]


mats = [(n, transfer_matrix(X)) for n, X in REPS]
allv = np.concatenate([M.ravel() for _, M in mats])
vmin, vmax = allv.min(), allv.max()
offm = lambda M: float(M[~np.eye(5, dtype=bool)].mean())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
for ax, (n, M) in zip(axes[:2], mats):
    im = ax.imshow(M, cmap="cividis", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(LAB); ax.set_yticklabels(LAB)
    ax.set_xlabel("test day (days since first session)", fontsize=8)
    ax.set_ylabel("train day", fontsize=8)
    ax.set_title(f"AJILE12 sub-06 — {n}\ncross-session decode  (off-diag mean = {offm(M):.2f})", fontsize=9)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="k" if M[i, j] > (vmin + vmax) / 2 else "w")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04).set_label("transfer accuracy", fontsize=8)

axc = axes[2]
for (n, M), c in zip(mats, ["#2c6fbb", "#dd8452"]):
    dts, vals = dt_curve(M); axc.plot(dts, vals, "o-", color=c, lw=2, label=n)
axc.axhline(0.5, ls="--", c="grey", lw=1, label="chance (0.5)")
axc.set_xlabel("Δt  (day gap between train & test)", fontsize=9)
axc.set_ylabel("transfer accuracy", fontsize=9); axc.set_xticks([0, 1, 2, 3])
axc.set_title("Δt drift curve  (cross-session decoding)", fontsize=10)
axc.legend(fontsize=8); axc.grid(alpha=.3); axc.set_ylim(0.45, 0.9)
fig.suptitle("Cross-session decoding transfer: Ours vs SPaRCNet  (AJILE12 sub-06)", fontsize=13, y=1.02)
fig.tight_layout()
for ext in ("svg", "png"):
    fig.savefig(f"{HERE}/sim_transfer_compare.{ext}",
                **({"format": "svg"} if ext == "svg" else {"dpi": 165}), bbox_inches="tight")
print("=== off-diag (cross-session) transfer acc ===")
for n, M in mats: print(f"  {n}: off-diag={offm(M):.3f}  diag(within)={np.diag(M).mean():.3f}")
print("WROTE sim_transfer_compare.svg/.png | vmin=%.2f vmax=%.2f\nDONE" % (vmin, vmax))
