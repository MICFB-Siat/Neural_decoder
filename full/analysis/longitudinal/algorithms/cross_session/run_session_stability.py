



import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
d = dict(np.load(f"{HERE}/sub06_layers_repr.npz"))
ses, y = d["session"], d["y"]
DAYMAP = {3: 0, 4: 0, 5: 1, 6: 2, 7: 3}
SESSIONS = [3, 4, 5, 6, 7]; DAYS = np.array([DAYMAP[s] for s in SESSIONS])
REPS = [("Ours (ea_combo)", d["ours"].astype(np.float64)),
        ("SPaRCNet (pre-pool)", d["sparc_prepool"].astype(np.float64))]



def cka_sessions(X):
    covs = []
    for s in SESSIONS:
        Xi = X[ses == s]; Xi = Xi - Xi.mean(0)
        covs.append(Xi.T @ Xi / len(Xi))
    M = np.eye(5)
    for i in range(5):
        for j in range(5):
            a, b = covs[i], covs[j]
            M[i, j] = (a * b).sum() / (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + 1e-12)
    return M



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



def session_identifiability(X):
    Xs = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = LogisticRegression(max_iter=500, C=1.0, multi_class="auto")
    return float(cross_val_score(clf, Xs, ses, cv=cv, scoring="balanced_accuracy").mean())


cka = {n: cka_sessions(X) for n, X in REPS}
trans = {n: transfer_matrix(X) for n, X in REPS}
sid = {n: session_identifiability(X) for n, X in REPS}



def dt_curve(M):
    out = {}
    for i in range(5):
        for j in range(5):
            dt = abs(DAYS[i] - DAYS[j])
            out.setdefault(dt, []).append(M[i, j])
    dts = sorted(out); return dts, [np.mean(out[t]) for t in dts]



fig, axes = plt.subplots(2, 3, figsize=(14, 8.8))
lab = [str(x) for x in DAYS]

for ax, (n, _) in zip(axes[0, :2], REPS):
    M = cka[n]; off = M[~np.eye(5, dtype=bool)]
    im = ax.imshow(M, cmap="cividis", vmin=off.min(), vmax=1, interpolation="nearest")
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(lab); ax.set_yticklabels(lab)
    ax.set_title(f"CKA(协方差)  {n}\noff-diag={off.mean():.2f}", fontsize=9)
    ax.set_xlabel("days since first session", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
axc = axes[0, 2]
for n, _ in REPS:
    dts, vals = dt_curve(trans[n]); axc.plot(dts, vals, "o-", label=n)
axc.axhline(0.5, ls="--", c="grey", lw=.8, label="chance"); axc.set_xlabel("Δt (days)")
axc.set_ylabel("transfer accuracy"); axc.set_title("③ Δt 漂移曲线 (解码迁移)", fontsize=10)
axc.set_xticks([0, 1, 2, 3]); axc.legend(fontsize=7); axc.grid(alpha=.3)

tv = np.concatenate([trans[n][~np.eye(5, dtype=bool)] for n, _ in REPS])
for ax, (n, _) in zip(axes[1, :2], REPS):
    M = trans[n]; off = M[~np.eye(5, dtype=bool)]
    im = ax.imshow(M, cmap="cividis", vmin=tv.min(), vmax=tv.max(), interpolation="nearest")
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(lab); ax.set_yticklabels(lab)
    ax.set_title(f"② 解码迁移 train→test  {n}\noff-diag={off.mean():.2f}", fontsize=9)
    ax.set_xlabel("test day"); ax.set_ylabel("train day", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
axb = axes[1, 2]
names = [n for n, _ in REPS]; vals = [sid[n] for n in names]
axb.bar(range(len(names)), vals, color=["#4c72b0", "#dd8452"])
axb.axhline(0.2, ls="--", c="grey", label="chance (1/5)")
axb.set_xticks(range(len(names))); axb.set_xticklabels(names, fontsize=7, rotation=10)
axb.set_ylabel("balanced acc"); axb.set_title("④ session 可识别性 (越低越好)", fontsize=10)
axb.legend(fontsize=7); axb.grid(alpha=.3, axis="y")
fig.suptitle("AJILE12 sub-06 — 方案一: 跨 session 稳定性 (Ours vs SPaRCNet)", fontsize=13, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.97])
for ext in ("svg", "png"):
    fig.savefig(f"{HERE}/sim_session_stability.{ext}",
                **({"format": "svg"} if ext == "svg" else {"dpi": 160}), bbox_inches="tight")
print("=== 结果 ===")
for n, _ in REPS:
    print(f"{n}: CKA off-diag={cka[n][~np.eye(5,dtype=bool)].mean():.3f} | "
          f"transfer off-diag={trans[n][~np.eye(5,dtype=bool)].mean():.3f} | session-id={sid[n]:.3f}")
print("WROTE sim_session_stability.svg/.png\nDONE")
