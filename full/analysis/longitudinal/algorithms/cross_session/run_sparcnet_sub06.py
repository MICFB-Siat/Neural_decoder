



import os, sys, numpy as np, h5py, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = "/home/guoyi/nas/ugreen/Language/AJILE12/code_my"
SRC  = "/home/guoyi/nas/ugreen/Language/AJILE12/preprocess/stage2_bp200_sleep"
sys.path.insert(0, CODE)
from run_sparcnet import SPaRCNet

SUBJ = "sub-06"; DMAP = {0: 3, 1: 5, 2: 6, 3: 7}
DEV  = torch.device("cuda:2")
SEED = 0; EPOCHS = 45; PATIENCE = 15; BS = 32
DIM_K = 5; DIM_SEEDS = 40


with h5py.File(f"{SRC}/{SUBJ}.h5", "r") as f:
    rawA = f["eeg_raw"][:].astype(np.float32)
    ses  = f["session"][:].astype(int)
    yA   = f["labels"][:].astype(np.int64)
sel, dts = [], []
for dt in sorted(DMAP):
    idx = np.where(ses == DMAP[dt])[0]; sel.append(idx); dts += [dt] * len(idx)
sel = np.concatenate(sel); dts = np.array(dts)
raw = rawA[sel]; y = yA[sel]; C = raw.shape[1]
tr = np.where(dts == 0)[0]
idx_by = {dt: np.where(dts == dt)[0] for dt in sorted(DMAP)}
print(f"[data] {SUBJ} C={C} sel={len(sel)} day0(train pool)={len(tr)} "
      f"per-dt={ {k: len(v) for k, v in idx_by.items()} }", flush=True)


torch.manual_seed(SEED); np.random.seed(SEED)
ai, bi = train_test_split(tr, test_size=.2, random_state=SEED, stratify=y[tr])
mu = raw[ai].mean((0, 2), keepdims=True); st = raw[ai].std((0, 2), keepdims=True) + 1e-6
Z = lambda X: torch.tensor((X - mu) / st, dtype=torch.float32)
Xtr, ytr = Z(raw[ai]).to(DEV), torch.tensor(y[ai]).to(DEV)
Xva, yva = Z(raw[bi]).to(DEV), y[bi]
net = SPaRCNet(C).to(DEV)
opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
lf = nn.CrossEntropyLoss(); best, bs_state, bad = 0., None, 0
for ep in range(EPOCHS):
    net.train(); perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), BS):
        b = perm[i:i + BS]; opt.zero_grad(); lf(net(Xtr[b]), ytr[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        va = (net(Xva).argmax(1).cpu().numpy() == yva).mean()
    if va > best:
        best = va; bs_state = {k: v.clone() for k, v in net.state_dict().items()}; bad = 0
    else:
        bad += 1
        if bad >= PATIENCE: break
net.load_state_dict(bs_state); net.eval()
print(f"[train] best val acc={best:.3f} (ep used={ep+1})", flush=True)


torch.save(bs_state, f"{HERE}/sparcnet_{SUBJ}_seed{SEED}.pt")
np.savez(f"{HERE}/norm_stats.npz", mu=mu, st=st, ai=ai, bi=bi, sel=sel, dt=dts, y=y)
print(f"[saved] weights -> sparcnet_{SUBJ}_seed{SEED}.pt", flush=True)


def acc(idx):
    with torch.no_grad():
        return float((net(Z(raw[idx]).to(DEV)).argmax(1).cpu().numpy() == y[idx]).mean())
line = f"  Δ0(val):{acc(bi)*100:.0f}  " + "  ".join(f"Δ{dt}:{acc(idx_by[dt])*100:.0f}"
                                                    for dt in [1, 2, 3])
print("[sanity cross-day acc]" + line, flush=True)


feats = []
with torch.no_grad():
    for i in range(0, len(raw), 64):
        feats.append(net.f(Z(raw[i:i + 64]).to(DEV)).cpu().numpy())
feats = np.concatenate(feats, 0).astype(np.float32)
np.savez(f"{HERE}/sparcnet_{SUBJ}_repr.npz", feat=feats, dt=dts, y=y)
print(f"[repr] f(x) feature {feats.shape} -> sparcnet_{SUBJ}_repr.npz", flush=True)


def dim_sim_5x5(X, k=DIM_K, n_seed=DIM_SEEDS):
    acc = np.zeros((k, k))
    for s in range(n_seed):
        R = np.random.default_rng(s).standard_normal((X.shape[1], k))
        Zp = X @ R
        Zn = Zp / (np.linalg.norm(Zp, axis=0) + 1e-12)
        acc += np.abs(Zn.T @ Zn)
    return acc / n_seed

S = dim_sim_5x5(feats.astype(np.float64))
off = S[~np.eye(DIM_K, dtype=bool)]
om = float(off.mean())
fig, ax = plt.subplots(figsize=(5.0, 4.6))
im = ax.imshow(S, cmap="cividis", vmin=off.min(), vmax=off.max(), interpolation="nearest")
ax.set_xticks(range(DIM_K)); ax.set_yticks(range(DIM_K))
ax.set_xticklabels([f"d{i+1}" for i in range(DIM_K)], fontsize=8)
ax.set_yticklabels([f"d{i+1}" for i in range(DIM_K)], fontsize=8)
ax.set_xlabel("latent dim", fontsize=8)
ax.set_title(f"AJILE12 {SUBJ} — SPaRCNet (pre-clf f(x), {feats.shape[1]}-D)\n"
             f"off-diagonal mean = {om:.2f}", fontsize=9)
fig.colorbar(im, ax=ax, fraction=.046, pad=.04).set_label("|cosine|", fontsize=8)
fig.suptitle("Low-level (SPaRCNet) representation similarity  (5-D, cividis)", fontsize=11, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{HERE}/sim_dim5x5_sparcnet.png", dpi=180, bbox_inches="tight")
print(f"[fig] off-diag mean |cos| = {om:.3f} -> sim_dim5x5_sparcnet.png", flush=True)
print("DONE", flush=True)
