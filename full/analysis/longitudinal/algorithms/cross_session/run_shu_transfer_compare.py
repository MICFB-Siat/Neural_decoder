





import os, sys, importlib.util, numpy as np, h5py, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/guoyi/a800/share/code/Eigen_brain_decoding"
os.environ["LABRAM_DIR"] = f"{REPO}/data_check_20260507/Exp_tyf_Results/Models/LaBraM"
LABMOD = f"{REPO}/data_check_20260507/uncertainty_seedv/tyf_20260526/labram_session_run.py"
H5 = "/home/guoyi/nas/ugreen/NoLanguageData/SHU Multi-session Dataset/preprocess/stage2/sub-006.h5"
OURS_NPZ = f"{REPO}/data_check_20260507/uncertainty_seedv/tyf_20260601/repr_cache/sub006_repr.npz"
CACHE = f"{HERE}/sub006_shu_repr.npz"; CKPT = f"{HERE}/labram_sub006_seed0.pt"
SESSIONS = [1, 2, 3, 4, 5]; DEV = torch.device("cuda:2")
SEED, BS, EPOCHS, PAT = 0, 32, 40, 10
LR_BB, LR_HEAD, WD, DP, LS, CLIP = 5e-5, 5e-4, 5e-2, 0.1, 0.1, 3.0


def extract_and_cache():
    spec = importlib.util.spec_from_file_location("LB", LABMOD)
    LB = importlib.util.module_from_spec(spec); spec.loader.exec_module(LB)
    pre = LB.PRESETS["SHU_MS"]; ic = LB.get_input_chans(pre["ch_names"])
    X, y, ses = LB.load_subject_session(H5, pre["keep_idx"], pre["src_sr"])


    torch.manual_seed(SEED); np.random.seed(SEED)
    s1 = np.where(ses == 1)[0]
    tr_used, _ = train_test_split(s1, test_size=0.5, random_state=SEED, stratify=y[s1])
    tr, va = train_test_split(tr_used, test_size=0.2, random_state=SEED, stratify=y[tr_used])
    model = LB.build_labram(n_classes=2, drop_path=DP).to(DEV)
    LB.load_pretrained(model, f"{os.environ['LABRAM_DIR']}/checkpoints/labram-base.pth")
    hp = [p for n, p in model.named_parameters() if n.startswith("head")]
    bp = [p for n, p in model.named_parameters() if not n.startswith("head")]
    opt = torch.optim.AdamW([{"params": bp, "lr": LR_BB}, {"params": hp, "lr": LR_HEAD}], weight_decay=WD)
    nb = max(1, (len(tr) + BS - 1) // BS)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * nb, eta_min=1e-6)
    crit = nn.CrossEntropyLoss(label_smoothing=LS).to(DEV)
    Xt = torch.from_numpy(X)

    def acc_on(idx):
        model.eval(); p = []
        with torch.no_grad():
            for i in range(0, len(idx), 64):
                xb = Xt[idx[i:i + 64]].to(DEV)
                p.append(model(xb, input_chans=ic).argmax(1).cpu().numpy())
        return (np.concatenate(p) == y[idx]).mean()

    best_v, best_s, ni = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]
            xb = Xt[b].to(DEV); yb = torch.from_numpy(y[b]).to(DEV)
            opt.zero_grad(); loss = crit(model(xb, input_chans=ic), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step(); sched.step()
        va_acc = acc_on(va)
        if va_acc > best_v:
            best_v = va_acc; best_s = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; ni = 0
        else:
            ni += 1
            if ni >= PAT: break
    model.load_state_dict(best_s); model.eval()
    torch.save(best_s, CKPT)
    print(f"[train] best val acc={best_v:.3f} (ep {ep+1}) -> {os.path.basename(CKPT)}", flush=True)
    for s in SESSIONS[1:]:
        idx = np.where(ses == s)[0]; print(f"  sanity S{s} acc={acc_on(idx):.3f}", flush=True)


    feats = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            feats.append(model.forward_features(Xt[i:i + 64].to(DEV), input_chans=ic).cpu().numpy())
    labram = np.concatenate(feats).astype(np.float32)


    o = np.load(OURS_NPZ); ours = o["ours"].astype(np.float32); oy, oses = o["y"], o["ses"]
    assert np.array_equal(y, oy) and np.array_equal(ses, oses), "trial order mismatch"
    np.savez(CACHE, ours=ours, labram=labram, session=ses, y=y)
    print(f"[cache] saved {CACHE}  ours{ours.shape} labram{labram.shape}", flush=True)
    return dict(ours=ours, labram=labram, session=ses, y=y)


d = dict(np.load(CACHE)) if os.path.exists(CACHE) else extract_and_cache()
ses, y = d["session"], d["y"]
REPS = [("Ours (Eigen, 512-D)", d["ours"].astype(np.float64)),
        ("LaBraM (forward_features, 200-D)", d["labram"].astype(np.float64))]


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
            out.setdefault(abs(i - j), []).append(M[i, j])
    dts = sorted(out); return dts, [np.mean(out[t]) for t in dts]


mats = [(n, transfer_matrix(X)) for n, X in REPS]
allv = np.concatenate([M.ravel() for _, M in mats]); vmin, vmax = allv.min(), allv.max()
offm = lambda M: float(M[~np.eye(5, dtype=bool)].mean())
LAB = ["0", "1", "2", "3", "4"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
for ax, (n, M) in zip(axes[:2], mats):
    im = ax.imshow(M, cmap="cividis", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(LAB); ax.set_yticklabels(LAB)
    ax.set_xlabel("test session (recording day, ~2–3d apart)", fontsize=8); ax.set_ylabel("train session", fontsize=8)
    ax.set_title(f"SHU sub-006 — {n}\ncross-session decode  (off-diag mean = {offm(M):.2f})", fontsize=9)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="k" if M[i, j] > (vmin + vmax) / 2 else "w")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04).set_label("transfer accuracy", fontsize=8)
axc = axes[2]
for (n, M), c in zip(mats, ["#2c6fbb", "#dd8452"]):
    dts, vals = dt_curve(M); axc.plot(dts, vals, "o-", color=c, lw=2, label=n)
axc.axhline(0.5, ls="--", c="grey", lw=1, label="chance (0.5)")
axc.set_xlabel("Δ sessions (each ~2–3 days)", fontsize=9); axc.set_ylabel("transfer accuracy", fontsize=9)
axc.set_xticks([0, 1, 2, 3, 4]); axc.set_title("Δt drift curve (cross-session decoding)", fontsize=10)
axc.legend(fontsize=7); axc.grid(alpha=.3); axc.set_ylim(0.45, 0.95)
fig.suptitle("Cross-session decoding transfer: Ours vs LaBraM  (SHU sub-006)", fontsize=13, y=1.02)
fig.tight_layout()
for ext in ("svg", "png"):
    fig.savefig(f"{HERE}/sim_transfer_compare_shu.{ext}",
                **({"format": "svg"} if ext == "svg" else {"dpi": 165}), bbox_inches="tight")
print("=== off-diag (cross-session) transfer acc ===")
for n, M in mats: print(f"  {n}: off-diag={offm(M):.3f}  within(diag)={np.diag(M).mean():.3f}")
print("WROTE sim_transfer_compare_shu.svg/.png\nDONE")
