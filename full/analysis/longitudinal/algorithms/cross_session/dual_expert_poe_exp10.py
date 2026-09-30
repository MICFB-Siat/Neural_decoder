















from __future__ import annotations
import os, sys, json, argparse, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, h5py
from sklearn.model_selection import StratifiedShuffleSplit

SPARC_DIR = "/home/guoyi/a800/share/code/Eigen_brain_decoding/data_check_20260507/uncertainty_seedv/tyf_20260518"
sys.path.insert(0, SPARC_DIR)
import sparcnet_run as S


@torch.no_grad()
def softmax_probs(model, X, dev, bs=128):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).to(dev)
        out.append(F.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out, 0)


def obs_confidence(p_obs):
    K = p_obs.shape[1]
    H = -(p_obs * np.log(p_obs + 1e-12)).sum(1) / np.log(K)
    return 1.0 - H


def fit_gate(p_obs, p_prior, y, grid_obs=None, grid_prior=None, gamma_grid=None):
    if grid_obs   is None: grid_obs   = np.linspace(0.0,  5.0, 26)
    if grid_prior is None: grid_prior = np.linspace(0.0, 10.0, 51)
    if gamma_grid is None: gamma_grid = [0.0, 0.5, 1.0, 2.0, 4.0]
    c  = obs_confidence(p_obs)
    lo = np.log(p_obs + 1e-8); lp = np.log(p_prior + 1e-8)
    idx = np.arange(len(y))
    best, best_nll = (1.0, 1.0, 0.0), 1e9
    for g in gamma_grid:
        lam = c ** g
        a = lam[:, None] * lo
        b = (1.0 - lam)[:, None] * lp
        for ao in grid_obs:
            for ap in grid_prior:
                if ao == 0 and ap == 0: continue
                logf = ao * a + ap * b
                logf = logf - logf.max(1, keepdims=True)
                logZ = np.log(np.exp(logf).sum(1, keepdims=True))
                nll = -(logf - logZ)[idx, y].mean()
                if nll < best_nll:
                    best_nll, best = nll, (float(ao), float(ap), float(g))
    return best


def fuse_pred(p_obs, p_prior, ao, ap, gamma):
    lam = obs_confidence(p_obs) ** gamma
    lo = np.log(p_obs + 1e-8); lp = np.log(p_prior + 1e-8)
    return (lam[:, None] * ao * lo + (1.0 - lam)[:, None] * ap * lp).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir",  required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--seed",     type=int, required=True)
    ap.add_argument("--device",   default="cuda:0")
    ap.add_argument("--n_train_days",      type=int,   default=3)
    ap.add_argument("--n_classes",         type=int,   default=6)
    ap.add_argument("--src_sr",            type=int,   default=256)
    ap.add_argument("--epochs",            type=int,   default=60)
    ap.add_argument("--batch_size",        type=int,   default=64)
    ap.add_argument("--lr",                type=float, default=5e-4)
    ap.add_argument("--weight_decay",      type=float, default=1e-4)
    ap.add_argument("--label_smoothing",   type=float, default=0.1)
    ap.add_argument("--clip_value",        type=float, default=1.0)
    ap.add_argument("--patience",          type=int,   default=12)
    ap.add_argument("--drop_rate",         type=float, default=0.2)
    ap.add_argument("--drop_fc",           type=float, default=0.5)
    ap.add_argument("--growth_rate",       type=int,   default=32)
    ap.add_argument("--block_config",      type=int, nargs="+", default=[4,4,4,4,4,4,4])
    ap.add_argument("--num_init_features", type=int,   default=64)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights").mkdir(exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with h5py.File(os.path.join(run_dir, "sub-01.h5"), "r") as f:
        eeg  = f["eeg_raw"][:].astype(np.float32)
        y    = f["labels"][:].astype(np.int64)
        sess = f["session"][:].astype(np.int64)
    X = S.zscore(S.resample_to(S.TARGET_SR, eeg, args.src_sr))
    PRIOR_DIR = os.path.join(run_dir, "prior_expert_geom_bandpower_np")
    p_prior = np.load(os.path.join(PRIOR_DIR, "p_prior.npy"))
    y_chk   = np.load(os.path.join(PRIOR_DIR, "labels.npy"))
    assert np.array_equal(y, y_chk), "labels order mismatch"

    days = sorted(np.unique(sess).tolist())
    train_days = days[:args.n_train_days]
    test_days  = days[args.n_train_days:]
    tr_all = np.where(np.isin(sess, train_days))[0]

    import logging; logging.basicConfig(level=logging.INFO); lg = logging.getLogger()
    lg.info(f"[seed{args.seed}] 训练日(Δt)={train_days} n_train_all={len(tr_all)} 测试日={test_days}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    core_rel, val_rel = next(sss.split(tr_all, y[tr_all]))
    tr, va = tr_all[core_rel], tr_all[val_rel]
    lg.info(f"[seed{args.seed}] train={len(tr)} val={len(va)}")

    _, state = S.train_one_fold(X, y, tr, va, va, args.n_classes, dev, args, lg)
    model = S.SPaRCNet(
        in_channels=X.shape[1], num_classes=args.n_classes,
        growth_rate=args.growth_rate, block_config=tuple(args.block_config),
        num_init_features=args.num_init_features,
        drop_rate=args.drop_rate, drop_fc=args.drop_fc,
        batch_norm=True, conv_bias=False
    ).to(dev)
    model.load_state_dict(state)
    p_obs = softmax_probs(model, X, dev)
    wpath = out_dir / "weights" / f"seed{args.seed}.pt"
    torch.save({"state_dict": state, "seed": args.seed, "train_days": train_days}, wpath)

    ao, ap_val, gamma = fit_gate(p_obs[va], p_prior[va], y[va])
    lg.info(f"[seed{args.seed}] Exp-4 a_obs={ao:.2f} a_prior={ap_val:.2f} gamma={gamma:.2f}")

    per_dt, accs_f, accs_s = {}, [], []
    for dt in test_days:
        idx = np.where(sess == dt)[0]
        acc_obs = float((p_obs[idx].argmax(1) == y[idx]).mean())
        acc_pri = float((p_prior[idx].argmax(1) == y[idx]).mean())
        acc_fus = float((fuse_pred(p_obs[idx], p_prior[idx], ao, ap_val, gamma) == y[idx]).mean())
        per_dt[dt] = {"n": int(len(idx)), "sparcnet": acc_obs, "prior": acc_pri, "fused": acc_fus}
        accs_f.append(acc_fus); accs_s.append(acc_obs)
    mean_fused = float(np.mean(accs_f)); mean_sparc = float(np.mean(accs_s))

    res = {
        "seed": args.seed, "exp": "exp10_fusion_geom",
        "n_train_days": args.n_train_days, "train_days": train_days, "test_days": test_days,
        "n_train": int(len(tr)), "n_val": int(len(va)),
        "a_obs": ao, "a_prior": ap_val, "gamma": gamma,
        "chance": 1 / args.n_classes,
        "mean_test_acc_fused": mean_fused, "mean_test_acc_sparcnet": mean_sparc,
        "per_dt": per_dt,
        "weights": str(wpath),
    }
    out_path = out_dir / f"seed{args.seed}.json"
    json.dump(res, open(out_path, "w"), indent=2)
    lg.info(f"[seed{args.seed}] mean_fused={mean_fused*100:.2f}% mean_sparc={mean_sparc*100:.2f}% saved {out_path}")

    print(f"\nseed{args.seed}  Fusion(ours)  a_obs={ao:.2f} a_prior={ap_val:.2f} gamma={gamma:.2f}")
    print(f"  mean_test: Fused={mean_fused*100:.2f}%  SPaRCNet={mean_sparc*100:.2f}%")
    print(f"{'Δt':>4} {'n':>4} {'SPaRC%':>7} {'prior%':>7} {'fused%':>7}")
    for dt in test_days:
        d = per_dt[dt]
        print(f"{dt:>4} {d['n']:>4} {d['sparcnet']*100:7.1f} {d['prior']*100:7.1f} {d['fused']*100:7.1f}")


if __name__ == "__main__":
    main()
