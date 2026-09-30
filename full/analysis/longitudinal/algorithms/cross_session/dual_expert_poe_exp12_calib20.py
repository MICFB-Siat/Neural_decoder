


















from __future__ import annotations
import os, sys, json, argparse, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, h5py
from sklearn.model_selection import StratifiedShuffleSplit

SPARC_DIR = str(Path(__file__).resolve().parent)
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


def build_split(sess, y, first_day, later_days, calib_frac, seed):

    train_idx = list(np.where(sess == first_day)[0])
    test_per_session = {}
    for d in later_days:
        idx_d = np.where(sess == d)[0]
        sss = StratifiedShuffleSplit(n_splits=1, train_size=calib_frac, random_state=seed + int(d))
        tr_rel, te_rel = next(sss.split(idx_d, y[idx_d]))
        train_idx.extend(idx_d[tr_rel].tolist())
        test_per_session[d] = idx_d[te_rel]
    return np.array(sorted(train_idx)), test_per_session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir",  required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--seed",     type=int, required=True)
    ap.add_argument("--device",   default="cuda:1")
    ap.add_argument("--calib_frac",        type=float, default=0.20)
    ap.add_argument("--val_frac",          type=float, default=0.15)
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
    first_day, later_days = days[0], days[1:]

    import logging; logging.basicConfig(level=logging.INFO); lg = logging.getLogger()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    train_pool, test_per_session = build_split(sess, y, first_day, later_days, args.calib_frac, args.seed)
    lg.info(f"[seed{args.seed}] first_session={first_day}(all)  later={later_days}  "
            f"calib_frac={args.calib_frac}  n_train_pool={len(train_pool)}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    core_rel, val_rel = next(sss.split(train_pool, y[train_pool]))
    tr, va = train_pool[core_rel], train_pool[val_rel]
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
    torch.save({"state_dict": state, "seed": args.seed, "train_pool": train_pool.tolist()}, wpath)

    ao, ap_val, gamma = fit_gate(p_obs[va], p_prior[va], y[va])
    lg.info(f"[seed{args.seed}] gate a_obs={ao:.2f} a_prior={ap_val:.2f} gamma={gamma:.2f}")


    per_session, accs_f, accs_s = {}, [], []
    n_calib = {int(d): int(args.calib_frac * len(np.where(sess == d)[0]) + 0.5) for d in later_days}
    for d in later_days:
        idx = test_per_session[d]
        acc_obs = float((p_obs[idx].argmax(1) == y[idx]).mean())
        acc_pri = float((p_prior[idx].argmax(1) == y[idx]).mean())
        acc_fus = float((fuse_pred(p_obs[idx], p_prior[idx], ao, ap_val, gamma) == y[idx]).mean())
        leaky = bool(d in (days[1], days[2]))
        per_session[str(int(d))] = {"n_test": int(len(idx)), "n_calib_train": n_calib[int(d)],
                                    "sparcnet": acc_obs, "prior": acc_pri, "fused": acc_fus,
                                    "prior_inSample": leaky}
        accs_f.append(acc_fus); accs_s.append(acc_obs)
    mean_fused = float(np.mean(accs_f)); mean_sparc = float(np.mean(accs_s))

    res = {
        "seed": args.seed, "exp": "exp12_calib20_perSession",
        "paradigm": "first_session_all + 20% calib per later session; test=remaining 80%",
        "prior": "reused first-3-day fit (sessions 1&6 in-sample -> prior optimistic there)",
        "first_session": int(first_day), "later_sessions": [int(d) for d in later_days],
        "calib_frac": args.calib_frac,
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_train_pool": int(len(train_pool)),
        "a_obs": ao, "a_prior": ap_val, "gamma": gamma, "chance": 1 / args.n_classes,
        "mean_test_acc_fused": mean_fused, "mean_test_acc_sparcnet": mean_sparc,
        "per_session": per_session,
        "weights": str(wpath),
    }
    json.dump(res, open(out_dir / f"seed{args.seed}.json", "w"), indent=2)
    lg.info(f"[seed{args.seed}] mean_fused={mean_fused*100:.2f}% mean_sparc={mean_sparc*100:.2f}% "
            f"saved {out_dir}/seed{args.seed}.json")

    print(f"\nseed{args.seed}  Exp-12 calib20  a_obs={ao:.2f} a_prior={ap_val:.2f} gamma={gamma:.2f}")
    print(f"  mean_test(over {len(later_days)} sessions): Fused={mean_fused*100:.2f}%  SPaRCNet={mean_sparc*100:.2f}%")
    print(f"{'sess':>5} {'n_te':>5} {'calib':>5} {'SPaRC%':>7} {'prior%':>7} {'fused%':>7}")
    for d in later_days:
        x = per_session[str(int(d))]
        flag = " *" if x["prior_inSample"] else ""
        print(f"{int(d):>5} {x['n_test']:>5} {x['n_calib_train']:>5} "
              f"{x['sparcnet']*100:7.1f} {x['prior']*100:7.1f} {x['fused']*100:7.1f}{flag}")


if __name__ == "__main__":
    main()
