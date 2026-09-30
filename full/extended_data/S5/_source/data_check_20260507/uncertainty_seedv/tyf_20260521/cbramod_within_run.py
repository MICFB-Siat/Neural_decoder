















from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))

import argparse
import copy
import datetime
import json
import logging
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CBRAMOD_DIR = Path(os.environ.get(
    "CBRAMOD_DIR",
    str(_LOCAL_SOURCE / 'data_check_20260507/uncertainty_seedv/tyf_20260518/CBraMod'),
))
sys.path.insert(0, str(CBRAMOD_DIR))
from models.cbramod import CBraMod

THIS_DIR = Path(__file__).resolve().parent
STAGE2_ROOT = str(_LOCAL_SOURCE / 'data_check_20260507/full_run_5fold/stage2_new')
EXP1_ROOT = str(_LOCAL_SOURCE / 'guoyi_exp/Exp1')
SOMATO_H5_DIR = str(THIS_DIR / "somatomotor_patched")

DATASETS = {
    "BCIC": dict(
        h5_dir=f"{STAGE2_ROOT}/BCIC_correct",
        label_key="labels", n_classes=4, src_sr=256, splits_json=None,
    ),
    "MOTOR": dict(
        h5_dir=f"{STAGE2_ROOT}/MOTOR",
        label_key="labels", n_classes=2, src_sr=256, splits_json=None,
    ),
    "somatomotor": dict(
        h5_dir=SOMATO_H5_DIR,
        label_key="labels", n_classes=2,
        src_sr=256, splits_json=None,
    ),
    "INNER_class2": dict(
        h5_dir=f"{STAGE2_ROOT}/INNER_class2",
        label_key="labels", n_classes=2,
        src_sr=256, splits_json=None,
    ),
    "INNER_class8": dict(
        h5_dir=f"{STAGE2_ROOT}/INNER_class8",
        label_key="labels", n_classes=8,
        src_sr=256, splits_json=None,
    ),
}
TARGET_SR = 200
PATCH_SIZE = 200






class CBraModClassifier(nn.Module):
    def __init__(self, n_channels: int, seq_len: int, n_classes: int,
                 backbone_ckpt: str, dropout: float = 0.1):
        super().__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30, n_layer=12, nhead=8,
        )
        if backbone_ckpt and os.path.isfile(backbone_ckpt):
            self.backbone.load_state_dict(torch.load(backbone_ckpt, map_location="cpu"))
        self.backbone.proj_out = nn.Identity()
        self.head = nn.Sequential(
            Rearrange("b c s d -> b d c s"),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, n_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))






def setup_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)
    torch.backends.cudnn.deterministic = True


def setup_logger(path):
    lg = logging.getLogger(f"cbramod_within_{path.parent.name}")
    lg.setLevel(logging.INFO); lg.handlers = []
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(path); fh.setFormatter(fmt); lg.addHandler(fh)
    sh = logging.StreamHandler(); sh.setFormatter(fmt); lg.addHandler(sh)
    return lg


def resample_to(target_sr, x, src_sr):
    if src_sr == target_sr:
        return x.astype(np.float32, copy=False)
    from math import gcd
    g = gcd(target_sr, src_sr)
    return resample_poly(x, target_sr // g, src_sr // g, axis=-1).astype(np.float32)


def zscore(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)


def make_patches(x, patch_size=PATCH_SIZE):
    n, ch, T = x.shape
    npat = T // patch_size
    return x[..., : npat * patch_size].reshape(n, ch, npat, patch_size).astype(np.float32)


def load_subject(h5_path, label_key, src_sr):
    with h5py.File(h5_path, "r") as f:
        eeg = f["eeg_raw"][:]
        y = f[label_key][:]
    return make_patches(zscore(resample_to(TARGET_SR, eeg, src_sr))), y.astype(np.int64)


def make_within_5fold(y, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return [{"train": tr.tolist(), "test": te.tolist()} for tr, te in skf.split(np.zeros_like(y), y)]


def split_train_val(train_idx, y_all, seed):
    train_idx = np.asarray(train_idx, dtype=int)
    if len(train_idx) < 10:
        return train_idx[:-1].tolist(), train_idx[-1:].tolist()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    a, b = next(sss.split(np.zeros_like(y_all[train_idx]), y_all[train_idx]))
    return train_idx[a].tolist(), train_idx[b].tolist()


def safe_save(obj, path, retries=3, lg=None):
    last = None
    for i in range(retries):
        try:
            torch.save(obj, path); return True
        except (OSError, RuntimeError) as e:
            last = e
            if lg: lg.warning(f"  torch.save retry {i+1}/{retries}: {e}")
            time.sleep(0.5 * (i + 1))
    if lg: lg.error(f"  torch.save permanently failed for {path}: {last}")
    return False


@torch.no_grad()
def predict(model, dl, dev):
    model.eval(); pr, gt = [], []
    for x, y in dl:
        pr.append(model(x.to(dev, non_blocking=True)).argmax(1).cpu().numpy())
        gt.append(y.numpy())
    pr = np.concatenate(pr) if pr else np.zeros(0, dtype=np.int64)
    gt = np.concatenate(gt) if gt else np.zeros(0, dtype=np.int64)
    return pr, gt


def scores_from(pr, gt, default_cm_dim=1):
    if not len(pr):
        return 0., 0., 0., np.zeros((default_cm_dim, default_cm_dim), int)
    return (accuracy_score(gt, pr),
            f1_score(gt, pr, average="macro", zero_division=0),
            cohen_kappa_score(gt, pr),
            confusion_matrix(gt, pr))


def train_one_fold(X, y, tr, va, te, n_classes, n_channels, seq_len, args, dev, lg):
    def dl(idx, sh, bs):
        return DataLoader(
            TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(y[idx])),
            batch_size=bs, shuffle=sh, num_workers=0, pin_memory=True,
        )
    train_l = dl(tr, True,  args.batch_size)
    val_l   = dl(va, False, args.batch_size)
    test_l  = dl(te, False, args.batch_size)

    model = CBraModClassifier(
        n_channels=n_channels, seq_len=seq_len, n_classes=n_classes,
        backbone_ckpt=args.backbone_ckpt, dropout=args.dropout,
    ).to(dev)


    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train_p = sum(p.numel() for p in trainable)
    n_total_p = sum(p.numel() for p in model.parameters())
    lg.info(f"      FROZEN backbone — trainable params: {n_train_p/1e6:.3f}M / "
            f"{n_total_p/1e6:.3f}M  ({100*n_train_p/n_total_p:.1f}%)")

    opt = torch.optim.AdamW(trainable, lr=args.lr_head, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs * len(train_l)), eta_min=1e-6,
    )
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(dev)

    best_v, best_s, best_e, ni = -1., None, -1, 0
    for ep in range(args.epochs):
        model.train()
        model.backbone.eval()
        losses, t0 = [], time.time()
        for xb, yb in train_l:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.no_grad():
                feats = model.backbone(xb)
            logits = model.head(feats)
            loss = crit(logits, yb)
            loss.backward()
            if args.clip_value > 0:
                nn.utils.clip_grad_norm_(trainable, args.clip_value)
            opt.step(); sched.step()
            losses.append(loss.item())
        va_pr, va_gt = predict(model, val_l, dev)
        va_a, va_f, va_k, _ = scores_from(va_pr, va_gt, default_cm_dim=n_classes)
        lg.info(f"      epoch {ep+1:03d}/{args.epochs} loss={np.mean(losses):.4f} "
                f"val_acc={va_a:.4f} val_f1={va_f:.4f} val_kappa={va_k:.4f} "
                f"({time.time()-t0:.1f}s)")
        if va_a > best_v:
            best_v, best_s, best_e, ni = va_a, copy.deepcopy(model.state_dict()), ep + 1, 0
        else:
            ni += 1
            if args.patience > 0 and ni >= args.patience:
                lg.info(f"      early stop at epoch {ep+1}"); break
    if best_s is not None:
        model.load_state_dict(best_s)

    pr, gt = predict(model, test_l, dev)
    test_acc, test_f1, test_kappa, test_cm = scores_from(pr, gt, default_cm_dim=n_classes)

    return {
        "best_epoch": best_e, "best_val_acc": float(best_v),
        "test_acc": float(test_acc), "test_f1": float(test_f1),
        "test_kappa": float(test_kappa), "test_cm": test_cm.tolist(),
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
    }, model.state_dict()


def write_txt_summary(out, fr, by, by_fold, dataset, n_classes):
    lines, chance = [], 1.0 / n_classes
    lines.append("=" * 92)
    lines.append(f"CBraMod (FROZEN backbone) {dataset} within-subject 5-fold — per-fold + aggregates")
    lines.append(f"Run dir : {out}")
    lines.append(f"Dataset : {dataset}  (n_classes={n_classes}, chance={chance:.4f})")
    lines.append("Model   : CBraMod (~4.92M params, FROZEN) + MLP head (200→200→n_classes)")
    lines.append(f"Stats   : {len(fr)} folds across {len(by)} subjects, 5-fold within-subject CV")
    lines.append("=" * 92)
    lines.append("")
    lines.append(f"{'subject':<10} {'fold':>4} {'test_acc':>10} {'test_f1':>10} {'test_kappa':>11} "
                 f"{'n_train':>8} {'n_val':>6} {'n_test':>7} {'best_ep':>8}")
    lines.append("-" * 92)
    for r in sorted(fr, key=lambda x: (x["subject"], x["fold"])):
        lines.append(f"{r['subject']:<10} {r['fold']:>4} {r['test_acc']:>10.4f} {r['test_f1']:>10.4f} "
                     f"{r['test_kappa']:>11.4f} {r['n_train']:>8} {r['n_val']:>6} "
                     f"{r['n_test']:>7} {r['best_epoch']:>8}")
    lines.append("-" * 92)
    lines.append("")
    lines.append("Per-subject mean ± std (5 folds each):")
    lines.append("")
    lines.append(f"{'subject':<10} {'acc_mean':>10} {'acc_std':>10} {'acc_min':>10} {'acc_max':>10} "
                 f"{'f1_mean':>10} {'kappa_mean':>11} {'n_folds':>8}")
    lines.append("-" * 92)
    sub_means = []
    for s in sorted(by):
        rs = by[s]
        a = np.array([r["test_acc"] for r in rs])
        f = np.array([r["test_f1"] for r in rs])
        k = np.array([r["test_kappa"] for r in rs])
        sub_means.append(a.mean())
        lines.append(f"{s:<10} {a.mean():>10.4f} {a.std():>10.4f} {a.min():>10.4f} "
                     f"{a.max():>10.4f} {f.mean():>10.4f} {k.mean():>11.4f} {len(rs):>8d}")
    lines.append("-" * 92)
    lines.append("")
    lines.append("Per-fold mean ± std (across subjects, for each CV fold index 0..4):")
    lines.append("")
    lines.append(f"{'fold':<6} {'acc_mean':>10} {'acc_std':>10} {'acc_min':>10} {'acc_max':>10} "
                 f"{'f1_mean':>10} {'kappa_mean':>11} {'n_subj':>8}")
    lines.append("-" * 92)
    fold_means = []
    for fi in sorted(by_fold):
        rs = by_fold[fi]
        a = np.array([r["test_acc"] for r in rs])
        f1 = np.array([r["test_f1"] for r in rs])
        k = np.array([r["test_kappa"] for r in rs])
        fold_means.append(a.mean())
        lines.append(f"{fi:<6d} {a.mean():>10.4f} {a.std():>10.4f} {a.min():>10.4f} "
                     f"{a.max():>10.4f} {f1.mean():>10.4f} {k.mean():>11.4f} {len(rs):>8d}")
    lines.append("-" * 92)
    lines.append("")
    lines.append("Overall aggregate (across all (subject, fold) data points):")
    lines.append("")
    accs = np.array([r["test_acc"] for r in fr])
    f1s = np.array([r["test_f1"] for r in fr])
    kappas = np.array([r["test_kappa"] for r in fr])
    lines.append(f"  n_folds_total   : {len(fr)}")
    lines.append(f"  n_subjects      : {len(by)}")
    lines.append(f"  test_acc        : mean={accs.mean():.4f}  std={accs.std():.4f}  "
                 f"min={accs.min():.4f}  max={accs.max():.4f}  median={np.median(accs):.4f}")
    lines.append(f"  test_f1 (macro) : mean={f1s.mean():.4f}  std={f1s.std():.4f}")
    lines.append(f"  test_kappa      : mean={kappas.mean():.4f}  std={kappas.std():.4f}")
    sm = np.array(sub_means); fm = np.array(fold_means)
    lines.append("")
    lines.append("Cross-subject aggregate (per-subject mean acc, then mean ± std across subjects):")
    lines.append(f"  per-subject acc means : {[f'{x:.4f}' for x in sm]}")
    lines.append(f"  mean ± std across subjects = {sm.mean():.4f} ± {sm.std():.4f}")
    lines.append("")
    lines.append("Cross-fold aggregate (per-fold mean acc, then mean ± std across folds):")
    lines.append(f"  per-fold acc means    : {[f'{x:.4f}' for x in fm]}")
    lines.append(f"  mean ± std across folds    = {fm.mean():.4f} ± {fm.std():.4f}")
    lines.append("")
    lines.append(f"  chance                = {chance:.4f}")
    lines.append(f"  above chance          = {accs.mean() - chance:+.4f}")
    lines.append("=" * 92)
    (out / "per_fold_accuracy_summary.txt").write_text("\n".join(lines))


def run_dataset(args, lg):
    cfg = DATASETS[args.dataset]
    h5d = Path(args.h5_dir if args.h5_dir else cfg["h5_dir"])
    subs = sorted(p.stem for p in h5d.glob("sub-*.h5"))
    if args.subjects:
        keep = set(args.subjects)
        subs = [s for s in subs if s in keep]
    if args.max_subjects > 0:
        subs = subs[: args.max_subjects]
    lg.info(f"[{args.dataset}] subjects: {len(subs)}  classes={cfg['n_classes']}  src_sr={cfg['src_sr']}")

    if cfg["splits_json"] and Path(cfg["splits_json"]).is_file():
        with open(cfg["splits_json"]) as f:
            preset = json.load(f)
        lg.info(f"  using preset splits: {cfg['splits_json']}")
        splits = {s: preset[s] for s in subs if s in preset}
    else:
        splits = {}
        lg.info("  no preset splits — generating 5-fold StratifiedKFold per subject")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    final_splits, fr = {}, []
    n_channels_global, seq_len_global = None, None
    for i, s in enumerate(subs):
        p = h5d / f"{s}.h5"
        if not p.is_file():
            lg.warning(f"  skip {s}: missing"); continue
        lg.info(f"[{args.dataset}] ({i+1}/{len(subs)}) {s}")
        X, y = load_subject(str(p), cfg["label_key"], cfg["src_sr"])
        lg.info(f"  X={X.shape} y_counts={np.bincount(y).tolist()}")
        n_channels = X.shape[1]
        seq_len    = X.shape[2]
        if n_channels_global is None:
            n_channels_global, seq_len_global = n_channels, seq_len
            with open(out / "model_config.json", "w") as f:
                json.dump({
                    "model": "CBraMod(in=200,out=200,d_model=200,ff=800,n_layer=12,nhead=8)",
                    "params_M": 4.92, "frozen_backbone": True,
                    "patch_size": PATCH_SIZE, "target_sr": TARGET_SR,
                    "backbone_ckpt": args.backbone_ckpt,
                    "n_classes": cfg["n_classes"], "n_channels": n_channels, "seq_len": seq_len,
                }, f, indent=2)

        folds = splits.get(s) or make_within_5fold(y, args.seed)
        final_splits[s] = folds
        sd = out / s; sd.mkdir(exist_ok=True)
        for fi, fd in enumerate(folds):
            te = np.asarray(fd["test"], int)
            tr, va = split_train_val(np.asarray(fd["train"], int), y, args.seed + fi)
            tr, va = np.asarray(tr), np.asarray(va)
            lg.info(f"    fold {fi}: train={len(tr)} val={len(va)} test={len(te)}")
            t0 = time.time()
            try:
                metrics, st = train_one_fold(
                    X, y, tr, va, te, cfg["n_classes"], n_channels, seq_len, args, dev, lg,
                )
            except Exception as e:
                lg.exception(f"    fold {fi} TRAIN FAILED: {e}"); continue
            fld = sd / f"fold_{fi:02d}"; fld.mkdir(exist_ok=True)
            head_state = {k: v.cpu() for k, v in st.items() if k.startswith("head.")}
            safe_save(head_state, fld / "head_weights.pth", lg=lg)
            with open(fld / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            lg.info(f"    fold {fi} DONE  test_acc={metrics['test_acc']:.4f} "
                    f"test_f1={metrics['test_f1']:.4f} test_kappa={metrics['test_kappa']:.4f} "
                    f"({(time.time()-t0)/60:.2f}min)")
            fr.append({"subject": s, "fold": fi, **metrics})
        with open(out / "splits.json", "w") as f:
            json.dump(final_splits, f, indent=2)
        with open(out / "fold_records.json", "w") as f:
            json.dump(fr, f, indent=2)

    if fr:
        by, by_fold = defaultdict(list), defaultdict(list)
        for r in fr:
            by[r["subject"]].append(r)
            by_fold[r["fold"]].append(r)
        per_subject = {
            s: {"acc_mean": float(np.mean([r["test_acc"] for r in rs])),
                "acc_std":  float(np.std([r["test_acc"] for r in rs])),
                "f1_mean":  float(np.mean([r["test_f1"] for r in rs])),
                "kappa_mean": float(np.mean([r["test_kappa"] for r in rs])),
                "n_folds": len(rs)}
            for s, rs in by.items()
        }
        per_fold = {
            int(fi): {"acc_mean": float(np.mean([r["test_acc"] for r in rs])),
                       "acc_std":  float(np.std([r["test_acc"] for r in rs])),
                       "f1_mean":  float(np.mean([r["test_f1"] for r in rs])),
                       "kappa_mean": float(np.mean([r["test_kappa"] for r in rs])),
                       "n_subjects": len(rs)}
            for fi, rs in by_fold.items()
        }
        smr = {
            "dataset": args.dataset, "model": "CBraMod-frozen",
            "n_subjects": len(by), "n_folds_total": len(fr),
            "test_acc_mean": float(np.mean([r["test_acc"] for r in fr])),
            "test_acc_std":  float(np.std([r["test_acc"] for r in fr])),
            "test_f1_mean":  float(np.mean([r["test_f1"] for r in fr])),
            "test_f1_std":   float(np.std([r["test_f1"] for r in fr])),
            "test_kappa_mean": float(np.mean([r["test_kappa"] for r in fr])),
            "test_kappa_std":  float(np.std([r["test_kappa"] for r in fr])),
            "per_subject": per_subject,
            "per_fold":    per_fold,
        }
        with open(out / "summary.json", "w") as f:
            json.dump(smr, f, indent=2)
        lg.info("===== SUMMARY =====")
        lg.info(json.dumps({k: v for k, v in smr.items() if k not in ("per_subject", "per_fold")}, indent=2))
        lg.info("per_fold (cross-subject aggregate per CV fold):")
        lg.info(json.dumps(per_fold, indent=2))
        write_txt_summary(out, fr, by, by_fold, args.dataset, cfg["n_classes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--h5_dir", default=None,
                    help="Override h5_dir from DATASETS registry.")
    ap.add_argument("--backbone_ckpt",
                    default=str(CBRAMOD_DIR / "pretrained_weights" / "pretrained_weights.pth"))
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--clip_value", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--max_subjects", type=int, default=0)
    ap.add_argument("--subjects", nargs="+", default=None)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    lg = setup_logger(out / "log.txt"); setup_seed(args.seed)
    lg.info(f"Run launched at {datetime.datetime.now().isoformat()}")
    lg.info(f"args: {vars(args)}")
    with open(out / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    run_dataset(args, lg)
    lg.info("ALL DONE")


if __name__ == "__main__":
    main()
