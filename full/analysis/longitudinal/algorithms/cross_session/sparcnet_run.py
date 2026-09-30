












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
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)









class _DenseLayer(nn.Sequential):
    def __init__(self, num_in, growth_rate, bn_size, drop_rate, conv_bias, batch_norm):
        super().__init__()
        if batch_norm:
            self.add_module("norm1", nn.BatchNorm1d(num_in))
        self.add_module("elu1", nn.ELU())
        self.add_module("conv1", nn.Conv1d(num_in, bn_size * growth_rate,
                                            kernel_size=1, stride=1, bias=conv_bias))
        if batch_norm:
            self.add_module("norm2", nn.BatchNorm1d(bn_size * growth_rate))
        self.add_module("elu2", nn.ELU())
        self.add_module("conv2", nn.Conv1d(bn_size * growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1, bias=conv_bias))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_in, bn_size, growth_rate, drop_rate, conv_bias, batch_norm):
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"denselayer{i + 1}",
                            _DenseLayer(num_in + i * growth_rate, growth_rate, bn_size,
                                        drop_rate, conv_bias, batch_norm))


class _Transition(nn.Sequential):
    def __init__(self, num_in, num_out, conv_bias, batch_norm):
        super().__init__()
        if batch_norm:
            self.add_module("norm", nn.BatchNorm1d(num_in))
        self.add_module("elu", nn.ELU())
        self.add_module("conv", nn.Conv1d(num_in, num_out, kernel_size=1, stride=1, bias=conv_bias))
        self.add_module("pool", nn.AvgPool1d(kernel_size=2, stride=2))


class SPaRCNet(nn.Module):


    def __init__(self, in_channels, num_classes, growth_rate=32,
                 block_config=(4, 4, 4, 4, 4, 4, 4), num_init_features=64,
                 bn_size=4, drop_rate=0.2, conv_bias=True, batch_norm=False,
                 drop_fc=0.5):
        super().__init__()
        first = OrderedDict([
            ("conv0", nn.Conv1d(in_channels, num_init_features, kernel_size=7,
                                 stride=2, padding=3, bias=conv_bias)),
            ("elu0", nn.ELU()),
            ("pool0", nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
        ])
        self.densenet = nn.Sequential(first)
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            self.densenet.add_module(
                f"denseblock{i + 1}",
                _DenseBlock(num_layers, num_features, bn_size, growth_rate,
                            drop_rate, conv_bias, batch_norm),
            )
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                self.densenet.add_module(
                    f"transition{i + 1}",
                    _Transition(num_features, num_features // 2, conv_bias, batch_norm),
                )
                num_features = num_features // 2
        if batch_norm:
            self.densenet.add_module(f"norm{len(block_config) + 1}", nn.BatchNorm1d(num_features))
        self.densenet.add_module(f"relu{len(block_config) + 1}", nn.ReLU())
        self.densenet.add_module(f"pool{len(block_config) + 1}", nn.AdaptiveAvgPool1d(1))
        self.num_features = num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_fc),
            nn.Linear(num_features, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        feats = self.densenet(x).flatten(1)
        return self.classifier(feats)






STAGE2_ROOT = "/home/guoyi/a800/share/code/Eigen_brain_decoding/data_check_20260507/full_run_5fold/stage2_new"
EXP1_ROOT = "/home/guoyi/a800/share/code/Eigen_brain_decoding/guoyi_exp/Exp1"
FACED_PRESET = "/home/guoyi/a800/share/code/Eigen_brain_decoding/data_check_20260507/uncertainty_seedv/tyf_20260518/FACED_3cls_eeg_within_frozen_cache_20260518_105117/splits.json"

DATASETS = {
    "BCIC": dict(h5_dir=f"{STAGE2_ROOT}/BCIC_correct", label_key="labels", n_classes=4, src_sr=256,
                 splits_json=f"{EXP1_ROOT}/per_dataset_20260515_034656/BCIC_correct_eeg_within_frozen_cache_20260515_034703/splits.json"),
    "FACED": dict(h5_dir=f"{STAGE2_ROOT}/FACED_new", label_key="label2", n_classes=3, src_sr=256,
                  splits_json=FACED_PRESET),
    "INNER_class2": dict(h5_dir=f"{STAGE2_ROOT}/INNER_class2", label_key="labels", n_classes=2, src_sr=256,
                         splits_json=None),
    "INNER_class8": dict(h5_dir=f"{STAGE2_ROOT}/INNER_class8", label_key="labels", n_classes=8, src_sr=256,
                         splits_json=f"{EXP1_ROOT}/per_dataset_20260515_034656/INNER_class8_eeg_within_frozen_cache_20260515_034704/splits.json"),
    "MOTOR": dict(h5_dir=f"{STAGE2_ROOT}/MOTOR", label_key="labels", n_classes=2, src_sr=256,
                  splits_json=f"{EXP1_ROOT}/per_dataset_20260515_034656/MOTOR_eeg_within_frozen_cache_20260515_034704/splits.json"),
    "SEED-V": dict(h5_dir=f"{STAGE2_ROOT}/SEED-V", label_key="labels", n_classes=5, src_sr=256,
                   splits_json=None),
    "somatomotor": dict(h5_dir=f"{STAGE2_ROOT}/somatomotor", label_key="labels", n_classes=2, src_sr=256,
                        splits_json=f"{EXP1_ROOT}/somatomotor-MEG/run_20260514_213214/splits.json"),
}
TARGET_SR = 200


def setup_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)
    torch.backends.cudnn.deterministic = True


def setup_logger(path):
    lg = logging.getLogger(f"sparcnet_{path.parent.name}")
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


def load_subject(h5_path, label_key, src_sr):
    with h5py.File(h5_path, "r") as f:
        eeg = f["eeg_raw"][:]
        y = f[label_key][:]
    X = zscore(resample_to(TARGET_SR, eeg, src_sr))
    return X.astype(np.float32), y.astype(np.int64)


def make_within_5fold(y, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return [{"train": tr.tolist(), "test": te.tolist()} for tr, te in skf.split(np.zeros_like(y), y)]


def split_train_val(train_idx, y_all, seed):
    train_idx = np.asarray(train_idx, dtype=int)
    if len(train_idx) < 10:
        return train_idx[:-1].tolist(), train_idx[-1:].tolist()
    y_tr = y_all[train_idx]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    a, b = next(sss.split(np.zeros_like(y_tr), y_tr))
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
def evaluate(m, loader, dev):
    m.eval(); pr, gt = [], []
    for x, y in loader:
        pr.append(m(x.to(dev, non_blocking=True)).argmax(1).cpu().numpy())
        gt.append(y.numpy())
    pr = np.concatenate(pr) if pr else np.zeros(0, dtype=np.int64)
    gt = np.concatenate(gt) if gt else np.zeros(0, dtype=np.int64)
    if not len(pr):
        return 0., 0., 0., np.zeros((1, 1), int)
    return (accuracy_score(gt, pr),
            f1_score(gt, pr, average="macro", zero_division=0),
            cohen_kappa_score(gt, pr),
            confusion_matrix(gt, pr))


def train_one_fold(X, y, tr, va, te, nc, dev, args, lg):
    n, ch, T = X.shape
    dl = lambda I, sh: DataLoader(TensorDataset(torch.from_numpy(X[I]), torch.from_numpy(y[I])),
                                  batch_size=args.batch_size, shuffle=sh, num_workers=0, pin_memory=True)
    train_l, val_l, test_l = dl(tr, True), dl(va, False), dl(te, False)

    model = SPaRCNet(in_channels=ch, num_classes=nc,
                     growth_rate=args.growth_rate,
                     block_config=tuple(args.block_config),
                     num_init_features=args.num_init_features,
                     drop_rate=args.drop_rate, drop_fc=args.drop_fc,
                     batch_norm=True, conv_bias=False).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs * len(train_l)), eta_min=1e-6,
    )
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(dev)

    best_v, best_s, best_e, ni = -1., None, -1, 0
    for ep in range(args.epochs):
        model.train(); losses, t0 = [], time.time()
        for xb, yb in train_l:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad()
            l = crit(model(xb), yb); l.backward()
            if args.clip_value > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
            opt.step(); sched.step()
            losses.append(l.item())
        va_a, va_f, va_k, _ = evaluate(model, val_l, dev)
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
    ta, tf, tk, tcm = evaluate(model, test_l, dev)
    return {
        "best_epoch": best_e, "best_val_acc": float(best_v),
        "test_acc": float(ta), "test_f1": float(tf), "test_kappa": float(tk),
        "test_cm": tcm.tolist(),
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
    }, model.state_dict()


def run_dataset(args, lg):
    cfg = DATASETS[args.dataset]
    h5d = Path(cfg["h5_dir"])
    subs = sorted(p.stem for p in h5d.glob("sub-*.h5"))
    if args.max_subjects > 0:
        subs = subs[: args.max_subjects]
    lg.info(f"[{args.dataset}] subjects: {len(subs)}  classes={cfg['n_classes']}")
    if cfg["splits_json"] and Path(cfg["splits_json"]).is_file():
        with open(cfg["splits_json"]) as f:
            preset = json.load(f)
        lg.info(f"  using preset splits: {cfg['splits_json']}")
        splits = {s: preset[s] for s in subs if s in preset}
    else:
        splits = {}
        lg.info("  no preset splits — generating 5-fold StratifiedKFold per subject")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "model_config.json", "w") as f:
        json.dump({
            "model": "SPaRCNet (1-D DenseNet)",
            "growth_rate": args.growth_rate, "block_config": list(args.block_config),
            "num_init_features": args.num_init_features,
            "drop_rate": args.drop_rate, "drop_fc": args.drop_fc,
            "n_classes": cfg["n_classes"], "label_key": cfg["label_key"],
            "src_sr": cfg["src_sr"], "target_sr": TARGET_SR,
        }, f, indent=2)
    dev = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    final_splits, fr = {}, []
    for i, s in enumerate(subs):
        p = h5d / f"{s}.h5"
        if not p.is_file():
            lg.warning(f"  skip {s}: missing"); continue
        lg.info(f"[{args.dataset}] ({i+1}/{len(subs)}) {s}")
        X, y = load_subject(str(p), cfg["label_key"], cfg["src_sr"])
        lg.info(f"  X={X.shape} y_counts={np.bincount(y).tolist()}")
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
                metrics, st = train_one_fold(X, y, tr, va, te, cfg["n_classes"], dev, args, lg)
            except Exception as e:
                lg.exception(f"    fold {fi} TRAIN FAILED: {e}"); continue
            fld = sd / f"fold_{fi:02d}"; fld.mkdir(exist_ok=True)
            safe_save(st, fld / "weights.pth", lg=lg)
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
        from collections import defaultdict
        by = defaultdict(list)
        for r in fr:
            by[r["subject"]].append(r)
        smr = {
            "dataset": args.dataset, "n_subjects": len(by), "n_folds_total": len(fr),
            "test_acc_mean": float(np.mean([r["test_acc"] for r in fr])),
            "test_acc_std": float(np.std([r["test_acc"] for r in fr])),
            "test_f1_mean": float(np.mean([r["test_f1"] for r in fr])),
            "test_f1_std": float(np.std([r["test_f1"] for r in fr])),
            "test_kappa_mean": float(np.mean([r["test_kappa"] for r in fr])),
            "test_kappa_std": float(np.std([r["test_kappa"] for r in fr])),
            "per_subject": {
                s: {"acc_mean": float(np.mean([r["test_acc"] for r in rs])),
                    "f1_mean": float(np.mean([r["test_f1"] for r in rs])),
                    "kappa_mean": float(np.mean([r["test_kappa"] for r in rs])),
                    "n_folds": len(rs)}
                for s, rs in by.items()
            },
        }
        with open(out / "summary.json", "w") as f:
            json.dump(smr, f, indent=2)
        lg.info("===== SUMMARY =====")
        lg.info(json.dumps({k: v for k, v in smr.items() if k != "per_subject"}, indent=2))

        write_txt_summary(out, fr, by, args.dataset, cfg["n_classes"])


def write_txt_summary(out, fr, by, dataset, n_classes):
    lines = []
    lines.append("=" * 92)
    lines.append(f"SPaRCNet {dataset} within-subject 5-fold — per-fold accuracy and aggregates")
    lines.append(f"Run dir : {out}")
    lines.append(f"Dataset : {dataset}  (n_classes={n_classes}, chance={1.0/n_classes:.4f})")
    lines.append("Model   : SPaRCNet (1-D DenseNet, 7 blocks × 4 layers, growth_rate=32)")
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
    lines.append("Per-subject mean ± std (across folds):")
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
    lines.append("Overall aggregate (across all (subject, fold) data points):")
    lines.append("")
    accs = np.array([r["test_acc"] for r in fr])
    f1s = np.array([r["test_f1"] for r in fr])
    kappas = np.array([r["test_kappa"] for r in fr])
    lines.append(f"  n_folds_total   : {len(fr)}")
    lines.append(f"  n_subjects      : {len(by)}")
    lines.append(f"  test_acc        : mean={accs.mean():.4f}  std={accs.std():.4f}  "
                 f"min={accs.min():.4f}  max={accs.max():.4f}  median={np.median(accs):.4f}")
    lines.append(f"  test_f1 (macro) : mean={f1s.mean():.4f}  std={f1s.std():.4f}  "
                 f"min={f1s.min():.4f}  max={f1s.max():.4f}")
    lines.append(f"  test_kappa      : mean={kappas.mean():.4f}  std={kappas.std():.4f}  "
                 f"min={kappas.min():.4f}  max={kappas.max():.4f}")
    lines.append("")
    sm = np.array(sub_means)
    lines.append("Cross-subject aggregate (per-subject mean acc, then mean ± std across subjects):")
    lines.append(f"  per-subject acc means : {[f'{x:.4f}' for x in sm]}")
    lines.append(f"  mean ± std across subjects = {sm.mean():.4f} ± {sm.std():.4f}")
    lines.append(f"  chance                = {1.0/n_classes:.4f}")
    lines.append(f"  above chance          = {accs.mean() - 1.0/n_classes:+.4f}")
    lines.append("=" * 92)
    (out / "per_fold_accuracy_summary.txt").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--clip_value", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--drop_rate", type=float, default=0.2)
    ap.add_argument("--drop_fc", type=float, default=0.5)
    ap.add_argument("--growth_rate", type=int, default=32)
    ap.add_argument("--block_config", type=int, nargs="+", default=[4, 4, 4, 4, 4, 4, 4])
    ap.add_argument("--num_init_features", type=int, default=64)
    ap.add_argument("--max_subjects", type=int, default=0)
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
