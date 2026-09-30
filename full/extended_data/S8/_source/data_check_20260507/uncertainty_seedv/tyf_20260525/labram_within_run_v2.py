



















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
from collections import OrderedDict, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

THIS_DIR = Path(__file__).resolve().parent
LABRAM_DIR = THIS_DIR / "LaBraM"
sys.path.insert(0, str(LABRAM_DIR))
import modeling_finetune
from timm.models import create_model






STANDARD_1020 = [
    "FP1", "FPZ", "FP2",
    "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10",
    "O1", "OZ", "O2", "O9", "CB1", "CB2",
    "IZ", "O10", "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2",
    "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6", "CFC7", "CFC8",
    "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6", "CCP7", "CCP8",
    "T1", "T2", "FTT9h", "TTP7h", "TPP9h", "FTT10h", "TPP8h", "TPP10h",
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]

def get_input_chans(ch_names):
    return [0] + [STANDARD_1020.index(c) + 1 for c in ch_names]


INNER_CH_NAMES = [

    "FP1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7",
    "FC5", "FC3", "FC1", "C1", "C3", "C5", "T7", "TP7",
    "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9",
    "PO7", "PO3", "O1", "IZ", "OZ", "POZ", "PZ", "CPZ",
    "FPZ", "FP2", "AF8", "AF4", "AFZ", "FZ", "F2", "F4",
    "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCZ", "CZ",
    "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
]


STAGE2_ROOT = str(_LOCAL_SOURCE / 'data_check_20260507/full_run_5fold/stage2_new')

DATASETS = {
    "INNER_class8": dict(
        h5_dir=f"{STAGE2_ROOT}/INNER_class8",
        label_key="labels", n_classes=8, src_sr=256,
        ch_names=INNER_CH_NAMES, splits_json=None,
    ),
    "INNER_class2": dict(
        h5_dir=f"{STAGE2_ROOT}/INNER_class2",
        label_key="labels", n_classes=2, src_sr=256,
        ch_names=INNER_CH_NAMES, splits_json=None,
    ),
}
TARGET_SR = 200
PATCH_SIZE = 200






class ChannelPreservingHeadV2(nn.Module):







    def __init__(self, n_channels: int, n_patches: int, embed_dim: int,
                 n_classes: int, hidden: int = 256, dropout: float = 0.3,
                 arch: str = "linear"):
        super().__init__()
        self.n_channels = n_channels
        self.n_patches  = n_patches
        self.arch = arch
        flat = n_channels * embed_dim
        self.norm = nn.LayerNorm(flat)
        if arch == "linear":
            self.fc = nn.Linear(flat, n_classes)
        elif arch == "mlp":
            self.fc1  = nn.Linear(flat, hidden)
            self.act  = nn.GELU()
            self.drop = nn.Dropout(dropout)
            self.fc2  = nn.Linear(hidden, n_classes)
        else:
            raise ValueError(f"unknown head arch: {arch}")

    def forward(self, tokens):

        B, NA, D = tokens.shape
        assert NA == self.n_channels * self.n_patches, f"NA={NA} expected {self.n_channels*self.n_patches}"
        x = tokens.reshape(B, self.n_channels, self.n_patches, D)
        x = x.mean(dim=2)
        x = x.reshape(B, -1)
        x = self.norm(x)
        if self.arch == "linear":
            return self.fc(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


class LaBraMClassifierV2(nn.Module):

    def __init__(self, n_channels: int, n_patches: int, n_classes: int,
                 pretrained_ckpt: str, hidden: int = 256, dropout: float = 0.3,
                 drop_path: float = 0.1, head_arch: str = "linear", lg=None):
        super().__init__()

        self.backbone = create_model(
            "labram_base_patch200_200",
            pretrained=False, num_classes=0,
            in_chans=1, out_chans=8,
            use_mean_pooling=True,

            drop_path_rate=drop_path,
            use_abs_pos_emb=True,
            init_values=0.1, qkv_bias=False,
        )
        if pretrained_ckpt:
            _load_labram_pretrained(self.backbone, pretrained_ckpt, lg=lg)

        self.head = ChannelPreservingHeadV2(
            n_channels=n_channels, n_patches=n_patches,
            embed_dim=200, n_classes=n_classes,
            hidden=hidden, dropout=dropout, arch=head_arch,
        )

    def forward(self, x, input_chans=None):

        tokens = self.backbone.forward_features(
            x, input_chans=input_chans, return_patch_tokens=True,
        )
        return self.head(tokens)


def _load_labram_pretrained(model, ckpt_path, lg=None):

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck.get("model", ck.get("module", ck))
    new = OrderedDict()
    for k, v in state.items():
        if k.startswith("student."):
            new[k[len("student."):]] = v
    state = new if new else state
    msd = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in state and k in msd and state[k].shape != msd[k].shape:
            if lg: lg.info(f"    dropping {k} (shape mismatch)")
            del state[k]
        elif k in state and k not in msd:

            if lg: lg.info(f"    dropping {k} (head removed in v2)")
            del state[k]
    for k in list(state.keys()):
        if "relative_position_index" in k:
            state.pop(k)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if lg:
        lg.info(f"    load_pretrained: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            lg.info(f"      first 5 missing keys: {missing[:5]}")






def setup_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)
    torch.backends.cudnn.deterministic = True


def setup_logger(path):
    lg = logging.getLogger(f"labram_within_v2_{path.parent.name}")
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
def evaluate(m, dl, dev, input_chans):
    m.eval(); pr, gt = [], []
    for x, y in dl:
        pr.append(m(x.to(dev, non_blocking=True), input_chans=input_chans).argmax(1).cpu().numpy())
        gt.append(y.numpy())
    pr = np.concatenate(pr) if pr else np.zeros(0, dtype=np.int64)
    gt = np.concatenate(gt) if gt else np.zeros(0, dtype=np.int64)
    if not len(pr):
        return 0., 0., 0., np.zeros((1, 1), int)
    return (accuracy_score(gt, pr),
            f1_score(gt, pr, average="macro", zero_division=0),
            cohen_kappa_score(gt, pr),
            confusion_matrix(gt, pr))


def train_one_fold(X, y, tr, va, te, n_classes, input_chans, args, dev, lg):
    n, ch, npat, _ = X.shape
    dl = lambda I, sh: DataLoader(
        TensorDataset(torch.from_numpy(X[I]), torch.from_numpy(y[I])),
        batch_size=args.batch_size, shuffle=sh, num_workers=0, pin_memory=True,
    )
    train_l, val_l, test_l = dl(tr, True), dl(va, False), dl(te, False)

    model = LaBraMClassifierV2(
        n_channels=ch, n_patches=npat, n_classes=n_classes,
        pretrained_ckpt=args.pretrained_ckpt,
        hidden=args.hidden, dropout=args.dropout,
        drop_path=args.drop_path, head_arch=args.head_arch, lg=lg,
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
        model.train(); model.backbone.eval()
        losses, t0 = [], time.time()
        for xb, yb in train_l:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.no_grad():
                tokens = model.backbone.forward_features(
                    xb, input_chans=input_chans, return_patch_tokens=True,
                )
            logits = model.head(tokens)
            loss = crit(logits, yb)
            loss.backward()
            if args.clip_value > 0:
                nn.utils.clip_grad_norm_(trainable, args.clip_value)
            opt.step(); sched.step()
            losses.append(loss.item())
        va_a, va_f, va_k, _ = evaluate(model, val_l, dev, input_chans)
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
    ta, tf, tk, tcm = evaluate(model, test_l, dev, input_chans)
    return {
        "best_epoch": best_e, "best_val_acc": float(best_v),
        "test_acc": float(ta), "test_f1": float(tf), "test_kappa": float(tk),
        "test_cm": tcm.tolist(),
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
    }, model.state_dict()


def write_txt_summary(out, fr, by, by_fold, dataset, n_classes):
    lines, chance = [], 1.0 / n_classes
    lines.append("=" * 92)
    lines.append(f"LaBraM-v2 (FROZEN backbone, channel-preserving head) {dataset} within-subject 5-fold")
    lines.append(f"Run dir : {out}")
    lines.append(f"Dataset : {dataset}  (n_classes={n_classes}, chance={chance:.4f})")
    lines.append("Model   : LaBraM-base (~5.82M params, FROZEN) + LN→MLP(C*200→256→C) head")
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
    input_chans = get_input_chans(cfg["ch_names"])
    lg.info(f"[{args.dataset}] subjects: {len(subs)}  classes={cfg['n_classes']}")
    lg.info(f"  ch_names ({len(cfg['ch_names'])}): {cfg['ch_names']}")
    lg.info(f"  input_chans (cls=0 + electrodes): {input_chans}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "model_config.json", "w") as f:
        head_desc = ("v2-linear: LN→Linear(C*200,C)" if args.head_arch == "linear"
                     else "v2-mlp: LN→Linear(C*200,hidden)→GELU→Dropout→Linear(hidden,C)")
        json.dump({
            "model": "labram_base_patch200_200",
            "head": head_desc, "head_arch": args.head_arch,
            "params_M": 5.82, "frozen_backbone": True,
            "patch_size": PATCH_SIZE, "target_sr": TARGET_SR,
            "pretrained_ckpt": args.pretrained_ckpt,
            "hidden": args.hidden, "dropout": args.dropout,
            "n_classes": cfg["n_classes"], "label_key": cfg["label_key"],
            "src_sr": cfg["src_sr"], "ch_names": cfg["ch_names"],
            "input_chans": input_chans,
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
        if len(cfg["ch_names"]) != X.shape[1]:
            raise ValueError(f"{s}: ch_names len={len(cfg['ch_names'])} ≠ X[1]={X.shape[1]}")
        folds = make_within_5fold(y, args.seed)
        final_splits[s] = folds
        sd = out / s; sd.mkdir(exist_ok=True)
        for fi, fd in enumerate(folds):
            te = np.asarray(fd["test"], int)
            tr, va = split_train_val(np.asarray(fd["train"], int), y, args.seed + fi)
            tr, va = np.asarray(tr), np.asarray(va)
            lg.info(f"    fold {fi}: train={len(tr)} val={len(va)} test={len(te)}")
            t0 = time.time()
            try:
                metrics, st = train_one_fold(X, y, tr, va, te, cfg["n_classes"],
                                             input_chans, args, dev, lg)
            except Exception as e:
                lg.exception(f"    fold {fi} TRAIN FAILED: {e}"); continue
            fld = sd / f"fold_{fi:02d}"; fld.mkdir(exist_ok=True)
            head_state = {k: v for k, v in st.items() if k.startswith("head.")}
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
            "dataset": args.dataset, "model": "LaBraM-v2-frozen",
            "n_subjects": len(by), "n_folds_total": len(fr),
            "test_acc_mean": float(np.mean([r["test_acc"] for r in fr])),
            "test_acc_std": float(np.std([r["test_acc"] for r in fr])),
            "test_f1_mean": float(np.mean([r["test_f1"] for r in fr])),
            "test_kappa_mean": float(np.mean([r["test_kappa"] for r in fr])),
            "per_subject": per_subject,
            "per_fold":    per_fold,
        }
        with open(out / "summary.json", "w") as f:
            json.dump(smr, f, indent=2)
        lg.info("===== SUMMARY =====")
        lg.info(json.dumps({k: v for k, v in smr.items() if k not in ("per_subject", "per_fold")}, indent=2))
        write_txt_summary(out, fr, by, by_fold, args.dataset, cfg["n_classes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="INNER_class8", choices=list(DATASETS.keys()))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--h5_dir", default=None)
    ap.add_argument("--pretrained_ckpt", default=str(LABRAM_DIR / "checkpoints" / "labram-base.pth"))
    ap.add_argument("--cuda", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr_head", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--head_arch", choices=["linear", "mlp"], default="linear",
                    help="linear = LN+Linear(C*200,C); mlp = + hidden GELU layer")
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--clip_value", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=20)
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
