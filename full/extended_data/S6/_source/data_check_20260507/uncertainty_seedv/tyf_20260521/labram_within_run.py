

















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
from collections import OrderedDict
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

LABRAM_DIR = Path(__file__).resolve().parent / "LaBraM"
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



BCIC_CH_NAMES = [
    "FZ", "FC3", "FC1", "FCZ", "FC2", "FC4",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP3", "CP1", "CPZ", "CP2", "CP4",
    "P1", "PZ", "P2", "POZ",
]






FACED_CH_NAMES = [
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8",
    "FC1", "FC2", "FC5", "FC6",
    "CZ", "C3", "C4", "T7", "T8",
    "CP1", "CP2", "CP5", "CP6",
    "PZ", "P3", "P4", "P7", "P8",
    "PO3", "PO4", "OZ", "O1", "O2",
    "A1", "A2",
]







STAGE2_ROOT = str(_LOCAL_SOURCE / 'data_check_20260507/full_run_5fold/stage2_new')
EXP1_ROOT = str(_LOCAL_SOURCE / 'guoyi_exp/Exp1')
FACED_PRESET_SPLITS = str(_LOCAL_SOURCE / 'data_check_20260507/uncertainty_seedv/tyf_20260518/FACED_3cls_eeg_within_frozen_cache_20260518_105117/splits.json')




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




SEEDV_CH_NAMES = [
    "FP1", "FPZ", "FP2",
    "AF3", "AF4",
    "F7",  "F5",  "F3",  "F1",  "FZ",  "F2",  "F4",  "F6",  "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7",  "C5",  "C3",  "C1",  "CZ",  "C2",  "C4",  "C6",  "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7",  "P5",  "P3",  "P1",  "PZ",  "P2",  "P4",  "P6",  "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "O1",  "OZ",  "O2",
]










MOTOR_CH_NAMES = [
    "O1", "FP2", "P5", "AF4", "C3", "C4", "AF3", "P6", "FP1", "O2",
    "TP7", "F8", "T7", "T8", "F7", "TP8", "FPZ", "CZ", "OZ", "OZ",
    "P1", "F2", "F1", "P2", "CP5", "FC6", "FC5", "CP6", "FT9", "TP10",
    "OZ", "PO3", "AF4", "C1", "C2", "AF3", "PO4", "O1", "AF4", "CP3",
    "F4", "F3", "CP4", "AF3", "O2", "P5", "F6", "C5", "C6", "F5",
    "P6", "PO7", "AF6", "T7", "FT8", "FT7", "T8", "AF5", "PO8", "TP9",
    "FT10", "FPZ", "OZ",
]



MOTOR_H5_DIR_LOCAL = f"{STAGE2_ROOT}/MOTOR"
MOTOR_SPLITS_LOCAL = str(_LOCAL_SOURCE / 'data_check_20260507/uncertainty_seedv/tyf_20260520/MOTOR_within_unifiedv3_eeg_within_frozen_cache_20260520_181156/splits.json')








SOMATO_PER_SUBJECT_CH_NAMES = {
    "sub-sm04": [
        "C2", "CZ", "C1", "CP4", "CP2", "CZ", "CP1", "CP3", "CP4", "CP4",
        "CP2", "CPZ", "CPZ", "CP1", "CP1", "CP3", "CP3", "P6", "P4", "P4",
        "CP2", "CP2", "CPZ", "CP1", "CP1", "P3", "P3", "P5", "PO6", "P4",
        "P4", "P2", "P2", "PZ", "P1", "P1", "P3", "P3", "PO5", "PO6",
        "PO4", "P4", "P2", "P2", "PZ", "P1", "P1", "P3", "PO3", "PO5",
        "PO6", "PO4", "PO4", "PO2", "PO2", "PZ", "P1", "PO1", "PO1", "PO3",
        "O1", "PO4", "PO2", "POZ", "PO1", "PO1", "PO2", "POZ", "PO1", "OZ",
    ],
    "sub-sm06": [
        "C1", "CZ", "C2", "CP3", "C1", "CZ", "C2", "CP4", "CP3", "CP1",
        "CP1", "CP1", "CPZ", "CP2", "CP2", "CP2", "CP4", "P5", "CP3", "CP3",
        "CP1", "CP1", "CPZ", "CP2", "CP2", "CP4", "P4", "P6", "P5", "P3",
        "P3", "CP1", "CP1", "CPZ", "CP2", "P2", "P4", "P4", "P6", "PO5",
        "P3", "P3", "P1", "P1", "PZ", "P2", "P2", "P4", "PO4", "PO6",
        "PO5", "PO3", "PO3", "P1", "P1", "PZ", "P2", "P2", "PO4", "PO4",
        "PO2", "PO4", "PO1", "POZ", "PO2", "OZ", "OZ", "OZ", "OZ", "OZ",
    ],
    "sub-sm07": [
        "C2", "CZ", "C1", "CP4", "C2", "CZ", "C1", "C3", "CP4", "CP4",
        "CP2", "CP2", "CZ", "CPZ", "CP1", "CP1", "CP3", "P6", "P4", "CP4",
        "CP2", "CP2", "CPZ", "CPZ", "CP1", "CP3", "P3", "P5", "PO6", "P4",
        "P4", "P2", "CP2", "CPZ", "CPZ", "CP1", "P3", "P3", "P5", "PO6",
        "PO4", "P4", "P2", "P2", "PZ", "PZ", "P1", "P1", "P3", "PO3",
        "PO4", "PO4", "PO2", "P2", "P2", "PZ", "PZ", "P1", "P1", "PO3",
        "PO1", "PO1", "PO2", "POZ", "PO1", "OZ", "OZ", "OZ", "OZ", "OZ",
    ],
    "sub-sm09": [
        "C1", "CZ", "CP2", "CP3", "CP1", "CPZ", "CP2", "CP4", "P3", "CP3",
        "CP1", "CPZ", "CPZ", "CP2", "CP2", "CP4", "P4", "PO5", "P3", "P3",
        "CP1", "CPZ", "CPZ", "CP2", "CP2", "P4", "P4", "P6", "PO5", "P3",
        "P3", "P1", "P1", "CPZ", "P2", "P2", "P4", "P4", "PO6", "PO5",
        "PO3", "P3", "P1", "P1", "PZ", "P2", "P2", "P4", "PO4", "PO6",
        "PO3", "PO3", "PO1", "P1", "P1", "PZ", "PZ", "P2", "PO2", "PO4",
        "PO4", "PO1", "PO1", "POZ", "PO2", "PO2", "PO1", "POZ", "PO2", "OZ",
    ],
    "sub-sm12": [
        "FC1", "FCZ", "FC2", "C3", "C1", "CZ", "C2", "C4", "CP3", "CP3",
        "C1", "C1", "CZ", "C2", "C2", "CP4", "CP4", "CP5", "CP3", "CP3",
        "CP1", "CP1", "CPZ", "CP2", "CP2", "CP4", "CP6", "P6", "P5", "P5",
        "CP3", "CP1", "CP1", "CPZ", "CP2", "CP2", "P4", "P6", "P6", "PO5",
        "P3", "P3", "P1", "P1", "PZ", "P2", "P2", "P4", "P4", "PO6",
        "PO5", "PO3", "P3", "P1", "P1", "PZ", "P2", "P2", "P4", "PO4",
        "PO6", "PO3", "PO1", "POZ", "PO2", "PO4", "PO1", "POZ", "PO2", "OZ",
    ],
}
SOMATO_DEFAULT_CH_NAMES = SOMATO_PER_SUBJECT_CH_NAMES["sub-sm04"]



SOMATO_H5_DIR = (str(_LOCAL_SOURCE / 'data_check_20260507/uncertainty_seedv/tyf_20260521/somatomotor_patched'))


DATASETS = {
    "BCIC": dict(
        h5_dir=f"{STAGE2_ROOT}/BCIC_correct",
        label_key="labels",
        n_classes=4,
        src_sr=256,
        ch_names=BCIC_CH_NAMES,
        splits_json=f"{EXP1_ROOT}/per_dataset_20260515_034656/BCIC_correct_eeg_within_frozen_cache_20260515_034703/splits.json",
    ),
    "FACED": dict(
        h5_dir=f"{STAGE2_ROOT}/FACED_new",
        label_key="label2",
        n_classes=3,
        src_sr=256,
        ch_names=FACED_CH_NAMES,
        splits_json=FACED_PRESET_SPLITS,
    ),
    "INNER_class2": dict(
        h5_dir=f"{STAGE2_ROOT}/INNER_class2",
        label_key="labels",
        n_classes=2,
        src_sr=256,
        ch_names=INNER_CH_NAMES,
        splits_json=None,
    ),
    "INNER_class8": dict(
        h5_dir=f"{STAGE2_ROOT}/INNER_class8",
        label_key="labels",
        n_classes=8,
        src_sr=256,
        ch_names=INNER_CH_NAMES,
        splits_json=None,
    ),
    "SEED-V": dict(
        h5_dir=f"{STAGE2_ROOT}/SEED-V",
        label_key="labels",
        n_classes=5,
        src_sr=256,
        ch_names=SEEDV_CH_NAMES,
        splits_json=None,
    ),
    "MOTOR": dict(
        h5_dir=MOTOR_H5_DIR_LOCAL,
        label_key="labels",
        n_classes=2,
        src_sr=256,
        ch_names=MOTOR_CH_NAMES,
        splits_json=MOTOR_SPLITS_LOCAL,
    ),
    "somatomotor": dict(
        h5_dir=SOMATO_H5_DIR,
        label_key="labels",
        n_classes=2,



        src_sr=256,
        ch_names=SOMATO_DEFAULT_CH_NAMES,
        ch_names_per_subject=SOMATO_PER_SUBJECT_CH_NAMES,
        splits_json=None,
    ),
}
TARGET_SR = 200
PATCH_SIZE = 200






def setup_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)
    torch.backends.cudnn.deterministic = True


def setup_logger(path):
    lg = logging.getLogger(f"labram_{path.parent.name}")
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


def build_labram(n_classes, drop_path=0.1, qkv_bias=False, init_values=0.1,
                 use_mean_pooling=True, use_abs_pos_emb=True):
    return create_model(
        "labram_base_patch200_200",
        pretrained=False,
        num_classes=n_classes,
        in_chans=1,
        out_chans=8,
        use_mean_pooling=use_mean_pooling,
        drop_path_rate=drop_path,
        use_abs_pos_emb=use_abs_pos_emb,
        init_values=init_values,
        qkv_bias=qkv_bias,
    )


def load_pretrained(model, ckpt_path, lg=None):

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck.get("model", ck.get("module", ck))
    new = OrderedDict()
    for k, v in state.items():
        if k.startswith("student."):
            new[k[len("student."):]] = v
    state = new if new else state
    msd = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in state and state[k].shape != msd[k].shape:
            if lg: lg.info(f"    dropping {k} (shape mismatch, new classifier)")
            del state[k]
    for k in list(state.keys()):
        if "relative_position_index" in k:
            state.pop(k)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if lg:
        lg.info(f"    load_pretrained: missing={len(missing)}  unexpected={len(unexpected)}")
        if missing:
            lg.info(f"      first 5 missing keys: {missing[:5]}")


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

    model = build_labram(n_classes=n_classes, drop_path=args.drop_path,
                         qkv_bias=False, init_values=0.1).to(dev)
    if args.pretrained_ckpt:
        load_pretrained(model, args.pretrained_ckpt, lg=lg)

    if args.frozen:
        for name, p in model.named_parameters():
            if not name.startswith("head"):
                p.requires_grad = False

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if name.startswith("head") else backbone_params).append(p)

    if args.frozen or not backbone_params:
        opt = torch.optim.AdamW(head_params, lr=args.lr_head, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(
            [{"params": backbone_params, "lr": args.lr_backbone},
             {"params": head_params,     "lr": args.lr_head}],
            weight_decay=args.weight_decay,
        )
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
            logits = model(xb, input_chans=input_chans)
            loss = crit(logits, yb)
            loss.backward()
            if args.clip_value > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
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
    lines.append(f"LaBraM {dataset} within-subject 5-fold — per-fold accuracy and aggregates")
    lines.append(f"Run dir : {out}")
    lines.append(f"Dataset : {dataset}  (n_classes={n_classes}, chance={chance:.4f})")
    lines.append("Model   : LaBraM-base (12-layer transformer, ~5.8M params), pretrained on ~2500h EEG")
    lines.append("Head    : Linear(embed_dim=200, n_classes), backbone unfrozen + cosine LR + early stop")
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
    lines.append(f"  test_f1 (macro) : mean={f1s.mean():.4f}  std={f1s.std():.4f}  "
                 f"min={f1s.min():.4f}  max={f1s.max():.4f}")
    lines.append(f"  test_kappa      : mean={kappas.mean():.4f}  std={kappas.std():.4f}  "
                 f"min={kappas.min():.4f}  max={kappas.max():.4f}")
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
    per_sub_chs = cfg.get("ch_names_per_subject") or {}
    lg.info(f"[{args.dataset}] subjects: {len(subs)}  classes={cfg['n_classes']}")
    lg.info(f"  ch_names ({len(cfg['ch_names'])}): {cfg['ch_names']}")
    lg.info(f"  input_chans (cls=0 + electrodes, default): {input_chans}")
    if per_sub_chs:
        lg.info(f"  per-subject ch_names override available for {len(per_sub_chs)} subjects")
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
            "model": "labram_base_patch200_200",
            "params_M": 5.82, "patch_size": PATCH_SIZE, "target_sr": TARGET_SR,
            "pretrained_ckpt": args.pretrained_ckpt,
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
        sub_chs = per_sub_chs.get(s, cfg["ch_names"])
        if len(sub_chs) != X.shape[1]:
            raise ValueError(f"{s}: ch_names len={len(sub_chs)} ≠ X[1]={X.shape[1]}")
        sub_input_chans = get_input_chans(sub_chs)
        if s in per_sub_chs:
            lg.info(f"    per-subject input_chans ({len(sub_chs)} ch, {len(set(sub_chs))} unique 1020 names)")
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
                metrics, st = train_one_fold(X, y, tr, va, te, cfg["n_classes"],
                                             sub_input_chans, args, dev, lg)
            except Exception as e:
                lg.exception(f"    fold {fi} TRAIN FAILED: {e}"); continue
            fld = sd / f"fold_{fi:02d}"; fld.mkdir(exist_ok=True)
            head_state = {k: v for k, v in st.items() if k.startswith("head.")}
            safe_save(head_state, fld / "head_weights.pth", lg=lg)
            if args.save_full_state:
                safe_save(st, fld / "full_state.pth", lg=lg)
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
            "dataset": args.dataset, "model": "LaBraM-base",
            "n_subjects": len(by), "n_folds_total": len(fr),
            "test_acc_mean": float(np.mean([r["test_acc"] for r in fr])),
            "test_acc_std": float(np.std([r["test_acc"] for r in fr])),
            "test_f1_mean": float(np.mean([r["test_f1"] for r in fr])),
            "test_f1_std": float(np.std([r["test_f1"] for r in fr])),
            "test_kappa_mean": float(np.mean([r["test_kappa"] for r in fr])),
            "test_kappa_std": float(np.std([r["test_kappa"] for r in fr])),
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
                    help="Override h5_dir from DATASETS registry (useful when data is on a different mount)")
    ap.add_argument("--pretrained_ckpt", default=str(LABRAM_DIR / "checkpoints" / "labram-base.pth"))
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_backbone", type=float, default=5e-4)
    ap.add_argument("--lr_head", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--clip_value", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--max_subjects", type=int, default=0)
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="Restrict to a subset of subjects, e.g. --subjects sub-06 sub-07 ...")
    ap.add_argument("--save_full_state", action="store_true")
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
