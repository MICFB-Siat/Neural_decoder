












import argparse
import copy
import datetime
import json
import logging
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
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
LABRAM_DIR = SCRIPT_DIR / "LaBraM"
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
assert len(SEEDV_CH_NAMES) == 60

TARGET_SR = 200
PATCH_SIZE = 200
SRC_SR = 256
N_CLASSES = 5
LABEL_KEY = "labels"


def get_input_chans(ch_names):
    return [0] + [STANDARD_1020.index(c) + 1 for c in ch_names]


def setup_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)
    torch.backends.cudnn.deterministic = True


def setup_logger(path):
    lg = logging.getLogger(f"labram_perturb_{path.parent.name}")
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


def load_subject(h5_path):
    with h5py.File(h5_path, "r") as f:
        eeg = f["eeg_raw"][:]
        y = f[LABEL_KEY][:]
    return make_patches(zscore(resample_to(TARGET_SR, eeg, SRC_SR))), y.astype(np.int64)


def build_labram(n_classes, drop_path=0.1):
    return create_model(
        "labram_base_patch200_200",
        pretrained=False, num_classes=n_classes,
        in_chans=1, out_chans=8, use_mean_pooling=True,
        drop_path_rate=drop_path, use_abs_pos_emb=True,
        init_values=0.1, qkv_bias=False,
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
            del state[k]
    for k in list(state.keys()):
        if "relative_position_index" in k:
            state.pop(k)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if lg:
        lg.info(f"    load_pretrained: missing={len(missing)}  unexpected={len(unexpected)}")


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
def predict(model, dl, dev, input_chans):
    model.eval(); pr, gt = [], []
    for x, y in dl:
        pr.append(model(x.to(dev, non_blocking=True), input_chans=input_chans).argmax(1).cpu().numpy())
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


def train_one_fold(X_tr, y_tr, X_va, y_va, X_te, y_te, subj_te,
                   n_classes, input_chans, args, dev, lg):
    def dl(Xs, ys, sh, bs):
        return DataLoader(
            TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys)),
            batch_size=bs, shuffle=sh, num_workers=0, pin_memory=True,
        )
    train_l = dl(X_tr, y_tr, True, args.batch_size)
    val_l   = dl(X_va, y_va, False, args.batch_size)
    test_l  = dl(X_te, y_te, False, args.batch_size)

    model = build_labram(n_classes=n_classes, drop_path=args.drop_path).to(dev)
    if args.pretrained_ckpt:
        load_pretrained(model, args.pretrained_ckpt, lg=lg)
    if args.frozen:
        for name, p in model.named_parameters():
            if not name.startswith("head"):
                p.requires_grad = False
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        (head_params if name.startswith("head") else backbone_params).append(p)
    n_train = sum(p.numel() for p in backbone_params) + sum(p.numel() for p in head_params)
    n_total = sum(p.numel() for p in model.parameters())
    lg.info(f"    {'FROZEN' if args.frozen else 'unfrozen'} — trainable {n_train/1e6:.3f}M / {n_total/1e6:.3f}M")
    opt = (torch.optim.AdamW(head_params, lr=args.lr_head, weight_decay=args.weight_decay)
           if args.frozen or not backbone_params
           else torch.optim.AdamW(
               [{"params": backbone_params, "lr": args.lr_backbone},
                {"params": head_params,     "lr": args.lr_head}],
               weight_decay=args.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs * len(train_l)), eta_min=1e-6)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(dev)

    best_v, best_s, best_e, ni = -1., None, -1, 0
    for ep in range(args.epochs):
        model.train()
        if args.frozen:
            for name, mod in model.named_modules():
                if not (name == "head" or name.startswith("head.") or
                        name == "fc_norm" or name.startswith("fc_norm.")):
                    mod.eval()
        losses, t0 = [], time.time()
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
        va_pr, va_gt = predict(model, val_l, dev, input_chans)
        va_a, va_f, va_k, _ = scores_from(va_pr, va_gt, default_cm_dim=n_classes)
        lg.info(f"      epoch {ep+1:03d}/{args.epochs} loss={np.mean(losses):.4f} "
                f"val_acc={va_a:.4f} ({time.time()-t0:.1f}s)")
        if va_a > best_v:
            best_v, best_s, best_e, ni = va_a, copy.deepcopy(model.state_dict()), ep + 1, 0
        else:
            ni += 1
            if args.patience > 0 and ni >= args.patience:
                lg.info(f"      early stop at epoch {ep+1}"); break
    if best_s is not None:
        model.load_state_dict(best_s)

    pr, gt = predict(model, test_l, dev, input_chans)
    test_acc, test_f1, test_kappa, test_cm = scores_from(pr, gt, default_cm_dim=n_classes)
    per_sub = {}
    for s in np.unique(subj_te):
        mask = subj_te == s
        if mask.sum() == 0: continue
        a, f1, k, _ = scores_from(pr[mask], gt[mask], default_cm_dim=n_classes)
        per_sub[str(s)] = {"n": int(mask.sum()),
                           "test_acc": float(a), "test_f1": float(f1), "test_kappa": float(k)}
    return {
        "best_epoch": best_e, "best_val_acc": float(best_v),
        "test_acc": float(test_acc), "test_f1": float(test_f1),
        "test_kappa": float(test_kappa), "test_cm": test_cm.tolist(),
        "n_train": int(len(X_tr)), "n_val": int(len(X_va)), "n_test": int(len(X_te)),
        "per_test_subject": per_sub,
    }, model.state_dict(), pr, gt


def write_txt_summary(out, fold_records, per_subj_records, tag, n_classes,
                      frozen=True, splits_json=""):
    lines, chance = [], 1.0 / n_classes
    bar = "=" * 92
    dash = "-" * 92
    lines += [bar,
              f"LaBraM{' (FROZEN)' if frozen else ''} SEED-V[{tag}] cross-subject 5-fold",
              f"Run dir : {out}",
              f"Dataset : SEED-V/{tag}  ({n_classes}-class, chance = {chance:.4f})",
              f"Splits  : preset ({splits_json})", bar, ""]
    lines.append(f"{'fold':<5}  {'test_subjects':<30}  {'n_train':>7}  {'n_val':>5}  "
                 f"{'n_test':>6}  {'best_ep':>7}  {'test_acc':>8}  {'test_f1':>8}  {'test_kappa':>10}")
    lines.append(dash)
    fold_accs, fold_f1s, fold_kappas = [], [], []
    fold_to_subs = {}
    for r in sorted(per_subj_records, key=lambda x: x["fold"]):
        fold_to_subs.setdefault(r["fold"], []).append(r)
    for r in sorted(fold_records, key=lambda x: x["fold"]):
        ts = ",".join(r["test_subjects"][:5]) + ("..." if len(r["test_subjects"]) > 5 else "")
        lines.append(f"{r['fold']:>4d}  {ts:<30}  {r['n_train']:>7d}  {r['n_val']:>5d}  "
                     f"{r['n_test']:>6d}  {r['best_epoch']:>7d}  "
                     f"{r['test_acc']:>8.4f}  {r['test_f1']:>8.4f}  {r['test_kappa']:>10.4f}")
        fold_accs.append(r['test_acc']); fold_f1s.append(r['test_f1']); fold_kappas.append(r['test_kappa'])
    fold_accs = np.asarray(fold_accs); fold_f1s = np.asarray(fold_f1s); fold_kappas = np.asarray(fold_kappas)
    lines += [dash, "", "Per-fold mean ± std (across test subjects):"]
    lines.append(f"{'fold':<5}  {'acc_mean':>10}  {'acc_std':>9}  {'f1_mean':>9}  {'kappa_mean':>11}  {'n_subj':>6}")
    lines.append(dash)
    for fi in sorted(fold_to_subs):
        rs = fold_to_subs[fi]
        a = np.array([r["test_acc"] for r in rs])
        f1 = np.array([r["test_f1"] for r in rs])
        k = np.array([r["test_kappa"] for r in rs])
        lines.append(f"{fi:>4d}  {a.mean():>10.4f}  {a.std():>9.4f}  "
                     f"{f1.mean():>9.4f}  {k.mean():>11.4f}  {len(rs):>6d}")
    lines += [dash, "", "Per-test-subject breakdown:"]
    lines.append(f"{'subject':<10}  {'fold':>4}  {'n':>5}  {'acc':>7}  {'f1':>7}  {'kappa':>8}")
    lines.append(dash)
    sub_accs = []
    for r in sorted(per_subj_records, key=lambda x: (x["fold"], x["subject"])):
        lines.append(f"{r['subject']:<10}  {r['fold']:>4d}  {r['n']:>5d}  "
                     f"{r['test_acc']:>7.4f}  {r['test_f1']:>7.4f}  {r['test_kappa']:>8.4f}")
        sub_accs.append(r['test_acc'])
    sub_accs = np.asarray(sub_accs)
    lines += [dash, "",
              "Per-fold aggregate:",
              f"  test_acc : mean={fold_accs.mean():.4f}  std={fold_accs.std():.4f}  "
              f"min={fold_accs.min():.4f}  max={fold_accs.max():.4f}  median={np.median(fold_accs):.4f}",
              f"  test_f1  : mean={fold_f1s.mean():.4f}  std={fold_f1s.std():.4f}",
              f"  test_kap : mean={fold_kappas.mean():.4f}  std={fold_kappas.std():.4f}",
              "",
              f"Subject-level aggregate ({len(sub_accs)} subjects):",
              f"  mean ± std = {sub_accs.mean():.4f} ± {sub_accs.std():.4f}",
              f"  chance level = {chance:.4f}",
              f"  above chance = {fold_accs.mean()-chance:+.4f} (per-fold) / "
              f"{sub_accs.mean()-chance:+.4f} (per-subject)",
              bar]
    (out / "per_fold_accuracy_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def run(args, lg):
    h5d = Path(args.h5_dir)
    input_chans = get_input_chans(SEEDV_CH_NAMES)
    lg.info(f"[SEED-V/{args.tag}]  h5={h5d}  classes={N_CLASSES}  ch={len(SEEDV_CH_NAMES)}")

    with open(args.splits_json) as f:
        splits = json.load(f)
    lg.info(f"  preset splits: {args.splits_json}  ({len(splits)} folds)")
    all_subs = sorted({s for fd in splits
                       for s in fd.get("train_subjects", []) + (fd.get("test_subjects") or [])})

    cache = {}
    for s in all_subs:
        p = h5d / f"{s}.h5"
        if not p.is_file():
            lg.warning(f"  skip {s}: file missing"); continue
        X, y = load_subject(str(p))
        cache[s] = (X, y)
        lg.info(f"  loaded {s}: X={X.shape}  y_counts={np.bincount(y).tolist()}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "model_config.json", "w") as f:
        json.dump({
            "model": "labram_base_patch200_200", "params_M": 5.82,
            "pretrained_ckpt": args.pretrained_ckpt, "frozen": args.frozen,
            "n_classes": N_CLASSES, "ch_names": SEEDV_CH_NAMES,
            "input_chans": input_chans, "splits_json": args.splits_json,
            "h5_dir": str(h5d), "tag": args.tag, "src_sr": SRC_SR,
            "target_sr": TARGET_SR, "patch_size": PATCH_SIZE,
        }, f, indent=2)

    dev = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    fold_records, per_subj_records = [], []
    for fold in splits:
        fi = fold["fold"]
        if args.fold is not None and fi != args.fold:
            continue
        fld = out / f"fold_{fi:02d}"; fld.mkdir(exist_ok=True)
        metrics_path = fld / "metrics.json"
        if metrics_path.is_file():
            lg.info(f"==== fold {fi}: RESUME (metrics.json exists) ====")
            with open(metrics_path) as f:
                m = json.load(f)
            fold_records.append({k: m[k] for k in m if k != "per_test_subject"} |
                                {"train_subjects": fold["train_subjects"],
                                 "test_subjects": fold["test_subjects"], "fold": fi})
            for s, sm in m.get("per_test_subject", {}).items():
                per_subj_records.append({"fold": fi, "subject": s, **sm})
            continue

        train_subs = [s for s in fold["train_subjects"] if s in cache]
        test_subs  = [s for s in (fold.get("test_subjects") or []) if s in cache]
        lg.info(f"==== fold {fi}: train={len(train_subs)} subs  test={test_subs} ====")

        def stack(subs):
            Xs, ys, sids = [], [], []
            for s in subs:
                if s not in cache: continue
                X, y = cache[s]
                Xs.append(X); ys.append(y)
                sids.append(np.array([s] * len(y)))
            if not Xs:
                first = next(iter(cache.values()))[0]
                return (np.zeros((0,) + first.shape[1:], dtype=np.float32),
                        np.zeros(0, dtype=np.int64), np.array([], dtype=object))
            return (np.concatenate(Xs, axis=0).astype(np.float32),
                    np.concatenate(ys, axis=0).astype(np.int64),
                    np.concatenate(sids, axis=0))

        Xtr_all, ytr_all, _ = stack(train_subs)
        Xte, yte, subj_te = stack(test_subs)
        if len(Xtr_all) == 0 or len(Xte) == 0:
            lg.warning(f"  fold {fi}: empty — skip"); continue

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=args.seed + fi)
        a, b = next(sss.split(np.zeros_like(ytr_all), ytr_all))
        X_tr, y_tr = Xtr_all[a], ytr_all[a]
        X_va, y_va = Xtr_all[b], ytr_all[b]
        lg.info(f"  shapes  X_tr={X_tr.shape}  X_va={X_va.shape}  X_te={Xte.shape}")

        t0 = time.time()
        try:
            metrics, st, pr, gt = train_one_fold(
                X_tr, y_tr, X_va, y_va, Xte, yte, subj_te,
                N_CLASSES, input_chans, args, dev, lg,
            )
        except Exception as e:
            lg.exception(f"  fold {fi} TRAIN FAILED: {e}"); continue

        head_state = {k: v.cpu() for k, v in st.items() if k.startswith("head.")}
        safe_save(head_state, fld / "head_weights.pth", lg=lg)
        if args.save_full_state:
            safe_save({k: v.cpu() for k, v in st.items()}, fld / "full_state.pth", lg=lg)
        np.savez(fld / "test_predictions.npz", pred=pr, true=gt, subject=subj_te)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        lg.info(f"  fold {fi} DONE  test_acc={metrics['test_acc']:.4f}  "
                f"test_f1={metrics['test_f1']:.4f}  ({(time.time()-t0)/60:.2f} min)")

        fold_records.append({
            "fold": fi, "train_subjects": train_subs, "test_subjects": test_subs,
            **{k: metrics[k] for k in
               ("best_epoch", "best_val_acc", "test_acc", "test_f1", "test_kappa",
                "test_cm", "n_train", "n_val", "n_test")},
        })
        for s, sm in metrics["per_test_subject"].items():
            per_subj_records.append({"fold": fi, "subject": s, **sm})
        with open(out / "fold_records.json", "w") as f:
            json.dump(fold_records, f, indent=2)
        with open(out / "per_subject_records.json", "w") as f:
            json.dump(per_subj_records, f, indent=2)

    if fold_records:
        fold_accs = np.array([r["test_acc"] for r in fold_records])
        fold_f1s  = np.array([r["test_f1"] for r in fold_records])
        fold_kaps = np.array([r["test_kappa"] for r in fold_records])
        sub_accs  = np.array([r["test_acc"] for r in per_subj_records]) if per_subj_records else np.array([])
        smr = {
            "tag": args.tag, "model": "LaBraM-base",
            "frozen": args.frozen, "n_folds": len(fold_records),
            "fold_test_acc_mean": float(fold_accs.mean()),
            "fold_test_acc_std":  float(fold_accs.std()),
            "fold_test_f1_mean":  float(fold_f1s.mean()),
            "fold_test_kappa_mean": float(fold_kaps.mean()),
            "subj_test_acc_mean": float(sub_accs.mean()) if len(sub_accs) else None,
            "subj_test_acc_std":  float(sub_accs.std())  if len(sub_accs) else None,
        }
        with open(out / "summary.json", "w") as f:
            json.dump(smr, f, indent=2)
        lg.info("===== SUMMARY =====")
        lg.info(json.dumps(smr, indent=2))
        write_txt_summary(out, fold_records, per_subj_records, args.tag,
                          N_CLASSES, frozen=args.frozen, splits_json=args.splits_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_dir", required=True,
                    help="Dir containing sub-*.h5 for one perturbation tag.")
    ap.add_argument("--tag", required=True,
                    help="clean / noise_03 / drop_05 / red_10 / ...")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits_json", default=str(SCRIPT_DIR / "seedv_cross_splits.json"))
    ap.add_argument("--pretrained_ckpt",
                    default=str(LABRAM_DIR / "checkpoints" / "labram-base.pth"))
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr_backbone", type=float, default=5e-4)
    ap.add_argument("--lr_head", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--clip_value", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=8)

    ap.add_argument("--frozen", dest="frozen", action="store_true", default=True)
    ap.add_argument("--no_frozen", dest="frozen", action="store_false")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--save_full_state", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    lg = setup_logger(out / "log.txt"); setup_seed(args.seed)
    lg.info(f"Run launched at {datetime.datetime.now().isoformat()}")
    lg.info(f"args: {vars(args)}")
    with open(out / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    run(args, lg)
    lg.info("ALL DONE")


if __name__ == "__main__":
    main()
