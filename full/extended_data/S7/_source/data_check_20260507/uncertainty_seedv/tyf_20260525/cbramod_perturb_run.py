






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
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
CBRAMOD_DIR = Path(os.environ.get("CBRAMOD_DIR", SCRIPT_DIR / "CBraMod"))
sys.path.insert(0, str(CBRAMOD_DIR))
from models.cbramod import CBraMod


CBRAMOD_TARGET_SR = 200
CBRAMOD_PATCH_SIZE = 200
SRC_SR = 256
N_CLASSES = 5
LABEL_KEY = "labels"


def setup_seed(seed: int):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"cbramod_perturb_{log_path.parent.name}")
    logger.setLevel(logging.INFO); logger.handlers = []
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


def resample_to_200hz(eeg: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == CBRAMOD_TARGET_SR:
        return eeg.astype(np.float32, copy=False)
    from math import gcd
    g = gcd(CBRAMOD_TARGET_SR, src_sr)
    return resample_poly(eeg, CBRAMOD_TARGET_SR // g, src_sr // g, axis=-1).astype(np.float32)


def make_patches(eeg_200: np.ndarray, patch_size: int = CBRAMOD_PATCH_SIZE) -> np.ndarray:
    n, ch, T = eeg_200.shape
    npat = T // patch_size
    return eeg_200[..., : npat * patch_size].reshape(n, ch, npat, patch_size).astype(np.float32)


def zscore_per_channel(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)


def load_subject(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        eeg = f["eeg_raw"][:]; y = f[LABEL_KEY][:]
    return make_patches(zscore_per_channel(resample_to_200hz(eeg, SRC_SR))).astype(np.float32), y.astype(np.int64)


def load_all_subjects(h5_dir: Path, subjects, logger):
    bank = {}
    for si, sub in enumerate(subjects):
        h5_path = h5_dir / f"{sub}.h5"
        if not h5_path.is_file():
            logger.warning(f"  skip {sub}: file missing"); continue
        X, y = load_subject(str(h5_path))
        logger.info(f"  ({si+1}/{len(subjects)}) {sub}: X={X.shape}  y_counts={np.bincount(y).tolist()}")
        bank[sub] = (X, y)
    return bank


def stack_subjects(bank, subs):
    Xs, ys, owners = [], [], []
    for s in subs:
        if s not in bank: continue
        X, y = bank[s]
        Xs.append(X); ys.append(y)
        owners.extend([s] * len(y))
    if not Xs:
        first = next(iter(bank.values()))[0]
        return (np.zeros((0,) + first.shape[1:], dtype=np.float32),
                np.zeros(0, dtype=np.int64), np.array([], dtype=object))
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0), np.asarray(owners, dtype=object)


def normalize_splits(raw, subjects):
    out = []
    for entry in raw:
        tr = [s for s in (entry.get("train_subjects") or []) if s in subjects]
        te = [s for s in (entry.get("test_subjects") or entry.get("val_subjects") or []) if s in subjects]
        out.append({"fold": int(entry.get("fold", len(out))),
                    "train_subjects": tr, "test_subjects": te})
    return out


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
            nn.Linear(200, 200), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, n_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def safe_torch_save(obj, path: Path, retries: int = 3, logger=None):
    last_err = None
    for i in range(retries):
        try:
            torch.save(obj, path); return True
        except (OSError, RuntimeError) as e:
            last_err = e
            if logger: logger.warning(f"  torch.save retry {i+1}/{retries}: {e}")
            time.sleep(0.5 * (i + 1))
    if logger: logger.error(f"  torch.save permanently failed for {path}: {last_err}")
    return False


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); preds, gts = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        preds.append(model(xb).argmax(dim=1).cpu().numpy())
        gts.append(yb.numpy())
    preds = np.concatenate(preds) if preds else np.zeros(0, dtype=np.int64)
    gts = np.concatenate(gts) if gts else np.zeros(0, dtype=np.int64)
    if len(preds) == 0:
        return 0.0, 0.0, 0.0, np.zeros((1, 1), dtype=int), preds, gts
    return (float(accuracy_score(gts, preds)),
            float(f1_score(gts, preds, average="macro", zero_division=0)),
            float(cohen_kappa_score(gts, preds)),
            confusion_matrix(gts, preds), preds, gts)


def train_one_fold(X_tr, y_tr, X_va, y_va, X_te, y_te,
                   backbone_ckpt, device, args, logger):
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                            batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
                             batch_size=args.batch_size, shuffle=False)

    n, ch, seq, ps = X_tr.shape
    model = CBraModClassifier(ch, seq, N_CLASSES, backbone_ckpt, dropout=args.dropout).to(device)
    if args.frozen:
        for p in model.backbone.parameters():
            p.requires_grad = False

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        (backbone_params if "backbone" in name else head_params).append(p)
    n_train = sum(p.numel() for p in backbone_params) + sum(p.numel() for p in head_params)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"  {'FROZEN' if args.frozen else 'unfrozen'} — trainable {n_train/1e6:.3f}M / {n_total/1e6:.3f}M")

    optimizer = (torch.optim.AdamW(head_params, lr=args.lr_head, weight_decay=args.weight_decay)
                 if args.frozen or not backbone_params
                 else torch.optim.AdamW(
                     [{"params": backbone_params, "lr": args.lr_backbone},
                      {"params": head_params,     "lr": args.lr_head}],
                     weight_decay=args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * len(train_loader)), eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    best_val_acc, best_state, best_epoch, no_improve = -1.0, None, -1, 0
    for epoch in range(args.epochs):
        model.train()
        losses, t0 = [], time.time()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb); loss.backward()
            if args.clip_value > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
            optimizer.step(); scheduler.step()
            losses.append(loss.item())
        val_acc, val_f1, val_kappa, _, _, _ = evaluate(model, val_loader, device)
        logger.info(f"    epoch {epoch+1:03d}/{args.epochs} loss={np.mean(losses):.4f} "
                    f"val_acc={val_acc:.4f} ({time.time()-t0:.1f}s)")
        if val_acc > best_val_acc:
            best_val_acc, best_state, best_epoch, no_improve = (
                val_acc, copy.deepcopy(model.state_dict()), epoch + 1, 0)
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                logger.info(f"    early stop at epoch {epoch+1}"); break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, test_f1, test_kappa, test_cm, preds, gts = evaluate(model, test_loader, device)
    return {
        "best_epoch": best_epoch, "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc), "test_f1": float(test_f1),
        "test_kappa": float(test_kappa), "test_cm": test_cm.tolist(),
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "n_test": int(len(y_te)),
    }, model.state_dict(), preds, gts


def per_subject_breakdown(preds, gts, owners):
    out = {}
    for sub in sorted(set(owners.tolist())):
        mask = owners == sub
        if mask.sum() == 0: continue
        out[sub] = {"n": int(mask.sum()),
                    "acc": float(accuracy_score(gts[mask], preds[mask])),
                    "f1": float(f1_score(gts[mask], preds[mask], average="macro", zero_division=0)),
                    "kappa": float(cohen_kappa_score(gts[mask], preds[mask]))}
    return out


def write_per_fold_summary(out_root: Path, fold_records, tag, args, splits_path):
    chance = 1.0 / max(1, N_CLASSES)
    lines, bar, dash = [], "=" * 92, "-" * 92
    lines += [bar, f"CBraMod ({'FROZEN' if args.frozen else 'unfrozen'}) "
              f"SEED-V[{tag}] cross-subject 5-fold",
              f"Run dir : {out_root}",
              f"Dataset : SEED-V/{tag}  ({N_CLASSES}-class, chance = {chance:.4f})",
              f"Splits  : preset ({splits_path})", bar, ""]
    lines.append(f"{'fold':<5}  {'test_subjects':<30}  {'n_train':>7}  {'n_val':>5}  "
                 f"{'n_test':>6}  {'best_ep':>7}  {'test_acc':>8}  {'test_f1':>8}  {'test_kappa':>10}")
    lines.append(dash)
    accs = []
    for r in sorted(fold_records, key=lambda x: x["fold"]):
        ts = ",".join(r["test_subjects"][:5]) + ("..." if len(r["test_subjects"]) > 5 else "")
        lines.append(f"{r['fold']:>4d}  {ts:<30}  {r['n_train']:>7d}  {r['n_val']:>5d}  "
                     f"{r['n_test']:>6d}  {r['best_epoch']:>7d}  "
                     f"{r['test_acc']:>8.4f}  {r['test_f1']:>8.4f}  {r['test_kappa']:>10.4f}")
        accs.append(r["test_acc"])
    accs = np.asarray(accs)
    lines += [dash, "",
              "Per-fold aggregate:",
              f"  test_acc : mean={accs.mean():.4f}  std={accs.std():.4f}  "
              f"min={accs.min():.4f}  max={accs.max():.4f}  median={np.median(accs):.4f}",
              f"  chance level = {chance:.4f}",
              f"  above chance = {accs.mean() - chance:+.4f}",
              bar]

    sub_lines = [bar, "Per-test-subject breakdown:",
                 f"{'subject':<10}  {'fold':>4}  {'n':>5}  {'acc':>7}  {'f1':>7}  {'kappa':>8}", dash]
    sub_accs = []
    for r in sorted(fold_records, key=lambda x: x["fold"]):
        for sub, sm in sorted(r.get("per_subject_test", {}).items()):
            sub_lines.append(f"{sub:<10}  {r['fold']:>4d}  {sm['n']:>5d}  "
                             f"{sm['acc']:>7.4f}  {sm['f1']:>7.4f}  {sm['kappa']:>8.4f}")
            sub_accs.append(sm["acc"])
    sub_accs = np.asarray(sub_accs)
    sub_lines += [dash,
                  f"Subject-level aggregate ({len(sub_accs)} subjects): "
                  f"mean ± std = {sub_accs.mean():.4f} ± {sub_accs.std():.4f}"
                  if len(sub_accs) else "(no per-subject data)",
                  bar]
    (out_root / "per_fold_accuracy_summary.txt").write_text("\n".join(lines + [""] + sub_lines), encoding="utf-8")


def run(args, logger):
    h5_dir = Path(args.h5_dir)
    splits_json_path = args.splits_json
    subjects = sorted([p.stem for p in h5_dir.glob("sub-*.h5")])
    logger.info(f"[SEED-V/{args.tag}] subjects: {len(subjects)}  classes={N_CLASSES}")

    with open(splits_json_path) as f:
        raw = json.load(f)
    folds = normalize_splits(raw, set(subjects))
    logger.info(f"  preset splits: {splits_json_path}")

    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "model_config.json", "w") as f:
        json.dump({
            "backbone": "CBraMod(in_dim=200,out_dim=200,d_model=200,dim_feedforward=800,seq_len=30,n_layer=12,nhead=8)",
            "backbone_ckpt": args.backbone_ckpt, "frozen": args.frozen,
            "n_classes": N_CLASSES, "label_key": LABEL_KEY,
            "src_sr": SRC_SR, "target_sr": CBRAMOD_TARGET_SR,
            "splits_json": splits_json_path, "h5_dir": str(h5_dir), "tag": args.tag,
        }, f, indent=2)
    with open(out_root / "splits.json", "w") as f:
        json.dump(folds, f, indent=2)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    logger.info("Loading all subjects …")
    all_subs_needed = sorted({s for fd in folds for s in fd["train_subjects"] + fd["test_subjects"]})
    bank = load_all_subjects(h5_dir, all_subs_needed, logger)

    fold_records = []
    for fold in folds:
        fi = fold["fold"]
        if args.fold is not None and fi != args.fold:
            continue
        fold_dir = out_root / f"fold_{fi:02d}"
        fold_dir.mkdir(exist_ok=True)
        metrics_path = fold_dir / "metrics.json"
        if metrics_path.is_file():
            logger.info(f"  fold {fi}: RESUME (metrics.json exists)")
            with open(metrics_path) as f:
                fold_records.append(json.load(f))
            continue

        tr_subs, te_subs = fold["train_subjects"], fold["test_subjects"]
        if not tr_subs or not te_subs:
            logger.warning(f"  fold {fi}: empty — skip"); continue
        logger.info(f"\n====== FOLD {fi}  train={len(tr_subs)} subs  test={te_subs} ======")

        X_tr_full, y_tr_full, _ = stack_subjects(bank, tr_subs)
        X_te, y_te, owners_te = stack_subjects(bank, te_subs)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio,
                                     random_state=args.seed + fi)
        inner_tr, inner_val = next(sss.split(np.zeros_like(y_tr_full), y_tr_full))
        X_tr, y_tr = X_tr_full[inner_tr], y_tr_full[inner_tr]
        X_va, y_va = X_tr_full[inner_val], y_tr_full[inner_val]
        logger.info(f"  trials: train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}")

        t0 = time.time()
        try:
            metrics, state, preds, gts = train_one_fold(
                X_tr, y_tr, X_va, y_va, X_te, y_te,
                backbone_ckpt=args.backbone_ckpt, device=device, args=args, logger=logger,
            )
        except Exception as e:
            logger.exception(f"  fold {fi} TRAIN FAILED: {e}"); continue

        head_state = {k: v for k, v in state.items() if k.startswith("head.")}
        safe_torch_save(head_state, fold_dir / "head_weights.pth", logger=logger)
        if args.save_full_state:
            safe_torch_save(state, fold_dir / "full_state.pth", logger=logger)
        per_sub = per_subject_breakdown(preds, gts, owners_te)
        metrics.update({"fold": fi, "train_subjects": tr_subs, "test_subjects": te_subs,
                        "per_subject_test": per_sub})
        np.savez(fold_dir / "test_predictions.npz",
                 preds=preds, gts=gts, owners=owners_te.astype("U"))
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"  fold {fi} DONE  test_acc={metrics['test_acc']:.4f} "
                    f"test_f1={metrics['test_f1']:.4f}  ({(time.time()-t0)/60:.2f}min)")
        fold_records.append(metrics)
        with open(out_root / "fold_records.json", "w") as f:
            json.dump(fold_records, f, indent=2)

    if fold_records:
        accs = [r["test_acc"] for r in fold_records]
        f1s = [r["test_f1"] for r in fold_records]
        kappas = [r["test_kappa"] for r in fold_records]
        summary = {
            "tag": args.tag, "model": "CBraMod", "frozen": args.frozen,
            "cv_mode": "cross-subject", "n_folds": len(fold_records),
            "test_acc_mean": float(np.mean(accs)), "test_acc_std": float(np.std(accs)),
            "test_f1_mean": float(np.mean(f1s)), "test_f1_std": float(np.std(f1s)),
            "test_kappa_mean": float(np.mean(kappas)), "test_kappa_std": float(np.std(kappas)),
        }
        with open(out_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("===== SUMMARY =====")
        logger.info(json.dumps(summary, indent=2))
        write_per_fold_summary(out_root, fold_records, args.tag, args, splits_json_path)
    else:
        logger.warning("no fold succeeded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--backbone_ckpt",
                    default=str(CBRAMOD_DIR / "pretrained_weights" / "pretrained_weights.pth"))
    ap.add_argument("--splits_json", default=str(SCRIPT_DIR / "seedv_cross_splits.json"))
    ap.add_argument("--cuda", type=int, default=1)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--clip_value", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--frozen", dest="frozen", action="store_true", default=True)
    ap.add_argument("--no_frozen", dest="frozen", action="store_false")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--save_full_state", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "log.txt")
    setup_seed(args.seed)
    logger.info(f"Run launched at {datetime.datetime.now().isoformat()}")
    logger.info(f"args: {vars(args)}")
    logger.info(f"backbone_ckpt exists: {os.path.isfile(args.backbone_ckpt)}")
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    run(args, logger)
    logger.info("ALL DONE")


if __name__ == "__main__":
    main()
