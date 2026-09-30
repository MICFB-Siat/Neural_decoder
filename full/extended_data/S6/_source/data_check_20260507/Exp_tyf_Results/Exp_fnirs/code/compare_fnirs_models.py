


from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))

import argparse
import glob
import importlib.util
import json
import math
import os
import random
import sys
import types
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from einops import repeat


DEFAULT_H5_DIR = '/media/wsqlab/nas2/Dataset/wholehead-cocktail-party-fnirs/preprocess/stage2_fs32k_poe'
DEFAULT_SPLIT_DIR = str(_LOCAL_SOURCE / 'data_check_20260507/Exp_tyf_Results/Exp_fnirs/result/cocktail_fs32k_poe_final_fnirs_within_frozen_cache_20260703_094109')
DEFAULT_FNIRSNET_DIR = str(_LOCAL_SOURCE / 'data_check_20260507/Exp_tyf_Results/Models/fNIRSNet')
DEFAULT_TRANSFORMER_DIR = str(_LOCAL_SOURCE / 'data_check_20260507/Exp_tyf_Results/Models/fNIRS-Transformer')
DEFAULT_OUT_DIR = str(_LOCAL_SOURCE / 'data_check_20260507/Exp_tyf_Results/Exp_fnirs/result')


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        return (self.confidence * nll_loss + self.smoothing * smooth_loss).mean()


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, idx: int):
        x = self.x[idx]
        std = x.std()
        x = (x - x.mean()) / (std + 1e-6)
        return x, self.y[idx]


def import_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_subject_h5(h5_dir: str, subject: str, label_key: str = "labels"):
    h5_path = os.path.join(h5_dir, f"{subject}.h5")
    with h5py.File(h5_path, "r") as f:
        raw = np.asarray(f["fnirs_raw"], dtype=np.float32)
        labels = np.asarray(f[label_key], dtype=np.int64)
    if raw.ndim != 3:
        raise ValueError(f"{h5_path}: expected fnirs_raw [N,C,T], got {raw.shape}")
    n_chan_total = raw.shape[1]
    if n_chan_total % 2 != 0:
        raise ValueError(f"{h5_path}: fnirs_raw channel dimension is not even: {raw.shape}")
    return raw, labels


def as_fnirsnet_input(raw: np.ndarray, max_channels: int | None = None) -> np.ndarray:

    if max_channels:
        c = raw.shape[1] // 2
        keep = np.linspace(0, c - 1, min(max_channels, c), dtype=np.int64)
        raw = np.concatenate([raw[:, keep, :], raw[:, c + keep, :]], axis=1)
    return raw[:, None, :, :]


def as_transformer_input(raw: np.ndarray, max_channels: int | None = None) -> np.ndarray:

    c = raw.shape[1] // 2
    hbo, hbr = raw[:, :c, :], raw[:, c:, :]
    if max_channels:
        keep = np.linspace(0, c - 1, min(max_channels, c), dtype=np.int64)
        hbo, hbr = hbo[:, keep, :], hbr[:, keep, :]
    return np.stack([hbo, hbr], axis=1)


def patch_fnirs_t_position_embeddings(model: nn.Module, n_channels: int, dim: int):


    n_patch_tokens = n_channels - 5 + 1
    n_channel_tokens = n_channels
    if n_patch_tokens <= 0:
        raise ValueError(f"fNIRS-T requires at least 5 channels, got {n_channels}")
    model.pos_embedding_patch = nn.Parameter(torch.randn(1, n_patch_tokens + 1, dim) * 0.02)
    model.pos_embedding_channel = nn.Parameter(torch.randn(1, n_channel_tokens + 1, dim) * 0.02)

    def safe_forward(self, img, mask=None):
        x = self.to_patch_embedding(img)
        x2 = self.to_channel_embedding(img)

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token_patch, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding_patch[:, :(n + 1)]
        x = self.dropout_patch(x)
        x = self.transformer_patch(x, mask)

        b, n, _ = x2.shape
        cls_tokens = repeat(self.cls_token_channel, "() n d -> b n d", b=b)
        x2 = torch.cat((cls_tokens, x2), dim=1)
        x2 = x2 + self.pos_embedding_channel[:, :(n + 1)]
        x2 = self.dropout_channel(x2)
        x2 = self.transformer_channel(x2, mask)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x2 = x2.mean(dim=1) if self.pool == "mean" else x2[:, 0]
        x3 = torch.cat((self.to_latent(x), self.to_latent(x2)), 1)
        return self.mlp_head(x3)

    model.forward = types.MethodType(safe_forward, model)


def build_model(model_name: str, args, input_shape):
    if model_name == "fnirsnet":
        mod = import_from_path("fnirsnet_repo", os.path.join(args.fnirsnet_dir, "fNIRSNet.py"))
        _, _, height, width = input_shape
        return mod.fNIRSNet(
            num_class=args.n_classes,
            DHRConv_width=width,
            DWConv_height=height,
            num_DHRConv=args.fnirsnet_dhr,
            num_DWConv=args.fnirsnet_dw,
        )

    if model_name == "fnirs_transformer":
        mod = import_from_path("fnirs_transformer_repo", os.path.join(args.transformer_dir, "model.py"))
        _, _, channels, sampling_points = input_shape
        model = mod.fNIRS_T(
            n_class=args.n_classes,
            sampling_point=sampling_points,
            dim=args.transformer_dim,
            depth=args.transformer_depth,
            heads=args.transformer_heads,
            mlp_dim=args.transformer_mlp_dim,
        )
        patch_fnirs_t_position_embeddings(model, channels, args.transformer_dim)
        return model

    raise ValueError(f"Unknown model: {model_name}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balacc": float(balanced_accuracy_score(y_true, y_pred)),
        "n": int(y_true.shape[0]),
    }


def train_one_fold(model_name: str, x_train, y_train, x_test, y_test, args, device):
    batch_size = args.transformer_batch_size if model_name == "fnirs_transformer" else args.fnirsnet_batch_size
    train_set = ArrayDataset(x_train, y_train)
    test_set = ArrayDataset(x_test, y_test)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=len(train_set) > batch_size,
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    seed_everything(args.seed)
    model = build_model(model_name, args, x_train.shape).to(device)
    criterion = LabelSmoothing(args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if model_name == "fnirsnet":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[max(1, int(args.epochs * 0.5)), max(1, int(args.epochs * 0.75))],
            gamma=0.1,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, min(30, args.epochs)))

    best = {"acc": -1.0, "balacc": -1.0, "epoch": 0}
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        scheduler.step()

        metrics = evaluate(model, test_loader, device)
        if metrics["acc"] > best["acc"] + 1e-8:
            best = {**metrics, "epoch": epoch}
            no_improve = 0
        else:
            no_improve += 1
        if args.patience > 0 and no_improve >= args.patience:
            break
    return best


def load_existing(path: Path):
    if not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def write_summary(records: list[dict], txt_path: Path, json_path: Path, args):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    by_model_subject = defaultdict(list)
    for r in records:
        by_model_subject[(r["model"], r["subject"])].append(r)

    dataset_tag = args.tag or Path(args.h5_dir.rstrip("/")).name
    lines = [
        f"fNIRS comparison models on {dataset_tag}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"H5 dir: {args.h5_dir}",
        f"Split source: {args.split_dir}",
        f"Models: fNIRSNet={args.fnirsnet_dir}; fNIRS-Transformer={args.transformer_dir}",
        f"Protocol: within-subject 5-fold, using the exact split.json files from our method",
        f"Epochs: {args.epochs}; patience: {args.patience}; seed: {args.seed}",
        "",
    ]

    for model in args.models:
        lines.append(f"===== {model} =====")
        subj_means = []
        fold_accs = []
        for subject in sorted({r["subject"] for r in records if r["model"] == model}):
            rows = sorted(by_model_subject[(model, subject)], key=lambda x: x["fold"])
            if not rows:
                continue
            accs = [r["acc"] for r in rows]
            balaccs = [r["balacc"] for r in rows]
            subj_means.append(float(np.mean(accs)))
            fold_accs.extend(accs)
            fold_text = ", ".join([f"f{r['fold']}={r['acc']*100:.2f}%" for r in rows])
            lines.append(
                f"{subject}: mean_acc={np.mean(accs)*100:.2f}% "
                f"mean_balacc={np.mean(balaccs)*100:.2f}% folds=[{fold_text}]"
            )
        if subj_means:
            lines.append(
                f"{model} subject_mean_acc={np.mean(subj_means)*100:.2f}% "
                f"+/- {np.std(subj_means)*100:.2f}%"
            )
            lines.append(
                f"{model} fold_mean_acc={np.mean(fold_accs)*100:.2f}% "
                f"+/- {np.std(fold_accs)*100:.2f}%"
            )
        else:
            lines.append("No completed folds yet.")
        lines.append("")

    with txt_path.open("w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_dir", default=DEFAULT_H5_DIR)
    ap.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR)
    ap.add_argument("--fnirsnet_dir", default=DEFAULT_FNIRSNET_DIR)
    ap.add_argument("--transformer_dir", default=DEFAULT_TRANSFORMER_DIR)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--run_id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--tag", default=None,
                    help="Dataset tag used in the output directory and summary title.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["fnirsnet", "fnirs_transformer"],
                    choices=["fnirsnet", "fnirs_transformer"])
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument("--n_classes", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_channels", type=int, default=0,
                    help="Optional deterministic HbO/HbR channel-pair subsampling; 0 uses all channels.")
    ap.add_argument("--fnirsnet_batch_size", type=int, default=16)
    ap.add_argument("--transformer_batch_size", type=int, default=4)
    ap.add_argument("--fnirsnet_dhr", type=int, default=4)
    ap.add_argument("--fnirsnet_dw", type=int, default=8)
    ap.add_argument("--transformer_dim", type=int, default=64)
    ap.add_argument("--transformer_depth", type=int, default=6)
    ap.add_argument("--transformer_heads", type=int, default=8)
    ap.add_argument("--transformer_mlp_dim", type=int, default=64)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset_tag = args.tag or Path(args.h5_dir.rstrip("/")).name
    run_dir = Path(args.out_dir) / f"comparison_fnirs_models_{dataset_tag}_{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    txt_path = run_dir / "comparison_summary.txt"
    json_path = run_dir / "per_fold_metrics.json"
    records = load_existing(json_path)
    done = {(r["model"], r["subject"], int(r["fold"])) for r in records}

    subjects = args.subjects
    if subjects is None:
        subjects = [Path(p).stem for p in sorted(glob.glob(os.path.join(args.h5_dir, "sub-*.h5")))]
    folds = args.folds if args.folds is not None else list(range(5))

    print(f"Run dir: {run_dir}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Subjects: {len(subjects)}", flush=True)
    print(f"Models: {args.models}", flush=True)

    max_channels = args.max_channels if args.max_channels > 0 else None
    for subject in subjects:
        raw, labels = load_subject_h5(args.h5_dir, subject)
        print(f"\n===== {subject} raw={raw.shape} labels={np.bincount(labels, minlength=args.n_classes).tolist()} =====", flush=True)
        x_by_model = {
            "fnirsnet": as_fnirsnet_input(raw, max_channels=max_channels),
            "fnirs_transformer": as_transformer_input(raw, max_channels=max_channels),
        }
        for fold in folds:
            split_path = Path(args.split_dir) / subject / f"fold_{fold:02d}" / "split.json"
            with split_path.open() as f:
                split = json.load(f)
            train_idx = np.asarray(split["train_idx"], dtype=np.int64)
            test_idx = np.asarray(split["test_idx"], dtype=np.int64)
            for model_name in args.models:
                key = (model_name, subject, fold)
                if key in done:
                    print(f"[skip] {model_name} {subject} f{fold}: already complete", flush=True)
                    continue
                x = x_by_model[model_name]
                print(f"[start] {model_name} {subject} f{fold} train={len(train_idx)} test={len(test_idx)} input={x.shape[1:]}", flush=True)
                metrics = train_one_fold(
                    model_name,
                    x[train_idx], labels[train_idx],
                    x[test_idx], labels[test_idx],
                    args, device,
                )
                row = {
                    "model": model_name,
                    "subject": subject,
                    "fold": fold,
                    "train_n": int(len(train_idx)),
                    "test_n": int(len(test_idx)),
                    **metrics,
                }
                records.append(row)
                done.add(key)
                write_summary(records, txt_path, json_path, args)
                print(
                    f"[done] {model_name} {subject} f{fold}: "
                    f"acc={metrics['acc']*100:.2f}% balacc={metrics['balacc']*100:.2f}% "
                    f"best_epoch={metrics['epoch']}",
                    flush=True,
                )

    write_summary(records, txt_path, json_path, args)
    print(f"\nWrote summary: {txt_path}", flush=True)
    print(f"Wrote metrics: {json_path}", flush=True)


if __name__ == "__main__":
    main()
