


































































import argparse
import glob
import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
)
sys.path.insert(0, PROJECT_ROOT)

from configs.time_config import (
    N_TOKENS, D_MODEL,
    EEGMEG_PRIOR_PATCH_SIZE, EEG_MEG_N_SAMPLES,
    FMRI_N_MODES, FMRI_N_GROUPS, FMRI_N_TIMEPOINTS,
    FUSION_MODE,
)
from models.prior.eeg_meg_prior import EEGMEGPriorEncoder
from models.prior.fmri_prior import FMRIPriorEncoder
from models.encoders.brainomni_encoder import BrainOmniEncoder
from models.encoders.fmri_cifti_encoder import (
    CortexCIFTIEncoder, N_CORTEX_DEFAULT, N_GROUPS_DEFAULT,
)
from models.fusion.cross_attn_gate import CrossAttnGateFusion

DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "stage2")

ABLATION_MODES = ["prior_only", "obs_only", "full"]
ABLATION_LABELS = {
    "prior_only": "先验        (Prior Only)",
    "obs_only":   "观测        (Obs Only)",
    "full":       "观测+先验   (Obs+Prior)",
}




def _setup_logging(out_dir: str) -> str:
    log_path = os.path.join(out_dir, "train_log.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(sh)
    return log_path


log = logging.getLogger(__name__)




EEGMEG_KEYS = ("modes", "modes_indiv", "raw", "pos", "sensor_type")
FMRI_KEYS   = ("modes", "modes_indiv", "cifti")


class MultiModalH5Dataset(Dataset):









    def __init__(self, h5_paths: List[str], n_samples: int,
                 modality_filter: Optional[str] = None):





        self.records = []
        self.n_samples       = n_samples
        self.modality_filter = modality_filter or "eeg"
        self.modalities      = set()
        self.subject_meta    = {}
        self.label_dtype     = np.int64
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

        labels_all = []
        for path in h5_paths:
            with h5py.File(path, "r") as f:
                if "labels" not in f:
                    log.warning("  %s — no /labels, skip", os.path.basename(path))
                    continue
                n_trials = f["labels"].shape[0]
                meta = {}
                avail = set()
                cache: Dict[str, np.ndarray] = {}

                cache["labels"] = f["labels"][:].astype(np.int64)
                labels_all.append(cache["labels"])

                if self.modality_filter in ("eeg", "meg"):
                    mod = self.modality_filter
                    if any(f"{mod}_{k}" in f for k in EEGMEG_KEYS):
                        avail.add(mod)
                        if f"{mod}_pos" in f:
                            meta[f"{mod}_pos"] = torch.tensor(
                                f[f"{mod}_pos"][:], dtype=torch.float32)
                        if f"{mod}_sensor_type" in f:
                            meta[f"{mod}_sensor_type"] = torch.tensor(
                                f[f"{mod}_sensor_type"][:], dtype=torch.long)
                        for key in (f"{mod}_modes", f"{mod}_modes_indiv", f"{mod}_raw"):
                            if key in f:
                                cache[key] = f[key][:].astype(np.float32)
                elif self.modality_filter == "fmri":
                    if any(f"fmri_{k}" in f for k in FMRI_KEYS):
                        avail.add("fmri")
                        for key in ("fmri_modes", "fmri_modes_indiv", "fmri_cifti"):
                            if key in f:
                                cache[key] = f[key][:].astype(np.float32)

                self.modalities.update(avail)
                self.subject_meta[path] = (meta, avail)
                self._cache[path] = cache
                for i in range(n_trials):
                    self.records.append((path, i, avail))
                log.info("  %s  loaded %d trials  mods=%s",
                         os.path.basename(path), n_trials, sorted(avail))

        if not self.records:
            raise RuntimeError("No usable trials found.")

        labels_all = np.concatenate(labels_all, axis=0)
        self.n_classes = int(labels_all.max()) + 1

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, trial_i, avail = self.records[idx]
        meta, _ = self.subject_meta[path]
        cache = self._cache[path]
        out: Dict[str, torch.Tensor] = {}

        out["label"] = torch.tensor(int(cache["labels"][trial_i]), dtype=torch.long)

        T_eeg = self.n_samples
        for mod in ("eeg", "meg"):
            if mod not in avail:
                continue
            if f"{mod}_modes" in cache:
                out[f"{mod}_modes"] = torch.from_numpy(
                    cache[f"{mod}_modes"][trial_i, :, :T_eeg])
            if f"{mod}_modes_indiv" in cache:
                out[f"{mod}_modes_indiv"] = torch.from_numpy(
                    cache[f"{mod}_modes_indiv"][trial_i, :, :T_eeg])
            if f"{mod}_raw" in cache:
                out[f"{mod}_raw"] = torch.from_numpy(
                    cache[f"{mod}_raw"][trial_i, :, :T_eeg])
            for k in (f"{mod}_pos", f"{mod}_sensor_type"):
                if k in meta:
                    out[k] = meta[k]

        if "fmri" in avail:
            for key in ("fmri_modes", "fmri_modes_indiv", "fmri_cifti"):
                if key in cache:
                    out[key] = torch.from_numpy(cache[key][trial_i])

        return out

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:

        keys = set().union(*[s.keys() for s in batch])
        out = {}
        for k in keys:
            vals = [s.get(k) for s in batch]
            if any(v is None for v in vals):
                continue
            try:
                out[k] = torch.stack(vals, dim=0)
            except RuntimeError:
                out[k] = vals
        return out




class BCIStage2Model(nn.Module):











    def __init__(
        self,
        eegmeg_prior: Optional[EEGMEGPriorEncoder],
        fmri_prior:   Optional[FMRIPriorEncoder],
        brainomni:    Optional[BrainOmniEncoder],
        cifti:        Optional[CortexCIFTIEncoder],
        fusion:       CrossAttnGateFusion,
        n_classes:    int,
        d_model:      int   = D_MODEL,
        dropout:      float = 0.1,
        fusion_mode:  str   = "full",
    ):
        super().__init__()
        self.eegmeg_prior = eegmeg_prior
        self.fmri_prior   = fmri_prior
        self.brainomni    = brainomni
        self.cifti        = cifti
        self.fusion       = fusion

        def _head():
            return nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )
        self.head_prior = _head()
        self.head_obs   = _head()
        self.head_full  = _head()

    @staticmethod
    def _blend(x_g, x_i, alpha: float):
        if x_i is not None and alpha > 0.0:
            return (1.0 - alpha) * x_g + alpha * x_i
        return x_g

    def _get_branches(self, batch: Dict, alpha: float):

        H_obs_list, H_prior_list = [], []

        for mod in ("eeg", "meg"):
            x_modes = self._blend(
                batch.get(f"{mod}_modes"),
                batch.get(f"{mod}_modes_indiv"),
                alpha,
            )
            if self.eegmeg_prior is not None and x_modes is not None:
                H_prior_list.append(self.eegmeg_prior(x_modes))

            x_raw = batch.get(f"{mod}_raw")
            pos   = batch.get(f"{mod}_pos")
            stype = batch.get(f"{mod}_sensor_type")
            if (self.brainomni is not None
                    and x_raw is not None and pos is not None and stype is not None):
                H_obs_list.append(self.brainomni(x_raw, pos, stype))

        x_fmri = self._blend(
            batch.get("fmri_modes"), batch.get("fmri_modes_indiv"), alpha)
        if self.fmri_prior is not None and x_fmri is not None:
            H_prior_list.append(self.fmri_prior(x_fmri))

        x_cifti = batch.get("fmri_cifti")
        if self.cifti is not None and x_cifti is not None:
            H_obs_list.append(self.cifti(x_cifti))

        H_obs   = torch.stack(H_obs_list,   0).mean(0) if H_obs_list   else None
        H_prior = torch.stack(H_prior_list, 0).mean(0) if H_prior_list else None
        return H_obs, H_prior

    def forward_three(self, batch: Dict, alpha: float = 0.0):

        H_obs, H_prior = self._get_branches(batch, alpha)

        logits_prior = (self.head_prior(H_prior.mean(1))
                        if H_prior is not None else None)
        logits_obs   = (self.head_obs(H_obs.mean(1))
                        if H_obs is not None else None)

        if H_obs is not None and H_prior is not None:
            H_full = self.fusion(H_obs, H_prior, "full")
        elif H_obs is not None:
            H_full = H_obs
        else:
            H_full = H_prior
        logits_full = (self.head_full(H_full.mean(1))
                       if H_full is not None else None)

        return logits_prior, logits_obs, logits_full




def _load_state(state, prefer_keys: Tuple[str, ...]):
    for k in prefer_keys:
        if isinstance(state, dict) and k in state and isinstance(state[k], dict):
            return state[k]
    return state


def load_eegmeg_prior(ckpt_path: str, args) -> EEGMEGPriorEncoder:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    enc = EEGMEGPriorEncoder(
        d_model    = ckpt.get("d_model",    args.d_model),
        d_patch    = ckpt.get("d_inner",    args.d_inner),
        patch_size = ckpt.get("patch_size", EEGMEG_PRIOR_PATCH_SIZE),
    )
    state = _load_state(ckpt, ("eegmeg_prior",))
    enc.load_state_dict(state)
    log.info("  EEGMEGPrior loaded ← %s (epoch=%s val=%.5f)",
             ckpt_path, ckpt.get("epoch", "?"), ckpt.get("val_loss", float("nan")))
    return enc


def load_fmri_prior(ckpt_path: str, args) -> FMRIPriorEncoder:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    enc = FMRIPriorEncoder(
        n_modes  = ckpt.get("n_modes",  args.fmri_n_modes),
        n_groups = ckpt.get("n_groups", args.fmri_n_groups),
        d_model  = ckpt.get("d_model",  args.d_model),
        d_inner  = ckpt.get("d_inner",  args.d_inner),
    )
    state = _load_state(ckpt, ("fmri_prior",))
    enc.load_state_dict(state)
    log.info("  FMRIPrior loaded ← %s (epoch=%s val=%.5f)",
             ckpt_path, ckpt.get("epoch", "?"), ckpt.get("val_loss", float("nan")))
    return enc


def load_fmri_cifti(ckpt_path: str, args) -> CortexCIFTIEncoder:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    enc = CortexCIFTIEncoder(
        n_cortex = ckpt.get("n_cortex", args.cifti_n_cortex),
        n_groups = ckpt.get("n_groups", args.cifti_n_groups),
        T        = ckpt.get("T",        args.cifti_T),
        d_model  = ckpt.get("d_model",  args.d_model),
        d_inner  = ckpt.get("d_inner",  args.d_inner),
        n_heads  = ckpt.get("n_heads",  8),
        n_layers = ckpt.get("n_layers", 4),
    )
    state = _load_state(ckpt, ("fmri_cifti",))
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if missing:
        log.info("  CIFTI missing keys (will train from scratch): %d", len(missing))
    if unexpected:
        log.info("  CIFTI unexpected keys: %d", len(unexpected))
    log.info("  CortexCIFTIEncoder loaded ← %s (epoch=%s val=%.5f)",
             ckpt_path, ckpt.get("epoch", "?"), ckpt.get("val_loss", float("nan")))
    return enc




def build_model(args, n_classes: int, modalities: set, device: torch.device) -> BCIStage2Model:
    needs_eegmeg = bool(modalities & {"eeg", "meg"})
    needs_fmri   = "fmri" in modalities

    eegmeg_prior = None
    brainomni    = None
    if needs_eegmeg:
        if not args.eegmeg_prior_ckpt:
            raise ValueError("--eegmeg_prior_ckpt required (EEG/MEG modality detected)")
        eegmeg_prior = load_eegmeg_prior(args.eegmeg_prior_ckpt, args)
        for p in eegmeg_prior.parameters():
            p.requires_grad_(False)
        if not args.no_brainomni:
            try:
                brainomni = BrainOmniEncoder(
                    d_model=args.d_model, n_tokens_out=N_TOKENS, freeze_backbone=True,
                    variant=args.brainomni_variant,
                )
                log.info("  BrainOmniEncoder loaded (backbone frozen, variant=%s)",
                         args.brainomni_variant)
            except Exception as e:
                log.warning("  BrainOmni unavailable (%s)", e)

    fmri_prior = None
    cifti      = None
    if needs_fmri:
        if not args.fmri_prior_ckpt:
            log.warning("fMRI present but --fmri_prior_ckpt not given; prior path disabled")
        else:
            fmri_prior = load_fmri_prior(args.fmri_prior_ckpt, args)
            for p in fmri_prior.parameters():
                p.requires_grad_(False)
        if not args.no_cifti:
            if not args.fmri_cifti_ckpt:
                log.warning("fMRI present but --fmri_cifti_ckpt not given; CIFTI obs disabled")
            else:
                cifti = load_fmri_cifti(args.fmri_cifti_ckpt, args)
                for name, p in cifti.named_parameters():
                    p.requires_grad_(name.startswith("out_proj"))

    fusion = CrossAttnGateFusion(d_model=args.d_model, n_heads=8, dropout=args.dropout)

    return BCIStage2Model(
        eegmeg_prior=eegmeg_prior, fmri_prior=fmri_prior,
        brainomni=brainomni, cifti=cifti,
        fusion=fusion, n_classes=n_classes,
        d_model=args.d_model, dropout=args.dropout,
        fusion_mode="full",
    ).to(device)




def extract_bci_state(model: BCIStage2Model) -> Dict:
    return {
        "eegmeg_prior":      model.eegmeg_prior.state_dict() if model.eegmeg_prior else None,
        "fmri_prior":        model.fmri_prior.state_dict()   if model.fmri_prior   else None,
        "brainomni_adapter": model.brainomni.adapter.state_dict() if model.brainomni else None,
        "cifti":             model.cifti.state_dict()        if model.cifti        else None,
        "fusion":            model.fusion.state_dict(),
        "head_prior":        model.head_prior.state_dict(),
        "head_obs":          model.head_obs.state_dict(),
        "head_full":         model.head_full.state_dict(),
    }




def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def run_epoch_three(
    model: BCIStage2Model,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    device: torch.device,
    train: bool,
    eval_alpha: float = 0.0,
    grad_clip: float  = 1.0,
) -> Dict[str, Tuple[float, float]]:




    model.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()
    stats: Dict[str, List] = {
        "prior": [0.0, 0, 0],
        "obs":   [0.0, 0, 0],
        "full":  [0.0, 0, 0],
    }
    with ctx:
        for batch in loader:
            batch  = _to_device(batch, device)
            labels = batch["label"]
            alpha  = random.uniform(0.0, 1.0) if train else eval_alpha

            lp, lo, lf = model.forward_three(batch, alpha=alpha)

            loss = torch.tensor(0.0, device=device)
            for key, logits in [("prior", lp), ("obs", lo), ("full", lf)]:
                if logits is None:
                    continue
                l = F.cross_entropy(logits, labels)
                loss = loss + l
                stats[key][0] += l.item() * labels.size(0)
                stats[key][1] += (logits.argmax(-1) == labels).sum().item()
                stats[key][2] += labels.size(0)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

    return {
        k: (s[0] / max(s[2], 1), s[1] / max(s[2], 1))
        for k, s in stats.items()
    }


def _eval_per_class(
    model: "BCIStage2Model",
    loader: DataLoader,
    device: torch.device,
    eval_alpha: float,
    n_classes: int,
    mode: str = "full",
):

    model.eval()
    correct = np.zeros(n_classes, dtype=int)
    totals  = np.zeros(n_classes, dtype=int)
    mode_idx = {"prior": 0, "obs": 1, "full": 2}[mode]
    with torch.no_grad():
        for batch in loader:
            batch  = _to_device(batch, device)
            labels = batch["label"]
            logits = model.forward_three(batch, alpha=eval_alpha)[mode_idx]
            if logits is None:
                continue
            preds = logits.argmax(-1).cpu().numpy()
            lbls  = labels.cpu().numpy()
            for p, l in zip(preds, lbls):
                totals[l]  += 1
                correct[l] += int(p == l)
    return correct, totals


def _print_stage2_table(
    dataset_name: str,
    splits: list,
    correct: np.ndarray,
    totals:  np.ndarray,
):

    W = 58
    hdr  = f"  Stage-2 Classification  ─  {dataset_name}"
    rows = [
        f"  {'Split':<9} {'Loss':>10}   {'Acc':>9}   {'Samples':>8}",
        f"  {'─'*9} {'─'*10}   {'─'*9}   {'─'*8}",
    ]
    for name, loss, acc, n in splits:
        rows.append(f"  {name:<9} {loss:>10.4f}   {acc*100:>8.2f}%   {n:>8d}")

    print()
    print("┌" + "─" * W + "┐")
    print(f"│{hdr:<{W}}│")
    print("├" + "─" * W + "┤")
    for r in rows:
        print(f"│{r:<{W}}│")
    print("└" + "─" * W + "┘")

    n_classes = len(correct)
    if n_classes == 0 or totals.sum() == 0:
        return
    print()
    print(f"  Per-class Test Accuracy  (n_classes={n_classes})")
    print(f"  ┌{'─'*8}┬{'─'*10}┬{'─'*15}┐")
    print(f"  │{'Class':^8}│{'Acc':^10}│{'Correct / N':^15}│")
    print(f"  ├{'─'*8}┼{'─'*10}┼{'─'*15}┤")
    for c in range(n_classes):
        n  = totals[c]
        ok = correct[c]
        acc_c = ok / n * 100 if n > 0 else float("nan")
        print(f"  │{c:^8d}│{acc_c:^9.1f}%│{f'{ok} / {n}':^15}│")
    print(f"  └{'─'*8}┴{'─'*10}┴{'─'*15}┘")
    print()


def _print_summary_table(
    dataset_name: str,
    exp_mode: str,
    n_folds: int,
    results: Dict[str, List[float]],
):

    W = 62
    title = f"  Stage-2 Results  ─  {dataset_name}  ─  {exp_mode}  ─  {n_folds}-fold"
    header = f"  {'Mode':<15}{'Test Acc (mean ± std)':>40}"
    sep_inner = f"{'─'*15}┬{'─'*44}"
    sep_data  = f"{'─'*15}┼{'─'*44}"

    print()
    print("┌" + "─" * W + "┐")
    print(f"│{title:<{W}}│")
    print("├" + "─" * W + "┤")
    print(f"│{header:<{W}}│")
    print("├" + sep_inner + "┤")
    for i, mode in enumerate(ABLATION_MODES):
        accs = results.get(mode, [])
        if accs:
            mean = np.mean(accs) * 100
            std  = np.std(accs)  * 100
            val_str = f"{mean:.2f} ± {std:.2f} %"
        else:
            val_str = "N/A"
        label = ABLATION_LABELS.get(mode, mode)
        row = f"  {label:<22}│  {val_str:<38}"
        print(f"│{row:<{W}}│")
        if i < len(ABLATION_MODES) - 1:
            print("├" + sep_data + "┤")
    print("└" + "─" * W + "┘")
    print()




def _within_fold_indices(
    h5_paths: List[str],
    cache_map: Dict[str, Dict],
    fold_idx: int,
    seed: int,
) -> Tuple[List[int], List[int]]:







    rng = np.random.default_rng(seed + fold_idx)
    train_idx, test_idx = [], []
    offset = 0
    for path in h5_paths:
        n = cache_map[path]["labels"].shape[0]
        perm = rng.permutation(n)
        n_train = int(n * 0.8)
        for i in perm[:n_train]:
            train_idx.append(offset + int(i))
        for i in perm[n_train:]:
            test_idx.append(offset + int(i))
        offset += n
    return train_idx, test_idx


def _cross_fold_file_groups(h5_paths: List[str], n_folds: int) -> List[List[str]]:

    sorted_paths = sorted(h5_paths)
    return [grp.tolist() for grp in np.array_split(sorted_paths, n_folds)]




def _make_loaders(
    full_ds: MultiModalH5Dataset,
    train_idx: List[int],
    test_idx: List[int],
    args,
) -> Tuple[DataLoader, DataLoader]:
    train_set = torch.utils.data.Subset(full_ds, train_idx)
    test_set  = torch.utils.data.Subset(full_ds, test_idx)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=MultiModalH5Dataset.collate_fn,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=MultiModalH5Dataset.collate_fn,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, test_loader


def _train_fold(
    model: BCIStage2Model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    args,
    device: torch.device,
    fold_label: str,
    dataset_name: str,
    n_classes: int,
) -> Dict[str, float]:





    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps  = max(1, args.epochs * len(train_loader))
    warmup_steps = min(args.warmup_epochs * len(train_loader), max(1, total_steps - 1))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    best_full_acc = 0.0
    best_state    = None
    last_tr: Dict[str, Tuple[float, float]] = {}
    last_te: Dict[str, Tuple[float, float]] = {}

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch_three(
            model, train_loader, optimizer, scheduler, device, train=True,
        )
        te = run_epoch_three(
            model, test_loader, None, None, device, train=False,
            eval_alpha=args.alpha,
        )
        last_tr, last_te = tr, te

        if epoch % 5 == 0 or epoch == 1:
            log.info(
                "[%s][%s] Epoch %03d/%03d\n"
                "         prior  — train acc=%.4f  test acc=%.4f\n"
                "         obs    — train acc=%.4f  test acc=%.4f\n"
                "         full   — train acc=%.4f  test acc=%.4f",
                dataset_name, fold_label, epoch, args.epochs,
                tr["prior"][1], te["prior"][1],
                tr["obs"][1],   te["obs"][1],
                tr["full"][1],  te["full"][1],
            )

        full_te_acc = te["full"][1]
        if full_te_acc > best_full_acc:
            best_full_acc = full_te_acc
            best_state    = extract_bci_state(model)
            log.info("  ✓ [%s] new best full test_acc=%.4f  (prior=%.4f  obs=%.4f)",
                     fold_label, full_te_acc, te["prior"][1], te["obs"][1])


    if best_state is not None:
        for k, sd in best_state.items():
            if sd is None:
                continue
            if k == "eegmeg_prior" and model.eegmeg_prior is not None:
                model.eegmeg_prior.load_state_dict(sd)
            elif k == "fmri_prior" and model.fmri_prior is not None:
                model.fmri_prior.load_state_dict(sd)
            elif k == "brainomni_adapter" and model.brainomni is not None:
                model.brainomni.adapter.load_state_dict(sd)
            elif k == "cifti" and model.cifti is not None:
                model.cifti.load_state_dict(sd)
            elif k == "fusion":
                model.fusion.load_state_dict(sd)
            elif k == "head_prior":
                model.head_prior.load_state_dict(sd)
            elif k == "head_obs":
                model.head_obs.load_state_dict(sd)
            elif k == "head_full":
                model.head_full.load_state_dict(sd)


    te_final = run_epoch_three(
        model, test_loader, None, None, device, train=False,
        eval_alpha=args.alpha,
    )
    mode_accs = {
        "prior_only": te_final["prior"][1],
        "obs_only":   te_final["obs"][1],
        "full":       te_final["full"][1],
    }
    log.info(
        "[%s][%s] ── Final best-ckpt results ──\n"
        "         prior_only  test acc=%.4f\n"
        "         obs_only    test acc=%.4f\n"
        "         full        test acc=%.4f",
        dataset_name, fold_label,
        mode_accs["prior_only"], mode_accs["obs_only"], mode_accs["full"],
    )


    per_class_correct, per_class_totals = _eval_per_class(
        model, test_loader, device, args.alpha, n_classes, mode="full",
    )
    _print_stage2_table(
        f"{dataset_name} [{fold_label}]",
        [
            ("Train(full)", last_tr["full"][0], last_tr["full"][1],
             len(train_loader.dataset)),
            ("Test(full)",  te_final["full"][0], te_final["full"][1],
             sum(per_class_totals)),
        ],
        per_class_correct,
        per_class_totals,
    )

    return mode_accs, best_state, None




def train_one_dataset(
    h5_dir: str,
    args,
    device: torch.device,
    run_id: str,
    train_files: Optional[List[str]] = None,
    test_files:  Optional[List[str]] = None,
):
    dataset_name = os.path.basename(h5_dir.rstrip("/\\"))
    out_dir = os.path.join(args.out_dir, f"{dataset_name}_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    _setup_logging(out_dir)

    log.info("=" * 70)
    log.info("Dataset : %s", dataset_name)
    log.info("Out dir : %s", out_dir)

    h5_paths = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
    if not h5_paths:
        log.warning("No .h5 files in %s — skip", h5_dir)
        return

    custom_split = (train_files is not None and test_files is not None)


    if custom_split:
        log.info("Custom split: %d train files, %d test files",
                 len(train_files), len(test_files))

        train_ds = MultiModalH5Dataset(train_files, n_samples=args.n_samples, modality_filter=args.modality)
        test_ds  = MultiModalH5Dataset(test_files,  n_samples=args.n_samples, modality_filter=args.modality)
        n_classes = max(train_ds.n_classes, test_ds.n_classes)
        modalities = train_ds.modalities | test_ds.modalities

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=MultiModalH5Dataset.collate_fn,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=MultiModalH5Dataset.collate_fn,
            num_workers=args.num_workers, pin_memory=True,
        )

        model = build_model(args, n_classes, modalities, device)
        n_total     = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("Params   : %.2fM total, %.2fM trainable",
                 n_total / 1e6, n_trainable / 1e6)

        mode_accs, best_state, best_head = _train_fold(
            model, train_loader, test_loader, args, device,
            "custom", dataset_name, n_classes,
        )

        results = {mode: [mode_accs[mode]] for mode in ABLATION_MODES}
        _print_summary_table(dataset_name, "custom", 1, results)

        _save_outputs(
            out_dir, dataset_name, run_id, model, best_state,
            modalities, n_classes, args, results,
            fold_details={"custom": mode_accs},
        )
        return


    full_ds = MultiModalH5Dataset(h5_paths, n_samples=args.n_samples, modality_filter=args.modality)
    log.info("Trials   : %d", len(full_ds))
    log.info("Modalities: %s", sorted(full_ds.modalities))
    log.info("N classes: %d", full_ds.n_classes)
    log.info("Exp mode : %s  n_folds=%d", args.exp_mode, args.n_folds)


    cache_map: Dict[str, Dict] = {p: full_ds._cache[p] for p in h5_paths
                                   if p in full_ds._cache}


    fold_results: Dict[str, List[float]] = {mode: [] for mode in ABLATION_MODES}
    fold_details: Dict[str, Dict[str, float]] = {}

    last_model      = None
    last_best_state = None

    for fold_idx in range(args.n_folds):
        fold_label = f"fold{fold_idx}"
        log.info("-" * 60)
        log.info("Starting %s  [%s]", fold_label, args.exp_mode)

        if args.exp_mode == "within":
            valid_paths = [p for p in h5_paths if p in full_ds._cache]
            train_idx, test_idx = _within_fold_indices(
                valid_paths, cache_map, fold_idx, args.seed,
            )
            log.info("  train=%d  test=%d", len(train_idx), len(test_idx))
            train_loader, test_loader = _make_loaders(
                full_ds, train_idx, test_idx, args,
            )
            n_classes = full_ds.n_classes
            modalities = full_ds.modalities

        else:
            groups = _cross_fold_file_groups(h5_paths, args.n_folds)
            test_paths  = groups[fold_idx]
            train_paths = [p for i, grp in enumerate(groups)
                           for p in grp if i != fold_idx]
            log.info("  train subjects=%d  test subjects=%d",
                     len(train_paths), len(test_paths))
            if not train_paths or not test_paths:
                log.warning("  Skipping fold %d: empty split", fold_idx)
                continue

            train_ds_fold = MultiModalH5Dataset(train_paths, n_samples=args.n_samples, modality_filter=args.modality)
            test_ds_fold  = MultiModalH5Dataset(test_paths,  n_samples=args.n_samples, modality_filter=args.modality)
            n_classes  = max(train_ds_fold.n_classes, test_ds_fold.n_classes)
            modalities = train_ds_fold.modalities | test_ds_fold.modalities

            train_loader = DataLoader(
                train_ds_fold, batch_size=args.batch_size, shuffle=True,
                collate_fn=MultiModalH5Dataset.collate_fn,
                num_workers=args.num_workers, pin_memory=True, drop_last=True,
            )
            test_loader = DataLoader(
                test_ds_fold, batch_size=args.batch_size, shuffle=False,
                collate_fn=MultiModalH5Dataset.collate_fn,
                num_workers=args.num_workers, pin_memory=True,
            )

        model = build_model(args, n_classes, modalities, device)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("  Trainable params: %.2fM", n_trainable / 1e6)

        mode_accs, best_state, best_head = _train_fold(
            model, train_loader, test_loader, args, device,
            fold_label, dataset_name, n_classes,
        )

        for mode in ABLATION_MODES:
            fold_results[mode].append(mode_accs[mode])
        fold_details[fold_label] = mode_accs

        last_model      = model
        last_best_state = best_state

    _print_summary_table(dataset_name, args.exp_mode, args.n_folds, fold_results)

    if last_model is not None:
        _save_outputs(
            out_dir, dataset_name, run_id,
            last_model, last_best_state,
            last_model.modalities if hasattr(last_model, "modalities") else full_ds.modalities,
            full_ds.n_classes, args, fold_results,
            fold_details=fold_details,
        )


def _save_outputs(
    out_dir: str,
    dataset_name: str,
    run_id: str,
    model: BCIStage2Model,
    best_state: Optional[Dict],
    modalities,
    n_classes: int,
    args,
    fold_results: Dict[str, List[float]],
    fold_details: Dict[str, Dict[str, float]],
):
    modality  = getattr(args, "modality", "unknown")
    variant   = getattr(args, "brainomni_variant", "tiny")
    exp_mode  = getattr(args, "exp_mode", "custom")
    stem      = f"{dataset_name}_{modality}_{variant}_{exp_mode}_{run_id}"

    bci_path     = os.path.join(out_dir, f"bci_{stem}.pth")
    head_path    = os.path.join(out_dir, f"head_{stem}.pth")
    summary_path = os.path.join(out_dir, f"summary_{stem}.json")

    bci_save = best_state if best_state is not None else extract_bci_state(model)

    torch.save({
        "dataset":             dataset_name,
        "run_id":              run_id,
        "modalities":          sorted(modalities),
        "n_classes":           n_classes,
        "brainomni_variant":   getattr(args, "brainomni_variant", "tiny"),
        "modality":            getattr(args, "modality", "eeg"),
        "args":                vars(args),
        "bci":                 bci_save,
    }, bci_path)

    summary = {
        "dataset_name": dataset_name,
        "run_id":       run_id,
        "exp_mode":     getattr(args, "exp_mode", "custom"),
        "n_folds":      getattr(args, "n_folds", 1),
        "per_fold":     fold_details,
        "mean_std":     {
            mode: {
                "mean": float(np.mean(accs)) if accs else None,
                "std":  float(np.std(accs))  if accs else None,
                "accs": [float(a) for a in accs],
            }
            for mode, accs in fold_results.items()
        },
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    log.info("Saved BCI model        → %s", bci_path)
    log.info("Saved classifier head  → %s", head_path)
    log.info("Saved summary          → %s", summary_path)




def main():
    parser = argparse.ArgumentParser(
        description="Stage-2 multi-modal supervised fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


    parser.add_argument("--h5_dirs", nargs="+", required=True,
                        help="One or more dataset directories; each is trained as a separate run.")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR,
                        help="Saves to {out_dir}/{dataset_name}_{run_id}/")


    parser.add_argument("--train_files", nargs="+", default=None,
                        help="Explicit list of .h5 files for training (skips fold loop).")
    parser.add_argument("--test_files", nargs="+", default=None,
                        help="Explicit list of .h5 files for testing (skips fold loop).")


    parser.add_argument("--eegmeg_prior_ckpt", type=str, default=None,
                        help="stage1/.../eegmeg_prior_<run>.pt  (required when EEG/MEG present)")
    parser.add_argument("--fmri_prior_ckpt",   type=str, default=None,
                        help="stage1/.../fmri_prior_<run>.pt    (optional, fMRI only)")
    parser.add_argument("--fmri_cifti_ckpt",   type=str, default=None,
                        help="stage1/.../fmri_cifti_<run>.pt    (optional, fMRI only)")

    parser.add_argument("--no_brainomni", action="store_true")
    parser.add_argument("--no_cifti",     action="store_true")


    parser.add_argument("--exp_mode", type=str, default="within",
                        choices=["within", "cross"],
                        help="within: per-subject 80/20 split pooled; "
                             "cross: leave-N-subjects-out.")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of CV folds.")
    parser.add_argument("--seed",    type=int, default=42)


    parser.add_argument("--d_model",           type=int, default=D_MODEL)
    parser.add_argument("--d_inner",           type=int, default=512)
    parser.add_argument("--n_samples",         type=int, default=EEG_MEG_N_SAMPLES)
    parser.add_argument("--fmri_n_modes",      type=int, default=FMRI_N_MODES)
    parser.add_argument("--fmri_n_groups",     type=int, default=FMRI_N_GROUPS)
    parser.add_argument("--cifti_n_cortex",    type=int, default=N_CORTEX_DEFAULT)
    parser.add_argument("--cifti_n_groups",    type=int, default=N_GROUPS_DEFAULT)
    parser.add_argument("--cifti_T",           type=int, default=FMRI_N_TIMEPOINTS)
    parser.add_argument("--dropout",           type=float, default=0.1)
    parser.add_argument("--brainomni_variant", type=str, default="tiny",
                        choices=["base", "tiny"],
                        help="BrainOmniEncoder architecture variant (base: lm_dim=512, tiny: lm_dim=256).")
    parser.add_argument("--modality", type=str, required=True,
                        choices=["eeg", "meg", "fmri"],
                        help="Single modality to train on: "
                             "eeg (EEG only), meg (MEG only), fmri (fMRI only).")


    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int,   default=5)
    parser.add_argument("--alpha",         type=float, default=0.0,
                        help="Eval alpha (0=group only). Train uses Uniform(0,1) per batch.")
    parser.add_argument("--num_workers",   type=int,   default=2)
    parser.add_argument("--device",        type=str,   default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    custom_split = (args.train_files is not None and args.test_files is not None)

    for h5_dir in args.h5_dirs:
        train_one_dataset(
            h5_dir, args, device, run_id,
            train_files=args.train_files if custom_split else None,
            test_files=args.test_files   if custom_split else None,
        )


if __name__ == "__main__":
    main()
