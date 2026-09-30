




import argparse

import copy

import glob

import io

import json

import logging

import os

import random

import shutil

import sys

from copy import deepcopy

from datetime import datetime

import h5py

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import balanced_accuracy_score, recall_score

from pathlib import Path

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))

BUNDLE_ROOT = str(Path(__file__).resolve().parents[2])

MODELS_DIR    = os.path.join(BUNDLE_ROOT, "Models")

BRAINOMNI_SRC = os.path.join(MODELS_DIR, "BrainOmni/BrainOmni-main")

TINY_CKPT_DIR = os.path.join(MODELS_DIR, "BrainOmni/BrainOmni/tiny")

BUNDLED_PRIOR_CKPT = os.path.join(
    BUNDLE_ROOT, "Checkpoints/prior_unified_v3_20260514_020311.pt")

sys.path.insert(0, BRAINOMNI_SRC)

import importlib.util as _ilu

_PRIOR_PKG_DIR = os.path.join(MODELS_DIR, "BT-ND")

_prior_spec = _ilu.spec_from_file_location(
    "prior", os.path.join(_PRIOR_PKG_DIR, "__init__.py"),
    submodule_search_locations=[_PRIOR_PKG_DIR])

_prior_pkg = _ilu.module_from_spec(_prior_spec)

sys.modules["prior"] = _prior_pkg

_prior_spec.loader.exec_module(_prior_pkg)

from prior.eeg_meg_prior_eigenval4 import (
    EEGMEGPriorEigenval4, compute_eigval_from_D,
)

from prior.eeg_meg_prior_eigenval3 import (
    EEGMEGPriorEigenval3, EEGMEGMAEDecoderEigenval3,
)

from prior.eeg_meg_prior_eigenval import (
    PATCH_SIZE, brainomni_2d_mask, patchify,
)

RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_PATH = os.path.join(SCRIPT_DIR, f"classify_baseline_{RUN_ID}.log")

_LOCAL_LOG_PATH = None

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)

def load_prior_from_ckpt(ckpt_path: str, device: torch.device,
                         trainable: bool,
                         n_samples_override: int | None = None) -> EEGMEGPriorEigenval4:






    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = EEGMEGPriorEigenval4(
        patch_size      = ckpt.get("patch_size", PATCH_SIZE),
        d_model         = ckpt["d_model"],
        n_heads         = 8,
        n_factor_layers = ckpt.get("n_factor_layers", 2),
        n_time_layers   = ckpt.get("n_time_layers", 4),
        dropout         = 0.0,
    ).to(device)

    state = ckpt["encoder"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if missing or unexpected:
        log.warning(f"  load_state_dict — missing={missing}  unexpected={unexpected}")
    for p in enc.parameters():
        p.requires_grad = bool(trainable)
    enc.eval()

    enc.expected_n_samples = (n_samples_override if n_samples_override is not None
                              else ckpt.get("n_samples", 2560))
    enc.d_model_dim        = ckpt["d_model"]
    log.info(f"  prior loaded ({'trainable' if trainable else 'frozen'}): "
             f"d_model={ckpt['d_model']}  "
             f"pretrain K={ckpt.get('n_modes','?')}  "
             f"mode_sig={ckpt.get('mode_signature','?')}  "
             f"n_samples={enc.expected_n_samples}  "
             f"epoch={ckpt.get('epoch','?')}  "
             f"val_loss={ckpt.get('val_loss', float('nan')):.4f}")
    return enc

def load_brainomni_tiny(device: torch.device, finetune: bool):
    from brainomni.model import BrainOmni
    with open(os.path.join(TINY_CKPT_DIR, "model_cfg.json")) as f:
        cfg = json.load(f)
    model = BrainOmni(**cfg)
    ckpt  = torch.load(os.path.join(TINY_CKPT_DIR, "BrainOmni.pt"), map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)

    if finetune:
        for p in model.parameters():
            p.requires_grad = True
        for p in model.tokenizer.parameters():
            p.requires_grad = False
        model.tokenizer.eval()
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        log.info(f"  BrainOmni-tiny loaded  finetune=ON  (tokenizer frozen)  "
                 f"lm_dim={cfg['lm_dim']}  trainable={n_tr:.2f}M")
    else:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        log.info(f"  BrainOmni-tiny loaded & FROZEN  lm_dim={cfg['lm_dim']}")
    return model, cfg["lm_dim"]

class ObsEncoder(nn.Module):





    def __init__(self, backbone, lm_dim: int, n_tok: int, d_model: int,
                 backbone_trainable: bool):
        super().__init__()
        self.backbone = backbone
        self.n_tok    = n_tok
        self.backbone_trainable = backbone_trainable
        self.adapter  = nn.Sequential(
            nn.Linear(lm_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def encode_raw(self, x_raw, eeg_pos, sensor_type) -> torch.Tensor:

        if self.backbone_trainable:
            enc = self.backbone.encode(x_raw, eeg_pos, sensor_type)
        else:
            with torch.no_grad():
                enc = self.backbone.encode(x_raw, eeg_pos, sensor_type)
        feat = enc.float().mean(dim=1)
        feat = feat.permute(0, 2, 1)
        feat = F.adaptive_avg_pool1d(feat, self.n_tok)
        return feat.permute(0, 2, 1)

    def apply_adapter(self, feat: torch.Tensor) -> torch.Tensor:
        return self.adapter(feat)

    def forward(self, x_raw, eeg_pos, sensor_type):
        return self.apply_adapter(self.encode_raw(x_raw, eeg_pos, sensor_type))

class GaussHead(nn.Module):
    def __init__(self, d_in: int, lat_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden, lat_dim)
        self.lv_head = nn.Linear(hidden, lat_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu_head(h), self.lv_head(h).clamp(-8, 4)

class ClsHead(nn.Module):
    def __init__(self, lat_dim: int, hidden: int = 256,
                 n_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(lat_dim),
            nn.Linear(lat_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, mu):
        return self.net(mu.mean(dim=1))

class ExpertModel(nn.Module):






    def __init__(self, encoder, ghead: GaussHead, cls_head: ClsHead,
                 is_obs: bool, encoder_trainable: bool):
        super().__init__()
        self.encoder, self.ghead, self.cls_head = encoder, ghead, cls_head
        self.is_obs            = is_obs
        self.encoder_trainable = encoder_trainable

        self.eigval: torch.Tensor | None = None

    def set_eigval(self, eigval: torch.Tensor | None):
        self.eigval = eigval

    def encode(self, x_modes=None, x_raw=None, eeg_pos=None, sensor_type=None):
        if self.is_obs:
            tok = self.encoder(x_raw, eeg_pos, sensor_type)
        elif self.encoder_trainable:
            tok = self.encoder.encode_pooled(x_modes, eigval=self.eigval)
        else:
            with torch.no_grad():
                tok = self.encoder.encode_pooled(x_modes, eigval=self.eigval)
        mu, lv = self.ghead(tok)
        return mu, lv

    def forward(self, x_modes=None, x_raw=None, eeg_pos=None, sensor_type=None):
        mu, lv = self.encode(x_modes, x_raw, eeg_pos, sensor_type)
        return self.cls_head(mu), mu, lv

    def cached_forward(self, cached: torch.Tensor):

        if self.is_obs:
            tok = self.encoder.apply_adapter(cached)
        else:
            tok = cached
        mu, lv = self.ghead(tok)
        return self.cls_head(mu), mu, lv

def precompute_features(bundle: dict, idx: np.ndarray,
                        backbone, prior_enc: EEGMEGPriorEigenval4,
                        n_tok: int, eigval: torch.Tensor | None,
                        device: torch.device, batch_size: int = 64
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:



    backbone.eval(); prior_enc.eval()
    obs_feats, prior_toks, ys = [], [], []
    N = len(idx)
    idx_t = torch.as_tensor(idx, dtype=torch.long)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, N, batch_size):
            sub = idx_t[s: s + batch_size]
            xm  = bundle["modes"][sub].to(device)
            xr  = bundle["raw"][sub].to(device)
            pos = bundle["pos"][sub].to(device)
            stp = bundle["stype"][sub].to(device)
            yb  = bundle["labels"][sub]


            enc  = backbone.encode(xr, pos, stp)
            feat = enc.float().mean(dim=1).permute(0, 2, 1)
            feat = F.adaptive_avg_pool1d(feat, n_tok).permute(0, 2, 1)
            obs_feats.append(feat.cpu())


            tok = prior_enc.encode_pooled(xm, eigval=eigval)
            prior_toks.append(tok.float().cpu())

            ys.append(yb)

    return torch.cat(obs_feats), torch.cat(prior_toks), torch.cat(ys)

def _stratified_half_split(y: np.ndarray, n_classes: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    A, B = [], []
    for c in range(n_classes):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        mid = len(idx_c) // 2
        A.extend(idx_c[:mid].tolist())
        B.extend(idx_c[mid:].tolist())
    return np.array(sorted(A)), np.array(sorted(B))

def load_subject(h5_dir: str, subj: str, n_samples: int, modality: str = "eeg",
                 label_key: str = "labels",
                 label_offset: int = 0,
                 n_classes: int | None = None,
                 override_eigval: torch.Tensor | None = None):


    h5_path = os.path.join(h5_dir, f"{subj}.h5")
    if not os.path.isfile(h5_path):
        log.warning(f"  {subj}: not found — skip")
        return None
    ev_key = f"{modality}_mode_eigval"
    with h5py.File(h5_path, "r") as f:
        modes  = torch.tensor(f[f"{modality}_modes"][:],      dtype=torch.float32)
        raw    = torch.tensor(f[f"{modality}_raw"][:],         dtype=torch.float32)
        labels = torch.tensor(f[label_key][:], dtype=torch.long) - label_offset
        pos    = torch.tensor(f[f"{modality}_pos"][:],         dtype=torch.float32)
        stype  = torch.tensor(f[f"{modality}_sensor_type"][:], dtype=torch.long)
        if override_eigval is None:
            if ev_key not in f:
                raise KeyError(
                    f"{h5_path}: missing '{ev_key}' — provide --d_path or pre-patch h5."
                )
            eigval = torch.tensor(f[ev_key][:], dtype=torch.float32)
        else:
            K_in = modes.shape[1]
            if override_eigval.shape[0] != K_in:
                raise ValueError(
                    f"{subj}: modes K={K_in} ≠ --d_path eigval K={override_eigval.shape[0]}"
                )
            eigval = override_eigval.detach().cpu().float()
    if labels.numel() and n_classes is not None:
        lo, hi = int(labels.min()), int(labels.max())
        if lo < 0 or hi >= n_classes:
            raise ValueError(
                f"{subj}: '{label_key}' after --label_offset={label_offset} has "
                f"range [{lo}, {hi}], outside [0, {n_classes - 1}]"
            )
    modes = modes[:, :, :n_samples]
    raw   = raw[:,   :, :n_samples]
    N = len(labels)
    return {
        "modes":  modes,
        "raw":    raw,
        "labels": labels,
        "pos":    pos.unsqueeze(0).expand(N, -1, -1).contiguous(),
        "stype":  stype.unsqueeze(0).expand(N, -1).contiguous(),
        "eigval": eigval,
        "subj":   subj,
        "K":      modes.shape[1],
    }
