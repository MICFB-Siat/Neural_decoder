
















from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.time_config import N_TOKENS, D_MODEL
from models.prior.eeg_meg_prior import EEGMEGPriorEncoder
from models.prior.fmri_prior import FMRIPriorEncoder
from models.encoders.brainomni_encoder import BrainOmniEncoder
from training.finetune.train_stage2_20260504 import MultiModalH5Dataset





class DeeperHead(nn.Module):


    def __init__(self, d_in: int, n_classes: int, hidden: int = 512, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionPool(nn.Module):


    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.q, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        q = self.q.expand(B, -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


class SubjectEmbed(nn.Module):


    def __init__(self, n_subjects: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(n_subjects, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, sid: torch.Tensor) -> torch.Tensor:

        return self.embed(sid).unsqueeze(1)


class CrossAttnGateFusionV2(nn.Module):











    def __init__(self, d_model: int = 1024, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(2 * d_model, d_model)
        self.norm_mid = nn.LayerNorm(d_model)


        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
        )
        self.norm_post = nn.LayerNorm(d_model)

    def forward(
        self,
        H_obs: torch.Tensor,
        H_prior: torch.Tensor,
        mode: str = "full",
    ) -> torch.Tensor:
        if mode == "obs_only":
            return H_obs
        if mode == "prior_only":
            return H_prior

        H_tilde, _ = self.cross_attn(H_obs, H_prior, H_prior)
        H_tilde = self.attn_norm(H_tilde)

        g = torch.sigmoid(self.gate_proj(torch.cat([H_obs, H_tilde], dim=-1)))
        H_mid = self.norm_mid(H_obs + g * H_tilde)
        H_post = self.norm_post(H_mid + self.ffn(H_mid))
        return H_post





class BCIStage2ModelV2(nn.Module):





    def __init__(
        self,
        eegmeg_prior: Optional[EEGMEGPriorEncoder],
        fmri_prior: Optional[FMRIPriorEncoder],
        brainomni: Optional[BrainOmniEncoder],
        cifti,
        fusion: CrossAttnGateFusionV2,
        n_classes: int,
        n_subjects: int,
        d_model: int = D_MODEL,
        dropout: float = 0.5,
        head_hidden: int = 512,
    ):
        super().__init__()
        self.eegmeg_prior = eegmeg_prior
        self.fmri_prior = fmri_prior
        self.brainomni = brainomni
        self.cifti = cifti
        self.fusion = fusion


        self.subject_embed = SubjectEmbed(n_subjects, d_model) if n_subjects > 0 else None


        self.pool = AttentionPool(d_model)

        self.head = DeeperHead(d_model, n_classes, hidden=head_hidden, dropout=dropout)

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
            pos = batch.get(f"{mod}_pos")
            stype = batch.get(f"{mod}_sensor_type")
            if (
                self.brainomni is not None
                and x_raw is not None
                and pos is not None
                and stype is not None
            ):
                H_obs_list.append(self.brainomni(x_raw, pos, stype))

        x_fmri = self._blend(
            batch.get("fmri_modes"), batch.get("fmri_modes_indiv"), alpha
        )
        if self.fmri_prior is not None and x_fmri is not None:
            H_prior_list.append(self.fmri_prior(x_fmri))

        x_cifti = batch.get("fmri_cifti")
        if self.cifti is not None and x_cifti is not None:
            H_obs_list.append(self.cifti(x_cifti))

        H_obs = torch.stack(H_obs_list, 0).mean(0) if H_obs_list else None
        H_prior = torch.stack(H_prior_list, 0).mean(0) if H_prior_list else None
        return H_obs, H_prior

    def forward_three(self, batch: Dict, alpha: float = 0.0):




        H_obs, H_prior = self._get_branches(batch, alpha)

        sid = batch.get("subject_idx")
        if sid is not None and self.subject_embed is not None:
            sub_emb = self.subject_embed(sid)
            if H_obs is not None:
                H_obs = H_obs + sub_emb
            if H_prior is not None:
                H_prior = H_prior + sub_emb

        if H_obs is not None and H_prior is not None:
            H_full = self.fusion(H_obs, H_prior, "full")
        elif H_obs is not None:
            H_full = H_obs
        else:
            H_full = H_prior

        logits_prior = self.head(self.pool(H_prior)) if H_prior is not None else None
        logits_obs   = self.head(self.pool(H_obs))   if H_obs   is not None else None
        logits_full  = self.head(self.pool(H_full))  if H_full  is not None else None

        return logits_prior, logits_obs, logits_full





class MultiModalH5DatasetV2(MultiModalH5Dataset):





    def __init__(
        self,
        h5_paths: List[str],
        n_samples: int,
        modality_filter: Optional[str] = None,
        subject_idx_map: Optional[Dict[str, int]] = None,
    ):
        super().__init__(h5_paths, n_samples, modality_filter=modality_filter)
        if subject_idx_map is None:
            subject_idx_map = {p: i for i, p in enumerate(sorted(h5_paths))}
        self.subject_idx_map = subject_idx_map
        self.n_subjects = max(subject_idx_map.values()) + 1

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        path, _, _ = self.records[idx]
        item["subject_idx"] = torch.tensor(
            self.subject_idx_map[path], dtype=torch.long
        )
        return item





def stratified_within_fold_indices(
    h5_paths: List[str],
    cache_map: Dict[str, Dict],
    fold_idx: int,
    n_folds: int,
    seed: int,
) -> Tuple[List[int], List[int]]:








    train_idx: List[int] = []
    test_idx: List[int] = []
    offset = 0
    for path in h5_paths:
        labels = cache_map[path]["labels"]
        n = len(labels)
        path_seed = (seed + hash(path)) & 0xFFFFFFFF
        rng = np.random.default_rng(path_seed)

        n_classes = int(labels.max()) + 1
        train_local: List[int] = []
        test_local: List[int] = []

        if n_folds <= 1:

            for c in range(n_classes):
                cls_idx = np.where(labels == c)[0]
                rng.shuffle(cls_idx)
                n_train = int(len(cls_idx) * 0.8)
                train_local.extend(int(i) for i in cls_idx[:n_train])
                test_local.extend(int(i) for i in cls_idx[n_train:])
        else:
            for c in range(n_classes):
                cls_idx = np.where(labels == c)[0]
                rng.shuffle(cls_idx)
                buckets = np.array_split(cls_idx, n_folds)
                for fi, bucket in enumerate(buckets):
                    if fi == fold_idx:
                        test_local.extend(int(i) for i in bucket)
                    else:
                        train_local.extend(int(i) for i in bucket)

        train_idx.extend(offset + i for i in train_local)
        test_idx.extend(offset + i for i in test_local)
        offset += n
    return train_idx, test_idx


def stratified_holdout_712_indices(
    h5_paths: List[str],
    cache_map: Dict[str, Dict],
    seed: int,
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> Tuple[List[int], List[int], List[int]]:




    r_tr, r_va, r_te = ratios
    assert abs(r_tr + r_va + r_te - 1.0) < 1e-6
    train_idx, val_idx, test_idx = [], [], []
    offset = 0
    for path in h5_paths:
        labels = cache_map[path]["labels"]
        n = len(labels)
        path_seed = (seed + hash(path)) & 0xFFFFFFFF
        rng = np.random.default_rng(path_seed)
        n_classes = int(labels.max()) + 1
        for c in range(n_classes):
            cls_idx = np.where(labels == c)[0]
            rng.shuffle(cls_idx)
            n_c = len(cls_idx)
            n_tr = int(round(n_c * r_tr))
            n_va = int(round(n_c * r_va))
            tr_local = cls_idx[:n_tr]
            va_local = cls_idx[n_tr:n_tr + n_va]
            te_local = cls_idx[n_tr + n_va:]
            train_idx.extend(offset + int(i) for i in tr_local)
            val_idx.extend(offset + int(i) for i in va_local)
            test_idx.extend(offset + int(i) for i in te_local)
        offset += n
    return train_idx, val_idx, test_idx


def compute_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:

    counts = np.bincount(labels.astype(np.int64), minlength=n_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    weights = labels.shape[0] / (n_classes * counts)

    weights = np.clip(weights, 0.5, 3.0)
    return torch.tensor(weights, dtype=torch.float32)
