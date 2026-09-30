






















from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)





class AttentionPoolV3(nn.Module):


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


class GaussHead(nn.Module):








    def __init__(self, d_in: int, lat_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden, lat_dim)
        self.lv_head = nn.Linear(hidden, lat_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.mu_head(h)
        lv = self.lv_head(h).clamp(-8.0, 4.0)
        return mu, lv


class ClsHeadV3(nn.Module):





    def __init__(
        self,
        lat_dim: int,
        n_classes: int,
        hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(lat_dim),
            nn.Linear(lat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        return self.net(mu)


class SubjectEmbedV3(nn.Module):


    def __init__(self, n_subjects: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(n_subjects, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, sid: torch.Tensor) -> torch.Tensor:
        return self.embed(sid).unsqueeze(1)





class BCIPoEModelV3(nn.Module):










    def __init__(
        self,
        eegmeg_prior: nn.Module,
        brainomni: nn.Module,
        n_classes: int,
        n_subjects: int,
        d_model: int = 1024,
        lat_dim: int = 256,
        dropout: float = 0.3,
        head_hidden: int = 256,
    ):
        super().__init__()
        self.eegmeg_prior = eegmeg_prior
        self.brainomni = brainomni
        self.n_classes = n_classes

        self.subject_embed = (
            SubjectEmbedV3(n_subjects, d_model) if n_subjects > 0 else None
        )

        self.pool_prior = AttentionPoolV3(d_model)
        self.pool_obs = AttentionPoolV3(d_model)

        self.prior_gauss = GaussHead(d_model, lat_dim)
        self.obs_gauss = GaussHead(d_model, lat_dim)
        self.poe_gauss = GaussHead(lat_dim * 2, lat_dim)

        self.prior_cls = ClsHeadV3(lat_dim, n_classes, head_hidden, dropout)
        self.obs_cls = ClsHeadV3(lat_dim, n_classes, head_hidden, dropout)
        self.poe_cls = ClsHeadV3(lat_dim, n_classes, head_hidden, dropout)



    @staticmethod
    def gaussian_poe(
        mu_p: torch.Tensor,
        lv_p: torch.Tensor,
        mu_o: torch.Tensor,
        lv_o: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        prec_p = (-lv_p).exp().clamp(max=1e6)
        prec_o = (-lv_o).exp().clamp(max=1e6)
        prec_t = prec_p + prec_o
        mu_poe = (mu_p * prec_p + mu_o * prec_o) / prec_t
        lv_poe = -prec_t.log().clamp(min=-8.0)
        return mu_poe, lv_poe

    @staticmethod
    def kl_loss(mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:

        return 0.5 * (lv.exp() + mu.pow(2) - 1.0 - lv).mean()

    def _encode_prior(
        self,
        eeg_modes: torch.Tensor,
        sid: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            H = self.eegmeg_prior(eeg_modes)
        if self.subject_embed is not None and sid is not None:
            H = H + self.subject_embed(sid)
        h = self.pool_prior(H)
        return self.prior_gauss(h)

    def _encode_obs(
        self,
        eeg_raw: torch.Tensor,
        eeg_pos: torch.Tensor,
        eeg_stype: torch.Tensor,
        sid: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H = self.brainomni(eeg_raw, eeg_pos, eeg_stype)
        if self.subject_embed is not None and sid is not None:
            H = H + self.subject_embed(sid)
        h = self.pool_obs(H)
        return self.obs_gauss(h)



    def forward(self, batch: Dict) -> Tuple:







        sid = batch.get("subject_idx")
        mu_p, lv_p = self._encode_prior(batch["eeg_modes"], sid)
        mu_o, lv_o = self._encode_obs(
            batch["eeg_raw"], batch["eeg_pos"], batch["eeg_sensor_type"], sid
        )

        mu_poe_raw, _ = self.gaussian_poe(mu_p, lv_p, mu_o, lv_o)

        mu_poe_ref, _ = self.poe_gauss(torch.cat([mu_p, mu_o], dim=-1))

        mu_poe = 0.7 * mu_poe_raw + 0.3 * mu_poe_ref

        logits_prior = self.prior_cls(mu_p)
        logits_obs = self.obs_cls(mu_o)
        logits_poe = self.poe_cls(mu_poe)

        return logits_prior, logits_obs, logits_poe, mu_p, lv_p, mu_o, lv_o

    def forward_conf_poe(self, batch: Dict) -> torch.Tensor:




        sid = batch.get("subject_idx")
        mu_p, lv_p = self._encode_prior(batch["eeg_modes"], sid)
        mu_o, lv_o = self._encode_obs(
            batch["eeg_raw"], batch["eeg_pos"], batch["eeg_sensor_type"], sid
        )
        logits_p = self.prior_cls(mu_p)
        logits_o = self.obs_cls(mu_o)

        log_p = F.log_softmax(logits_p.float(), dim=-1)
        log_o = F.log_softmax(logits_o.float(), dim=-1)
        prob_p = log_p.exp()
        prob_o = log_o.exp()
        chance = 1.0 / self.n_classes

        w_p = (prob_p.max(1).values - chance).clamp(min=0.0)
        w_o = (prob_o.max(1).values - chance).clamp(min=1e-4)
        w_sum = (w_p + w_o).clamp(min=1e-8)
        wp = (w_p / w_sum).unsqueeze(1)
        wo = (w_o / w_sum).unsqueeze(1)

        return wp * log_p + wo * log_o





import h5py


class SEEDVPoEDataset(Dataset):















    def __init__(
        self,
        h5_paths: list,
        n_modes: int = 48,
        n_samples: int = 2560,
        subject_idx_map: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.n_modes = n_modes
        self.n_samples = n_samples

        if subject_idx_map is None:
            subject_idx_map = {p: i for i, p in enumerate(sorted(h5_paths))}
        self.subject_idx_map = subject_idx_map
        self.n_subjects = max(subject_idx_map.values()) + 1


        self.records = []
        self._labels_per_path: Dict[str, np.ndarray] = {}

        for path in sorted(h5_paths):
            sid = subject_idx_map[path]
            with h5py.File(path, "r") as f:
                modes = torch.tensor(
                    f["eeg_modes"][:, :n_modes, :n_samples], dtype=torch.float32
                )
                raw = torch.tensor(
                    f["eeg_raw"][:, :, :n_samples], dtype=torch.float32
                )
                labels = torch.tensor(f["labels"][:], dtype=torch.long)
                pos = torch.tensor(f["eeg_pos"][:], dtype=torch.float32)
                stype = torch.tensor(f["eeg_sensor_type"][:], dtype=torch.long)

            N = len(labels)
            pos_exp = pos.unsqueeze(0).expand(N, -1, -1)
            stype_exp = stype.unsqueeze(0).expand(N, -1)
            sid_t = torch.full((N,), sid, dtype=torch.long)

            self._labels_per_path[path] = labels.numpy().astype(np.int64)

            for i in range(N):
                self.records.append((
                    modes[i], raw[i], pos_exp[i], stype_exp[i],
                    labels[i], sid_t[i],
                ))

        self.n_classes = int(
            max(r[4].item() for r in self.records) + 1
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        modes, raw, pos, stype, label, sid = self.records[idx]
        return {
            "eeg_modes":       modes,
            "eeg_raw":         raw,
            "eeg_pos":         pos,
            "eeg_sensor_type": stype,
            "label":           label,
            "subject_idx":     sid,
        }

    @staticmethod
    def collate_fn(batch):
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) for k in keys}





def per_subject_stratified_split_80_20(
    h5_paths: list,
    labels_per_path: Dict[str, np.ndarray],
    seed: int = 42,
) -> Tuple[list, list]:




    train_idx, test_idx = [], []
    offset = 0
    for path in sorted(h5_paths):
        labels = labels_per_path[path]
        n = len(labels)
        path_seed = (seed + hash(path)) & 0xFFFFFFFF
        rng = np.random.default_rng(path_seed)
        n_classes = int(labels.max()) + 1
        tr_local, te_local = [], []
        for c in range(n_classes):
            cls_idx = np.where(labels == c)[0]
            rng.shuffle(cls_idx)
            n_train = int(round(len(cls_idx) * 0.8))
            tr_local.extend(int(i) for i in cls_idx[:n_train])
            te_local.extend(int(i) for i in cls_idx[n_train:])
        train_idx.extend(offset + i for i in tr_local)
        test_idx.extend(offset + i for i in te_local)
        offset += n
    return train_idx, test_idx


def compute_class_weights_v3(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=n_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    weights = labels.shape[0] / (n_classes * counts)
    weights = np.clip(weights, 0.5, 3.0)
    return torch.tensor(weights, dtype=torch.float32)
