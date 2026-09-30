















from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from ..prior.eeg_meg_prior_eigenval import (
    sinusoidal_pos, FactorizedBlock, TimeBlock,
)

from ..prior.fmri_prior_eigenval_v2 import MultiQueryGPool

from .fmri_cifti_encoder_xyz import (
    N_LH_DEFAULT, N_RH_DEFAULT, N_CORTEX_DEFAULT, N_GROUPS_PER_HEMISPHERE,
    LH_SURF_DEFAULT, RH_SURF_DEFAULT,
    cifti_2d_mask, compute_group_centroids,
    CortexCIFTIMAEDecoderXYZ as CortexCIFTIMAEDecoderXYZV2,
)


__all__ = [
    "CortexCIFTIEncoderXYZV2",
    "CortexCIFTIMAEDecoderXYZV2",
    "cifti_2d_mask",
    "compute_group_centroids",
]


class CortexCIFTIEncoderXYZV2(nn.Module):


















    def __init__(self,
                 n_lh: int = N_LH_DEFAULT, n_rh: int = N_RH_DEFAULT,
                 n_groups_per_hemi: int = N_GROUPS_PER_HEMISPHERE,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_factor_layers: int = 2,
                 n_time_layers: int = 2,
                 n_pool_queries: int = 4,
                 dropout: float = 0.1,
                 group_centroids: torch.Tensor | None = None,
                 use_xyz_pos: bool = True,
                 xyz_scale: float = 1.0):
        super().__init__()
        self.n_lh = n_lh
        self.n_rh = n_rh
        self.n_cortex = n_lh + n_rh
        self.n_groups_per_hemi = n_groups_per_hemi
        self.n_groups = 2 * n_groups_per_hemi
        self.g_lh = n_lh // n_groups_per_hemi
        self.g_rh = n_rh // n_groups_per_hemi
        self.max_group_size = max(self.g_lh, self.g_rh)
        self.d_model = d_model
        self.n_pool_queries = n_pool_queries
        self.use_xyz_pos = use_xyz_pos
        self.xyz_scale = xyz_scale


        self.patch_embed = nn.Linear(self.max_group_size, d_model, bias=False)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)

        if group_centroids is not None:
            self.register_buffer('group_centroids', group_centroids.float())
        else:
            self.register_buffer('group_centroids', torch.zeros(self.n_groups, 3))

        self.factor_blocks = nn.ModuleList([
            FactorizedBlock(d_model, n_heads, dropout)
            for _ in range(n_factor_layers)
        ])
        self.g_pool = MultiQueryGPool(
            d_model=d_model, n_queries=n_pool_queries,
            n_heads=n_heads, dropout=dropout)
        self.time_blocks = nn.ModuleList([
            TimeBlock(d_model, n_heads, dropout)
            for _ in range(n_time_layers)
        ])
        self.norm = nn.LayerNorm(d_model)


        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)


    def set_group_centroids(self, centroids: torch.Tensor):
        if centroids.shape != (self.n_groups, 3):
            raise ValueError(f"centroids must be ({self.n_groups}, 3), got {tuple(centroids.shape)}")
        self.group_centroids = centroids.float().to(self.group_centroids.device)

    def auto_load_centroids(self,
                              lh_surf: str = LH_SURF_DEFAULT,
                              rh_surf: str = RH_SURF_DEFAULT):
        cents, _ = compute_group_centroids(
            lh_surf, rh_surf,
            self.n_lh, self.n_rh, self.n_groups_per_hemi)
        self.set_group_centroids(torch.from_numpy(cents))

    def _group_pos(self, dtype) -> torch.Tensor:

        if not self.use_xyz_pos:
            return torch.zeros(self.n_groups, self.d_model,
                               device=self.group_centroids.device, dtype=dtype)
        d_per_axis = (self.d_model // 3 // 2) * 2
        d_remainder = self.d_model - 3 * d_per_axis
        xyz = self.group_centroids.to(dtype) * self.xyz_scale
        parts = []
        for axis in range(3):
            parts.append(sinusoidal_pos(xyz[:, axis], d_per_axis))
        if d_remainder > 0:
            parts.append(torch.zeros(self.n_groups, d_remainder,
                                      device=xyz.device, dtype=dtype))
        return torch.cat(parts, dim=-1)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:

        B, N, T = x.shape
        if N != self.n_cortex:
            raise ValueError(f"x has {N} cortex verts, expected {self.n_cortex}")
        lh = x[:, :self.n_lh, :]
        rh = x[:, self.n_lh:, :]
        lh = lh.reshape(B, self.n_groups_per_hemi, self.g_lh, T)
        rh = rh.reshape(B, self.n_groups_per_hemi, self.g_rh, T)
        if self.g_lh < self.max_group_size:
            pad = self.max_group_size - self.g_lh
            lh = F.pad(lh, (0, 0, 0, pad), value=0.0)
        if self.g_rh < self.max_group_size:
            pad = self.max_group_size - self.g_rh
            rh = F.pad(rh, (0, 0, 0, pad), value=0.0)
        x = torch.cat([lh, rh], dim=1)
        x = x.permute(0, 3, 1, 2)
        return x

    def _embed(self, x: torch.Tensor, loss_mask: torch.Tensor | None):

        patches = self._patchify(x)
        B, T, G, _ = patches.shape
        tokens = self.patch_embed(patches)
        if loss_mask is not None:
            mt = self.mask_token.to(tokens.dtype).view(1, 1, 1, self.d_model)
            tokens = torch.where(loss_mask.unsqueeze(-1), mt.expand_as(tokens), tokens)
        gpos = self._group_pos(tokens.dtype)
        tokens = tokens + gpos.view(1, 1, G, self.d_model)
        t_idx = torch.arange(T, device=x.device, dtype=tokens.dtype)
        tpos = sinusoidal_pos(t_idx, self.d_model)
        tokens = tokens + tpos.view(1, T, 1, self.d_model)
        return tokens, T


    def encode_full_grid(self, x: torch.Tensor,
                         loss_mask: torch.Tensor | None = None) -> torch.Tensor:

        tokens, _ = self._embed(x, loss_mask)
        tokens = tokens.permute(0, 2, 1, 3).contiguous()
        for blk in self.factor_blocks:
            tokens = blk(tokens)
        tokens = tokens.permute(0, 2, 1, 3).contiguous()
        return self.norm(tokens)

    def encode_pooled(self, x: torch.Tensor,
                      loss_mask: torch.Tensor | None = None,
                      n_out_tokens: int | None = None) -> torch.Tensor:

        tokens, T = self._embed(x, loss_mask)
        tokens = tokens.permute(0, 2, 1, 3).contiguous()
        for blk in self.factor_blocks:
            tokens = blk(tokens)
        x = self.g_pool(tokens)
        for blk in self.time_blocks:
            x = blk(x)
        x = self.norm(x)
        if n_out_tokens is not None and n_out_tokens != x.shape[1]:
            x = x.permute(0, 2, 1)
            x = F.adaptive_avg_pool1d(x, n_out_tokens)
            x = x.permute(0, 2, 1)
        return x

    def forward(self, x: torch.Tensor,
                loss_mask: torch.Tensor | None = None,
                n_out_tokens: int | None = None) -> torch.Tensor:
        return self.encode_pooled(x, loss_mask, n_out_tokens)
