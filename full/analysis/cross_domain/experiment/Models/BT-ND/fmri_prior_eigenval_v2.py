































from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from .eeg_meg_prior_eigenval import (
    sinusoidal_pos, FactorizedBlock, TimeBlock,
)

from .fmri_prior_eigenval import (
    fmri_2d_mask, compute_fmri_group_eigval,
    FMRIMAEDecoderEigenval as FMRIMAEDecoderEigenvalV2,
)


__all__ = [
    "FMRIPriorEigenvalV2",
    "FMRIMAEDecoderEigenvalV2",
    "MultiQueryGPool",
    "fmri_2d_mask",
    "compute_fmri_group_eigval",
]


class MultiQueryGPool(nn.Module):













    def __init__(self, d_model: int, n_queries: int = 4,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_queries = n_queries
        self.d_model = d_model
        self.queries = nn.Parameter(torch.zeros(1, n_queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:

        B, G, T, D = tokens.shape
        x = self.norm(tokens)

        x = x.permute(0, 2, 1, 3).reshape(B * T, G, D)
        q = self.queries.expand(B * T, -1, -1)
        out, _ = self.attn(q, x, x, need_weights=False)
        out = out.reshape(B, T, self.n_queries, D)
        out = out.reshape(B, T * self.n_queries, D)
        return out


class FMRIPriorEigenvalV2(nn.Module):













    def __init__(self,
                 n_modes: int = 2000, n_groups: int = 8,
                 d_model: int = 512, d_inner: int = 512,
                 n_heads: int = 8,
                 n_factor_layers: int = 2, n_time_layers: int = 4,
                 dropout: float = 0.1,
                 n_pool_queries: int = 4,
                 use_mode_signature: bool = True,
                 mode_pos_normalize: float = 1024.0):
        super().__init__()
        if n_modes % n_groups != 0:
            raise ValueError(f"n_modes={n_modes} must be divisible by n_groups={n_groups}")
        self.n_modes = n_modes
        self.n_groups = n_groups
        self.group_size = n_modes // n_groups
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_pool_queries = n_pool_queries
        self.use_mode_signature = use_mode_signature
        self.mode_pos_normalize = mode_pos_normalize



        self.post_proj = nn.Sequential(
            nn.LayerNorm(d_inner),
            nn.Linear(d_inner, d_inner),
            nn.GELU(),
            nn.LayerNorm(d_inner),
            nn.Linear(d_inner, d_model),
            nn.LayerNorm(d_model),
        )

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


        self.mask_token = nn.Parameter(torch.zeros(d_inner))
        nn.init.trunc_normal_(self.mask_token, std=0.02)


    def _mode_pos_table(self, full_eigval: torch.Tensor, dtype) -> torch.Tensor:



        ev = full_eigval.to(dtype).clamp(min=1e-12)
        ev_log = torch.log(ev)
        ev_log = ev_log - ev_log.min()
        span = ev_log.max().clamp(min=1e-6)
        ev_log = ev_log * (self.mode_pos_normalize / span)
        return sinusoidal_pos(ev_log, self.d_inner)

    def _embed(self, x: torch.Tensor,
               full_eigval: torch.Tensor | None,
               loss_mask: torch.Tensor | None) -> tuple[torch.Tensor, int]:


        B, K, T = x.shape
        if K != self.n_modes:
            raise ValueError(f"K={K} mismatch n_modes={self.n_modes}")


        if self.use_mode_signature and full_eigval is not None:
            pos_table = self._mode_pos_table(full_eigval, x.dtype)
        else:
            pos_table = torch.zeros(
                self.n_modes, self.d_inner,
                device=x.device, dtype=x.dtype)


        x_t = x.permute(0, 2, 1)
        coeff = x_t.reshape(B, T, self.n_groups, self.group_size)
        pos = pos_table.reshape(self.n_groups, self.group_size, self.d_inner)





        tokens = torch.einsum('btgs,gsd->btgd', coeff, pos)


        if loss_mask is not None:
            mt = self.mask_token.to(tokens.dtype).view(1, 1, 1, self.d_inner)
            tokens = torch.where(
                loss_mask.unsqueeze(-1), mt.expand_as(tokens), tokens)


        tokens = self.post_proj(tokens)


        t_idx = torch.arange(T, device=x.device, dtype=tokens.dtype)
        time_pos = sinusoidal_pos(t_idx, self.d_model)
        tokens = tokens + time_pos.view(1, T, 1, self.d_model)
        return tokens, T


    def encode_full_grid(self, x: torch.Tensor,
                         full_eigval: torch.Tensor | None = None,
                         loss_mask: torch.Tensor | None = None) -> torch.Tensor:

        tokens, _ = self._embed(x, full_eigval, loss_mask)

        tokens = tokens.permute(0, 2, 1, 3).contiguous()
        for blk in self.factor_blocks:
            tokens = blk(tokens)
        tokens = tokens.permute(0, 2, 1, 3).contiguous()
        return self.norm(tokens)

    def encode_pooled(self, x: torch.Tensor,
                      full_eigval: torch.Tensor | None = None,
                      loss_mask: torch.Tensor | None = None,
                      n_out_tokens: int | None = None) -> torch.Tensor:




        tokens, T = self._embed(x, full_eigval, loss_mask)
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
                full_eigval: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None,
                n_out_tokens: int | None = None) -> torch.Tensor:
        return self.encode_pooled(x, full_eigval, loss_mask, n_out_tokens)
