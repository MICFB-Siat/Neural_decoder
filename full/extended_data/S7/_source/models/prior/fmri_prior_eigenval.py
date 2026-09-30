








































from __future__ import annotations
import math
import torch
import torch.nn as nn


from .eeg_meg_prior_eigenval import (
    sinusoidal_pos, FactorizedBlock, TimeBlock, KAttentionPool
)


def fmri_2d_mask(B: int, T: int, G: int, mask_ratio: float, device):

    keep = torch.rand(B, T, G, device=device) > mask_ratio
    return ~keep


def grouped_patchify(x_t_first: torch.Tensor, n_groups: int, group_size: int) -> torch.Tensor:

    B, T, K = x_t_first.shape
    if K != n_groups * group_size:
        raise ValueError(f"K={K} ≠ G*group_size={n_groups}*{group_size}")
    return x_t_first.reshape(B, T, n_groups, group_size)


class FMRIPriorEigenval(nn.Module):











    def __init__(self, n_modes: int = 2000, n_groups: int = 8,
                 d_model: int = 512, d_inner: int = 512,
                 n_heads: int = 8, n_factor_layers: int = 2, n_time_layers: int = 4,
                 dropout: float = 0.1,
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
        self.use_mode_signature = use_mode_signature
        self.mode_pos_normalize = mode_pos_normalize


        self.group_encoder = nn.Sequential(
            nn.Linear(self.group_size, d_inner),
            nn.GELU(),
            nn.LayerNorm(d_inner),
            nn.Linear(d_inner, d_model),
            nn.LayerNorm(d_model),
        )

        self.factor_blocks = nn.ModuleList([
            FactorizedBlock(d_model, n_heads, dropout) for _ in range(n_factor_layers)
        ])
        self.g_pool = KAttentionPool(d_model, n_heads, dropout)
        self.time_blocks = nn.ModuleList([
            TimeBlock(d_model, n_heads, dropout) for _ in range(n_time_layers)
        ])
        self.norm = nn.LayerNorm(d_model)


        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _mode_pos(self, group_eigval: torch.Tensor, dtype) -> torch.Tensor:




        ev = group_eigval.to(dtype).clamp(min=1e-12)
        ev_log = torch.log(ev)
        ev_log = ev_log - ev_log.min()
        span = ev_log.max().clamp(min=1e-6)
        ev_log = ev_log * (self.mode_pos_normalize / span)
        return sinusoidal_pos(ev_log, self.d_model)

    def _embed(self, x: torch.Tensor, group_eigval: torch.Tensor | None,
               loss_mask: torch.Tensor | None):




        B, K, T = x.shape
        if K != self.n_modes:
            raise ValueError(f"K={K} mismatch n_modes={self.n_modes}")

        x = x.permute(0, 2, 1)
        x = grouped_patchify(x, self.n_groups, self.group_size)
        tokens = self.group_encoder(x)

        if loss_mask is not None:
            mt = self.mask_token.to(tokens.dtype).view(1, 1, 1, self.d_model)
            tokens = torch.where(loss_mask.unsqueeze(-1), mt.expand_as(tokens), tokens)

        if self.use_mode_signature and group_eigval is not None:
            mode_pos = self._mode_pos(group_eigval, tokens.dtype)
            tokens = tokens + mode_pos.view(1, 1, self.n_groups, self.d_model)
        t_idx = torch.arange(T, device=x.device, dtype=tokens.dtype)
        time_pos = sinusoidal_pos(t_idx, self.d_model)
        tokens = tokens + time_pos.view(1, T, 1, self.d_model)
        return tokens, T

    def encode_full_grid(self, x: torch.Tensor,
                         group_eigval: torch.Tensor | None = None,
                         loss_mask: torch.Tensor | None = None) -> torch.Tensor:

        tokens, _ = self._embed(x, group_eigval, loss_mask)





        tokens = tokens.permute(0, 2, 1, 3).contiguous()
        for blk in self.factor_blocks:
            tokens = blk(tokens)

        tokens = tokens.permute(0, 2, 1, 3).contiguous()
        return self.norm(tokens)

    def encode_pooled(self, x: torch.Tensor,
                      group_eigval: torch.Tensor | None = None,
                      loss_mask: torch.Tensor | None = None,
                      n_out_tokens: int | None = None) -> torch.Tensor:





        tokens, T = self._embed(x, group_eigval, loss_mask)

        tokens = tokens.permute(0, 2, 1, 3).contiguous()
        for blk in self.factor_blocks:
            tokens = blk(tokens)


        x = self.g_pool(tokens)
        for blk in self.time_blocks:
            x = blk(x)
        x = self.norm(x)

        if n_out_tokens is not None and n_out_tokens != T:
            x = x.permute(0, 2, 1)
            x = nn.functional.adaptive_avg_pool1d(x, n_out_tokens)
            x = x.permute(0, 2, 1)
        return x

    def forward(self, x: torch.Tensor,
                group_eigval: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None,
                n_out_tokens: int | None = None) -> torch.Tensor:
        return self.encode_pooled(x, group_eigval, loss_mask, n_out_tokens)


class FMRIMAEDecoderEigenval(nn.Module):

    def __init__(self, group_size: int, d_model: int = 512, d_dec: int = 256,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.group_size = group_size
        self.enc_proj = nn.Linear(d_model, d_dec, bias=False)
        self.blocks = nn.ModuleList([
            FactorizedBlock(d_dec, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_dec)
        self.head = nn.Linear(d_dec, group_size)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:


        x = encoded.permute(0, 2, 1, 3).contiguous()
        x = self.enc_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)
        return x.permute(0, 2, 1, 3).contiguous()




def compute_fmri_group_eigval(lh_eval_path: str, rh_eval_path: str,
                               n_modes: int = 2000, n_groups: int = 8):












    import numpy as np
    half = n_modes // 2
    lh = np.load(lh_eval_path)[:half].astype(np.float64)
    rh = np.load(rh_eval_path)[:half].astype(np.float64)
    full = np.concatenate([lh, rh])
    group_size = n_modes // n_groups
    group_eigval = full.reshape(n_groups, group_size).mean(axis=1)
    return group_eigval.astype(np.float32), full.astype(np.float32)
