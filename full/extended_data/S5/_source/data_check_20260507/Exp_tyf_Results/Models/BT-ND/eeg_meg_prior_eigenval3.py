






































from __future__ import annotations
import torch
import torch.nn as nn

from .eeg_meg_prior_eigenval import (
    sinusoidal_pos, patchify, brainomni_2d_mask,
    FactorizedBlock, TimeBlock, KAttentionPool, PATCH_SIZE,
)


class EEGMEGPriorEigenval3(nn.Module):










    def __init__(self, patch_size: int = PATCH_SIZE, d_model: int = 512,
                 n_heads: int = 8, n_factor_layers: int = 2, n_time_layers: int = 4,
                 dropout: float = 0.1, mode_pos_normalize: float = 1024.0):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.mode_pos_normalize = mode_pos_normalize


        self.patch_proj = nn.Linear(patch_size, d_model, bias=False)
        nn.init.trunc_normal_(self.patch_proj.weight, std=0.02)

        self.factor_blocks = nn.ModuleList([
            FactorizedBlock(d_model, n_heads, dropout) for _ in range(n_factor_layers)
        ])
        self.k_pool = KAttentionPool(d_model, n_heads, dropout)
        self.time_blocks = nn.ModuleList([
            TimeBlock(d_model, n_heads, dropout) for _ in range(n_time_layers)
        ])
        self.norm = nn.LayerNorm(d_model)


        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _mode_pos(self, eigval: torch.Tensor, dtype) -> torch.Tensor:






        ev = eigval.to(dtype).clamp(min=1e-12)
        ev_log = torch.log(ev)


        ev_log = ev_log - ev_log.amin(dim=-1, keepdim=True)
        span   = ev_log.amax(dim=-1, keepdim=True).clamp(min=1e-6)
        ev_log = ev_log * (self.mode_pos_normalize / span)

        return sinusoidal_pos(ev_log, self.d_model)

    def _embed(self, x: torch.Tensor, eigval: torch.Tensor,
               loss_mask: torch.Tensor | None):

        B, K, T = x.shape
        if T % self.patch_size != 0:
            raise ValueError(f"T={T} not divisible by patch_size={self.patch_size}")
        if eigval is None:
            raise ValueError("eigval is required for v3 (use v2 if you want to skip it).")

        patches = patchify(x, self.patch_size)
        NP = patches.shape[2]
        tokens = self.patch_proj(patches)

        if loss_mask is not None:
            mt = self.mask_token.to(tokens.dtype).view(1, 1, 1, self.d_model)
            tokens = torch.where(loss_mask.unsqueeze(-1), mt.expand_as(tokens), tokens)


        mode_pos = self._mode_pos(eigval, tokens.dtype)
        if mode_pos.dim() == 2:
            tokens = tokens + mode_pos.view(1, K, 1, self.d_model)
        else:
            tokens = tokens + mode_pos.view(B, K, 1, self.d_model)

        t_idx    = torch.arange(NP, device=x.device, dtype=tokens.dtype)
        time_pos = sinusoidal_pos(t_idx, self.d_model)
        tokens   = tokens + time_pos.view(1, 1, NP, self.d_model)
        return tokens, NP

    def encode_full_grid(self, x: torch.Tensor, eigval: torch.Tensor,
                         loss_mask: torch.Tensor | None = None) -> torch.Tensor:

        tokens, _ = self._embed(x, eigval, loss_mask)
        for blk in self.factor_blocks:
            tokens = blk(tokens)
        return self.norm(tokens)

    def encode_pooled(self, x: torch.Tensor, eigval: torch.Tensor,
                      loss_mask: torch.Tensor | None = None) -> torch.Tensor:

        tokens, _ = self._embed(x, eigval, loss_mask)
        for blk in self.factor_blocks:
            tokens = blk(tokens)
        x = self.k_pool(tokens)
        for blk in self.time_blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor, eigval: torch.Tensor,
                loss_mask: torch.Tensor | None = None) -> torch.Tensor:

        return self.encode_pooled(x, eigval, loss_mask)


class EEGMEGMAEDecoderEigenval3(nn.Module):





    def __init__(self, d_model: int = 512, d_dec: int = 256,
                 n_heads: int = 4, n_layers: int = 2,
                 patch_size: int = PATCH_SIZE, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.enc_proj = nn.Linear(d_model, d_dec, bias=False)
        self.blocks = nn.ModuleList([
            FactorizedBlock(d_dec, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_dec)
        self.head = nn.Linear(d_dec, patch_size)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:

        x = self.enc_proj(encoded)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))
