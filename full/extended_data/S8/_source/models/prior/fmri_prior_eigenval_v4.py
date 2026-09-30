


















from __future__ import annotations
import torch
import torch.nn as nn

from .eeg_meg_prior_eigenval import sinusoidal_pos
from .fmri_prior_eigenval_v3_patch import FMRIPriorEigenvalPatch, patch_mask
from .fmri_prior_eigenval_v3_nogroup import SDPASelfAttn

__all__ = ["FMRIPriorEigenvalV4", "patch_mask"]


class PooledMAEDecoder(nn.Module):







    def __init__(self, d_model, max_ps, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.q_in = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
             for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model),
                          nn.GELU(), nn.Linear(2 * d_model, d_model))
            for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, max_ps))

    def forward(self, z, patch_pos, T):

        B = z.shape[0]
        P = patch_pos.shape[0]
        tpos = sinusoidal_pos(torch.arange(T, device=z.device, dtype=z.dtype), self.d_model)

        q = patch_pos.view(1, P, -1) + tpos.view(T, 1, -1)
        q = self.q_in(q).reshape(1, T * P, self.d_model).expand(B, -1, -1).contiguous()
        for attn, ln, ffn in zip(self.layers, self.norms, self.ffns):
            a, _ = attn(q, z, z, need_weights=False)
            q = ln(q + a)
            q = q + ffn(q)
        rec = self.head(q)
        return rec.reshape(B, T, P, -1)


class FMRIPriorEigenvalV4(FMRIPriorEigenvalPatch):
    def __init__(self, n_modes=2000, n_lh=1000, patch_size=25, d_model=512,
                 n_heads=8, n_mode_layers=2, n_time_layers=4, n_pool_queries=16,
                 dropout=0.1, use_mode_signature=True, use_hemisphere=True,
                 mode_pos_normalize=1024.0, patch_bounds=None):
        super().__init__(n_modes=n_modes, n_lh=n_lh, patch_size=patch_size,
                         d_model=d_model, n_heads=n_heads, n_mode_layers=n_mode_layers,
                         n_time_layers=n_time_layers, n_pool_queries=n_pool_queries,
                         dropout=dropout, use_mode_signature=use_mode_signature,
                         use_hemisphere=use_hemisphere,
                         mode_pos_normalize=mode_pos_normalize, patch_bounds=patch_bounds)

        self.recon_head = None
        self.mae_dec = PooledMAEDecoder(d_model, self.max_ps, n_heads=4, n_layers=2,
                                        dropout=dropout)

    def mae_forward(self, x, full_eigval=None, loss_mask=None):





        B, K, T = x.shape
        tok = self._tokens(x, full_eigval, loss_mask=None)
        h = self._backbone(tok)
        pooled = self._pool(h)
        z = pooled.reshape(B, T * self.n_pool_queries, self.d_model)
        for blk in self.out_blocks:
            z = blk(z)
        z = self.norm(z)
        patch_pos = self._patch_pos(full_eigval, x.dtype, x.device)
        rp = self.mae_dec(z, patch_pos, T)
        rec = x.new_zeros(B, T, K)
        for i in range(self.n_patch):
            a, b = self.bounds[i], self.bounds[i + 1]
            rec[:, :, a:b] = rp[:, :, i, :b - a]
        return rec
