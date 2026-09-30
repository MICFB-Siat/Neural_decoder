


























from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .eeg_meg_prior_eigenval import sinusoidal_pos
from .fmri_prior_eigenval_v3_nogroup import Block

__all__ = ["FMRIPriorEigenvalPatch", "patch_mask"]


class FMRIPriorEigenvalPatch(nn.Module):
    def __init__(self,
                 n_modes: int = 2000, n_lh: int = 1000, patch_size: int = 25,
                 d_model: int = 512, n_heads: int = 8,
                 n_mode_layers: int = 4, n_time_layers: int = 2,
                 n_pool_queries: int = 4, dropout: float = 0.1,
                 use_mode_signature: bool = True, use_hemisphere: bool = True,
                 mode_pos_normalize: float = 1024.0,
                 patch_bounds: list | None = None):
        super().__init__()
        self.n_modes, self.n_lh, self.d_model = n_modes, n_lh, d_model
        self.n_pool_queries = n_pool_queries
        self.use_mode_signature = use_mode_signature
        self.use_hemisphere = use_hemisphere
        self.mode_pos_normalize = mode_pos_normalize


        if patch_bounds is None:
            assert n_modes % patch_size == 0, "patch_size must divide n_modes"
            assert n_lh % patch_size == 0, "patch_size must divide n_lh (clean hemis)"
            bounds = list(range(0, n_modes + 1, patch_size))
        else:
            bounds = list(patch_bounds)
        self.bounds = bounds
        self.n_patch = len(bounds) - 1
        self.max_ps = max(bounds[i + 1] - bounds[i] for i in range(self.n_patch))
        self.uniform = (patch_bounds is None)
        self.patch_size = patch_size if self.uniform else None
        self.n_groups = self.n_patch


        hemi = torch.tensor([0 if (bounds[i] + bounds[i + 1]) // 2 < n_lh else 1
                             for i in range(self.n_patch)], dtype=torch.long)
        self.register_buffer("patch_hemi", hemi, persistent=False)


        self.patch_embed = nn.Linear(self.max_ps, d_model)
        self.hemi_embed = nn.Parameter(torch.zeros(2, d_model)); nn.init.trunc_normal_(self.hemi_embed, std=0.02)
        self.mask_embed = nn.Parameter(torch.zeros(d_model)); nn.init.trunc_normal_(self.mask_embed, std=0.02)

        self.mode_blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_mode_layers)])
        self.time_attn_blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_mode_layers)])
        self.enc_norm = nn.LayerNorm(d_model)

        self.queries = nn.Parameter(torch.zeros(1, n_pool_queries, d_model)); nn.init.trunc_normal_(self.queries, std=0.02)
        self.pool_norm = nn.LayerNorm(d_model)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.out_blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_time_layers)])
        self.norm = nn.LayerNorm(d_model)

        self.recon_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.max_ps))


    def _patch_pos(self, full_eigval, dtype, device):
        if self.use_mode_signature and full_eigval is not None:
            ev = full_eigval.to(device=device, dtype=torch.float32).clamp(min=1e-12)
            ev_log = torch.log(ev)
            pv = torch.stack([ev_log[self.bounds[i]:self.bounds[i + 1]].mean()
                              for i in range(self.n_patch)])
            pv = pv - pv.min(); pv = pv * (self.mode_pos_normalize / pv.max().clamp(min=1e-6))
            pos = sinusoidal_pos(pv.to(dtype), self.d_model)
        else:
            pos = torch.zeros(self.n_patch, self.d_model, device=device, dtype=dtype)
        if self.use_hemisphere:
            pos = pos + self.hemi_embed[self.patch_hemi].to(dtype)
        return pos

    def _patchify(self, coeff):

        B, T, K = coeff.shape
        out = coeff.new_zeros(B, T, self.n_patch, self.max_ps)
        for i in range(self.n_patch):
            a, b = self.bounds[i], self.bounds[i + 1]
            out[:, :, i, :b - a] = coeff[:, :, a:b]
        return out

    def _tokens(self, x, full_eigval, loss_mask):
        B, K, T = x.shape
        coeff = x.permute(0, 2, 1)
        patches = self._patchify(coeff)
        tok = self.patch_embed(patches) + self._patch_pos(full_eigval, x.dtype, x.device)
        if loss_mask is not None:
            mk = (self.mask_embed.view(1, 1, 1, -1)
                  + self._patch_pos(full_eigval, tok.dtype, x.device).view(1, 1, self.n_patch, -1))
            tok = torch.where(loss_mask.unsqueeze(-1), mk.expand_as(tok), tok)
        return tok

    def _backbone(self, tok):
        B, T, P, D = tok.shape
        h = tok
        for mblk, tblk in zip(self.mode_blocks, self.time_attn_blocks):
            h = mblk(h.reshape(B * T, P, D)).reshape(B, T, P, D)
            h = tblk(h.permute(0, 2, 1, 3).reshape(B * P, T, D)).reshape(B, P, T, D).permute(0, 2, 1, 3)
        return self.enc_norm(h)

    def _pool(self, h):
        B, T, P, D = h.shape
        x = self.pool_norm(h).reshape(B * T, P, D)
        q = self.queries.expand(B * T, -1, -1)
        out, _ = self.pool_attn(q, x, x, need_weights=False)
        return out.reshape(B, T, self.n_pool_queries, D)

    def encode_pooled(self, x, full_eigval=None, loss_mask=None, n_out_tokens=None):
        h = self._backbone(self._tokens(x, full_eigval, loss_mask))
        pooled = self._pool(h)
        B, T, nq, D = pooled.shape
        z = pooled.reshape(B, T * nq, D)
        for blk in self.out_blocks:
            z = blk(z)
        z = self.norm(z)
        if n_out_tokens is not None and n_out_tokens != z.shape[1]:
            z = F.adaptive_avg_pool1d(z.permute(0, 2, 1), n_out_tokens).permute(0, 2, 1)
        return z

    def forward(self, x, full_eigval=None, loss_mask=None, n_out_tokens=None):
        return self.encode_pooled(x, full_eigval, loss_mask, n_out_tokens)

    def mae_forward(self, x, full_eigval=None, loss_mask=None):

        B, K, T = x.shape
        h = self._backbone(self._tokens(x, full_eigval, loss_mask))
        rp = self.recon_head(h)
        rec = x.new_zeros(B, T, K)
        for i in range(self.n_patch):
            a, b = self.bounds[i], self.bounds[i + 1]
            rec[:, :, a:b] = rp[:, :, i, :b - a]
        return rec


def patch_mask(B, T, n_patch, mask_ratio, device):

    return torch.rand(B, T, n_patch, device=device) <= mask_ratio
