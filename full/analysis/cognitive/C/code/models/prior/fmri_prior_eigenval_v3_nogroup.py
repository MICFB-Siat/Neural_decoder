




























from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .eeg_meg_prior_eigenval import sinusoidal_pos

__all__ = ["FMRIPriorEigenvalNoGroup", "nogroup_mask"]



class SDPASelfAttn(nn.Module):
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.nh, self.hd, self.p = n_heads, d // n_heads, dropout
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        o = F.scaled_dot_product_attention(q, k, v,
                                           dropout_p=self.p if self.training else 0.0)
        return self.proj(o.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):

    def __init__(self, d, n_heads, dropout, mlp=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.attn = SDPASelfAttn(d, n_heads, dropout)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(mlp * d, d))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class FMRIPriorEigenvalNoGroup(nn.Module):
    def __init__(self,
                 n_modes: int = 2000, n_lh: int = 1000,
                 d_model: int = 512, n_heads: int = 8,
                 n_mode_layers: int = 4, n_time_layers: int = 2,
                 n_pool_queries: int = 4, dropout: float = 0.1,
                 use_mode_signature: bool = True, use_hemisphere: bool = True,
                 mode_pos_normalize: float = 1024.0,
                 use_checkpoint: bool = False):
        super().__init__()
        self.n_modes, self.n_lh = n_modes, n_lh
        self.d_model = d_model
        self.n_pool_queries = n_pool_queries
        self.use_checkpoint = use_checkpoint
        self.use_mode_signature = use_mode_signature
        self.use_hemisphere = use_hemisphere
        self.mode_pos_normalize = mode_pos_normalize
        self.n_groups = n_modes

        hemi = torch.zeros(n_modes, dtype=torch.long); hemi[n_lh:] = 1
        self.register_buffer("hemi_id", hemi, persistent=False)
        self.hemi_embed = nn.Parameter(torch.zeros(2, d_model))
        nn.init.trunc_normal_(self.hemi_embed, std=0.02)

        self.in_proj = nn.Linear(1, d_model)
        self.mask_embed = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.mask_embed, std=0.02)

        self.mode_blocks = nn.ModuleList(
            [Block(d_model, n_heads, dropout) for _ in range(n_mode_layers)])
        self.time_attn_blocks = nn.ModuleList(
            [Block(d_model, n_heads, dropout) for _ in range(n_mode_layers)])
        self.enc_norm = nn.LayerNorm(d_model)


        self.queries = nn.Parameter(torch.zeros(1, n_pool_queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.pool_norm = nn.LayerNorm(d_model)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True)
        self.out_blocks = nn.ModuleList(
            [Block(d_model, n_heads, dropout) for _ in range(n_time_layers)])
        self.norm = nn.LayerNorm(d_model)

        self.recon_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))


    def _pos(self, full_eigval, dtype, device):
        if self.use_mode_signature and full_eigval is not None:
            ev = full_eigval.to(device=device, dtype=dtype).clamp(min=1e-12)
            ev_log = torch.log(ev); ev_log = ev_log - ev_log.min()
            ev_log = ev_log * (self.mode_pos_normalize / ev_log.max().clamp(min=1e-6))
            pos = sinusoidal_pos(ev_log, self.d_model)
        else:
            pos = torch.zeros(self.n_modes, self.d_model, device=device, dtype=dtype)
        if self.use_hemisphere:
            pos = pos + self.hemi_embed[self.hemi_id].to(dtype)
        return pos

    def _tokens(self, x, full_eigval, loss_mask):

        B, K, T = x.shape
        if K != self.n_modes:
            raise ValueError(f"K={K} != n_modes={self.n_modes}")
        pos = self._pos(full_eigval, x.dtype, x.device)
        coeff = x.permute(0, 2, 1).unsqueeze(-1)
        tok = self.in_proj(coeff) + pos.view(1, 1, K, self.d_model)
        if loss_mask is not None:
            mk = (self.mask_embed.view(1, 1, 1, -1) + pos.view(1, 1, K, -1)).to(tok.dtype)
            tok = torch.where(loss_mask.unsqueeze(-1), mk.expand_as(tok), tok)
        return tok

    def _backbone(self, tok):

        B, T, K, D = tok.shape
        h = tok
        ckpt = self.use_checkpoint and self.training
        for mblk, tblk in zip(self.mode_blocks, self.time_attn_blocks):
            hm = h.reshape(B * T, K, D)
            hm = checkpoint(mblk, hm, use_reentrant=False) if ckpt else mblk(hm)
            h = hm.reshape(B, T, K, D)
            ht = h.permute(0, 2, 1, 3).reshape(B * K, T, D)
            ht = checkpoint(tblk, ht, use_reentrant=False) if ckpt else tblk(ht)
            h = ht.reshape(B, K, T, D).permute(0, 2, 1, 3)
        return self.enc_norm(h)

    def _pool(self, h):

        B, T, K, D = h.shape
        x = self.pool_norm(h).reshape(B * T, K, D)
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
        h = self._backbone(self._tokens(x, full_eigval, loss_mask))
        rec = self.recon_head(h).squeeze(-1)
        return rec


def nogroup_mask(B, T, K, mask_ratio, device):

    return torch.rand(B, T, K, device=device) <= mask_ratio
