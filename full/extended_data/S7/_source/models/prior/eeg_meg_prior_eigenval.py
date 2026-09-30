





























from __future__ import annotations
import math
import torch
import torch.nn as nn

PATCH_SIZE = 64


def sinusoidal_pos(values: torch.Tensor, d_model: int) -> torch.Tensor:



    if d_model % 2 != 0:
        raise ValueError("d_model must be even")
    device = values.device
    half = d_model // 2
    div = torch.exp(-math.log(10000.0) *
                    torch.arange(half, device=device, dtype=values.dtype) / half)
    angles = values.unsqueeze(-1) * div.unsqueeze(0)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def patchify(x: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:

    return x.unfold(dimension=-1, size=patch_size, step=patch_size)


def brainomni_2d_mask(B: int, K: int, NP: int, mask_ratio: float, device):

    keep = torch.rand(B, K, NP, device=device) > mask_ratio
    return ~keep


class FactorizedBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mode_norm = nn.LayerNorm(d_model)
        self.mode_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.time_norm = nn.LayerNorm(d_model)
        self.time_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))

    def forward(self, x):

        B, K, NP, D = x.shape
        h = self.mode_norm(x).permute(0, 2, 1, 3).reshape(B * NP, K, D)
        h, _ = self.mode_attn(h, h, h, need_weights=False)
        x = x + h.reshape(B, NP, K, D).permute(0, 2, 1, 3)

        h = self.time_norm(x).reshape(B * K, NP, D)
        h, _ = self.time_attn(h, h, h, need_weights=False)
        x = x + h.reshape(B, K, NP, D)

        x = x + self.ffn(self.ffn_norm(x))
        return x


class TimeBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 1)

    def forward(self, x):
        return self.transformer(x)


class KAttentionPool(nn.Module):


    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.norm  = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        B, K, NP, D = x.shape
        x = self.norm(x)
        x = x.permute(0, 2, 1, 3).reshape(B * NP, K, D)
        q = self.query.expand(B * NP, 1, D)
        out, _ = self.attn(q, x, x, need_weights=False)
        return out.reshape(B, NP, D)


class EEGMEGPriorEigenval(nn.Module):










    def __init__(self, patch_size: int = PATCH_SIZE, d_model: int = 512,
                 n_heads: int = 8, n_factor_layers: int = 2, n_time_layers: int = 4,
                 dropout: float = 0.1, use_mode_signature: bool = True,
                 mode_pos_normalize: float = 1024.0):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.use_mode_signature = use_mode_signature
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
        ev_log = ev_log - ev_log.min()
        span = ev_log.max().clamp(min=1e-6)
        ev_log = ev_log * (self.mode_pos_normalize / span)
        return sinusoidal_pos(ev_log, self.d_model)

    def _embed(self, x: torch.Tensor, eigval: torch.Tensor | None,
               loss_mask: torch.Tensor | None):

        B, K, T = x.shape
        if T % self.patch_size != 0:
            raise ValueError(f"T={T} not divisible by patch_size={self.patch_size}")
        patches = patchify(x, self.patch_size)
        NP = patches.shape[2]
        tokens = self.patch_proj(patches)

        if loss_mask is not None:
            mt = self.mask_token.to(tokens.dtype).view(1, 1, 1, self.d_model)
            tokens = torch.where(loss_mask.unsqueeze(-1), mt.expand_as(tokens), tokens)

        if self.use_mode_signature and eigval is not None:
            mode_pos = self._mode_pos(eigval, tokens.dtype)
            tokens = tokens + mode_pos.view(1, K, 1, self.d_model)
        t_idx = torch.arange(NP, device=x.device, dtype=tokens.dtype)
        time_pos = sinusoidal_pos(t_idx, self.d_model)
        tokens = tokens + time_pos.view(1, 1, NP, self.d_model)
        return tokens, NP

    def encode_full_grid(self, x: torch.Tensor, eigval: torch.Tensor | None = None,
                         loss_mask: torch.Tensor | None = None) -> torch.Tensor:

        tokens, _ = self._embed(x, eigval, loss_mask)
        for blk in self.factor_blocks:
            tokens = blk(tokens)
        return self.norm(tokens)

    def encode_pooled(self, x: torch.Tensor, eigval: torch.Tensor | None = None,
                      loss_mask: torch.Tensor | None = None) -> torch.Tensor:

        tokens, _ = self._embed(x, eigval, loss_mask)
        for blk in self.factor_blocks:
            tokens = blk(tokens)
        x = self.k_pool(tokens)
        for blk in self.time_blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor, eigval: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None) -> torch.Tensor:

        return self.encode_pooled(x, eigval, loss_mask)


class EEGMEGMAEDecoderEigenval(nn.Module):




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




def compute_E_lambda_from_phi(phi: torch.Tensor | "np.ndarray",
                               lam_cortex: torch.Tensor | "np.ndarray",
                               ratio: float = 1e-2):










    import numpy as np
    Phi = np.asarray(phi, dtype=np.float64)
    lam = np.asarray(lam_cortex, dtype=np.float64)
    if Phi.shape[1] != lam.shape[0]:
        raise ValueError(f"Phi K={Phi.shape[1]} mismatch lam_cortex K={lam.shape[0]}")
    U, S, Vt = np.linalg.svd(Phi, full_matrices=False)
    K_eff = int((S / S[0] >= ratio).sum())
    V = Vt.T
    E_lambda = (V[:, :K_eff] ** 2 * lam[:, None]).sum(axis=0)
    return E_lambda.astype(np.float32), S[:K_eff].astype(np.float32), K_eff
