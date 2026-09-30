from __future__ import annotations
import math
import torch
import torch.nn as nn
def sinusoidal_pos(values: torch.Tensor, d_model: int) -> torch.Tensor:
    if d_model % 2 != 0:
        raise ValueError('d_model must be even')
    device = values.device
    half = d_model // 2
    div = torch.exp(-math.log(10000.0) * torch.arange(half, device=device, dtype=values.dtype) / half)
    angles = values.unsqueeze(-1) * div.unsqueeze(0)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)

class FactorizedBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int=8, dropout: float=0.1):
        super().__init__()
        self.mode_norm = nn.LayerNorm(d_model)
        self.mode_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.time_norm = nn.LayerNorm(d_model)
        self.time_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))

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

    def __init__(self, d_model: int, n_heads: int=8, dropout: float=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 1)

    def forward(self, x):
        return self.transformer(x)

class KAttentionPool(nn.Module):

    def __init__(self, d_model: int, n_heads: int=8, dropout: float=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        B, K, NP, D = x.shape
        x = self.norm(x)
        x = x.permute(0, 2, 1, 3).reshape(B * NP, K, D)
        q = self.query.expand(B * NP, 1, D)
        out, _ = self.attn(q, x, x, need_weights=False)
        return out.reshape(B, NP, D)
