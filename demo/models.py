






from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from eigenmodes import eigenvalue_positional_encoding


class GaussianExpert(nn.Module):


    def __init__(self, d_in: int, d_latent: int, hidden: int, dropout: float):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden, d_latent)
        self.logvar = nn.Linear(hidden, d_latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        return self.mu(hidden), self.logvar(hidden).clamp(-5.0, 5.0)


class GaussianPoERegressor(nn.Module):


    def __init__(self, d_in: int, d_latent: int, hidden: int,
                 d_out: int, dropout: float):
        super().__init__()
        self.obs_expert = GaussianExpert(d_in, d_latent, hidden, dropout)
        self.indiv_expert = GaussianExpert(d_in, d_latent, hidden, dropout)
        self.obs_head = nn.Linear(d_latent, d_out)
        self.indiv_head = nn.Linear(d_latent, d_out)
        self.poe_head = nn.Linear(d_latent, d_out)

    def forward(self, obs: torch.Tensor, indiv: torch.Tensor,
                sample: bool = False) -> dict[str, torch.Tensor]:
        mu_obs, logvar_obs = self.obs_expert(obs)
        mu_indiv, logvar_indiv = self.indiv_expert(indiv)
        tau_obs = torch.exp(-logvar_obs)
        tau_indiv = torch.exp(-logvar_indiv)
        var_poe = (1.0 + tau_obs + tau_indiv).reciprocal()
        mu_poe = var_poe * (tau_obs * mu_obs + tau_indiv * mu_indiv)
        z_poe = mu_poe
        if sample and self.training:
            z_poe = mu_poe + torch.randn_like(mu_poe) * torch.sqrt(var_poe)
        return {
            "obs": self.obs_head(mu_obs),
            "indiv": self.indiv_head(mu_indiv),
            "poe": self.poe_head(z_poe),
            "mu_poe": mu_poe,
            "var_poe": var_poe,
        }


class EEGMEGPriorEncoder(nn.Module):


    def __init__(self, d_model=1024, d_patch=512, patch_size=64,
                 n_attn_heads=8, n_transformer_layers=2, dropout=0.1,
                 use_eigenvalue_pos=False):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_size
        self.use_eigenvalue_pos = use_eigenvalue_pos
        self.patch_embed = nn.Sequential(nn.Linear(patch_size, d_patch), nn.LayerNorm(d_patch))
        self.mode_query = nn.Parameter(torch.zeros(1, 1, d_patch))
        nn.init.trunc_normal_(self.mode_query, std=0.02)
        self.mode_attn = nn.MultiheadAttention(d_patch, n_attn_heads, dropout=dropout,
                                               batch_first=True)
        self.mode_norm = nn.LayerNorm(d_patch)
        layer = nn.TransformerEncoderLayer(d_patch, n_attn_heads, d_patch * 4,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.temporal_transformer = nn.TransformerEncoder(layer, n_transformer_layers)
        self.out_proj = (nn.Sequential(nn.Linear(d_patch, d_model), nn.LayerNorm(d_model))
                         if d_patch != d_model else nn.Identity())

    def forward(self, x: torch.Tensor, eigenvalues: torch.Tensor | None = None) -> torch.Tensor:
        batch, modes, _ = x.shape
        x = x.unfold(-1, self.patch_size, self.patch_stride)
        tokens = x.size(2)
        x = self.patch_embed(x)
        if self.use_eigenvalue_pos:
            values = (torch.arange(modes, device=x.device, dtype=x.dtype)
                      if eigenvalues is None else eigenvalues.to(device=x.device, dtype=x.dtype))
            position = torch.from_numpy(eigenvalue_positional_encoding(
                values.detach().cpu().numpy(), x.size(-1)
            )).to(device=x.device, dtype=x.dtype)
            x = x + position[None, :, None, :]
        width = x.size(-1)
        x = x.permute(0, 2, 1, 3).reshape(batch * tokens, modes, width)
        query = self.mode_query.expand(batch * tokens, -1, -1)
        x, _ = self.mode_attn(query, x, x)
        x = self.mode_norm(x.squeeze(1)).reshape(batch, tokens, width)
        return self.out_proj(self.temporal_transformer(x))


class SubjectLayers(nn.Module):


    def __init__(self, n_subjects: int, d: int):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_subjects, d, d))
        self.weights.data[:] = torch.eye(d)[None] / math.sqrt(d)
        self.bias = nn.Parameter(torch.zeros(n_subjects, d))

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        weight = self.weights.index_select(0, subject_ids)
        bias = self.bias.index_select(0, subject_ids).unsqueeze(1)
        return torch.einsum("btc,bcd->btd", x, weight) + bias


class PoEFusion(nn.Module):


    def __init__(self):
        super().__init__()
        self.raw_tau_a = nn.Parameter(torch.zeros(1))
        self.raw_tau_b = nn.Parameter(torch.zeros(1))

    def precisions(self):
        return F.softplus(self.raw_tau_a) + 1e-4, F.softplus(self.raw_tau_b) + 1e-4

    def forward(self, observation: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        tau_a, tau_b = self.precisions()
        return (tau_a * observation + tau_b * prior) / (tau_a + tau_b)


class QFormerBlock(nn.Module):
    def __init__(self, d_q: int, n_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm_sa = nn.LayerNorm(d_q)
        self.sa = nn.MultiheadAttention(d_q, n_heads, dropout=dropout, batch_first=True)
        self.norm_ca = nn.LayerNorm(d_q)
        self.norm_kv = nn.LayerNorm(d_q)
        self.ca = nn.MultiheadAttention(d_q, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_q)
        hidden = int(d_q * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(d_q, hidden), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(hidden, d_q), nn.Dropout(dropout))

    def forward(self, queries: torch.Tensor, key_values: torch.Tensor) -> torch.Tensor:
        x = self.norm_sa(queries)
        queries = queries + self.sa(x, x, x, need_weights=False)[0]
        x = self.norm_ca(queries)
        keys = self.norm_kv(key_values)
        queries = queries + self.ca(x, keys, keys, need_weights=False)[0]
        return queries + self.ff(self.norm_ff(queries))


class QFormer(nn.Module):


    def __init__(self, n_queries=20, d_q=768, d_in=512, d_out=1024,
                 n_layers=4, n_heads=8, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, n_queries, d_q))
        self.pos = nn.Parameter(torch.zeros(1, n_queries, d_q))
        nn.init.trunc_normal_(self.queries, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.input_proj = nn.Linear(d_in, d_q)
        self.input_norm = nn.LayerNorm(d_q)
        self.blocks = nn.ModuleList([QFormerBlock(d_q, n_heads, dropout=dropout)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_q)
        self.out_proj = nn.Linear(d_q, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        key_values = self.input_norm(self.input_proj(x))
        queries = (self.queries + self.pos).expand(batch, -1, -1).contiguous()
        for block in self.blocks:
            queries = block(queries, key_values)
        return self.out_proj(self.norm(queries))
