from typing import Literal

import torch
import torch.nn as nn


class CrossAttnGateFusion(nn.Module):



















    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()


        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)


        self.gate_proj = nn.Linear(2 * d_model, d_model)
        self.post_norm = nn.LayerNorm(d_model)


    def forward(
        self,
        H_obs: torch.Tensor,
        H_prior: torch.Tensor,
        mode: Literal["full", "obs_only", "prior_only"] = "full",
    ) -> torch.Tensor:










        if mode == "obs_only":
            return H_obs

        if mode == "prior_only":
            return H_prior




        H_tilde, _ = self.cross_attn(
            query=H_obs,
            key=H_prior,
            value=H_prior,
        )
        H_tilde = self.attn_norm(H_tilde)


        g = torch.sigmoid(
            self.gate_proj(torch.cat([H_obs, H_tilde], dim=-1))
        )
        H_post = self.post_norm(H_obs + g * H_tilde)

        return H_post
