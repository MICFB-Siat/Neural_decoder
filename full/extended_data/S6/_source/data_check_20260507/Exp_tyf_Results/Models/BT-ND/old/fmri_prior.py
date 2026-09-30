import torch
import torch.nn as nn


class FMRIPriorEncoder(nn.Module):




















    def __init__(
        self,
        n_modes: int = 2000,
        n_groups: int = 8,
        d_model: int = 1024,
        d_inner: int = 512,
        n_attn_heads: int = 8,
        n_transformer_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert n_modes % n_groups == 0, (
            f"n_modes ({n_modes}) must be divisible by n_groups ({n_groups})"
        )
        self.n_groups = n_groups
        self.group_size = n_modes // n_groups



        self.group_encoder = nn.Sequential(
            nn.Linear(self.group_size, d_inner),
            nn.GELU(),
            nn.LayerNorm(d_inner),
            nn.Linear(d_inner, d_inner),
            nn.LayerNorm(d_inner),
        )


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_inner,
            nhead=n_attn_heads,
            dim_feedforward=d_inner * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )


        self.out_proj = (
            nn.Sequential(nn.Linear(d_inner, d_model), nn.LayerNorm(d_model))
            if d_inner != d_model
            else nn.Identity()
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:






        B, n_modes, T = x.shape


        x = x.permute(0, 2, 1)


        x = x.reshape(B, T, self.n_groups, self.group_size)



        x = self.group_encoder(x)


        x = x.reshape(B, T * self.n_groups, -1)


        x = self.transformer(x)


        x = self.out_proj(x)

        return x
