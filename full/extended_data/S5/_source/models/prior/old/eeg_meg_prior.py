import torch
import torch.nn as nn


class EEGMEGPriorEncoder(nn.Module):


















    def __init__(
        self,
        d_model: int = 1024,
        d_patch: int = 512,
        patch_size: int = 64,
        n_attn_heads: int = 8,
        n_transformer_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_size


        self.patch_embed = nn.Sequential(
            nn.Linear(patch_size, d_patch),
            nn.LayerNorm(d_patch),
        )



        self.mode_query = nn.Parameter(torch.zeros(1, 1, d_patch))
        nn.init.trunc_normal_(self.mode_query, std=0.02)

        self.mode_attn = nn.MultiheadAttention(
            embed_dim=d_patch,
            num_heads=n_attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mode_norm = nn.LayerNorm(d_patch)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_patch,
            nhead=n_attn_heads,
            dim_feedforward=d_patch * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )


        self.out_proj = (
            nn.Sequential(nn.Linear(d_patch, d_model), nn.LayerNorm(d_model))
            if d_patch != d_model
            else nn.Identity()
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:






        B, M, _ = x.shape



        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        n_tokens = x.size(2)



        x = self.patch_embed(x)
        d_patch = x.size(-1)




        x = x.permute(0, 2, 1, 3).reshape(B * n_tokens, M, d_patch)

        query = self.mode_query.expand(B * n_tokens, -1, -1)
        attn_out, _ = self.mode_attn(query, x, x)
        attn_out = self.mode_norm(attn_out.squeeze(1))


        x = attn_out.reshape(B, n_tokens, d_patch)


        x = self.temporal_transformer(x)


        x = self.out_proj(x)

        return x
