
























import torch
import torch.nn as nn


N_CORTEX_DEFAULT = 59408
N_GROUPS_DEFAULT = 8


class CortexCIFTIEncoder(nn.Module):









    def __init__(
        self,
        n_cortex: int = N_CORTEX_DEFAULT,
        n_groups: int = N_GROUPS_DEFAULT,
        T: int = 5,
        max_T: int = 50,
        d_model: int = 1024,
        d_inner: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert n_cortex % n_groups == 0, (
            f"n_cortex ({n_cortex}) must be divisible by n_groups ({n_groups})"
        )
        self.n_cortex   = n_cortex
        self.n_groups   = n_groups
        self.T          = T
        self.group_size = n_cortex // n_groups
        self.n_patches  = n_groups * T
        self.d_inner    = d_inner
        self.d_model    = d_model

        ffn_dim = ffn_dim or d_inner * 2

        self.patch_embed = nn.Linear(self.group_size, d_inner, bias=False)

        self.pos_embed   = nn.Parameter(torch.zeros(1, n_groups * max_T, d_inner))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model         = d_inner,
            nhead           = n_heads,
            dim_feedforward = ffn_dim,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_inner)

        self.out_proj = (
            nn.Sequential(nn.Linear(d_inner, d_model), nn.LayerNorm(d_model))
            if d_inner != d_model
            else nn.Identity()
        )


    def patchify(self, x: torch.Tensor) -> torch.Tensor:





        B, _, T_actual = x.shape
        x = x[:, :self.n_cortex, :]
        x = x.reshape(B, self.n_groups, self.group_size, T_actual)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B, T_actual * self.n_groups, self.group_size)
        return x


    def encode_tokens(self, patches: torch.Tensor) -> torch.Tensor:




        n = patches.shape[1]
        tokens = self.patch_embed(patches) + self.pos_embed[:, :n, :]
        tokens = self.transformer(tokens)
        return self.norm(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(x)
        tokens  = self.encode_tokens(patches)
        return self.out_proj(tokens)
