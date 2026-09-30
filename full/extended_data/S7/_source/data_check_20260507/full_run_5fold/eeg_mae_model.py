














import torch
import torch.nn as nn

PATCH_SIZE = 64
N_SAMPLES  = 2560
N_PATCHES  = N_SAMPLES // PATCH_SIZE


def patchify(x: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:

    B, M, T = x.shape
    n = T // patch_size
    x = x.reshape(B, M, n, patch_size)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(B, n, M * patch_size)


class EEGMAEEncoder(nn.Module):








    def __init__(
        self,
        n_modes:    int   = 50,
        patch_size: int   = PATCH_SIZE,
        n_samples:  int   = N_SAMPLES,
        d_model:    int   = 512,
        n_heads:    int   = 8,
        n_layers:   int   = 6,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_modes    = n_modes
        self.patch_size = patch_size
        self.n_patches  = n_samples // patch_size
        self.d_model    = d_model

        in_dim = n_modes * patch_size
        self.patch_proj = nn.Linear(in_dim, d_model, bias=False)
        self.pos_embed  = nn.Embedding(self.n_patches, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward = d_model * 4,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

        nn.init.trunc_normal_(self.patch_proj.weight, std=0.02)

    def forward(
        self,
        x:           torch.Tensor,
        visible_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:





        tokens = self.patch_proj(patchify(x, self.patch_size))
        pos    = self.pos_embed(
            torch.arange(self.n_patches, device=x.device)
        )
        tokens = tokens + pos

        if visible_idx is not None:
            tokens = tokens[:, visible_idx, :]

        return self.norm(self.transformer(tokens))


class EEGMAEDecoder(nn.Module):







    def __init__(
        self,
        d_model:    int = 512,
        d_dec:      int = 256,
        n_heads:    int = 4,
        n_layers:   int = 4,
        n_patches:  int = N_PATCHES,
        target_dim: int = 50 * PATCH_SIZE,
    ):
        super().__init__()
        self.n_patches  = n_patches
        self.target_dim = target_dim

        self.enc_proj   = nn.Linear(d_model, d_dec, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_dec))
        self.pos_embed  = nn.Embedding(n_patches, d_dec)

        dec_layer = nn.TransformerEncoderLayer(
            d_model        = d_dec,
            nhead          = n_heads,
            dim_feedforward = d_dec * 4,
            dropout        = 0.1,
            batch_first    = True,
            norm_first     = True,
        )
        self.transformer = nn.TransformerEncoder(dec_layer, n_layers)
        self.norm = nn.LayerNorm(d_dec)
        self.head = nn.Linear(d_dec, target_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        encoded:     torch.Tensor,
        visible_idx: torch.Tensor,
        mask_idx:    torch.Tensor,
    ) -> torch.Tensor:

        B   = encoded.shape[0]
        dec = self.enc_proj(encoded)


        full = self.mask_token.expand(B, self.n_patches, -1).clone()
        full[:, visible_idx, :] = dec

        pos  = self.pos_embed(
            torch.arange(self.n_patches, device=encoded.device)
        )
        full = full + pos

        out  = self.norm(self.transformer(full))
        return self.head(out[:, mask_idx, :])
