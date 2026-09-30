




















import os
import sys
import json
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.realpath(__file__))
_BRAINOMNI_SRC = os.path.normpath(
    os.path.join(_HERE, "../../weights/BrainOmni/BrainOmni-main")
)
if _BRAINOMNI_SRC not in sys.path:
    sys.path.insert(0, _BRAINOMNI_SRC)

from brainomni.model import BrainOmni

_CKPT_BASE = os.path.normpath(
    os.path.join(_HERE, "../../weights/BrainOmni/BrainOmni/base")
)
_CKPT_TINY = os.path.normpath(
    os.path.join(_HERE, "../../weights/BrainOmni/BrainOmni/tiny")
)




class BrainOmniAdapter(nn.Module):












    def __init__(
        self,
        n_neuro: int = 16,
        d_enc: int = 512,
        d_model: int = 1024,
        n_tokens_out: int = 40,
        n_heads: int = 8,
    ):
        super().__init__()


        self.channel_query = nn.Parameter(torch.empty(1, 1, d_enc))
        nn.init.trunc_normal_(self.channel_query, std=0.02)
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=d_enc, num_heads=n_heads, batch_first=True
        )
        self.proj = nn.Linear(d_enc, d_model)
        self.norm1 = nn.LayerNorm(d_model)


        self.compress_queries = nn.Parameter(torch.empty(1, n_tokens_out, d_model))
        nn.init.trunc_normal_(self.compress_queries, std=0.02)
        self.compress_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, W, D = x.shape


        x = x.permute(0, 2, 1, 3).reshape(B * W, C, D)
        q = self.channel_query.expand(B * W, -1, -1)
        pooled, _ = self.channel_attn(q, x, x)
        pooled = self.norm1(self.proj(pooled.squeeze(1).reshape(B, W, D)))



        q = self.compress_queries.expand(B, -1, -1)
        out, _ = self.compress_attn(q, pooled, pooled)
        return self.norm2(out)




class BrainOmniEncoder(nn.Module):






    def __init__(
        self,
        d_model: int = 1024,
        n_tokens_out: int = 40,
        freeze_backbone: bool = True,
        variant: str = "tiny",
    ):
        super().__init__()

        ckpt_dir = _CKPT_TINY if variant == "tiny" else _CKPT_BASE
        cfg_path = os.path.join(ckpt_dir, "model_cfg.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        self.backbone = BrainOmni(**cfg)
        d_enc: int = cfg["lm_dim"]

        ckpt = torch.load(
            os.path.join(ckpt_dir, "BrainOmni.pt"), map_location="cpu"
        )
        missing, _ = self.backbone.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"[BrainOmniEncoder] missing keys: {len(missing)}")

        self.set_backbone_frozen(freeze_backbone)

        self.adapter = BrainOmniAdapter(
            n_neuro=cfg.get("n_neuro", 16),
            d_enc=d_enc,
            d_model=d_model,
            n_tokens_out=n_tokens_out,
        )

    def set_backbone_frozen(self, frozen: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not frozen

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
    ) -> torch.Tensor:











        no_grad = not any(p.requires_grad for p in self.backbone.parameters())
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            feats = self.backbone.encode(x, pos, sensor_type)
        return self.adapter(feats)
