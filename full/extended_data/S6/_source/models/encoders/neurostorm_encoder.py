
























import os
import sys
import types
import importlib.util
import torch
import torch.nn as nn
from einops import rearrange


_HERE = os.path.dirname(os.path.realpath(__file__))
_NEUROSTORM_SRC = os.path.normpath(
    os.path.join(_HERE, "../../weights/NeuroStorm/NeuroSTORM-main")
)


def _import_NeuroSTORMMAE():




    _pkg = "_neurostorm_internal"
    _models_dir = os.path.join(_NEUROSTORM_SRC, "models")

    if _pkg not in sys.modules:
        pkg = types.ModuleType(_pkg)
        pkg.__path__ = [_models_dir]
        pkg.__package__ = _pkg
        sys.modules[_pkg] = pkg

    _pe = f"{_pkg}.patchembedding"
    if _pe not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            _pe, os.path.join(_models_dir, "patchembedding.py")
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = _pkg
        sys.modules[_pe] = mod
        spec.loader.exec_module(mod)

    _ns = f"{_pkg}.neurostorm"
    if _ns not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            _ns, os.path.join(_models_dir, "neurostorm.py")
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = _pkg
        sys.modules[_ns] = mod
        if _NEUROSTORM_SRC not in sys.path:
            sys.path.insert(0, _NEUROSTORM_SRC)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(_ns, None)
            raise

    ns_mod = sys.modules[_ns]
    if not hasattr(ns_mod, "NeuroSTORMMAE"):
        sys.modules.pop(_ns, None)
        return _import_NeuroSTORMMAE()

    return ns_mod.NeuroSTORMMAE


_CKPT_FILE = os.path.normpath(
    os.path.join(
        _HERE,
        "../../weights/NeuroStorm/NeuroSTORM/neurostorm/"
        "pt_neurostorm_mae_ratio0.5.ckpt",
    )
)
_CKPT_FILE_08 = os.path.normpath(
    os.path.join(
        _HERE,
        "../../weights/NeuroStorm/NeuroSTORM/neurostorm/"
        "pt_neurostorm_mae_ratio0.8.ckpt",
    )
)


_SPATIAL_FLAT: int = 288 * 2 * 2 * 2




class NeuroStormAdapter(nn.Module):












    def __init__(
        self,
        spatial_flat: int = _SPATIAL_FLAT,
        n_groups: int = 8,
        d_model: int = 1024,
        d_inner: int = 512,
        n_heads: int = 8,
        n_transformer_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert spatial_flat % n_groups == 0
        self.n_groups = n_groups
        self.n_channels = spatial_flat // n_groups

        self.group_proj = nn.Sequential(
            nn.Linear(self.n_channels, d_inner),
            nn.LayerNorm(d_inner),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_inner, nhead=n_heads,
            dim_feedforward=d_inner * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        self.out_proj = nn.Sequential(
            nn.Linear(d_inner, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, D, H, W, T = x.shape
        x = rearrange(x, "B C D H W T -> B T (D H W) C")
        x = self.group_proj(x)
        x = x.reshape(B, T * self.n_groups, -1)
        x = self.transformer(x)
        return self.out_proj(x)




def _build_from_ckpt(ckpt_path: str):
    NeuroSTORMMAE = _import_NeuroSTORMMAE()
    state = torch.load(ckpt_path, map_location="cpu")
    hp = state["hyper_parameters"]

    model = NeuroSTORMMAE(
        img_size=tuple(hp["img_size"]),
        in_chans=hp["in_chans"],
        embed_dim=hp["embed_dim"],
        window_size=hp["window_size"],
        first_window_size=hp["first_window_size"],
        patch_size=hp["patch_size"],
        depths=hp["depths"],
        num_heads=hp["num_heads"],
        c_multiplier=hp.get("c_multiplier", 2),
        last_layer_full_MSA=hp.get("last_layer_full_MSA", False),
        drop_rate=hp.get("attn_drop_rate", 0.0),
        drop_path_rate=hp.get("attn_drop_rate", 0.0),
        attn_drop_rate=hp.get("attn_drop_rate", 0.0),
        mask_ratio=hp.get("mask_ratio", 0.5),
        spatial_mask=hp.get("spatial_mask", "random"),
        time_mask=hp.get("time_mask", "random"),
    )

    raw_sd = state["state_dict"]
    backbone_sd = {
        k.removeprefix("model."): v
        for k, v in raw_sd.items()
        if k.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    if missing:
        print(f"[NeuroStormEncoder] missing keys: {len(missing)}")

    t_window_size = int(hp["window_size"][3])
    return model, t_window_size




class NeuroStormEncoder(nn.Module):








    def __init__(
        self,
        d_model: int = 1024,
        n_groups: int = 8,
        freeze_backbone: bool = True,
        ckpt_path: str = None,
    ):
        super().__init__()


        ckpt = ckpt_path if ckpt_path is not None else _CKPT_FILE
        self.backbone, self._t_window = _build_from_ckpt(ckpt)


        self.set_backbone_frozen(freeze_backbone)


        self.adapter = NeuroStormAdapter(
            spatial_flat=_SPATIAL_FLAT,
            n_groups=n_groups,
            d_model=d_model,
        )

    def set_backbone_frozen(self, frozen: bool) -> None:

        for p in self.backbone.parameters():
            p.requires_grad = not frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:











        import torch.nn.functional as F

        T = x.shape[-1]
        ws = self._t_window
        pad = (ws - T % ws) % ws
        if pad > 0:
            x = F.pad(x, (0, pad))

        no_grad = not any(p.requires_grad for p in self.backbone.parameters())
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            feats, _mask = self.backbone.forward_encoder(x)

        feats = feats[..., :T]
        return self.adapter(feats)
