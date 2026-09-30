




















from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))

import argparse
import copy
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parents[3]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prior_eigenval_schemeA import (
    SchemeAMAEDecoder,
    SchemeAPriorEigenval,
    brainomni_2d_mask,
    patchify,
)


DEFAULT_RESULT_ROOT = Path(
    str(_LOCAL_SOURCE / 'data_check_20260507/Exp_tyf_Results/Exp_fnirs/result')
)
DEFAULT_TRANSFORMER_DIR = Path(
    str(_LOCAL_SOURCE / 'data_check_20260507/Exp_tyf_Results/Models/fNIRS-Transformer')
)
DEFAULT_COCKTAIL_H5 = Path(
    '/media/wsqlab/nas2/Dataset/wholehead-cocktail-party-fnirs/preprocess/stage2_fs32k_poe'
)
DEFAULT_DUBOIS_H5 = Path(
    '/media/wsqlab/nas2/Dataset/Reliability-Dubois2024/preprocess/stage2_fs32k_poe'
)
DEFAULT_SPLIT_SUITE = DEFAULT_RESULT_ROOT / "fnirs_comparisons_5seed_20260714_221535" / "runs"
DEFAULT_COCKTAIL_SPLITS = (
    DEFAULT_SPLIT_SUITE
    / "cocktail_fs32k_poe_schemeA_frozen_50seed_fnirs_within_frozen_cache_seed000"
)
DEFAULT_DUBOIS_SPLITS = (
    DEFAULT_SPLIT_SUITE
    / "dubois_fs32k_poe_schemeA_frozen_50seed_fnirs_within_frozen_cache_seed000"
)
DEFAULT_PRIOR_INIT = (
    DEFAULT_RESULT_ROOT
    / "cocktail_fs32k_poe_schemeA_p32q4_fnirs_within_frozen_cache_20260709_203848"
    / "pretrain"
    / "mae_schemeA_p32_q4_cocktail_fs32k_poe_schemeA_p32q4_fnirs.pt"
)


@dataclass(frozen=True)
class DomainSpec:
    name: str
    h5_dir: str
    split_dir: str
    n_classes: int
    channels: int
    samples: int


@dataclass(frozen=True)
class SubjectRef:
    domain: str
    subject: str
    path: str
    n_trials: int


_LOG_HANDLES: dict[str, object] = {}


def log(message: str, log_path: Path | None = None) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    print(line, flush=True)
    if log_path is not None:
        key = str(log_path)
        handle = _LOG_HANDLES.get(key)
        if handle is None:
            handle = log_path.open("a", encoding="utf-8", buffering=1)
            _LOG_HANDLES[key] = handle
        handle.write(line + "\n")
        handle.flush()


def atomic_json_dump(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_transformer_module(transformer_dir: Path):
    path = transformer_dir / "model.py"
    spec = importlib.util.spec_from_file_location("fnirs_transformer_joint_repo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import fNIRS-Transformer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def amp_context(device: torch.device):
    enabled = device.type == "cuda"
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        enabled=enabled,
    )


def normalize_raw(x: torch.Tensor) -> torch.Tensor:

    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def raw_to_chromophores(raw: np.ndarray) -> torch.Tensor:
    if raw.ndim != 3 or raw.shape[1] % 2:
        raise ValueError(f"Expected raw [N,2C,T], got {raw.shape}")
    channels = raw.shape[1] // 2
    out = np.stack([raw[:, :channels], raw[:, channels:]], axis=1)
    return torch.from_numpy(out.astype(np.float32, copy=False))


def discover_specs(args) -> dict[str, DomainSpec]:
    requested = {
        "cocktail": (Path(args.cocktail_h5), Path(args.cocktail_splits), 2),
        "dubois": (Path(args.dubois_h5), Path(args.dubois_splits), 3),
    }
    specs: dict[str, DomainSpec] = {}
    for name, (h5_dir, split_dir, n_classes) in requested.items():
        files = sorted(h5_dir.glob("sub-*.h5"))
        if not files:
            raise FileNotFoundError(f"No H5 files found in {h5_dir}")
        with h5py.File(files[0], "r") as handle:
            shape = tuple(handle["fnirs_raw"].shape)
        if shape[1] % 2:
            raise ValueError(f"{files[0]} has odd raw channel count: {shape}")
        specs[name] = DomainSpec(
            name=name,
            h5_dir=str(h5_dir),
            split_dir=str(split_dir),
            n_classes=n_classes,
            channels=shape[1] // 2,
            samples=shape[2],
        )
    return specs


def discover_subjects(
    specs: dict[str, DomainSpec], max_per_domain: int = 0
) -> dict[str, list[SubjectRef]]:
    result: dict[str, list[SubjectRef]] = {}
    for domain, spec in specs.items():
        files = sorted(Path(spec.h5_dir).glob("sub-*.h5"))
        if max_per_domain > 0:
            files = files[:max_per_domain]
        refs = []
        for path in files:
            with h5py.File(path, "r") as handle:
                n_trials = int(handle["labels"].shape[0])
            refs.append(SubjectRef(domain, path.stem, str(path), n_trials))
        result[domain] = refs
    return result


def choose_indices(n: int, count: int, rng: np.random.Generator) -> np.ndarray:
    count = min(max(1, count), n)
    return np.sort(rng.choice(n, size=count, replace=False).astype(np.int64))


def read_h5_rows(path: str, key: str, indices: np.ndarray) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[key][indices], dtype=np.float32)


def read_eigval(path: str) -> torch.Tensor:
    with h5py.File(path, "r") as handle:
        return torch.from_numpy(np.asarray(handle["fnirs_mode_eigval"], dtype=np.float32))


class DomainInputAdapter(nn.Module):


    def __init__(self, channels: int, samples: int, dim: int):
        super().__init__()
        if channels < 5 or samples < 30:
            raise ValueError(f"fNIRS-T needs C>=5,T>=30; got C={channels},T={samples}")
        temporal_tokens = math.floor((samples - 30) / 4) + 1
        embed_width = temporal_tokens * 8
        self.channels = channels
        self.samples = samples
        self.patch_conv = nn.Conv2d(2, 8, kernel_size=(5, 30), stride=(1, 4))
        self.patch_linear = nn.Linear(embed_width, dim)
        self.patch_norm = nn.LayerNorm(dim)
        self.channel_conv = nn.Conv2d(2, 8, kernel_size=(1, 30), stride=(1, 4))
        self.channel_linear = nn.Linear(embed_width, dim)
        self.channel_norm = nn.LayerNorm(dim)
        self.pos_patch = nn.Parameter(torch.randn(1, channels - 4 + 1, dim) * 0.02)
        self.pos_channel = nn.Parameter(torch.randn(1, channels + 1, dim) * 0.02)

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_conv(x).permute(0, 2, 1, 3).flatten(2)
        return self.patch_norm(self.patch_linear(x))

    def channel_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_conv(x).permute(0, 2, 1, 3).flatten(2)
        return self.channel_norm(self.channel_linear(x))


class JointFNIRSTransformer(nn.Module):


    def __init__(
        self,
        specs: dict[str, DomainSpec],
        transformer_module,
        dim: int = 64,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 64,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.adapters = nn.ModuleDict(
            {
                name: DomainInputAdapter(spec.channels, spec.samples, dim)
                for name, spec in specs.items()
            }
        )
        self.cls_token_patch = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cls_token_channel = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.transformer_patch = transformer_module.Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )
        self.transformer_channel = transformer_module.Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )
        self.out_norm = nn.LayerNorm(dim * 2)

    @property
    def output_dim(self) -> int:
        return self.dim * 2

    def encode_tokens(
        self, x: torch.Tensor, domain: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adapter = self.adapters[domain]
        patch = adapter.patch_tokens(x)
        channel = adapter.channel_tokens(x)
        batch = x.shape[0]
        patch_cls = self.cls_token_patch.expand(batch, -1, -1)
        channel_cls = self.cls_token_channel.expand(batch, -1, -1)
        patch = torch.cat([patch_cls, patch], dim=1) + adapter.pos_patch
        channel = torch.cat([channel_cls, channel], dim=1) + adapter.pos_channel
        patch = self.transformer_patch(patch)
        channel = self.transformer_channel(channel)
        return patch, channel

    def encode(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        patch, channel = self.encode_tokens(x, domain)
        return self.out_norm(torch.cat([patch[:, 0], channel[:, 0]], dim=-1))

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        return self.encode(x, domain)


class ObsMaskedAutoencoder(nn.Module):
    def __init__(self, backbone: JointFNIRSTransformer, specs: dict[str, DomainSpec]):
        super().__init__()
        self.backbone = backbone
        self.channel_decoders = nn.ModuleDict(
            {name: nn.Linear(backbone.dim, 2 * spec.samples) for name, spec in specs.items()}
        )
        self.patch_decoders = nn.ModuleDict(
            {name: nn.Linear(backbone.dim, 2 * spec.samples) for name, spec in specs.items()}
        )

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.unsqueeze(2).to(pred.dtype)
        denominator = weight.sum().clamp_min(1.0) * target.shape[2]
        return (((pred.float() - target.float()) ** 2) * weight).sum() / denominator

    def forward(
        self,
        x: torch.Tensor,
        domain: str,
        channel_mask_ratio: float,
        time_mask_ratio: float,
    ) -> torch.Tensor:
        batch, _, channels, samples = x.shape
        channel_mask = torch.rand(batch, channels, device=x.device) < channel_mask_ratio
        time_mask = torch.rand(batch, samples, device=x.device) < time_mask_ratio
        channel_mask[:, 0] = True
        full_mask = channel_mask.unsqueeze(-1) | time_mask.unsqueeze(1)
        masked = x.masked_fill(full_mask.unsqueeze(1), 0.0)
        patch_tokens, channel_tokens = self.backbone.encode_tokens(masked, domain)

        target_channel = x.permute(0, 2, 1, 3)
        pred_channel = self.channel_decoders[domain](channel_tokens[:, 1:])
        pred_channel = pred_channel.view(batch, channels, 2, samples)
        loss_channel = self._masked_mse(pred_channel, target_channel, full_mask)

        patch_count = channels - 4
        target_patch = x[:, :, 2 : 2 + patch_count].permute(0, 2, 1, 3)
        patch_mask = full_mask[:, 2 : 2 + patch_count]
        pred_patch = self.patch_decoders[domain](patch_tokens[:, 1:])
        pred_patch = pred_patch.view(batch, patch_count, 2, samples)
        loss_patch = self._masked_mse(pred_patch, target_patch, patch_mask)
        return 0.5 * (loss_channel + loss_patch)


def build_obs_backbone(args, specs, transformer_module) -> JointFNIRSTransformer:
    return JointFNIRSTransformer(
        specs,
        transformer_module,
        dim=args.transformer_dim,
        depth=args.transformer_depth,
        heads=args.transformer_heads,
        mlp_dim=args.transformer_mlp_dim,
        dim_head=args.transformer_dim_head,
    )


def validation_indices(ref: SubjectRef, count: int) -> np.ndarray:
    stable_seed = 7919 + sum((index + 1) * ord(char) for index, char in enumerate(
        f"{ref.domain}:{ref.subject}"
    ))
    rng = np.random.default_rng(stable_seed % (2**32))
    return choose_indices(ref.n_trials, count, rng)


def balanced_epoch_refs(
    subjects: dict[str, list[SubjectRef]], rng: np.random.Generator
) -> list[SubjectRef]:

    target = max(len(refs) for refs in subjects.values())
    output: list[SubjectRef] = []
    for refs in subjects.values():
        order = list(rng.permutation(refs))
        output.extend(order)
        while len(order) < target:
            take = min(target - len(order), len(refs))
            order.extend(list(rng.choice(refs, size=take, replace=False)))
        output.extend(order[len(refs) : target])
    rng.shuffle(output)
    return output


def pretrain_observation(
    args,
    specs: dict[str, DomainSpec],
    subjects: dict[str, list[SubjectRef]],
    transformer_module,
    checkpoint: Path,
    device: torch.device,
    log_path: Path,
) -> None:
    if checkpoint.exists() and not args.force_pretrain:
        log(f"Observation checkpoint exists; skip pretraining: {checkpoint}", log_path)
        return
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed + 101)
    backbone = build_obs_backbone(args, specs, transformer_module).to(device)
    model = ObsMaskedAutoencoder(backbone, specs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.obs_pretrain_lr, weight_decay=0.05, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.obs_pretrain_epochs)
    )
    best = float("inf")
    history = []
    log(
        f"Observation joint pretraining: epochs={args.obs_pretrain_epochs}, "
        f"batch={args.obs_pretrain_batch_size}, subjects="
        f"{sum(len(x) for x in subjects.values())}",
        log_path,
    )

    for epoch in range(1, args.obs_pretrain_epochs + 1):
        model.train()
        losses = []
        for ref in balanced_epoch_refs(subjects, rng):
            indices = choose_indices(ref.n_trials, args.obs_pretrain_batch_size, rng)
            raw = read_h5_rows(ref.path, "fnirs_raw", indices)
            x = normalize_raw(raw_to_chromophores(raw).to(device))
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                loss = model(
                    x,
                    ref.domain,
                    args.obs_channel_mask_ratio,
                    args.obs_time_mask_ratio,
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()

        val_loss = float("nan")
        if epoch == 1 or epoch == args.obs_pretrain_epochs or epoch % args.pretrain_val_every == 0:
            model.eval()
            values = []
            with torch.no_grad():
                for refs in subjects.values():
                    for ref in refs:
                        indices = validation_indices(ref, args.pretrain_val_batch_size)
                        raw = read_h5_rows(ref.path, "fnirs_raw", indices)
                        x = normalize_raw(raw_to_chromophores(raw).to(device))
                        with amp_context(device):
                            value = model(
                                x,
                                ref.domain,
                                args.obs_channel_mask_ratio,
                                args.obs_time_mask_ratio,
                            )
                        values.append(float(value.item()))
            val_loss = float(np.mean(values))
            if val_loss < best:
                best = val_loss
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "backbone": {k: v.cpu() for k, v in backbone.state_dict().items()},
                        "specs": {k: asdict(v) for k, v in specs.items()},
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "architecture": {
                            "dim": args.transformer_dim,
                            "depth": args.transformer_depth,
                            "heads": args.transformer_heads,
                            "mlp_dim": args.transformer_mlp_dim,
                            "dim_head": args.transformer_dim_head,
                        },
                        "objective": "masked_channel_time_reconstruction",
                        "labels_used": False,
                    },
                    checkpoint,
                )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "best_val": best,
        }
        history.append(row)
        atomic_json_dump(history, checkpoint.with_suffix(".history.json"))
        log(
            f"OBS-PRE {epoch:03d}/{args.obs_pretrain_epochs} "
            f"train={row['train_loss']:.6f} val={val_loss:.6f} best={best:.6f}",
            log_path,
        )
    del model, backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()


def build_prior(args) -> SchemeAPriorEigenval:
    return SchemeAPriorEigenval(
        patch_size=args.prior_patch_size,
        d_model=args.prior_d_model,
        n_heads=8,
        n_factor_layers=args.prior_factor_layers,
        n_time_layers=args.prior_time_layers,
        n_pool_queries=args.prior_pool_queries,
    )


def pretrain_prior(
    args,
    subjects: dict[str, list[SubjectRef]],
    checkpoint: Path,
    device: torch.device,
    log_path: Path,
) -> None:
    if checkpoint.exists() and not args.force_pretrain:
        log(f"Prior checkpoint exists; skip pretraining: {checkpoint}", log_path)
        return
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed + 211)
    encoder = build_prior(args).to(device)
    if args.prior_init and Path(args.prior_init).exists():
        initial = torch.load(args.prior_init, map_location="cpu", weights_only=False)
        state = initial.get("encoder", initial)
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        log(
            f"Initialized prior from {args.prior_init}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}",
            log_path,
        )
    decoder = SchemeAMAEDecoder(
        d_model=args.prior_d_model,
        d_dec=args.prior_d_model // 2,
        n_heads=4,
        n_layers=args.prior_decoder_layers,
        patch_size=args.prior_patch_size,
    ).to(device)
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.prior_pretrain_lr, weight_decay=0.05, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.prior_pretrain_epochs)
    )
    eigvals = {
        (ref.domain, ref.subject): read_eigval(ref.path).to(device)
        for refs in subjects.values()
        for ref in refs
    }
    best = float("inf")
    history = []
    log(
        f"Prior joint pretraining: epochs={args.prior_pretrain_epochs}, "
        f"Cocktail batch={args.prior_batch_cocktail}, "
        f"Dubois batch={args.prior_batch_dubois}",
        log_path,
    )

    for epoch in range(1, args.prior_pretrain_epochs + 1):
        encoder.train()
        decoder.train()
        losses = []
        for ref in balanced_epoch_refs(subjects, rng):
            batch_size = (
                args.prior_batch_cocktail if ref.domain == "cocktail" else args.prior_batch_dubois
            )
            indices = choose_indices(ref.n_trials, batch_size, rng)
            modes = torch.from_numpy(read_h5_rows(ref.path, "fnirs_modes", indices)).to(device)
            eigval = eigvals[(ref.domain, ref.subject)]
            npatches = modes.shape[-1] // args.prior_patch_size
            mask = brainomni_2d_mask(
                modes.shape[0], modes.shape[1], npatches, args.prior_mask_ratio, device
            )
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                target = patchify(modes, args.prior_patch_size)
                encoded = encoder.encode_full_grid(modes, eigval=eigval, loss_mask=mask)
                prediction = decoder(encoded)
                weight = mask.float().unsqueeze(-1)
                loss = (
                    ((prediction.float() - target.float()) ** 2 * weight).sum()
                    / (weight.sum() * args.prior_patch_size + 1e-8)
                )
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()

        val_loss = float("nan")
        if epoch == 1 or epoch == args.prior_pretrain_epochs or epoch % args.pretrain_val_every == 0:
            encoder.eval()
            decoder.eval()
            values = []
            with torch.no_grad():
                for refs in subjects.values():
                    for ref in refs:
                        batch_size = min(args.pretrain_val_batch_size, ref.n_trials)
                        indices = validation_indices(ref, batch_size)
                        modes = torch.from_numpy(
                            read_h5_rows(ref.path, "fnirs_modes", indices)
                        ).to(device)
                        eigval = eigvals[(ref.domain, ref.subject)]
                        npatches = modes.shape[-1] // args.prior_patch_size
                        mask = brainomni_2d_mask(
                            modes.shape[0],
                            modes.shape[1],
                            npatches,
                            args.prior_mask_ratio,
                            device,
                        )
                        with amp_context(device):
                            target = patchify(modes, args.prior_patch_size)
                            encoded = encoder.encode_full_grid(
                                modes, eigval=eigval, loss_mask=mask
                            )
                            prediction = decoder(encoded)
                            weight = mask.float().unsqueeze(-1)
                            value = (
                                ((prediction.float() - target.float()) ** 2 * weight).sum()
                                / (weight.sum() * args.prior_patch_size + 1e-8)
                            )
                        values.append(float(value.item()))
            val_loss = float(np.mean(values))
            if val_loss < best:
                best = val_loss
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "encoder": {k: v.cpu() for k, v in encoder.state_dict().items()},
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "patch_size": args.prior_patch_size,
                        "d_model": args.prior_d_model,
                        "n_factor_layers": args.prior_factor_layers,
                        "n_time_layers": args.prior_time_layers,
                        "n_pool_queries": args.prior_pool_queries,
                        "scheme": "A",
                        "pretrain_domains": sorted(subjects),
                        "labels_used": False,
                    },
                    checkpoint,
                )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "best_val": best,
        }
        history.append(row)
        atomic_json_dump(history, checkpoint.with_suffix(".history.json"))
        log(
            f"PRIOR-PRE {epoch:03d}/{args.prior_pretrain_epochs} "
            f"train={row['train_loss']:.6f} val={val_loss:.6f} best={best:.6f}",
            log_path,
        )
    del encoder, decoder
    if device.type == "cuda":
        torch.cuda.empty_cache()


class FrozenFeatureHead(nn.Module):


    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PriorAttentiveHead(nn.Module):


    def __init__(self, input_dim: int, n_classes: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weight = torch.softmax(self.attn(tokens), dim=1)
        pooled = (tokens * weight).sum(dim=1)
        return self.net(pooled)


def train_head(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, dict]:
    seed_everything(seed)
    model = model.to(device)
    x = features.to(device)
    y = labels.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.head_lr, weight_decay=args.head_weight_decay
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.head_label_smoothing)
    train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_t = torch.as_tensor(val_idx, dtype=torch.long, device=device)
    best_acc = -1.0
    best_loss = float("inf")
    best_state = None
    no_improve = 0
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    for epoch in range(1, args.head_epochs + 1):
        model.train()
        order = train_t[torch.randperm(len(train_t), generator=rng, device=device)]
        for start in range(0, len(order), args.head_batch_size):
            idx = order[start : start + args.head_batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[idx])
            loss = criterion(logits, y[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x[val_t])
            val_loss = float(F.cross_entropy(val_logits, y[val_t]).item())
            val_acc = float((val_logits.argmax(1) == y[val_t]).float().mean().item())
        improved = val_acc > best_acc + 1e-8 or (
            abs(val_acc - best_acc) <= 1e-8 and val_loss < best_loss
        )
        if improved:
            best_acc = val_acc
            best_loss = val_loss
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.head_patience:
            break
    if best_state is None:
        raise RuntimeError("Classification head did not produce a valid state")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, {"best_val_acc": best_acc, "best_val_loss": best_loss, "epochs": epoch}


@torch.no_grad()
def logits_for(model: nn.Module, features: torch.Tensor, indices: np.ndarray, device) -> torch.Tensor:
    idx = torch.as_tensor(indices, dtype=torch.long, device=device)
    return model(features.to(device)[idx]).float().cpu()


def classification_metrics(logits: torch.Tensor, labels: np.ndarray) -> dict:
    prediction = logits.argmax(1).numpy()
    return {
        "acc": float(accuracy_score(labels, prediction)),
        "balacc": float(balanced_accuracy_score(labels, prediction)),
    }


def select_poe_weight(
    obs_logits: torch.Tensor, prior_logits: torch.Tensor, labels: np.ndarray
) -> tuple[float, float]:
    y = torch.from_numpy(labels).long()
    obs_logp = F.log_softmax(obs_logits, dim=-1)
    prior_logp = F.log_softmax(prior_logits, dim=-1)
    best_alpha = 1.0
    best_nll = float("inf")
    for alpha in np.linspace(0.0, 1.0, 21):
        fused = float(alpha) * obs_logp + (1.0 - float(alpha)) * prior_logp
        nll = float(F.nll_loss(F.log_softmax(fused, dim=-1), y).item())
        if nll < best_nll:
            best_nll = nll
            best_alpha = float(alpha)
    return best_alpha, best_nll


def fused_logits(obs_logits: torch.Tensor, prior_logits: torch.Tensor, alpha: float) -> torch.Tensor:
    return alpha * F.log_softmax(obs_logits, dim=-1) + (1.0 - alpha) * F.log_softmax(
        prior_logits, dim=-1
    )


def load_frozen_models(args, specs, transformer_module, obs_ckpt, prior_ckpt, device):
    obs = build_obs_backbone(args, specs, transformer_module)
    obs_state = torch.load(obs_ckpt, map_location="cpu", weights_only=False)["backbone"]
    obs.load_state_dict(obs_state, strict=True)
    prior = build_prior(args)
    prior_state = torch.load(prior_ckpt, map_location="cpu", weights_only=False)["encoder"]
    prior.load_state_dict(prior_state, strict=True)
    for model in (obs, prior):
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.to(device).eval()
    return obs, prior


@torch.no_grad()
def extract_subject_features(
    ref: SubjectRef,
    obs: JointFNIRSTransformer,
    prior: SchemeAPriorEigenval,
    args,
    device: torch.device,
) -> dict:
    with h5py.File(ref.path, "r") as handle:
        labels = torch.from_numpy(np.asarray(handle["labels"], dtype=np.int64))
        eigval = torch.from_numpy(np.asarray(handle["fnirs_mode_eigval"], dtype=np.float32)).to(device)
        n_trials = len(labels)
    obs_features = []
    prior_features = []
    for start in range(0, n_trials, args.feature_batch_size):
        stop = min(n_trials, start + args.feature_batch_size)
        indices = np.arange(start, stop, dtype=np.int64)
        with h5py.File(ref.path, "r") as handle:
            raw = np.asarray(handle["fnirs_raw"][indices], dtype=np.float32)
            modes = np.asarray(handle["fnirs_modes"][indices], dtype=np.float32)
        x_raw = normalize_raw(raw_to_chromophores(raw).to(device))
        x_modes = torch.from_numpy(modes).to(device)
        with amp_context(device):
            obs_value = obs.encode(x_raw, ref.domain)
            prior_value = prior.encode_pooled(x_modes, eigval=eigval)
        obs_features.append(obs_value.float().cpu())
        prior_features.append(prior_value.float().cpu())
    return {
        "obs": torch.cat(obs_features),
        "prior": torch.cat(prior_features),
        "labels": labels,
    }


def load_or_extract_features(
    ref,
    cache_path: Path,
    obs,
    prior,
    args,
    device,
    log_path,
):
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    features = extract_subject_features(ref, obs, prior, args, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, cache_path)
    log(
        f"Cached {ref.domain}/{ref.subject}: obs={tuple(features['obs'].shape)}, "
        f"prior={tuple(features['prior'].shape)}",
        log_path,
    )
    return features


def make_train_validation_split(
    train_idx: np.ndarray, labels: np.ndarray, ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
    local_train, local_val = next(splitter.split(np.zeros(len(train_idx)), labels[train_idx]))
    return train_idx[local_train], train_idx[local_val]


def write_classification_summary(records: list[dict], output: Path, specs) -> None:
    methods = [
        "fnirs_transformer",
        "ours_obs",
        "ours_prior",
        "ours_poe_equal",
        "ours_poe_val_weighted",
    ]
    lines = [
        "Joint-pretrained fNIRS-Transformer + Scheme-A frozen PoE",
        "Protocol: unlabeled joint pretraining on Cocktail and Dubois; frozen backbones;",
        "within-subject 5-fold classification; validation split from training fold only.",
        "Scale: decimal accuracy",
        "",
    ]
    for domain in specs:
        domain_rows = [row for row in records if row["dataset"] == domain]
        lines.append("=" * 88)
        lines.append(f"{domain} ({specs[domain].n_classes} classes)")
        lines.append("=" * 88)
        width = 24
        header = f"{'Subject':<18}" + "".join(f"{name:>{width}}" for name in methods)
        lines.append(header)
        subject_values = defaultdict(lambda: defaultdict(list))
        for row in domain_rows:
            for name in methods:
                subject_values[row["subject"]][name].append(float(row[name]["acc"]))
        for subject in sorted(subject_values):
            values = [np.mean(subject_values[subject][name]) for name in methods]
            lines.append(f"{subject:<18}" + "".join(f"{value:{width}.4f}" for value in values))
        if subject_values:
            means = [
                np.mean([np.mean(value[name]) for value in subject_values.values()])
                for name in methods
            ]
            lines.append(f"{'Subject mean':<18}" + "".join(f"{value:{width}.4f}" for value in means))
        lines.append("")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_classification(
    args,
    specs,
    subjects,
    transformer_module,
    obs_ckpt,
    prior_ckpt,
    run_dir,
    device,
    log_path,
) -> None:
    obs, prior = load_frozen_models(
        args, specs, transformer_module, obs_ckpt, prior_ckpt, device
    )
    metrics_path = run_dir / "per_fold_metrics.json"
    records = []
    if metrics_path.exists():
        records = json.loads(metrics_path.read_text(encoding="utf-8"))
    done = {(row["dataset"], row["subject"], int(row["fold"])) for row in records}
    folds = args.folds if args.folds is not None else list(range(5))

    for domain, refs in subjects.items():
        spec = specs[domain]
        for ref in refs:
            features = load_or_extract_features(
                ref,
                run_dir / "feature_cache" / domain / f"{ref.subject}.pt",
                obs,
                prior,
                args,
                device,
                log_path,
            )
            labels_np = features["labels"].numpy()
            for fold in folds:
                key = (domain, ref.subject, int(fold))
                if key in done:
                    continue
                split_path = Path(spec.split_dir) / ref.subject / f"fold_{fold:02d}" / "split.json"
                if not split_path.exists():
                    raise FileNotFoundError(f"Missing split: {split_path}")
                split = json.loads(split_path.read_text(encoding="utf-8"))
                train_idx = np.asarray(split["train_idx"], dtype=np.int64)
                test_idx = np.asarray(split["test_idx"], dtype=np.int64)
                head_train_idx, val_idx = make_train_validation_split(
                    train_idx,
                    labels_np,
                    args.head_val_ratio,
                    args.seed * 10000 + fold * 101 + sum(map(ord, ref.subject)),
                )
                base_seed = args.seed * 100000 + fold * 1000 + sum(map(ord, ref.subject))
                obs_head, obs_train = train_head(
                    FrozenFeatureHead(obs.output_dim, spec.n_classes),
                    features["obs"],
                    features["labels"],
                    head_train_idx,
                    val_idx,
                    args,
                    device,
                    base_seed + 1,
                )
                prior_head, prior_train = train_head(
                    PriorAttentiveHead(args.prior_d_model, spec.n_classes),
                    features["prior"],
                    features["labels"],
                    head_train_idx,
                    val_idx,
                    args,
                    device,
                    base_seed + 2,
                )
                obs_val = logits_for(obs_head, features["obs"], val_idx, device)
                prior_val = logits_for(prior_head, features["prior"], val_idx, device)
                alpha, val_nll = select_poe_weight(obs_val, prior_val, labels_np[val_idx])
                obs_test = logits_for(obs_head, features["obs"], test_idx, device)
                prior_test = logits_for(prior_head, features["prior"], test_idx, device)
                obs_metrics = classification_metrics(obs_test, labels_np[test_idx])
                prior_metrics = classification_metrics(prior_test, labels_np[test_idx])
                equal_metrics = classification_metrics(
                    fused_logits(obs_test, prior_test, 0.5), labels_np[test_idx]
                )
                weighted_metrics = classification_metrics(
                    fused_logits(obs_test, prior_test, alpha), labels_np[test_idx]
                )
                row = {
                    "dataset": domain,
                    "subject": ref.subject,
                    "fold": int(fold),
                    "seed": args.seed,
                    "train_n": int(len(head_train_idx)),
                    "val_n": int(len(val_idx)),
                    "test_n": int(len(test_idx)),
                    "fnirs_transformer": obs_metrics,
                    "ours_obs": obs_metrics,
                    "ours_prior": prior_metrics,
                    "ours_poe_equal": equal_metrics,
                    "ours_poe_val_weighted": weighted_metrics,
                    "poe_obs_weight": alpha,
                    "poe_val_nll": val_nll,
                    "obs_head": obs_train,
                    "prior_head": prior_train,
                }
                records.append(row)
                done.add(key)
                atomic_json_dump(records, metrics_path)
                write_classification_summary(
                    records, run_dir / "summary_by_subject.txt", specs
                )
                log(
                    f"DONE {domain}/{ref.subject}/fold{fold}: "
                    f"fNIRS-T={obs_metrics['acc']:.4f}, prior={prior_metrics['acc']:.4f}, "
                    f"PoE={weighted_metrics['acc']:.4f}, alpha_obs={alpha:.2f}",
                    log_path,
                )
    del obs, prior


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "pretrain", "classify"], default="all")
    parser.add_argument("--cocktail_h5", default=str(DEFAULT_COCKTAIL_H5))
    parser.add_argument("--dubois_h5", default=str(DEFAULT_DUBOIS_H5))
    parser.add_argument("--cocktail_splits", default=str(DEFAULT_COCKTAIL_SPLITS))
    parser.add_argument("--dubois_splits", default=str(DEFAULT_DUBOIS_SPLITS))
    parser.add_argument("--transformer_dir", default=str(DEFAULT_TRANSFORMER_DIR))
    parser.add_argument("--prior_init", default=str(DEFAULT_PRIOR_INIT))
    parser.add_argument("--out_dir", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force_pretrain", action="store_true")
    parser.add_argument("--max_subjects_per_domain", type=int, default=0)
    parser.add_argument("--folds", type=int, nargs="+", default=None)

    parser.add_argument("--transformer_dim", type=int, default=64)
    parser.add_argument("--transformer_depth", type=int, default=6)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_mlp_dim", type=int, default=64)
    parser.add_argument("--transformer_dim_head", type=int, default=64)
    parser.add_argument("--obs_pretrain_epochs", type=int, default=50)
    parser.add_argument("--obs_pretrain_batch_size", type=int, default=1)
    parser.add_argument("--obs_pretrain_lr", type=float, default=3e-4)
    parser.add_argument("--obs_channel_mask_ratio", type=float, default=0.15)
    parser.add_argument("--obs_time_mask_ratio", type=float, default=0.20)

    parser.add_argument("--prior_patch_size", type=int, default=32)
    parser.add_argument("--prior_d_model", type=int, default=512)
    parser.add_argument("--prior_factor_layers", type=int, default=2)
    parser.add_argument("--prior_time_layers", type=int, default=4)
    parser.add_argument("--prior_decoder_layers", type=int, default=2)
    parser.add_argument("--prior_pool_queries", type=int, default=4)
    parser.add_argument("--prior_pretrain_epochs", type=int, default=50)
    parser.add_argument("--prior_pretrain_lr", type=float, default=1e-4)
    parser.add_argument("--prior_batch_cocktail", type=int, default=8)
    parser.add_argument("--prior_batch_dubois", type=int, default=2)
    parser.add_argument("--prior_mask_ratio", type=float, default=0.5)
    parser.add_argument("--pretrain_val_every", type=int, default=5)
    parser.add_argument("--pretrain_val_batch_size", type=int, default=1)

    parser.add_argument("--feature_batch_size", type=int, default=1)
    parser.add_argument("--head_epochs", type=int, default=300)
    parser.add_argument("--head_batch_size", type=int, default=32)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--head_weight_decay", type=float, default=1e-3)
    parser.add_argument("--head_label_smoothing", type=float, default=0.05)
    parser.add_argument("--head_patience", type=int, default=40)
    parser.add_argument("--head_val_ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / f"jointpre_fnirst_schemeA_seed{args.seed:03d}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.txt"
    specs = discover_specs(args)
    subjects = discover_subjects(specs, args.max_subjects_per_domain)
    transformer_module = import_transformer_module(Path(args.transformer_dir))
    obs_ckpt = run_dir / "pretrain" / "joint_fnirs_transformer_masked.pt"
    prior_ckpt = run_dir / "pretrain" / "joint_schemeA_prior_mae.pt"
    atomic_json_dump(
        {
            "args": vars(args),
            "specs": {name: asdict(spec) for name, spec in specs.items()},
            "subjects": {name: [asdict(ref) for ref in refs] for name, refs in subjects.items()},
            "run_dir": str(run_dir),
            "protocol": {
                "pretraining_labels_used": False,
                "backbones_frozen_for_classification": True,
                "validation_from_training_fold_only": True,
                "test_evaluated_once": True,
            },
        },
        run_dir / "manifest.json",
    )
    log(f"Run directory: {run_dir}", log_path)
    log(f"Device: {device}", log_path)
    log(
        "Domains: "
        + ", ".join(
            f"{name}[subjects={len(subjects[name])},C={spec.channels},T={spec.samples}]"
            for name, spec in specs.items()
        ),
        log_path,
    )
    start = time.time()
    if args.stage in ("all", "pretrain"):
        pretrain_observation(
            args, specs, subjects, transformer_module, obs_ckpt, device, log_path
        )
        pretrain_prior(args, subjects, prior_ckpt, device, log_path)
    if args.stage in ("all", "classify"):
        if not obs_ckpt.exists() or not prior_ckpt.exists():
            raise FileNotFoundError(
                f"Missing pretraining checkpoints: obs={obs_ckpt.exists()}, "
                f"prior={prior_ckpt.exists()}"
            )
        run_classification(
            args,
            specs,
            subjects,
            transformer_module,
            obs_ckpt,
            prior_ckpt,
            run_dir,
            device,
            log_path,
        )
    log(f"Finished stage={args.stage} in {(time.time() - start) / 3600:.2f} h", log_path)


if __name__ == "__main__":
    main()
