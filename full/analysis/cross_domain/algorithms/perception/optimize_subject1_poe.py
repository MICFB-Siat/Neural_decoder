








from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


NSD_ROOT = Path(
    "/home/guoyi/a800/share/code/Eigen_brain_decoding/lizhuo_exp/lizhuo_exp_NSD"
)
CODE_ROOT = NSD_ROOT / "test_cifti_modes/code"
BASE_ROOT = NSD_ROOT / "test_cifti_modes/result/subject1_poe_mindeye2_1024"
DEFAULT_ROOT = NSD_ROOT / "test_cifti_modes/result/subject1_poe_optimization"
DATA_PATH = Path(
    "/home/guoyi/nas2/share/Dataset/NSD-download/preprocess/stage2/sub-01.h5"
)
OLD_VERTICES = (
    NSD_ROOT / "test_cifti_modes/result/cache/cifti_ncsnr_top4096_vertices.npy"
)
NEW_VERTICES = (
    BASE_ROOT / "cache/cifti_actual_repeat_top4096_vertices_train_only.npy"
)
POE_CHECKPOINT = BASE_ROOT / "checkpoints/poe_seed42_best.pt"
CIFTI_CHECKPOINT = BASE_ROOT / "checkpoints/cifti_seed42_best.pt"
SHM_CACHE = Path("/dev/shm/subject1_poe_cache")
TEMP_PREFIX = "subject1_"

CIFTI_DIM = 4096
MODES_DIM = 2000
HIDDEN_DIM = 1024
CLIP_TOKENS = 256
CLIP_DIM = 1664
FUSION_VARIANTS = (
    "original_poe",
    "stable_poe",
    "global_poe",
    "anchored_poe",
    "cifti_mean",
)

sys.path.insert(0, str(CODE_ROOT))
import subject1_poe_mindeye2 as base


def directories(root: Path) -> dict[str, Path]:
    result = {
        "root": root,
        "cache": root / "cache",
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
        "metrics": root / "metrics",
    }
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    base.set_seed(seed)


def device_from_arg(value: str) -> torch.device:
    return torch.device(value if torch.cuda.is_available() else "cpu")


def load_split() -> dict[str, np.ndarray]:
    with np.load(BASE_ROOT / "cache/split_and_normalization.npz") as value:
        return {key: value[key] for key in value.files}


def cmd_prepare(args: argparse.Namespace) -> None:
    root = Path(args.result_dir)
    out = directories(root)
    split = load_split()
    vertices = np.load(args.vertices).astype(np.int64)
    if vertices.shape != (CIFTI_DIM,) or len(np.unique(vertices)) != CIFTI_DIM:
        raise ValueError(f"Expected {CIFTI_DIM} unique vertices, got {vertices.shape}")
    if vertices.min() < 0 or vertices.max() >= 59412:
        raise ValueError("CIFTI vertex indices are outside [0, 59412)")

    train_rows = split["train_rows"]
    with h5py.File(DATA_PATH, "r") as h5:
        selected_fit = h5["fmri_cift"][train_rows][:, vertices].astype(np.float32)
        mean = selected_fit.mean(0, keepdims=True, dtype=np.float64).astype(np.float32)
        std = selected_fit.std(0, keepdims=True, dtype=np.float64).astype(np.float32)
        std = np.maximum(std, 1e-6)
        del selected_fit
        for name in ("train", "val", "test"):
            rows = split[f"{name}_rows"]
            destination = np.lib.format.open_memmap(
                out["cache"] / f"cifti_{name}.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(rows), CIFTI_DIM),
            )
            for start in tqdm(
                range(0, len(rows), args.batch_size), desc=f"corrected CIFTI {name}"
            ):
                stop = min(start + args.batch_size, len(rows))
                block = h5["fmri_cift"][rows[start:stop]][:, vertices].astype(np.float32)
                destination[start:stop] = ((block - mean) / std).astype(np.float16)
            destination.flush()
            del destination

    np.save(out["cache"] / "vertices_train_repeat_top4096.npy", vertices)
    np.savez(
        out["cache"] / "protocol.npz",
        train_rows=split["train_rows"],
        val_rows=split["val_rows"],
        test_rows=split["test_rows"],
        vertices=vertices,
        cifti_mean=mean,
        cifti_std=std,
    )
    old_vertices = np.load(OLD_VERTICES).astype(np.int64)
    report = {
        "selection": "top-4096 mean pairwise repeat correlation on the 8100 fitting images",
        "selection_uses_shared1000": False,
        "vertices": str(Path(args.vertices)),
        "old_new_overlap": int(len(np.intersect1d(old_vertices, vertices))),
        "train_count": int(len(split["train_rows"])),
        "validation_count": int(len(split["val_rows"])),
        "test_count": int(len(split["test_rows"])),
        "normalization": "feature-wise mean/std from the 8100 fitting images only",
        "finite": True,
    }
    for name in ("train", "val", "test"):
        array = np.load(out["cache"] / f"cifti_{name}.npy", mmap_mode="r")
        report[f"{name}_shape"] = list(array.shape)
        report["finite"] = report["finite"] and bool(
            np.isfinite(np.asarray(array[: min(256, len(array))])).all()
        )
    write_json(out["metrics"] / "corrected_cifti_protocol.json", report)
    print(json.dumps(report, indent=2))


class StablePoE(nn.Module):






    def __init__(self, prior_precision: float = 0.05, logvar_limit: float = 4.0):
        super().__init__()
        self.cifti_mean = nn.Linear(CIFTI_DIM, HIDDEN_DIM)
        self.cifti_logvar = nn.Linear(CIFTI_DIM, HIDDEN_DIM)
        self.modes_mean = nn.Linear(MODES_DIM, HIDDEN_DIM)
        self.modes_logvar = nn.Linear(MODES_DIM, HIDDEN_DIM)
        self.prior_precision = prior_precision
        self.logvar_limit = logvar_limit

    def _logvar(self, layer: nn.Linear, value: torch.Tensor) -> torch.Tensor:
        raw = layer(value)
        return self.logvar_limit * torch.tanh(raw / self.logvar_limit)

    def forward(self, cifti: torch.Tensor, modes: torch.Tensor):
        cifti_mean = self.cifti_mean(cifti)
        modes_mean = self.modes_mean(modes)
        cifti_logvar = self._logvar(self.cifti_logvar, cifti)
        modes_logvar = self._logvar(self.modes_logvar, modes)
        cifti_precision = torch.exp(-cifti_logvar)
        modes_precision = torch.exp(-modes_logvar)
        total_precision = self.prior_precision + cifti_precision + modes_precision
        fused_variance = total_precision.reciprocal()
        fused_mean = fused_variance * (
            cifti_mean * cifti_precision + modes_mean * modes_precision
        )
        return fused_mean[:, None], {
            "fused_mean": fused_mean,
            "fused_variance": fused_variance,
            "cifti_precision": cifti_precision,
            "modes_precision": modes_precision,
        }


class DeterministicOriginalPoE(base.LinearGaussianPoE):


    def __init__(self):
        super().__init__("poe", hidden_dim=HIDDEN_DIM)

    def forward(self, cifti: torch.Tensor, modes: torch.Tensor):
        return super().forward(
            cifti,
            modes,
            sample=False,
            modality_dropout=False,
        )


class GlobalPoE(nn.Module):


    def __init__(self):
        super().__init__()
        self.cifti_mean = nn.Linear(CIFTI_DIM, HIDDEN_DIM)
        self.modes_mean = nn.Linear(MODES_DIM, HIDDEN_DIM)
        self.cifti_log_precision = nn.Parameter(torch.full((HIDDEN_DIM,), -math.log(2.0)))
        self.modes_log_precision = nn.Parameter(torch.full((HIDDEN_DIM,), -math.log(2.0)))

    def forward(self, cifti: torch.Tensor, modes: torch.Tensor):
        cifti_mean = self.cifti_mean(cifti)
        modes_mean = self.modes_mean(modes)
        cifti_precision = torch.exp(3.0 * torch.tanh(self.cifti_log_precision / 3.0))
        modes_precision = torch.exp(3.0 * torch.tanh(self.modes_log_precision / 3.0))
        total_precision = cifti_precision + modes_precision
        fused_mean = (
            cifti_mean * cifti_precision + modes_mean * modes_precision
        ) / total_precision
        fused_variance = total_precision.reciprocal().expand_as(fused_mean)
        return fused_mean[:, None], {
            "fused_mean": fused_mean,
            "fused_variance": fused_variance,
            "cifti_precision": cifti_precision.expand_as(fused_mean),
            "modes_precision": modes_precision.expand_as(fused_mean),
        }


class AnchoredPoE(GlobalPoE):


    def __init__(self):
        super().__init__()
        with torch.no_grad():
            self.cifti_log_precision.zero_()
            self.modes_log_precision.fill_(-2.0)


class CiftiMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.cifti_mean = nn.Linear(CIFTI_DIM, HIDDEN_DIM)

    def forward(self, cifti: torch.Tensor, modes: torch.Tensor):
        mean = self.cifti_mean(cifti)
        ones = torch.ones_like(mean)
        return mean[:, None], {
            "fused_mean": mean,
            "fused_variance": ones,
            "cifti_precision": ones,
        }


class FusionBackbone(nn.Module):
    def __init__(self, fusion: nn.Module):
        super().__init__()
        sys.path.insert(0, str(base.MINDEYE_SRC))
        from models import BrainNetwork

        self.fusion = fusion
        self.backbone = BrainNetwork(
            h=HIDDEN_DIM,
            in_dim=HIDDEN_DIM,
            seq_len=1,
            clip_size=CLIP_DIM,
            out_dim=CLIP_DIM * CLIP_TOKENS,
            n_blocks=4,
            blurry_recon=True,
        )

    def forward(self, cifti: torch.Tensor, modes: torch.Tensor):
        latent, statistics = self.fusion(cifti, modes)
        condition, projected, lowlevel = self.backbone(latent)
        return condition, projected, lowlevel, statistics


def source_checkpoint(variant: str) -> Path:
    cifti_source = variant in ("cifti_mean", "anchored_poe")
    source = CIFTI_CHECKPOINT if cifti_source else POE_CHECKPOINT
    local = Path(
        f"/dev/shm/{TEMP_PREFIX}cifti_seed42_best.pt"
        if cifti_source
        else f"/dev/shm/{TEMP_PREFIX}poe_seed42_best.pt"
    )
    return local if local.exists() else source


def make_fusion(variant: str) -> nn.Module:
    if variant == "original_poe":
        return DeterministicOriginalPoE()
    if variant == "stable_poe":
        return StablePoE()
    if variant == "global_poe":
        return GlobalPoE()
    if variant == "anchored_poe":
        return AnchoredPoE()
    if variant == "cifti_mean":
        return CiftiMean()
    raise ValueError(variant)


def _copy_remapped_cifti_layer(
    destination: nn.Linear,
    source_weight: torch.Tensor,
    source_bias: torch.Tensor,
    old_vertices: np.ndarray,
    new_vertices: np.ndarray,
) -> int:
    old_lookup = {int(vertex): index for index, vertex in enumerate(old_vertices)}
    copied = 0
    with torch.no_grad():
        destination.bias.copy_(source_bias.float())
        for new_index, vertex in enumerate(new_vertices):
            old_index = old_lookup.get(int(vertex))
            if old_index is not None:
                destination.weight[:, new_index].copy_(source_weight[:, old_index].float())
                copied += 1
    return copied


def build_model(variant: str, device: torch.device) -> tuple[FusionBackbone, dict[str, Any]]:
    checkpoint = source_checkpoint(variant)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = saved["model"]
    fusion = make_fusion(variant)
    model = FusionBackbone(fusion)
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    model.backbone.load_state_dict(backbone_state, strict=True)

    old_vertices = np.load(OLD_VERTICES).astype(np.int64)
    new_vertices = np.load(NEW_VERTICES).astype(np.int64)
    copied = _copy_remapped_cifti_layer(
        fusion.cifti_mean,
        state["fusion.cifti_mean.weight"],
        state["fusion.cifti_mean.bias"],
        old_vertices,
        new_vertices,
    )
    if hasattr(fusion, "cifti_logvar"):
        _copy_remapped_cifti_layer(
            fusion.cifti_logvar,
            state["fusion.cifti_logvar.weight"],
            state["fusion.cifti_logvar.bias"],
            old_vertices,
            new_vertices,
        )
    modes_source = None
    if hasattr(fusion, "modes_mean") and "fusion.modes_mean.weight" not in state:
        modes_saved = torch.load(
            source_checkpoint("original_poe"),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        modes_source = modes_saved["model"]
    else:
        modes_saved = None
        modes_source = state
    if hasattr(fusion, "modes_mean") and modes_source is not None:
        fusion.modes_mean.weight.data.copy_(
            modes_source["fusion.modes_mean.weight"].float()
        )
        fusion.modes_mean.bias.data.copy_(
            modes_source["fusion.modes_mean.bias"].float()
        )
    if hasattr(fusion, "modes_logvar") and "fusion.modes_logvar.weight" in state:
        fusion.modes_logvar.weight.data.copy_(state["fusion.modes_logvar.weight"].float())
        fusion.modes_logvar.bias.data.copy_(state["fusion.modes_logvar.bias"].float())

    del saved, state, backbone_state, modes_saved, modes_source
    model.backbone.requires_grad_(False).eval()
    model.fusion.train()
    model.to(device)
    return model, {
        "source_checkpoint": str(checkpoint),
        "remapped_cifti_columns": copied,
        "new_cifti_columns": CIFTI_DIM - copied,
    }


class Arrays:
    def __init__(self, root: Path, split: str, convnext: bool = True):
        local = SHM_CACHE

        def cached(name: str, fallback: Path) -> Path:
            candidate = local / name
            return candidate if candidate.exists() else fallback

        self.cifti = np.load(
            cached(f"cifti_{split}.npy", root / "cache" / f"cifti_{split}.npy"),
            mmap_mode="r",
        )
        self.modes = np.load(
            cached(
                f"modes_{split}.npy", BASE_ROOT / "cache" / f"modes_{split}.npy"
            ),
            mmap_mode="r",
        )
        self.bigg = np.load(
            cached(
                f"bigg_tokens_{split}.npy",
                BASE_ROOT / "cache" / f"bigg_tokens_{split}.npy",
            ),
            mmap_mode="r",
        )
        self.vae = np.load(
            cached(
                f"vae_latents_{split}.npy",
                BASE_ROOT / "cache" / f"vae_latents_{split}.npy",
            ),
            mmap_mode="r",
        )
        self.convnext_original = None
        self.convnext_augmented = None
        if convnext:
            self.convnext_original = np.load(
                cached(
                    f"convnext_original_{split}.npy",
                    BASE_ROOT / "cache" / f"convnext_original_{split}.npy",
                ),
                mmap_mode="r",
            )
            self.convnext_augmented = np.load(
                cached(
                    f"convnext_augmented_{split}.npy",
                    BASE_ROOT / "cache" / f"convnext_augmented_{split}.npy",
                ),
                mmap_mode="r",
            )
        lengths = {len(self.cifti), len(self.modes), len(self.bigg), len(self.vae)}
        if len(lengths) != 1:
            raise RuntimeError(f"Misaligned {split} arrays: {lengths}")

    def __len__(self) -> int:
        return len(self.cifti)

    def batch(self, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
        result = {
            "cifti": torch.from_numpy(np.asarray(self.cifti[indices], dtype=np.float32)).to(device),
            "modes": torch.from_numpy(np.asarray(self.modes[indices], dtype=np.float32)).to(device),
            "bigg": torch.from_numpy(np.asarray(self.bigg[indices], dtype=np.float32)).to(device),
            "vae": torch.from_numpy(np.asarray(self.vae[indices], dtype=np.float32)).to(device),
        }
        if self.convnext_original is not None:
            result["convnext_original"] = torch.from_numpy(
                np.asarray(self.convnext_original[indices], dtype=np.float32)
            ).to(device)
            result["convnext_augmented"] = torch.from_numpy(
                np.asarray(self.convnext_augmented[indices], dtype=np.float32)
            ).to(device)
        return result


def mix_lowlevel_target(
    target: torch.Tensor, permutation: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    shape = (-1,) + (1,) * (target.ndim - 1)
    return target * beta.reshape(shape) + target[permutation] * (1.0 - beta).reshape(shape)


def compute_stage1_losses(
    model: FusionBackbone,
    batch: dict[str, torch.Tensor],
    epoch: int,
    epochs: int,
    mixup_fraction: float,
    train: bool,
    convnext_weight: float,
):
    cifti, modes = batch["cifti"], batch["modes"]
    permutation = beta = None
    use_mixco = train and epoch < int(mixup_fraction * epochs)
    if use_mixco:
        cifti, modes, permutation, beta, _ = base.apply_mixco(cifti, modes)
    condition, projected, lowlevel, statistics = model(cifti, modes)
    projected_flat = F.normalize(projected.flatten(1).float(), dim=-1)
    target_flat = F.normalize(batch["bigg"].flatten(1).float(), dim=-1)
    if use_mixco:
        clip_loss = base.mixco_nce_loss(
            projected_flat, target_flat, permutation, beta, temperature=0.006
        )
        vae_target = mix_lowlevel_target(batch["vae"].float(), permutation, beta)
    else:
        temperature = 0.006 if not train else base.cosine_temperature(
            epoch, epochs, mixup_fraction
        )
        clip_loss = base.soft_clip_loss(projected_flat, target_flat, temperature)
        vae_target = batch["vae"].float()
    lowlevel_loss = F.l1_loss(lowlevel[0].float(), vae_target)
    structural_loss = torch.zeros((), device=cifti.device)
    if convnext_weight > 0:
        convnext_original = batch["convnext_original"].float()
        convnext_augmented = batch["convnext_augmented"].float()
        if use_mixco:
            convnext_original = mix_lowlevel_target(
                convnext_original, permutation, beta
            )
            convnext_augmented = mix_lowlevel_target(
                convnext_augmented, permutation, beta
            )
        structural_loss = base.soft_cont_loss(
            lowlevel[1].float(),
            convnext_original,
            convnext_augmented,
        )
    precision_regularizer = torch.zeros((), device=cifti.device)
    for key in ("cifti_precision", "modes_precision"):
        if key in statistics:
            precision_regularizer = precision_regularizer + torch.log(
                statistics[key].float().clamp_min(1e-6)
            ).square().mean()
    total = clip_loss + 0.5 * lowlevel_loss + convnext_weight * structural_loss
    total = total + 1e-5 * precision_regularizer
    losses = {
        "total": total,
        "clip": clip_loss,
        "lowlevel_l1": lowlevel_loss,
        "convnext": structural_loss,
        "precision_regularizer": precision_regularizer,
    }
    diagonal_cosine = (projected_flat * target_flat).sum(-1).mean()
    return total, losses, diagonal_cosine, projected, lowlevel, statistics


def aggregate(sums: dict[str, float], values: dict[str, torch.Tensor], count: int) -> None:
    for key, value in values.items():
        sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * count


def validate(
    model: FusionBackbone,
    arrays: Arrays,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> tuple[dict[str, float], dict[str, float]]:
    fusion_training = model.fusion.training
    backbone_training = model.backbone.training
    model.eval()
    sums: dict[str, float] = {}
    cosine_sum = 0.0
    count = 0
    cifti_precision = modes_precision = fused_variance = 0.0
    with torch.no_grad():
        for start in range(0, len(arrays), args.eval_batch_size):
            indices = np.arange(start, min(start + args.eval_batch_size, len(arrays)))
            batch = arrays.batch(indices, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                _, losses, cosine, _, _, statistics = compute_stage1_losses(
                    model,
                    batch,
                    epoch,
                    args.epochs,
                    args.mixup_fraction,
                    False,
                    args.convnext_weight,
                )
            n = len(indices)
            aggregate(sums, losses, n)
            cosine_sum += float(cosine.cpu()) * n
            fused_variance += float(statistics["fused_variance"].float().mean().cpu()) * n
            if "cifti_precision" in statistics:
                cifti_precision += float(
                    statistics["cifti_precision"].float().mean().cpu()
                ) * n
            if "modes_precision" in statistics:
                modes_precision += float(
                    statistics["modes_precision"].float().mean().cpu()
                ) * n
            count += n
    model.fusion.train(fusion_training)
    model.backbone.train(backbone_training)
    metrics = {key: value / count for key, value in sums.items()}
    metrics["diagonal_cosine"] = cosine_sum / count
    diagnostics = {
        "mean_fused_variance": fused_variance / count,
        "mean_cifti_precision": cifti_precision / count,
        "mean_modes_precision": modes_precision / count,
    }
    return metrics, diagnostics


def compact_state(module: nn.Module) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in module.state_dict().items():
        tensor = value.detach().cpu()
        result[key] = tensor.half() if tensor.is_floating_point() else tensor
    return result


def save_fusion(path: Path, model: FusionBackbone, metadata: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"fusion": compact_state(model.fusion), "metadata": metadata}, temporary)
    temporary.replace(path)


def load_fusion(path: Path, model: FusionBackbone) -> dict[str, Any]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model.fusion.load_state_dict(saved["fusion"], strict=True)
    return saved["metadata"]


def save_stage2(path: Path, model: FusionBackbone, metadata: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "fusion": compact_state(model.fusion),
            "backbone": compact_state(model.backbone),
            "metadata": metadata,
        },
        temporary,
    )
    temporary.replace(path)


def atomic_copy(source: Path, destination: Path) -> None:

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(destination)


def load_stage2(path: Path, model: FusionBackbone) -> dict[str, Any]:
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    model.fusion.load_state_dict(saved["fusion"], strict=True)
    model.backbone.load_state_dict(saved["backbone"], strict=True)
    return saved["metadata"]


def predict_split(
    root: Path,
    run_name: str,
    model: FusionBackbone,
    arrays: Arrays,
    device: torch.device,
    batch_size: int,
    split: str = "test",
) -> tuple[np.ndarray, np.ndarray]:
    out = directories(root)
    token_path = out["cache"] / f"{run_name}_{split}_direct_tokens.npy"
    lowlevel_path = out["cache"] / f"{run_name}_{split}_lowlevel_latents.npy"
    tokens = np.lib.format.open_memmap(
        token_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(arrays), CLIP_TOKENS, CLIP_DIM),
    )
    lowlevel = np.lib.format.open_memmap(
        lowlevel_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(arrays), 4, 28, 28),
    )
    model.eval()
    with torch.no_grad():
        for start in tqdm(
            range(0, len(arrays), batch_size), desc=f"{split} {run_name}"
        ):
            indices = np.arange(start, min(start + batch_size, len(arrays)))
            batch = arrays.batch(indices, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                _, projected, low, _ = model(batch["cifti"], batch["modes"])
            tokens[start : start + len(indices)] = projected.float().cpu().numpy().astype(np.float16)
            lowlevel[start : start + len(indices)] = low[0].float().cpu().numpy().astype(np.float16)
    tokens.flush()
    lowlevel.flush()
    return tokens, lowlevel


def lowlevel_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float32).reshape(len(prediction), -1)
    target = np.asarray(target, dtype=np.float32).reshape(len(target), -1)
    mse = float(np.mean(np.square(prediction - target)))
    cosine = np.sum(prediction * target, axis=1) / (
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1) + 1e-8
    )
    prediction_centered = prediction - prediction.mean(axis=1, keepdims=True)
    target_centered = target - target.mean(axis=1, keepdims=True)
    pearson = np.sum(prediction_centered * target_centered, axis=1) / (
        np.linalg.norm(prediction_centered, axis=1)
        * np.linalg.norm(target_centered, axis=1)
        + 1e-8
    )
    return {
        "mse": mse,
        "normalized_mse": float(mse / (np.var(target) + 1e-8)),
        "mean_cosine": float(cosine.mean()),
        "mean_pearson": float(pearson.mean()),
    }


def cmd_train_stage1(args: argparse.Namespace) -> None:
    root = Path(args.result_dir)
    out = directories(root)
    device = device_from_arg(args.device)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train_arrays = Arrays(root, "train", convnext=args.convnext_weight > 0)
    val_arrays = Arrays(root, "val", convnext=args.convnext_weight > 0)
    test_arrays = Arrays(root, "test", convnext=False)
    model, initialization = build_model(args.variant, device)
    train_count = len(train_arrays)
    if args.max_train_samples > 0:
        train_count = min(train_count, args.max_train_samples)
    optimizer = torch.optim.AdamW(
        model.fusion.parameters(), lr=args.max_lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(train_count / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        total_steps=args.epochs * steps_per_epoch,
        pct_start=min(0.2, 2.0 / args.epochs),
        final_div_factor=1000,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    run_name = args.run_name or args.variant
    checkpoint = out["checkpoints"] / f"{run_name}_fusion_best.pt"
    config = {
        "round": 1,
        "run_name": run_name,
        "variant": args.variant,
        "corrected_cifti": True,
        "brainnetwork_frozen": True,
        "diffusion_prior_used": False,
        "poe_sampling": False,
        "modality_dropout": False,
        "mixed_lowlevel_target_fixed": True,
        "epochs": args.epochs,
        "train_count": train_count,
        "batch_size": args.batch_size,
        "max_lr": args.max_lr,
        "convnext_weight": args.convnext_weight,
        "seed": args.seed,
        **initialization,
    }
    write_json(out["metrics"] / f"{run_name}_config.json", config)
    print(json.dumps(config, indent=2), flush=True)

    history = []
    best_cosine = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    started = time.time()
    for epoch in range(args.epochs):
        model.fusion.train()
        permutation = np.random.permutation(len(train_arrays))[:train_count]
        sums: dict[str, float] = {}
        count = 0
        progress = tqdm(
            range(0, len(permutation), args.batch_size),
            desc=f"{run_name} epoch {epoch + 1}",
        )
        for start in progress:
            indices = permutation[start : start + args.batch_size]
            batch = train_arrays.batch(indices, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                total, losses, _, _, _, _ = compute_stage1_losses(
                    model,
                    batch,
                    epoch,
                    args.epochs,
                    args.mixup_fraction,
                    True,
                    args.convnext_weight,
                )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            aggregate(sums, losses, len(indices))
            count += len(indices)
            if start % (args.batch_size * 20) == 0:
                progress.set_postfix(loss=f"{float(total.detach().cpu()):.4f}")
        train_metrics = {key: value / count for key, value in sums.items()}
        val_metrics, diagnostics = validate(model, val_arrays, device, args, epoch)
        item = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": val_metrics,
            "diagnostics": diagnostics,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(item)
        write_json(out["logs"] / f"{run_name}_history.json", {"config": config, "history": history})
        score = val_metrics["diagonal_cosine"]
        if score > best_cosine + args.min_delta:
            best_cosine = score
            best_epoch = epoch + 1
            bad_epochs = 0
            save_fusion(
                checkpoint,
                model,
                {**config, "best_epoch": best_epoch, "best_validation_cosine": best_cosine},
            )
        else:
            bad_epochs += 1
        print(
            f"[{run_name}] epoch={epoch + 1} train={train_metrics['total']:.5f} "
            f"val={val_metrics['total']:.5f} cosine={score:.6f} "
            f"best={best_cosine:.6f} bad={bad_epochs}",
            flush=True,
        )
        if epoch + 1 >= args.min_epochs and bad_epochs >= args.patience:
            break

    metadata = load_fusion(checkpoint, model)
    summary = {
        **config,
        "best_epoch": best_epoch,
        "best_validation_cosine": best_cosine,
        "last_epoch": len(history),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": metadata,
    }
    if not args.skip_test:
        direct, lowlevel = predict_split(
            root, run_name, model, test_arrays, device, args.eval_batch_size
        )
        summary["test_direct_token_metrics"] = base.token_identification(
            direct, test_arrays.bigg, device, args.identification_batch_size
        )
        summary["test_lowlevel_latent_metrics"] = lowlevel_metrics(
            lowlevel, test_arrays.vae
        )
    write_json(out["metrics"] / f"{run_name}_stage1_results.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def cmd_evaluate_stage1(args: argparse.Namespace) -> None:
    root = Path(args.result_dir)
    out = directories(root)
    device = device_from_arg(args.device)
    set_seed(args.seed)
    model, initialization = build_model(args.variant, device)
    metadata = load_fusion(Path(args.checkpoint), model)
    arrays = Arrays(root, args.split, convnext=False)
    run_name = args.run_name or metadata["run_name"]
    direct, lowlevel = predict_split(
        root,
        run_name,
        model,
        arrays,
        device,
        args.eval_batch_size,
        split=args.split,
    )
    report = {
        "run_name": run_name,
        "variant": args.variant,
        "checkpoint": str(Path(args.checkpoint)),
        "split": args.split,
        "count": len(arrays),
        "initialization": initialization,
        "checkpoint_metadata": metadata,
        f"{args.split}_direct_token_metrics": base.token_identification(
            direct, arrays.bigg, device, args.identification_batch_size
        ),
        f"{args.split}_lowlevel_latent_metrics": lowlevel_metrics(
            lowlevel, arrays.vae
        ),
    }
    write_json(out["metrics"] / f"{run_name}_{args.split}_results.json", report)
    print(json.dumps(report, indent=2), flush=True)


def cmd_train_stage2(args: argparse.Namespace) -> None:
    root = Path(args.result_dir)
    out = directories(root)
    device = device_from_arg(args.device)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train_arrays = Arrays(root, "train", convnext=args.convnext_weight > 0)
    val_arrays = Arrays(root, "val", convnext=args.convnext_weight > 0)
    model, initialization = build_model(args.variant, device)
    if args.resume_stage2:
        stage1_metadata = load_stage2(Path(args.resume_stage2), model)
        initialization_checkpoint = Path(args.resume_stage2)
        initialization_type = "stage2"
    else:
        stage1_metadata = load_fusion(Path(args.stage1_checkpoint), model)
        initialization_checkpoint = Path(args.stage1_checkpoint)
        initialization_type = "stage1"
    model.fusion.requires_grad_(True)
    model.backbone.requires_grad_(args.train_scope == "all")
    if args.train_scope == "selective":
        for name in (
            "mixer_blocks1",
            "mixer_blocks2",
            "clip_proj",
            "blin1",
            "bupsampler",
            "b_maps_projector",
        ):
            getattr(model.backbone, name).requires_grad_(True)
    model.train()

    train_count = len(train_arrays)
    if args.max_train_samples > 0:
        train_count = min(train_count, args.max_train_samples)
    parameter_groups = [
        {
            "params": [p for p in model.fusion.parameters() if p.requires_grad],
            "lr": args.fusion_lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for p in model.backbone.parameters() if p.requires_grad],
            "lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer_options: dict[str, Any] = {}
    if device.type == "cuda":
        optimizer_options["fused"] = True
    optimizer = torch.optim.AdamW(parameter_groups, **optimizer_options)
    steps_per_epoch = math.ceil(train_count / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.fusion_lr, args.backbone_lr],
        total_steps=args.epochs * steps_per_epoch,
        pct_start=min(0.2, 2.0 / args.epochs),
        final_div_factor=1000,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    run_name = args.run_name or f"r2_{args.variant}"
    temporary_checkpoint = Path("/dev/shm") / f"{TEMP_PREFIX}{run_name}_stage2_best.pt"
    final_checkpoint = out["checkpoints"] / f"{run_name}_stage2_best.pt"
    config = {
        "round": 2,
        "run_name": run_name,
        "variant": args.variant,
        "initialization_checkpoint": str(initialization_checkpoint),
        "initialization_checkpoint_type": initialization_type,
        "stage1_best_validation_cosine": stage1_metadata.get(
            "best_validation_cosine"
        ),
        "corrected_cifti": True,
        "brainnetwork_frozen": False,
        "brainnetwork_train_scope": args.train_scope,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "diffusion_prior_used": False,
        "poe_sampling": False,
        "modality_dropout": False,
        "epochs": args.epochs,
        "train_count": train_count,
        "batch_size": args.batch_size,
        "fusion_lr": args.fusion_lr,
        "backbone_lr": args.backbone_lr,
        "convnext_weight": args.convnext_weight,
        "seed": args.seed,
        **initialization,
    }
    write_json(out["metrics"] / f"{run_name}_config.json", config)
    print(json.dumps(config, indent=2), flush=True)

    history: list[dict[str, Any]] = []
    best_cosine = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        permutation = np.random.permutation(len(train_arrays))[:train_count]
        sums: dict[str, float] = {}
        count = 0
        progress = tqdm(
            range(0, len(permutation), args.batch_size),
            desc=f"{run_name} epoch {epoch + 1}",
        )
        for start in progress:
            indices = permutation[start : start + args.batch_size]
            batch = train_arrays.batch(indices, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                total, losses, _, _, _, _ = compute_stage1_losses(
                    model,
                    batch,
                    epoch,
                    args.epochs,
                    args.mixup_fraction,
                    True,
                    args.convnext_weight,
                )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            aggregate(sums, losses, len(indices))
            count += len(indices)
            if start % (args.batch_size * 20) == 0:
                progress.set_postfix(loss=f"{float(total.detach().cpu()):.4f}")
        train_metrics = {key: value / count for key, value in sums.items()}
        val_metrics, diagnostics = validate(model, val_arrays, device, args, epoch)
        item = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": val_metrics,
            "diagnostics": diagnostics,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(item)
        write_json(
            out["logs"] / f"{run_name}_history.json",
            {"config": config, "history": history},
        )
        score = val_metrics["diagonal_cosine"]
        if score > best_cosine + args.min_delta:
            best_cosine = score
            best_epoch = epoch + 1
            bad_epochs = 0
            save_stage2(
                temporary_checkpoint,
                model,
                {
                    **config,
                    "best_epoch": best_epoch,
                    "best_validation_cosine": best_cosine,
                },
            )
        else:
            bad_epochs += 1
        print(
            f"[{run_name}] epoch={epoch + 1} train={train_metrics['total']:.5f} "
            f"val={val_metrics['total']:.5f} cosine={score:.6f} "
            f"best={best_cosine:.6f} bad={bad_epochs}",
            flush=True,
        )
        if epoch + 1 >= args.min_epochs and bad_epochs >= args.patience:
            break

    metadata = load_stage2(temporary_checkpoint, model)
    atomic_copy(temporary_checkpoint, final_checkpoint)
    summary = {
        **config,
        "best_epoch": best_epoch,
        "best_validation_cosine": best_cosine,
        "last_epoch": len(history),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(final_checkpoint),
        "checkpoint_metadata": metadata,
    }
    write_json(out["metrics"] / f"{run_name}_stage2_results.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def cmd_cache_conditions(args: argparse.Namespace) -> None:
    root = Path(args.result_dir)
    out = directories(root)
    device = device_from_arg(args.device)
    set_seed(args.seed)
    model, initialization = build_model(args.variant, device)
    checkpoint = Path(args.brain_checkpoint)
    if args.checkpoint_type == "stage1":
        metadata = load_fusion(checkpoint, model)
    else:
        metadata = load_stage2(checkpoint, model)
    model.eval().requires_grad_(False)
    local = SHM_CACHE
    local.mkdir(parents=True, exist_ok=True)
    report_path = out["metrics"] / f"{args.run_name}_brain_results.json"
    report: dict[str, Any] = {
        "run_name": args.run_name,
        "variant": args.variant,
        "checkpoint_type": args.checkpoint_type,
        "brain_checkpoint": str(checkpoint),
        "checkpoint_metadata": metadata,
        "initialization": initialization,
        "ablation": args.ablate,
        "splits": {},
    }
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("brain_checkpoint") == str(checkpoint):
            report.update(existing)
    for split in args.splits:
        arrays = Arrays(root, split, convnext=False)
        condition_path = local / f"{args.run_name}_condition_{split}.npy"
        condition = np.lib.format.open_memmap(
            condition_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(arrays), CLIP_TOKENS, CLIP_DIM),
        )
        direct = lowlevel = None
        if split in ("val", "test"):
            direct = np.lib.format.open_memmap(
                out["cache"] / f"{args.run_name}_{split}_direct_tokens.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(arrays), CLIP_TOKENS, CLIP_DIM),
            )
            lowlevel = np.lib.format.open_memmap(
                out["cache"] / f"{args.run_name}_{split}_lowlevel_latents.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(arrays), 4, 28, 28),
            )
        with torch.no_grad():
            for start in tqdm(
                range(0, len(arrays), args.batch_size),
                desc=f"condition {split}",
            ):
                indices = np.arange(start, min(start + args.batch_size, len(arrays)))
                batch = arrays.batch(indices, device)
                with torch.autocast(
                    "cuda", dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    if args.ablate == "modes":
                        latent = model.fusion.cifti_mean(batch["cifti"])[:, None]
                        brain_condition, projected, low = model.backbone(latent)
                    elif args.ablate == "cifti":
                        if not hasattr(model.fusion, "modes_mean"):
                            raise ValueError("This fusion has no modes expert to retain")
                        latent = model.fusion.modes_mean(batch["modes"])[:, None]
                        brain_condition, projected, low = model.backbone(latent)
                    else:
                        brain_condition, projected, low, _ = model(
                            batch["cifti"], batch["modes"]
                        )
                stop = start + len(indices)
                condition[start:stop] = (
                    brain_condition.float().cpu().numpy().astype(np.float16)
                )
                if direct is not None and lowlevel is not None:
                    direct[start:stop] = (
                        projected.float().cpu().numpy().astype(np.float16)
                    )
                    lowlevel[start:stop] = (
                        low[0].float().cpu().numpy().astype(np.float16)
                    )
        condition.flush()
        if direct is not None and lowlevel is not None:
            direct.flush()
            lowlevel.flush()
            report[f"{split}_direct_token_metrics"] = base.token_identification(
                direct, arrays.bigg, device, args.identification_batch_size
            )
            report[f"{split}_lowlevel_latent_metrics"] = lowlevel_metrics(
                lowlevel, arrays.vae
            )
        report["splits"][split] = {
            "count": len(arrays),
            "condition_path": str(condition_path),
        }
    write_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)


def make_diffusion_prior() -> nn.Module:
    sys.path.insert(0, str(base.MINDEYE_SRC))
    from models import BrainDiffusionPrior, PriorNetwork

    prior_network = PriorNetwork(
        dim=CLIP_DIM,
        depth=6,
        dim_head=52,
        heads=CLIP_DIM // 52,
        causal=False,
        num_tokens=CLIP_TOKENS,
        learned_query_mode="pos_emb",
    )
    return BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=CLIP_DIM,
        condition_on_text_encodings=False,
        timesteps=100,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    )


def build_prior(device: torch.device, checkpoint: Path | None = None) -> nn.Module:
    prior = make_diffusion_prior()
    if checkpoint is None:
        saved = torch.load(
            source_checkpoint("original_poe"),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        state = {
            key.removeprefix("diffusion_prior."): value
            for key, value in saved["model"].items()
            if key.startswith("diffusion_prior.")
        }
    else:
        saved = torch.load(
            checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        state = saved["prior"]
    prior.load_state_dict(state, strict=True)
    del saved, state
    return prior.to(device)


class PriorArrays:
    def __init__(self, prefix: str, split: str):
        local = SHM_CACHE
        self.condition = np.load(
            local / f"{prefix}_condition_{split}.npy", mmap_mode="r"
        )
        local_target = local / f"bigg_tokens_{split}.npy"
        target = (
            local_target
            if local_target.exists()
            else BASE_ROOT / "cache" / f"bigg_tokens_{split}.npy"
        )
        self.target = np.load(target, mmap_mode="r")
        if len(self.condition) != len(self.target):
            raise RuntimeError(
                f"Prior condition/target mismatch: {len(self.condition)} != {len(self.target)}"
            )

    def __len__(self) -> int:
        return len(self.condition)

    def batch(self, indices: np.ndarray, device: torch.device):
        condition = torch.from_numpy(
            np.asarray(self.condition[indices], dtype=np.float32)
        ).to(device)
        target = torch.from_numpy(
            np.asarray(self.target[indices], dtype=np.float32)
        ).to(device)
        return condition, target


def validate_prior(
    prior: nn.Module,
    arrays: PriorArrays,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    prior.eval()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    loss_sum = cosine_sum = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(arrays), batch_size):
            indices = np.arange(start, min(start + batch_size, len(arrays)))
            condition, target = arrays.batch(indices, device)
            with torch.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                loss, prediction = prior(
                    text_embed=condition, image_embed=target
                )
            cosine = F.cosine_similarity(
                prediction.float().flatten(1), target.float().flatten(1), dim=-1
            ).mean()
            loss_sum += float(loss.float().cpu()) * len(indices)
            cosine_sum += float(cosine.cpu()) * len(indices)
            count += len(indices)
    prior.train()
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    return {
        "loss": loss_sum / count,
        "prediction_cosine": cosine_sum / count,
    }


def save_prior(path: Path, prior: nn.Module, metadata: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"prior": compact_state(prior), "metadata": metadata}, temporary)
    temporary.replace(path)


def cmd_train_prior(args: argparse.Namespace) -> None:
    root = Path(args.result_dir)
    out = directories(root)
    device = device_from_arg(args.device)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    train_arrays = PriorArrays(args.condition_prefix, "train")
    val_arrays = PriorArrays(args.condition_prefix, "val")
    prior = build_prior(device).train()
    train_count = len(train_arrays)
    if args.max_train_samples > 0:
        train_count = min(train_count, args.max_train_samples)
    optimizer_options: dict[str, Any] = {}
    if device.type == "cuda":
        optimizer_options["fused"] = True
    optimizer = torch.optim.AdamW(
        prior.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay,
        **optimizer_options,
    )
    steps_per_epoch = math.ceil(train_count / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        total_steps=args.epochs * steps_per_epoch,
        pct_start=min(0.2, 2.0 / args.epochs),
        final_div_factor=1000,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    run_name = args.run_name or f"r3_{args.condition_prefix}"
    temporary_checkpoint = Path("/dev/shm") / f"{TEMP_PREFIX}{run_name}_prior_best.pt"
    final_checkpoint = out["checkpoints"] / f"{run_name}_prior_best.pt"
    config = {
        "round": 3,
        "run_name": run_name,
        "condition_prefix": args.condition_prefix,
        "initialization": "same-subject baseline diffusion prior",
        "brain_model_frozen": True,
        "train_count": train_count,
        "validation_count": len(val_arrays),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_lr": args.max_lr,
        "seed": args.seed,
    }
    write_json(out["metrics"] / f"{run_name}_config.json", config)
    print(json.dumps(config, indent=2), flush=True)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    started = time.time()
    for epoch in range(args.epochs):
        prior.train()
        permutation = np.random.permutation(len(train_arrays))[:train_count]
        loss_sum = 0.0
        count = 0
        progress = tqdm(
            range(0, train_count, args.batch_size),
            desc=f"{run_name} epoch {epoch + 1}",
        )
        for start in progress:
            indices = permutation[start : start + args.batch_size]
            condition, target = train_arrays.batch(indices, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                loss, _ = prior(text_embed=condition, image_embed=target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(prior.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.float().detach().cpu()) * len(indices)
            count += len(indices)
            if start % (args.batch_size * 20) == 0:
                progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
        validation = validate_prior(
            prior,
            val_arrays,
            device,
            args.eval_batch_size,
            args.seed + 10000,
        )
        item = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / count,
            "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(item)
        write_json(
            out["logs"] / f"{run_name}_history.json",
            {"config": config, "history": history},
        )
        if validation["loss"] < best_loss - args.min_delta:
            best_loss = validation["loss"]
            best_epoch = epoch + 1
            bad_epochs = 0
            save_prior(
                temporary_checkpoint,
                prior,
                {
                    **config,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                },
            )
        else:
            bad_epochs += 1
        print(
            f"[{run_name}] epoch={epoch + 1} train={loss_sum / count:.6f} "
            f"val={validation['loss']:.6f} cosine={validation['prediction_cosine']:.6f} "
            f"best={best_loss:.6f} bad={bad_epochs}",
            flush=True,
        )
        if epoch + 1 >= args.min_epochs and bad_epochs >= args.patience:
            break
    atomic_copy(temporary_checkpoint, final_checkpoint)
    summary = {
        **config,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "last_epoch": len(history),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(final_checkpoint),
        "temporary_checkpoint": str(temporary_checkpoint),
    }
    write_json(out["metrics"] / f"{run_name}_prior_results.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def cmd_evaluate_prior(args: argparse.Namespace) -> None:
    root = Path(args.result_dir)
    out = directories(root)
    device = device_from_arg(args.device)
    set_seed(args.seed)
    checkpoint = Path(args.prior_checkpoint)
    prior = build_prior(device, checkpoint).eval().requires_grad_(False)
    arrays = PriorArrays(args.condition_prefix, args.split)
    prediction_path = (
        out["cache"] / f"{args.run_name}_{args.split}_prior_tokens.npy"
    )
    predictions = np.lib.format.open_memmap(
        prediction_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(arrays), CLIP_TOKENS, CLIP_DIM),
    )
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.float16, enabled=device.type == "cuda"
    ):
        for start in tqdm(
            range(0, len(arrays), args.batch_size), desc=f"prior {args.split}"
        ):
            stop = min(start + args.batch_size, len(arrays))
            set_seed(args.seed + start)
            condition = torch.from_numpy(
                np.asarray(arrays.condition[start:stop], dtype=np.float32)
            ).to(device)
            prediction = prior.p_sample_loop(
                condition.shape,
                text_cond={"text_embed": condition},
                cond_scale=args.cond_scale,
                timesteps=args.steps,
            )
            predictions[start:stop] = (
                prediction.float().cpu().numpy().astype(np.float16)
            )
    predictions.flush()
    metrics = base.token_identification(
        predictions, arrays.target, device, args.identification_batch_size
    )
    report = {
        "run_name": args.run_name,
        "condition_prefix": args.condition_prefix,
        "prior_checkpoint": str(checkpoint),
        "split": args.split,
        "count": len(arrays),
        "steps": args.steps,
        "condition_scale": args.cond_scale,
        "diffusion_prior_token_metrics": metrics,
        "prediction_path": str(prediction_path),
    }
    write_json(
        out["metrics"] / f"{args.run_name}_{args.split}_prior_metrics.json",
        report,
    )
    print(json.dumps(report, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--result-dir", default=str(DEFAULT_ROOT))
    sub = value.add_subparsers(dest="command", required=True)

    command = sub.add_parser("prepare")
    command.add_argument("--vertices", default=str(NEW_VERTICES))
    command.add_argument("--batch-size", type=int, default=128)
    command.set_defaults(func=cmd_prepare)

    command = sub.add_parser("train-stage1")
    command.add_argument("--variant", choices=FUSION_VARIANTS, required=True)
    command.add_argument("--run-name")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--epochs", type=int, default=30)
    command.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Use only this many fitting samples; zero uses all samples.",
    )
    command.add_argument("--min-epochs", type=int, default=12)
    command.add_argument("--patience", type=int, default=8)
    command.add_argument("--min-delta", type=float, default=1e-5)
    command.add_argument("--batch-size", type=int, default=48)
    command.add_argument("--eval-batch-size", type=int, default=48)
    command.add_argument("--identification-batch-size", type=int, default=16)
    command.add_argument("--skip-test", action="store_true")
    command.add_argument("--max-lr", type=float, default=3e-4)
    command.add_argument("--weight-decay", type=float, default=1e-2)
    command.add_argument("--grad-clip", type=float, default=1.0)
    command.add_argument("--mixup-fraction", type=float, default=0.33)
    command.add_argument("--convnext-weight", type=float, default=0.05)
    command.add_argument("--seed", type=int, default=42)
    command.set_defaults(func=cmd_train_stage1)

    command = sub.add_parser("evaluate-stage1")
    command.add_argument("--variant", choices=FUSION_VARIANTS, required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--run-name")
    command.add_argument("--split", choices=("val", "test"), default="test")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--eval-batch-size", type=int, default=48)
    command.add_argument("--identification-batch-size", type=int, default=16)
    command.add_argument("--seed", type=int, default=42)
    command.set_defaults(func=cmd_evaluate_stage1)

    command = sub.add_parser("train-stage2")
    command.add_argument("--variant", choices=FUSION_VARIANTS, required=True)
    command.add_argument("--stage1-checkpoint", required=True)
    command.add_argument("--resume-stage2")
    command.add_argument("--run-name")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--epochs", type=int, default=10)
    command.add_argument("--min-epochs", type=int, default=5)
    command.add_argument("--patience", type=int, default=4)
    command.add_argument("--min-delta", type=float, default=1e-5)
    command.add_argument("--batch-size", type=int, default=24)
    command.add_argument("--eval-batch-size", type=int, default=48)
    command.add_argument("--max-train-samples", type=int, default=0)
    command.add_argument(
        "--train-scope", choices=("all", "selective"), default="all"
    )
    command.add_argument("--fusion-lr", type=float, default=5e-5)
    command.add_argument("--backbone-lr", type=float, default=1e-5)
    command.add_argument("--weight-decay", type=float, default=1e-2)
    command.add_argument("--grad-clip", type=float, default=1.0)
    command.add_argument("--mixup-fraction", type=float, default=0.0)
    command.add_argument("--convnext-weight", type=float, default=0.02)
    command.add_argument("--seed", type=int, default=42)
    command.set_defaults(func=cmd_train_stage2)

    command = sub.add_parser("cache-conditions")
    command.add_argument("--variant", choices=FUSION_VARIANTS, required=True)
    command.add_argument("--brain-checkpoint", required=True)
    command.add_argument(
        "--checkpoint-type", choices=("stage1", "stage2"), required=True
    )
    command.add_argument("--run-name", required=True)
    command.add_argument(
        "--ablate", choices=("none", "modes", "cifti"), default="none"
    )
    command.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test")
    )
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--batch-size", type=int, default=48)
    command.add_argument("--identification-batch-size", type=int, default=16)
    command.add_argument("--seed", type=int, default=42)
    command.set_defaults(func=cmd_cache_conditions)

    command = sub.add_parser("train-prior")
    command.add_argument("--condition-prefix", required=True)
    command.add_argument("--run-name")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--epochs", type=int, default=12)
    command.add_argument("--min-epochs", type=int, default=6)
    command.add_argument("--patience", type=int, default=4)
    command.add_argument("--min-delta", type=float, default=1e-5)
    command.add_argument("--batch-size", type=int, default=16)
    command.add_argument("--eval-batch-size", type=int, default=16)
    command.add_argument("--max-train-samples", type=int, default=0)
    command.add_argument("--max-lr", type=float, default=1e-5)
    command.add_argument("--weight-decay", type=float, default=1e-2)
    command.add_argument("--grad-clip", type=float, default=1.0)
    command.add_argument("--seed", type=int, default=42)
    command.set_defaults(func=cmd_train_prior)

    command = sub.add_parser("evaluate-prior")
    command.add_argument("--condition-prefix", required=True)
    command.add_argument("--prior-checkpoint", required=True)
    command.add_argument("--run-name", required=True)
    command.add_argument("--split", choices=("val", "test"), default="test")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--batch-size", type=int, default=4)
    command.add_argument("--identification-batch-size", type=int, default=16)
    command.add_argument("--steps", type=int, default=20)
    command.add_argument("--cond-scale", type=float, default=1.0)
    command.add_argument("--seed", type=int, default=42)
    command.set_defaults(func=cmd_evaluate_prior)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
