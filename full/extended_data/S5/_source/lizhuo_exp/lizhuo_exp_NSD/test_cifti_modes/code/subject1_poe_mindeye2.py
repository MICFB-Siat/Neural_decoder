







from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))

import argparse
import gc
import json
import math
import os
import random
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


NSD_ROOT = Path(str(_LOCAL_SOURCE / 'lizhuo_exp/lizhuo_exp_NSD'))
CODE_ROOT = NSD_ROOT / "test_cifti_modes/code"
DEFAULT_ROOT = NSD_ROOT / "test_cifti_modes/result/subject1_poe_mindeye2_1024"
DATA_PATH = Path('/media/wsqlab/nas2/Dataset/NSD-download/preprocess/stage2/sub-01.h5')
MINDEYE_ROOT = NSD_ROOT / "fuxian/code/MindEyeV2"
MINDEYE_SRC = MINDEYE_ROOT / "src"
ARTIFACT_ROOT = Path('/media/wsqlab/nas2/Dataset/NSD-download/mindeyev2_hf_artifacts')
RCLONE_ARTIFACT_CACHE = Path(
    "/home/guoyi/.cache/rclone/vfs/nas2/share/Dataset/NSD-download/mindeyev2_hf_artifacts"
)
UNCLIP_CHECKPOINT_FAST = Path("/dev/shm/guoyi_mindeye2_unclip6.ckpt")
RELIABLE_VERTICES = NSD_ROOT / "test_cifti_modes/result/cache/cifti_ncsnr_top4096_vertices.npy"
RELIABLE_CIFTI_TRAIN = NSD_ROOT / "test_cifti_modes/result/cache/cifti_ncsnr_top4096_train.npy"
RELIABLE_CIFTI_TEST = NSD_ROOT / "test_cifti_modes/result/cache/cifti_ncsnr_top4096_test.npy"

CIFTI_DIM = 4096
MODES_DIM = 2000
HIDDEN_DIM = 1024
CLIP_TOKENS = 256
CLIP_DIM = 1664
VARIANTS = ("poe", "cifti", "modes")
SUBJECT_ID = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def directories(root: Path) -> dict[str, Path]:
    value = {
        "root": root,
        "cache": root / "cache",
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
        "metrics": root / "metrics",
    }
    for path in value.values():
        path.mkdir(parents=True, exist_ok=True)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def device_from_arg(value: str) -> torch.device:
    return torch.device(value if torch.cuda.is_available() else "cpu")


def resolve_artifact(name: str, preferred: Path | None = None) -> Path:
    candidates = [preferred, ARTIFACT_ROOT / name, RCLONE_ARTIFACT_CACHE / name]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        except OSError:
            continue
    raise FileNotFoundError(f"Missing required artifact: {name}")


def split_file(root: Path) -> Path:
    return directories(root)["cache"] / "split_and_normalization.npz"


def load_split(root: Path) -> dict[str, np.ndarray]:
    path = split_file(root)
    if not path.exists():
        raise FileNotFoundError(f"Run prepare first: {path}")
    with np.load(path) as value:
        return {name: value[name] for name in value.files}


def _write_input_split(
    cifti_source: np.ndarray,
    modes_source: np.ndarray,
    positions: np.ndarray,
    cifti_mean: np.ndarray,
    cifti_std: np.ndarray,
    modes_mean: np.ndarray,
    modes_std: np.ndarray,
    cache: Path,
    name: str,
    batch_size: int,
) -> None:
    cifti_path = cache / f"cifti_{name}.npy"
    modes_path = cache / f"modes_{name}.npy"
    cifti_out = np.lib.format.open_memmap(
        cifti_path, mode="w+", dtype=np.float16, shape=(len(positions), CIFTI_DIM)
    )
    modes_out = np.lib.format.open_memmap(
        modes_path, mode="w+", dtype=np.float16, shape=(len(positions), MODES_DIM)
    )
    for start in tqdm(range(0, len(positions), batch_size), desc=f"inputs {name}"):
        stop = min(start + batch_size, len(positions))
        selected = positions[start:stop]
        cifti = np.asarray(cifti_source[selected], dtype=np.float32)
        modes = np.asarray(modes_source[selected], dtype=np.float32)
        cifti_out[start:stop] = ((cifti - cifti_mean) / cifti_std).astype(np.float16)
        modes_out[start:stop] = ((modes - modes_mean) / modes_std).astype(np.float16)
    cifti_out.flush()
    modes_out.flush()


def cmd_prepare(args: argparse.Namespace) -> None:
    root = Path(args.result_dir)
    out = directories(root)
    set_seed(args.seed)
    vertices = np.load(RELIABLE_VERTICES).astype(np.int64)
    if len(vertices) != CIFTI_DIM:
        raise RuntimeError(f"Expected {CIFTI_DIM} reliable vertices, got {len(vertices)}")

    with h5py.File(DATA_PATH, "r") as h5:
        shared = h5["is_shared1000"][:].astype(bool)
        stim_ids = h5["stim_id"][:].astype(np.int64)
        nonshared_rows = np.where(~shared)[0]
        test_rows = np.where(shared)[0]
        rng = np.random.RandomState(args.seed)
        permutation = rng.permutation(len(nonshared_rows))
        val_count = int(round(len(nonshared_rows) * args.val_fraction))
        train_positions = np.sort(permutation[val_count:])
        val_positions = np.sort(permutation[:val_count])
        train_rows = nonshared_rows[train_positions]
        val_rows = nonshared_rows[val_positions]

        cifti_nonshared = np.load(RELIABLE_CIFTI_TRAIN, mmap_mode="r")
        cifti_test = np.load(RELIABLE_CIFTI_TEST, mmap_mode="r")
        modes_nonshared = h5["fmri_modes"][nonshared_rows].astype(np.float32)
        modes_test = h5["fmri_modes"][test_rows].astype(np.float32)
        cifti_train = np.asarray(cifti_nonshared[train_positions], dtype=np.float32)
        modes_train = modes_nonshared[train_positions]
        cifti_mean = cifti_train.mean(axis=0, keepdims=True).astype(np.float32)
        cifti_std = (cifti_train.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        modes_mean = modes_train.mean(axis=0, keepdims=True).astype(np.float32)
        modes_std = (modes_train.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        del cifti_train, modes_train

        np.savez(
            out["cache"] / "split_and_normalization.npz",
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            train_stim_ids=stim_ids[train_rows],
            val_stim_ids=stim_ids[val_rows],
            test_stim_ids=stim_ids[test_rows],
            reliable_vertices=vertices,
            cifti_mean=cifti_mean,
            cifti_std=cifti_std,
            modes_mean=modes_mean,
            modes_std=modes_std,
        )
        sources = {
            "train": (cifti_nonshared, modes_nonshared, train_positions),
            "val": (cifti_nonshared, modes_nonshared, val_positions),
            "test": (cifti_test, modes_test, np.arange(len(test_rows))),
        }
        for name, (cifti_source, modes_source, positions) in sources.items():
            _write_input_split(
                cifti_source,
                modes_source,
                positions,
                cifti_mean,
                cifti_std,
                modes_mean,
                modes_std,
                out["cache"],
                name,
                args.batch_size,
            )
        n_rep = h5["n_rep"][:]

    report = {
        "protocol": f"Subject-{SUBJECT_ID} only, no MindEye2 brain-model distillation or cross-subject fMRI",
        "data": str(DATA_PATH),
        "train_count": int(len(train_rows)),
        "validation_count": int(len(val_rows)),
        "test_count": int(len(test_rows)),
        "test_name": "NSD Shared1000",
        "normalization": "feature-wise statistics from the 8100 fit images only",
        "cifti_selection": "top-4096 train-repeat-reliability vertices; no validation/test responses used",
        "cifti_dim": CIFTI_DIM,
        "modes_dim": MODES_DIM,
        "repeat_count_values": np.unique(n_rep).astype(int).tolist(),
        "stage2_representation": "one response per image, averaged over all available repetitions",
        "seed": args.seed,
    }
    write_json(out["metrics"] / "data_protocol.json", report)
    print(json.dumps(report, indent=2))


def _load_bigg(device: torch.device):
    sys.path.insert(0, str(MINDEYE_SRC))
    from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

    model = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
        cache_dir=str(Path.home() / ".cache/huggingface/hub"),
    )
    return model.to(device).eval().requires_grad_(False)


def _load_image_vae(device: torch.device):
    from diffusers import AutoencoderKL

    model = AutoencoderKL(
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    model.load_state_dict(
        torch.load(
            resolve_artifact("sd_image_var_autoenc.pth"),
            map_location="cpu",
            weights_only=False,
        )
    )
    return model.to(device=device, dtype=torch.float16).eval().requires_grad_(False)


def _cache_bigg_and_vae(
    root: Path,
    split_name: str,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> None:
    out = directories(root)
    token_path = out["cache"] / f"bigg_tokens_{split_name}.npy"
    vae_path = out["cache"] / f"vae_latents_{split_name}.npy"
    need_tokens = force or not token_path.exists()
    need_vae = force or not vae_path.exists()
    if not need_tokens and not need_vae:
        print(f"[targets:{split_name}] BigG and VAE caches already exist")
        return

    bigg = _load_bigg(device) if need_tokens else None
    vae = _load_image_vae(device) if need_vae else None
    token_out = (
        np.lib.format.open_memmap(
            token_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(rows), CLIP_TOKENS, CLIP_DIM),
        )
        if need_tokens
        else None
    )
    vae_out = (
        np.lib.format.open_memmap(
            vae_path, mode="w+", dtype=np.float16, shape=(len(rows), 4, 28, 28)
        )
        if need_vae
        else None
    )
    with h5py.File(DATA_PATH, "r") as h5, torch.no_grad():
        for start in tqdm(range(0, len(rows), batch_size), desc=f"image targets {split_name}"):
            stop = min(start + batch_size, len(rows))
            image = torch.from_numpy(h5["labels"][rows[start:stop]]).permute(0, 3, 1, 2)
            image = image.to(device=device, dtype=torch.float16) / 127.5 - 1.0
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                if bigg is not None:
                    tokens = bigg(image)
                    token_out[start:stop] = tokens.float().cpu().numpy().astype(np.float16)
                if vae is not None:
                    resized = F.interpolate(
                        image, size=(224, 224), mode="bicubic", align_corners=False, antialias=True
                    )
                    latent = vae.encode(resized).latent_dist.mode() * 0.18215
                    vae_out[start:stop] = latent.float().cpu().numpy().astype(np.float16)
    if token_out is not None:
        token_out.flush()
    if vae_out is not None:
        vae_out.flush()
    del bigg, vae, token_out, vae_out
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _cache_convnext(
    root: Path,
    split_name: str,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
    force: bool,
    seed: int,
) -> None:
    out = directories(root)
    original_path = out["cache"] / f"convnext_original_{split_name}.npy"
    augmented_path = out["cache"] / f"convnext_augmented_{split_name}.npy"
    if not force and original_path.exists() and augmented_path.exists():
        print(f"[targets:{split_name}] ConvNeXt caches already exist")
        return
    checkpoint = resolve_artifact("convnext_xlarge_alpha0.75_fullckpt.pth")

    sys.path.insert(0, str(MINDEYE_SRC))
    import kornia
    from kornia.augmentation.container import AugmentationSequential
    from autoencoder.convnext import ConvnextXL

    model = ConvnextXL(str(checkpoint)).to(device).eval().requires_grad_(False)
    augment = AugmentationSequential(
        kornia.augmentation.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        kornia.augmentation.RandomGrayscale(p=0.1),
        kornia.augmentation.RandomSolarize(p=0.1),
        kornia.augmentation.RandomResizedCrop(
            (224, 224), scale=(0.9, 0.9), ratio=(1.0, 1.0), p=1.0
        ),
        data_keys=["input"],
    ).to(device)
    original_out = np.lib.format.open_memmap(
        original_path, mode="w+", dtype=np.float16, shape=(len(rows), 49, 512)
    )
    augmented_out = np.lib.format.open_memmap(
        augmented_path, mode="w+", dtype=np.float16, shape=(len(rows), 49, 512)
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.228, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    set_seed(seed)
    with h5py.File(DATA_PATH, "r") as h5, torch.no_grad():
        for start in tqdm(range(0, len(rows), batch_size), desc=f"ConvNeXt {split_name}"):
            stop = min(start + batch_size, len(rows))
            image = torch.from_numpy(h5["labels"][rows[start:stop]]).permute(0, 3, 1, 2)
            image = image.to(device=device, dtype=torch.float32) / 255.0
            image = F.interpolate(
                image, size=(224, 224), mode="bicubic", align_corners=False, antialias=True
            )
            image_aug = augment(image)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                _, original = model((image - mean) / std)
                _, augmented = model((image_aug - mean) / std)
            original_out[start:stop] = original.float().cpu().numpy().astype(np.float16)
            augmented_out[start:stop] = augmented.float().cpu().numpy().astype(np.float16)
    original_out.flush()
    augmented_out.flush()
    del model, original_out, augmented_out
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def cmd_cache_targets(args: argparse.Namespace) -> None:
    root = Path(args.result_dir)
    out = directories(root)
    split = load_split(root)
    device = device_from_arg(args.device)
    for name in args.splits:
        rows = split[f"{name}_rows"]
        _cache_bigg_and_vae(root, name, rows, device, args.batch_size, args.force)
    if args.with_convnext:
        for name in args.splits:
            _cache_convnext(
                root,
                name,
                split[f"{name}_rows"],
                device,
                args.convnext_batch_size,
                args.force,
                args.seed,
            )
    report = {
        "splits": args.splits,
        "bigg": "Frozen OpenCLIP ViT-bigG-14 laion2b_s39b_b160k, 256 x 1664 tokens",
        "vae": "Frozen sd-image-variations VAE, 4 x 28 x 28 mode latent",
        "convnext": bool(args.with_convnext),
        "frozen_image_teachers": True,
    }
    write_json(out["metrics"] / "target_cache.json", report)
    print(json.dumps(report, indent=2))


class LinearGaussianPoE(nn.Module):
    def __init__(
        self,
        variant: str,
        hidden_dim: int = HIDDEN_DIM,
        logvar_min: float = -4.0,
        logvar_max: float = 4.0,
        both_probability: float = 0.6,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.hidden_dim = hidden_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.both_probability = both_probability
        if variant in ("poe", "cifti"):
            self.cifti_mean = nn.Linear(CIFTI_DIM, hidden_dim)
            self.cifti_logvar = nn.Linear(CIFTI_DIM, hidden_dim)
            nn.init.zeros_(self.cifti_logvar.weight)
            nn.init.constant_(self.cifti_logvar.bias, math.log(2.0))
        if variant in ("poe", "modes"):
            self.modes_mean = nn.Linear(MODES_DIM, hidden_dim)
            self.modes_logvar = nn.Linear(MODES_DIM, hidden_dim)
            nn.init.zeros_(self.modes_logvar.weight)
            nn.init.constant_(self.modes_logvar.bias, math.log(2.0))

    def forward(
        self,
        cifti: torch.Tensor,
        modes: torch.Tensor,
        sample: bool,
        modality_dropout: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = len(cifti)
        dtype = cifti.dtype
        device = cifti.device
        total_precision = torch.ones(batch, self.hidden_dim, device=device, dtype=dtype)
        weighted_mean = torch.zeros_like(total_precision)
        statistics: dict[str, torch.Tensor] = {}

        if self.variant == "poe" and self.training and modality_dropout:
            draw = torch.rand(batch, device=device)
            side_probability = (1.0 - self.both_probability) / 2.0
            cifti_mask = (draw < self.both_probability + side_probability).to(dtype)[:, None]
            modes_mask = ((draw < self.both_probability) | (draw >= 1.0 - side_probability)).to(dtype)[:, None]
        else:
            cifti_mask = torch.ones(batch, 1, device=device, dtype=dtype)
            modes_mask = torch.ones(batch, 1, device=device, dtype=dtype)

        if self.variant in ("poe", "cifti"):
            cifti_mean = self.cifti_mean(cifti)
            cifti_logvar = self.cifti_logvar(cifti).clamp(self.logvar_min, self.logvar_max)
            cifti_precision = torch.exp(-cifti_logvar) * cifti_mask
            total_precision = total_precision + cifti_precision
            weighted_mean = weighted_mean + cifti_mean * cifti_precision
            statistics.update(
                cifti_mean=cifti_mean,
                cifti_logvar=cifti_logvar,
                cifti_precision=cifti_precision,
                cifti_active=cifti_mask,
            )
        if self.variant in ("poe", "modes"):
            modes_mean = self.modes_mean(modes)
            modes_logvar = self.modes_logvar(modes).clamp(self.logvar_min, self.logvar_max)
            modes_precision = torch.exp(-modes_logvar) * modes_mask
            total_precision = total_precision + modes_precision
            weighted_mean = weighted_mean + modes_mean * modes_precision
            statistics.update(
                modes_mean=modes_mean,
                modes_logvar=modes_logvar,
                modes_precision=modes_precision,
                modes_active=modes_mask,
            )

        fused_variance = total_precision.reciprocal()
        fused_mean = fused_variance * weighted_mean
        fused_logvar = torch.log(fused_variance)
        latent = fused_mean
        if sample:
            latent = fused_mean + torch.randn_like(fused_mean) * torch.sqrt(fused_variance)
        statistics.update(
            fused_mean=fused_mean,
            fused_logvar=fused_logvar,
            fused_variance=fused_variance,
            total_precision=total_precision,
        )
        return latent[:, None], statistics


class Subject1PoEDecoder(nn.Module):
    def __init__(
        self,
        variant: str,
        hidden_dim: int = HIDDEN_DIM,
        shared_initialization_seed: int = 20260710,
    ) -> None:
        super().__init__()
        sys.path.insert(0, str(MINDEYE_SRC))
        from models import BrainDiffusionPrior, BrainNetwork, PriorNetwork

        self.fusion = LinearGaussianPoE(variant=variant, hidden_dim=hidden_dim)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(shared_initialization_seed)
            self.backbone = BrainNetwork(
                h=hidden_dim,
                in_dim=hidden_dim,
                seq_len=1,
                clip_size=CLIP_DIM,
                out_dim=CLIP_DIM * CLIP_TOKENS,
                n_blocks=4,
                blurry_recon=True,
            )
            prior_network = PriorNetwork(
                dim=CLIP_DIM,
                depth=6,
                dim_head=52,
                heads=CLIP_DIM // 52,
                causal=False,
                num_tokens=CLIP_TOKENS,
                learned_query_mode="pos_emb",
            )
            self.diffusion_prior = BrainDiffusionPrior(
                net=prior_network,
                image_embed_dim=CLIP_DIM,
                condition_on_text_encodings=False,
                timesteps=100,
                cond_drop_prob=0.2,
                image_embed_scale=None,
            )

    def brain_outputs(
        self,
        cifti: torch.Tensor,
        modes: torch.Tensor,
        sample: bool,
        modality_dropout: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
        latent, statistics = self.fusion(cifti, modes, sample, modality_dropout)
        condition, projected, lowlevel = self.backbone(latent)
        return condition, projected, lowlevel, statistics


class SplitArrays:
    def __init__(self, root: Path, split_name: str, convnext: bool) -> None:
        cache_override = os.environ.get("MINDEYE2_BASELINE_CACHE_DIR")
        cache = Path(cache_override) if cache_override else directories(root)["cache"]
        self.cifti = np.load(cache / f"cifti_{split_name}.npy", mmap_mode="r")
        self.modes = np.load(cache / f"modes_{split_name}.npy", mmap_mode="r")
        self.bigg = np.load(cache / f"bigg_tokens_{split_name}.npy", mmap_mode="r")
        self.vae = np.load(cache / f"vae_latents_{split_name}.npy", mmap_mode="r")
        self.convnext_original = None
        self.convnext_augmented = None
        if convnext:
            self.convnext_original = np.load(
                cache / f"convnext_original_{split_name}.npy", mmap_mode="r"
            )
            self.convnext_augmented = np.load(
                cache / f"convnext_augmented_{split_name}.npy", mmap_mode="r"
            )
        lengths = {len(self.cifti), len(self.modes), len(self.bigg), len(self.vae)}
        if len(lengths) != 1:
            raise RuntimeError(f"Misaligned {split_name} caches: {lengths}")

    def __len__(self) -> int:
        return len(self.cifti)

    def batch(self, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
        value = {
            "cifti": torch.from_numpy(np.asarray(self.cifti[indices], dtype=np.float32)).to(device),
            "modes": torch.from_numpy(np.asarray(self.modes[indices], dtype=np.float32)).to(device),
            "bigg": torch.from_numpy(np.asarray(self.bigg[indices], dtype=np.float32)).to(device),
            "vae": torch.from_numpy(np.asarray(self.vae[indices], dtype=np.float32)).to(device),
        }
        if self.convnext_original is not None and self.convnext_augmented is not None:
            value["convnext_original"] = torch.from_numpy(
                np.asarray(self.convnext_original[indices], dtype=np.float32)
            ).to(device)
            value["convnext_augmented"] = torch.from_numpy(
                np.asarray(self.convnext_augmented[indices], dtype=np.float32)
            ).to(device)
        return value


def apply_mixco(
    cifti: torch.Tensor,
    modes: torch.Tensor,
    beta_parameter: float = 0.15,
    selection_probability: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    permutation = torch.randperm(len(cifti), device=cifti.device)
    beta = torch.distributions.Beta(beta_parameter, beta_parameter).sample((len(cifti),)).to(cifti.device)
    selected = torch.rand(len(cifti), device=cifti.device) <= selection_probability
    beta[~selected] = 1.0
    shape = (-1, 1)
    cifti = cifti * beta.reshape(shape) + cifti[permutation] * (1.0 - beta).reshape(shape)
    modes = modes * beta.reshape(shape) + modes[permutation] * (1.0 - beta).reshape(shape)
    return cifti, modes, permutation, beta, selected


def mixco_nce_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    permutation: torch.Tensor,
    beta: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    similarity = prediction @ target.T / temperature
    probabilities = torch.diag(beta)
    probabilities[torch.arange(len(prediction), device=prediction.device), permutation] = 1.0 - beta
    forward = -(similarity.log_softmax(-1) * probabilities).sum(-1).mean()
    backward = -(similarity.T.log_softmax(-1) * probabilities.T).sum(-1).mean()
    return (forward + backward) / 2.0


def soft_clip_loss(prediction: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    teacher = target @ target.T / temperature
    student = prediction @ target.T / temperature
    probabilities = teacher.softmax(-1)
    forward = -(student.log_softmax(-1) * probabilities).sum(-1).mean()
    backward = -(student.T.log_softmax(-1) * probabilities).sum(-1).mean()
    return (forward + backward) / 2.0


def soft_cont_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    augmented_target: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    prediction = F.normalize(prediction.flatten(0, 1), dim=-1)
    target = F.normalize(target.flatten(0, 1), dim=-1)
    augmented_target = F.normalize(augmented_target.flatten(0, 1), dim=-1)
    teacher_forward = target @ augmented_target.T / temperature
    teacher_backward = augmented_target @ target.T / temperature
    student_forward = prediction @ augmented_target.T / temperature
    student_backward = augmented_target @ prediction.T / temperature
    forward = -(student_forward.log_softmax(-1) * teacher_forward.softmax(-1)).sum(-1).mean()
    backward = -(student_backward.log_softmax(-1) * teacher_backward.softmax(-1)).sum(-1).mean()
    return (forward + backward) / 2.0


def gaussian_kl(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mean.square() + logvar.exp() - logvar - 1.0).mean()


def poe_kl_loss(statistics: dict[str, torch.Tensor]) -> torch.Tensor:
    loss = gaussian_kl(statistics["fused_mean"], statistics["fused_logvar"])
    count = 1
    for prefix in ("cifti", "modes"):
        mean_key = f"{prefix}_mean"
        if mean_key in statistics:
            loss = loss + gaussian_kl(statistics[mean_key], statistics[f"{prefix}_logvar"])
            count += 1
    return loss / count


def cosine_temperature(epoch: int, total_epochs: int, mixup_fraction: float) -> float:
    switch = int(mixup_fraction * total_epochs)
    remaining = max(1, total_epochs - switch)
    position = min(max(epoch - switch, 0), remaining - 1)
    if remaining == 1:
        return 0.0075
    return 0.0075 + (0.004 - 0.0075) / 2.0 * (
        1.0 + math.cos(math.pi * position / (remaining - 1))
    )


def compute_losses(
    model: Subject1PoEDecoder,
    batch: dict[str, torch.Tensor],
    epoch: int,
    total_epochs: int,
    mixup_fraction: float,
    train: bool,
    use_convnext: bool,
    prior_weight: float,
    clip_weight: float,
    lowlevel_weight: float,
    convnext_weight: float,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    cifti = batch["cifti"]
    modes = batch["modes"]
    permutation = beta = None
    use_mixco = train and epoch < int(mixup_fraction * total_epochs)
    if use_mixco:
        cifti, modes, permutation, beta, _selected = apply_mixco(cifti, modes)

    condition, projected, lowlevel, statistics = model.brain_outputs(
        cifti,
        modes,
        sample=train,
        modality_dropout=train,
    )
    prior_loss, prior_prediction = model.diffusion_prior(
        text_embed=condition, image_embed=batch["bigg"]
    )
    projected_flat = F.normalize(projected.flatten(1).float(), dim=-1)
    target_flat = F.normalize(batch["bigg"].flatten(1).float(), dim=-1)
    if use_mixco:
        assert permutation is not None and beta is not None
        clip_loss = mixco_nce_loss(projected_flat, target_flat, permutation, beta, 0.006)
    else:
        temperature = 0.006 if not train else cosine_temperature(epoch, total_epochs, mixup_fraction)
        clip_loss = soft_clip_loss(projected_flat, target_flat, temperature)
    latent_loss = F.l1_loss(lowlevel[0].float(), batch["vae"].float())
    structural_loss = torch.zeros((), device=cifti.device)
    if use_convnext:
        structural_loss = soft_cont_loss(
            lowlevel[1].float(),
            batch["convnext_original"].float(),
            batch["convnext_augmented"].float(),
        )
    kl_loss = poe_kl_loss(statistics)
    total = (
        prior_weight * prior_loss.float()
        + clip_weight * clip_loss
        + lowlevel_weight * latent_loss
        + convnext_weight * structural_loss
        + kl_weight * kl_loss
    )
    losses = {
        "total": total,
        "prior": prior_loss.float(),
        "clip": clip_loss,
        "lowlevel_l1": latent_loss,
        "convnext": structural_loss,
        "poe_kl": kl_loss,
    }
    outputs = {
        "condition": condition,
        "projected": projected,
        "prior_prediction": prior_prediction,
        "lowlevel_latent": lowlevel[0],
        "poe_mean": statistics["fused_mean"],
        "poe_variance": statistics["fused_variance"],
    }
    for prefix in ("cifti", "modes"):
        key = f"{prefix}_precision"
        if key in statistics:
            outputs[key] = statistics[key]
    return total, losses, outputs


def trainable_parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("bias") or "norm" in name.lower() or "layernorm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _aggregate_losses(sums: dict[str, float], losses: dict[str, torch.Tensor], count: int) -> None:
    for name, value in losses.items():
        sums[name] = sums.get(name, 0.0) + float(value.detach().cpu()) * count


def _mean_losses(sums: dict[str, float], count: int) -> dict[str, float]:
    return {name: value / max(1, count) for name, value in sums.items()}


def validate(
    model: Subject1PoEDecoder,
    arrays: SplitArrays,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    set_seed(args.validation_seed)
    sums: dict[str, float] = {}
    count = 0
    variance_sum = 0.0
    cifti_precision_sum = 0.0
    modes_precision_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(arrays), args.eval_batch_size):
            indices = np.arange(start, min(start + args.eval_batch_size, len(arrays)))
            batch = arrays.batch(indices, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                _total, losses, outputs = compute_losses(
                    model,
                    batch,
                    epoch,
                    args.epochs,
                    args.mixup_fraction,
                    False,
                    args.use_convnext,
                    args.prior_weight,
                    args.clip_weight,
                    args.lowlevel_weight,
                    args.convnext_weight,
                    args.kl_weight,
                )
            batch_count = len(indices)
            _aggregate_losses(sums, losses, batch_count)
            count += batch_count
            variance_sum += float(outputs["poe_variance"].float().mean().cpu()) * batch_count
            if "cifti_precision" in outputs:
                cifti_precision_sum += float(outputs["cifti_precision"].float().mean().cpu()) * batch_count
            if "modes_precision" in outputs:
                modes_precision_sum += float(outputs["modes_precision"].float().mean().cpu()) * batch_count
    diagnostics = {
        "poe_variance": variance_sum / count,
        "cifti_precision": cifti_precision_sum / count,
        "modes_precision": modes_precision_sum / count,
    }
    model.train()
    return _mean_losses(sums, count), diagnostics


def compact_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu()
        if tensor.is_floating_point() and tensor.numel() >= 4096:
            tensor = tensor.half()
        state[name] = tensor
    return state


def save_model(path: Path, model: nn.Module, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": compact_state_dict(model), "metadata": metadata}, temporary)
    temporary.replace(path)


def cmd_train(args: argparse.Namespace) -> None:
    root = Path(args.result_dir)
    out = directories(root)
    device = device_from_arg(args.device)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_arrays = SplitArrays(root, "train", args.use_convnext)
    val_arrays = SplitArrays(root, "val", args.use_convnext)
    train_count = min(len(train_arrays), args.max_train_samples or len(train_arrays))
    model = Subject1PoEDecoder(
        args.variant,
        hidden_dim=args.hidden_dim,
        shared_initialization_seed=args.shared_initialization_seed,
    ).to(device)
    optimizer = torch.optim.AdamW(
        trainable_parameter_groups(model, args.weight_decay), lr=args.max_lr
    )
    steps_per_epoch = math.ceil(train_count / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        total_steps=max(1, args.epochs * steps_per_epoch),
        final_div_factor=1000,
        pct_start=min(0.3, 2.0 / max(1, args.epochs)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    run_name = args.run_name or args.variant
    checkpoint_path = out["checkpoints"] / f"{run_name}_best.pt"
    history_path = out["logs"] / f"{run_name}_history.json"
    config_path = out["metrics"] / f"{run_name}_config.json"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = {
        "run_name": run_name,
        "variant": args.variant,
        "subject": SUBJECT_ID,
        "cross_subject_pretraining": False,
        "mindeye2_brain_distillation": False,
        "brain_model_initialization": "random",
        "hidden_dim": args.hidden_dim,
        "parameter_count": parameter_count,
        "train_count": train_count,
        "validation_count": len(val_arrays),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_lr": args.max_lr,
        "weight_decay": args.weight_decay,
        "loss_weights": {
            "diffusion_prior": args.prior_weight,
            "direct_clip": args.clip_weight,
            "lowlevel_l1": args.lowlevel_weight,
            "convnext": args.convnext_weight if args.use_convnext else 0.0,
            "poe_kl": args.kl_weight,
        },
        "mixup_fraction": args.mixup_fraction,
        "modality_training_probabilities": {
            "both": 0.6,
            "cifti_only": 0.2,
            "modes_only": 0.2,
        },
        "seed": args.seed,
        "shared_initialization_seed": args.shared_initialization_seed,
    }
    write_json(config_path, config)
    print(json.dumps(config, indent=2), flush=True)

    best_validation = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        permutation = np.random.permutation(len(train_arrays))[:train_count]
        sums: dict[str, float] = {}
        seen = 0
        epoch_started = time.time()
        progress = tqdm(range(0, train_count, args.batch_size), desc=f"{run_name} epoch {epoch + 1}")
        for start in progress:
            indices = permutation[start : start + args.batch_size]
            batch = train_arrays.batch(indices, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                total, losses, _outputs = compute_losses(
                    model,
                    batch,
                    epoch,
                    args.epochs,
                    args.mixup_fraction,
                    True,
                    args.use_convnext,
                    args.prior_weight,
                    args.clip_weight,
                    args.lowlevel_weight,
                    args.convnext_weight,
                    args.kl_weight,
                )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            batch_count = len(indices)
            _aggregate_losses(sums, losses, batch_count)
            seen += batch_count
            if start % (args.batch_size * 20) == 0:
                progress.set_postfix(loss=f"{float(total.detach().cpu()):.4f}")
        train_metrics = _mean_losses(sums, seen)
        val_metrics, diagnostics = validate(model, val_arrays, device, args, epoch)
        item = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": val_metrics,
            "diagnostics": diagnostics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - epoch_started,
        }
        history.append(item)
        write_json(history_path, {"config": config, "history": history})
        validation_score = val_metrics["total"]
        improved = validation_score < best_validation - args.min_delta
        if improved:
            best_validation = validation_score
            best_epoch = epoch + 1
            bad_epochs = 0
            save_model(
                checkpoint_path,
                model,
                {**config, "best_epoch": best_epoch, "best_validation_total": best_validation},
            )
        else:
            bad_epochs += 1
        print(
            f"[{run_name}] epoch={epoch + 1} train={train_metrics['total']:.5f} "
            f"val={validation_score:.5f} best={best_validation:.5f} bad={bad_epochs} "
            f"poe_var={diagnostics['poe_variance']:.4f}",
            flush=True,
        )
        if epoch + 1 >= args.min_epochs and bad_epochs >= args.patience:
            break

    summary = {
        **config,
        "best_epoch": best_epoch,
        "best_validation_total": best_validation,
        "last_epoch": len(history),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(checkpoint_path),
        "last_metrics": history[-1] if history else None,
    }
    write_json(out["metrics"] / f"{run_name}_train.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def load_trained_model(
    checkpoint: Path, device: torch.device
) -> tuple[Subject1PoEDecoder, dict[str, Any]]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    metadata = saved["metadata"]
    model = Subject1PoEDecoder(
        variant=metadata["variant"],
        hidden_dim=int(metadata["hidden_dim"]),
        shared_initialization_seed=int(metadata.get("shared_initialization_seed", 20260710)),
    )
    model.load_state_dict(saved["model"], strict=True)
    del saved
    return model.to(device).eval().requires_grad_(False), metadata


def identification_metrics(similarity: np.ndarray) -> dict[str, float]:
    order = np.argsort(-similarity, axis=1)
    positions = np.empty(len(similarity), dtype=np.int64)
    for index in range(len(similarity)):
        positions[index] = int(np.where(order[index] == index)[0][0])
    diagonal = np.diag(similarity)
    forward = ((similarity < diagonal[:, None]).sum(axis=1) - 0) / max(1, len(similarity) - 1)
    backward = ((similarity < diagonal[None, :]).sum(axis=0) - 0) / max(1, len(similarity) - 1)
    return {
        "R@1": float((positions < 1).mean()),
        "R@5": float((positions < 5).mean()),
        "R@10": float((positions < 10).mean()),
        "median_rank": float(np.median(positions + 1)),
        "mean_cosine": float(diagonal.mean()),
        "fwd_2way": float(forward.mean()),
        "bwd_2way": float(backward.mean()),
    }


def token_identification(
    prediction: np.ndarray, target: np.ndarray, device: torch.device, chunk_size: int = 16
) -> dict[str, float]:
    target_t = torch.from_numpy(np.asarray(target, dtype=np.float16)).to(device).flatten(1)
    target_t = F.normalize(target_t.float(), dim=-1).half()
    similarity = np.empty((len(prediction), len(target)), dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, len(prediction), chunk_size), desc="token identification"):
            stop = min(start + chunk_size, len(prediction))
            block = torch.from_numpy(np.asarray(prediction[start:stop], dtype=np.float16)).to(device)
            block = F.normalize(block.flatten(1).float(), dim=-1).half()
            similarity[start:stop] = (block @ target_t.T).float().cpu().numpy()
    del target_t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return identification_metrics(similarity)


def cmd_evaluate_tokens(args: argparse.Namespace) -> None:
    root = Path(args.result_dir)
    out = directories(root)
    device = device_from_arg(args.device)
    set_seed(args.seed)
    checkpoint = Path(args.checkpoint)
    model, metadata = load_trained_model(checkpoint, device)
    arrays = SplitArrays(root, args.split, convnext=False)
    count = len(arrays)
    run_name = args.run_name or metadata["run_name"]
    direct_path = out["cache"] / f"{run_name}_{args.split}_direct_tokens.npy"
    condition_path = out["cache"] / f"{run_name}_{args.split}_condition_tokens.npy"
    lowlevel_path = out["cache"] / f"{run_name}_{args.split}_lowlevel_latents.npy"
    prior_path = out["cache"] / f"{run_name}_{args.split}_prior_tokens.npy"
    direct = np.lib.format.open_memmap(
        direct_path, mode="w+", dtype=np.float16, shape=(count, CLIP_TOKENS, CLIP_DIM)
    )
    condition = np.lib.format.open_memmap(
        condition_path, mode="w+", dtype=np.float16, shape=(count, CLIP_TOKENS, CLIP_DIM)
    )
    lowlevel = np.lib.format.open_memmap(
        lowlevel_path, mode="w+", dtype=np.float16, shape=(count, 4, 28, 28)
    )
    variance_sum = 0.0
    precision_sums = {"cifti": 0.0, "modes": 0.0}
    with torch.no_grad():
        for start in tqdm(range(0, count, args.batch_size), desc="BrainNetwork test"):
            indices = np.arange(start, min(start + args.batch_size, count))
            batch = arrays.batch(indices, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                brain_condition, projected, low, statistics = model.brain_outputs(
                    batch["cifti"], batch["modes"], sample=False, modality_dropout=False
                )
            condition[start : start + len(indices)] = brain_condition.float().cpu().numpy().astype(np.float16)
            direct[start : start + len(indices)] = projected.float().cpu().numpy().astype(np.float16)
            lowlevel[start : start + len(indices)] = low[0].float().cpu().numpy().astype(np.float16)
            variance_sum += float(statistics["fused_variance"].float().mean().cpu()) * len(indices)
            for prefix in precision_sums:
                key = f"{prefix}_precision"
                if key in statistics:
                    precision_sums[prefix] += float(statistics[key].float().mean().cpu()) * len(indices)
    direct.flush()
    condition.flush()
    lowlevel.flush()

    prior = np.lib.format.open_memmap(
        prior_path, mode="w+", dtype=np.float16, shape=(count, CLIP_TOKENS, CLIP_DIM)
    )
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.float16, enabled=device.type == "cuda"
    ):
        for start in tqdm(range(0, count, args.prior_batch_size), desc="Diffusion Prior test"):
            stop = min(start + args.prior_batch_size, count)
            set_seed(args.seed + start)
            brain_condition = torch.from_numpy(
                np.asarray(condition[start:stop], dtype=np.float32)
            ).to(device)
            prediction = model.diffusion_prior.p_sample_loop(
                brain_condition.shape,
                text_cond={"text_embed": brain_condition},
                cond_scale=args.cond_scale,
                timesteps=args.prior_steps,
            )
            prior[start:stop] = prediction.float().cpu().numpy().astype(np.float16)
    prior.flush()
    target = arrays.bigg
    direct_metrics = token_identification(direct, target, device, args.identification_batch_size)
    prior_metrics = token_identification(prior, target, device, args.identification_batch_size)
    report = {
        "run_name": run_name,
        "checkpoint": str(checkpoint),
        "split": args.split,
        "count": count,
        "subject": SUBJECT_ID,
        "cross_subject_pretraining": False,
        "mindeye2_brain_distillation": False,
        "direct_brain_token_metrics": direct_metrics,
        "diffusion_prior_token_metrics": prior_metrics,
        "poe_diagnostics": {
            "mean_fused_variance": variance_sum / count,
            "mean_cifti_precision": precision_sums["cifti"] / count,
            "mean_modes_precision": precision_sums["modes"] / count,
        },
        "prior_sampling": {
            "steps": args.prior_steps,
            "condition_scale": args.cond_scale,
        },
    }
    write_json(out["metrics"] / f"{run_name}_{args.split}_token_metrics.json", report)
    print(json.dumps(report, indent=2))


def reconstruction_paths(
    root: Path, run_name: str, split_name: str
) -> dict[str, Path]:
    cache = directories(root)["cache"]
    prefix = f"{run_name}_{split_name}"
    return {
        "prior": cache / f"{prefix}_prior_tokens.npy",
        "latents": cache / f"{prefix}_lowlevel_latents.npy",
        "lowlevel": cache / f"{prefix}_lowlevel_images.npy",
        "semantic": cache / f"{prefix}_semantic_images.npy",
        "completed": cache / f"{prefix}_semantic_completed.npy",
    }


def cmd_prepare_reconstruction(args: argparse.Namespace) -> None:
    root = Path(args.result_dir)
    out = directories(root)
    device = device_from_arg(args.device)
    paths = reconstruction_paths(root, args.run_name, args.split)
    if not paths["prior"].exists() or not paths["latents"].exists():
        raise FileNotFoundError(
            f"Run evaluate-tokens first for {args.run_name} ({args.split})"
        )
    latents = np.load(paths["latents"], mmap_mode="r")
    count = len(latents)
    expected_image_shape = (count, 3, 256, 256)

    decode_lowlevel = args.force or not paths["lowlevel"].exists()
    if not decode_lowlevel:
        existing = np.load(paths["lowlevel"], mmap_mode="r")
        decode_lowlevel = existing.shape != expected_image_shape
        del existing
    if decode_lowlevel:
        destination = np.lib.format.open_memmap(
            paths["lowlevel"], mode="w+", dtype=np.float16, shape=expected_image_shape
        )
        autoencoder = _load_image_vae(device)
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            for start in tqdm(
                range(0, count, args.batch_size), desc=f"low-level decode {args.run_name}"
            ):
                stop = min(start + args.batch_size, count)
                latent = torch.from_numpy(
                    np.asarray(latents[start:stop], dtype=np.float16)
                ).to(device)
                image = autoencoder.decode(latent / 0.18215).sample / 2.0 + 0.5
                image = F.interpolate(
                    image, size=(256, 256), mode="bilinear", align_corners=False
                ).clamp(0, 1)
                destination[start:stop] = image.float().cpu().numpy().astype(np.float16)
        destination.flush()
        del autoencoder, destination
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reset_semantic = args.force or not paths["semantic"].exists()
    if not reset_semantic:
        semantic = np.load(paths["semantic"], mmap_mode="r")
        reset_semantic = semantic.shape != expected_image_shape
        del semantic
    if reset_semantic:
        semantic = np.lib.format.open_memmap(
            paths["semantic"], mode="w+", dtype=np.float16, shape=expected_image_shape
        )
        semantic[:] = 0
        semantic.flush()
        del semantic

    reset_completed = args.force or reset_semantic or not paths["completed"].exists()
    if not reset_completed:
        completed = np.load(paths["completed"], mmap_mode="r")
        reset_completed = completed.shape != (count,)
        del completed
    if reset_completed:
        completed = np.lib.format.open_memmap(
            paths["completed"], mode="w+", dtype=np.bool_, shape=(count,)
        )
        completed[:] = False
        completed.flush()
        del completed

    report = {
        "run_name": args.run_name,
        "split": args.split,
        "count": count,
        "lowlevel_images": str(paths["lowlevel"]),
        "semantic_images": str(paths["semantic"]),
        "semantic_completed": str(paths["completed"]),
        "stores_png_or_jpeg": False,
    }
    write_json(
        out["metrics"] / f"{args.run_name}_{args.split}_reconstruction_prepared.json",
        report,
    )
    print(json.dumps(report, indent=2))


def build_unclip_engine(device: torch.device, steps: int):
    sys.path.insert(0, str(MINDEYE_SRC))
    sys.path.insert(0, str(MINDEYE_SRC / "generative_models"))
    from generative_models.sgm.models.diffusion import DiffusionEngine
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(
        OmegaConf.load(MINDEYE_SRC / "generative_models/configs/unclip6.yaml"),
        resolve=True,
    )
    params = config["model"]["params"]
    first_stage = params["first_stage_config"]
    first_stage["target"] = "sgm.models.autoencoder.AutoencoderKL"
    first_stage["params"]["ddconfig"]["attn_type"] = "vanilla"
    params["sampler_config"]["params"]["num_steps"] = steps
    engine = DiffusionEngine(
        network_config=params["network_config"],
        denoiser_config=params["denoiser_config"],
        first_stage_config=first_stage,
        conditioner_config=params["conditioner_config"],
        sampler_config=params["sampler_config"],
        scale_factor=params["scale_factor"],
        disable_first_stage_autocast=params["disable_first_stage_autocast"],
    )
    checkpoint_path = resolve_artifact(
        "unclip6_epoch0_step110000.ckpt", preferred=UNCLIP_CHECKPOINT_FAST
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    engine.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    engine = engine.to(device).eval().requires_grad_(False)
    batch = {
        "jpg": torch.randn(1, 3, 1, 1, device=device),
        "original_size_as_tuple": torch.ones(1, 2, device=device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2, device=device),
    }
    vector_suffix = engine.conditioner(batch)["vector"].to(device)
    return engine, vector_suffix, checkpoint_path


def cmd_reconstruct_shard(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    root = Path(args.result_dir)
    out = directories(root)
    paths = reconstruction_paths(root, args.run_name, args.split)
    for name in ("prior", "semantic", "completed"):
        if not paths[name].exists():
            raise FileNotFoundError(
                f"Missing {paths[name]}; run evaluate-tokens and prepare-reconstruction first"
            )
    prior = np.load(paths["prior"], mmap_mode="r")
    semantic = np.lib.format.open_memmap(paths["semantic"], mode="r+")
    completed = np.lib.format.open_memmap(paths["completed"], mode="r+")
    if semantic.shape != (len(prior), 3, 256, 256) or completed.shape != (len(prior),):
        raise RuntimeError("Reconstruction cache shapes do not match prior-token count")
    pending = np.flatnonzero(~np.asarray(completed))
    pending = pending[pending % args.num_shards == args.shard_index]
    if len(pending) == 0:
        print(
            json.dumps(
                {
                    "run_name": args.run_name,
                    "shard_index": args.shard_index,
                    "pending": 0,
                    "completed_total": int(np.asarray(completed).sum()),
                },
                indent=2,
            )
        )
        return

    device = device_from_arg(args.device)
    set_seed(args.seed)
    sys.path.insert(0, str(MINDEYE_SRC))
    import utils as mindeye_utils

    engine, vector_suffix, checkpoint_path = build_unclip_engine(device, args.steps)
    with torch.no_grad():
        for sequence, index in enumerate(
            tqdm(pending, desc=f"unCLIP {args.run_name} shard {args.shard_index}")
        ):
            set_seed(args.seed + int(index))
            token = torch.from_numpy(
                np.asarray(prior[index : index + 1], dtype=np.float32)
            ).to(device=device, dtype=torch.float16)
            sample = mindeye_utils.unclip_recon(
                token, engine, vector_suffix, num_samples=1
            )[0]
            resized = F.interpolate(
                sample[None].float(),
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )[0].clamp(0, 1)
            semantic[index] = resized.cpu().numpy().astype(np.float16)
            completed[index] = True
            if sequence % 5 == 0:
                semantic.flush()
                completed.flush()
    semantic.flush()
    completed.flush()
    report = {
        "run_name": args.run_name,
        "split": args.split,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "generated_by_shard": int(len(pending)),
        "completed_total": int(np.asarray(completed).sum()),
        "steps": args.steps,
        "unclip_checkpoint": str(checkpoint_path),
    }
    write_json(
        out["metrics"]
        / f"{args.run_name}_{args.split}_reconstruction_shard{args.shard_index}.json",
        report,
    )
    print(json.dumps(report, indent=2))


def cmd_evaluate_images(args: argparse.Namespace) -> None:
    root = Path(args.result_dir)
    out = directories(root)
    paths = reconstruction_paths(root, args.run_name, args.split)
    completed = np.load(paths["completed"], mmap_mode="r")
    if not bool(np.asarray(completed).all()):
        raise RuntimeError(
            f"Only {int(np.asarray(completed).sum())}/{len(completed)} semantic images are complete"
        )
    semantic_array = np.load(paths["semantic"], mmap_mode="r")
    lowlevel_array = np.load(paths["lowlevel"], mmap_mode="r")
    if semantic_array.shape != lowlevel_array.shape:
        raise RuntimeError(
            f"Semantic/low-level shape mismatch: {semantic_array.shape} vs {lowlevel_array.shape}"
        )
    split = load_split(root)
    rows = split[f"{args.split}_rows"].astype(np.int64)
    if len(rows) != len(semantic_array):
        raise RuntimeError("Image count does not match the recorded data split")
    with h5py.File(DATA_PATH, "r") as h5:
        ground_truth = torch.from_numpy(np.asarray(h5["labels"][rows])).permute(0, 3, 1, 2)
    ground_truth = ground_truth.float() / 255.0
    semantic = torch.from_numpy(np.asarray(semantic_array, dtype=np.float32))
    lowlevel = torch.from_numpy(np.asarray(lowlevel_array, dtype=np.float32))
    reconstruction = (
        args.semantic_weight * semantic + (1.0 - args.semantic_weight) * lowlevel
    ).clamp(0, 1)
    del semantic, lowlevel, semantic_array, lowlevel_array

    sys.path.insert(0, str(CODE_ROOT))
    import evaluate_images

    device = device_from_arg(args.device)
    metrics = evaluate_images.evaluate_pair(ground_truth, reconstruction, device)
    token_path = out["metrics"] / f"{args.run_name}_{args.split}_token_metrics.json"
    token_metrics = (
        json.loads(token_path.read_text(encoding="utf-8")) if token_path.exists() else None
    )
    report = {
        "run_name": args.run_name,
        "split": args.split,
        "count": len(reconstruction),
        "semantic_branch": "frozen MindEye2 SDXL-unCLIP",
        "lowlevel_branch": "independently predicted SD image-variations VAE latent",
        "semantic_weight": args.semantic_weight,
        "lowlevel_weight": 1.0 - args.semantic_weight,
        "image_metrics": metrics,
        "token_metrics": token_metrics,
        "saved_test_images": False,
    }
    report_path = out["metrics"] / f"{args.run_name}_{args.split}_image_metrics.json"
    write_json(report_path, report)
    if args.delete_temporary_images:
        for name in ("lowlevel", "semantic", "completed"):
            paths[name].unlink(missing_ok=True)
        report["temporary_image_arrays_deleted"] = True
        write_json(report_path, report)
    print(json.dumps(report, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--result-dir", default=str(DEFAULT_ROOT))
    sub = value.add_subparsers(dest="command", required=True)

    command = sub.add_parser("prepare")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--val-fraction", type=float, default=0.1)
    command.add_argument("--batch-size", type=int, default=128)
    command.set_defaults(func=cmd_prepare)

    command = sub.add_parser("cache-targets")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--batch-size", type=int, default=4)
    command.add_argument("--convnext-batch-size", type=int, default=8)
    command.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    command.add_argument("--with-convnext", action="store_true")
    command.add_argument("--force", action="store_true")
    command.add_argument("--seed", type=int, default=42)
    command.set_defaults(func=cmd_cache_targets)

    command = sub.add_parser("train")
    command.add_argument("--variant", choices=VARIANTS, required=True)
    command.add_argument("--run-name")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    command.add_argument("--epochs", type=int, default=40)
    command.add_argument("--min-epochs", type=int, default=15)
    command.add_argument("--patience", type=int, default=8)
    command.add_argument("--min-delta", type=float, default=1e-4)
    command.add_argument("--batch-size", type=int, default=24)
    command.add_argument("--eval-batch-size", type=int, default=24)
    command.add_argument("--max-train-samples", type=int)
    command.add_argument("--max-lr", type=float, default=3e-4)
    command.add_argument("--weight-decay", type=float, default=1e-2)
    command.add_argument("--grad-clip", type=float, default=1.0)
    command.add_argument("--mixup-fraction", type=float, default=0.33)
    command.add_argument("--prior-weight", type=float, default=30.0)
    command.add_argument("--clip-weight", type=float, default=1.0)
    command.add_argument("--lowlevel-weight", type=float, default=0.5)
    command.add_argument("--convnext-weight", type=float, default=0.05)
    command.add_argument("--kl-weight", type=float, default=1e-3)
    command.add_argument("--use-convnext", action="store_true")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--shared-initialization-seed", type=int, default=20260710)
    command.add_argument("--validation-seed", type=int, default=20260710)
    command.set_defaults(func=cmd_train)

    command = sub.add_parser("evaluate-tokens")
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--run-name")
    command.add_argument("--split", choices=("val", "test"), default="test")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--batch-size", type=int, default=24)
    command.add_argument("--prior-batch-size", type=int, default=4)
    command.add_argument("--prior-steps", type=int, default=20)
    command.add_argument("--cond-scale", type=float, default=1.0)
    command.add_argument("--identification-batch-size", type=int, default=16)
    command.add_argument("--seed", type=int, default=42)
    command.set_defaults(func=cmd_evaluate_tokens)

    command = sub.add_parser("prepare-reconstruction")
    command.add_argument("--run-name", required=True)
    command.add_argument("--split", choices=("val", "test"), default="test")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--force", action="store_true")
    command.set_defaults(func=cmd_prepare_reconstruction)

    command = sub.add_parser("reconstruct-shard")
    command.add_argument("--run-name", required=True)
    command.add_argument("--split", choices=("val", "test"), default="test")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--steps", type=int, default=38)
    command.add_argument("--shard-index", type=int, default=0)
    command.add_argument("--num-shards", type=int, default=1)
    command.add_argument("--seed", type=int, default=42)
    command.set_defaults(func=cmd_reconstruct_shard)

    command = sub.add_parser("evaluate-images")
    command.add_argument("--run-name", required=True)
    command.add_argument("--split", choices=("val", "test"), default="test")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--semantic-weight", type=float, default=0.75)
    command.add_argument("--delete-temporary-images", action="store_true")
    command.set_defaults(func=cmd_evaluate_images)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
