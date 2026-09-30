












from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torchvision import transforms


ROOT = Path(__file__).resolve().parent
VENDOR_SRC = ROOT / "vendor/generative_runtime/src"
WEIGHT_ROOT = Path(
    os.environ.get("NSD_SUB01_WEIGHT_ROOT", str(ROOT / "weights/nsd_sub01"))
)
DEFAULT_DATA = ROOT / "assets/nsd_sub01_test.h5"
DEFAULT_OUTPUT = ROOT / "outputs/nsd_sub01"
MAX_SAVED_IMAGES = 3

CIFTI_DIM = 4096
MODES_DIM = 2000
HIDDEN_DIM = 1024
CLIP_TOKENS = 256
CLIP_DIM = 1664


def display_path(path: Path) -> str:

    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_weights(path: Path) -> dict[str, Any]:

    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a checkpoint dictionary: {path}")
    return value


def configure_vendor() -> None:
    if not (VENDOR_SRC / "models.py").is_file():
        raise FileNotFoundError(VENDOR_SRC / "models.py")
    for path in (VENDOR_SRC, VENDOR_SRC / "generative_models"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def vendor_models():

    configure_vendor()
    module_name = "_eigen_generative_runtime_models"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, VENDOR_SRC / "models.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {VENDOR_SRC / 'models.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class AnchoredPoE(nn.Module):


    def __init__(self) -> None:
        super().__init__()
        self.cifti_mean = nn.Linear(CIFTI_DIM, HIDDEN_DIM)
        self.modes_mean = nn.Linear(MODES_DIM, HIDDEN_DIM)
        self.cifti_log_precision = nn.Parameter(torch.zeros(HIDDEN_DIM))
        self.modes_log_precision = nn.Parameter(torch.full((HIDDEN_DIM,), -2.0))

    def forward(
        self, cifti: torch.Tensor, modes: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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


class FusionBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        BrainNetwork = vendor_models().BrainNetwork

        self.fusion = AnchoredPoE()
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


def load_brain(device: torch.device) -> FusionBackbone:
    path = WEIGHT_ROOT / "brain_final.pt"
    saved = load_weights(path)
    expected = {"fusion", "backbone"}
    if not expected.issubset(saved):
        raise KeyError(f"{path} is missing {sorted(expected.difference(saved))}")
    model = FusionBackbone()
    model.fusion.load_state_dict(saved["fusion"], strict=True)
    model.backbone.load_state_dict(saved["backbone"], strict=True)
    del saved
    return model.to(device).eval().requires_grad_(False)


def make_diffusion_prior() -> nn.Module:
    module = vendor_models()
    BrainDiffusionPrior = module.BrainDiffusionPrior
    PriorNetwork = module.PriorNetwork

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


def load_prior(device: torch.device) -> nn.Module:
    path = WEIGHT_ROOT / "diffusion_prior.pt"
    saved = load_weights(path)
    if "prior" not in saved:
        raise KeyError(f"{path} is missing key 'prior'")
    prior = make_diffusion_prior()
    prior.load_state_dict(saved["prior"], strict=True)
    del saved
    return prior.to(device).eval().requires_grad_(False)


def load_lowlevel_vae(device: torch.device) -> nn.Module:
    from diffusers import AutoencoderKL

    model = AutoencoderKL(
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    state = load_weights(WEIGHT_ROOT / "sd_image_var_autoenc.pth")
    model.load_state_dict(state, strict=True)
    del state
    return model.to(device=device, dtype=torch.float16).eval().requires_grad_(False)


def build_unclip_engine(device: torch.device, steps: int):
    configure_vendor()
    from generative_models.sgm.models.diffusion import DiffusionEngine
    from omegaconf import OmegaConf

    config_path = VENDOR_SRC / "generative_models/configs/unclip6.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
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
    path = WEIGHT_ROOT / "unclip6_epoch0_step110000.ckpt"
    saved = load_weights(path)
    if "state_dict" not in saved:
        raise KeyError(f"{path} is missing key 'state_dict'")
    engine.load_state_dict(saved["state_dict"], strict=True)
    del saved
    engine = engine.to(device).eval().requires_grad_(False)
    batch = {
        "jpg": torch.randn(1, 3, 1, 1, device=device),
        "original_size_as_tuple": torch.ones(1, 2, device=device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2, device=device),
    }
    vector_suffix = engine.conditioner(batch)["vector"].to(device)
    return engine, vector_suffix


def load_test_slice(path: Path, start: int, count: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as h5:
        n = len(h5["cifti"])
        if start < 0 or start >= n:
            raise ValueError(f"start must be in [0, {n}), got {start}")
        stop = n if count == 0 else min(start + count, n)
        result = {
            "cifti": h5["cifti"][start:stop].astype(np.float32),
            "modes": h5["modes"][start:stop].astype(np.float32),
            "rgb": h5["rgb"][start:stop].astype(np.uint8),
            "test_index": h5["test_index"][start:stop].astype(np.int64),
            "source_row": h5["source_row"][start:stop].astype(np.int64),
            "stim_id": h5["stim_id"][start:stop].astype(np.int64),
        }
    if result["cifti"].shape[1:] != (CIFTI_DIM,):
        raise ValueError(f"Unexpected CIFTI shape: {result['cifti'].shape}")
    if result["modes"].shape[1:] != (MODES_DIM,):
        raise ValueError(f"Unexpected modes shape: {result['modes'].shape}")
    return result


@torch.inference_mode()
def predict_brain(
    data: dict[str, np.ndarray], device: torch.device, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    model = load_brain(device)
    conditions, latents = [], []
    cifti_precision, modes_share = [], []
    for begin in range(0, len(data["cifti"]), batch_size):
        end = min(begin + batch_size, len(data["cifti"]))
        cifti = torch.from_numpy(data["cifti"][begin:end]).to(device)
        modes = torch.from_numpy(data["modes"][begin:end]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            condition, _, lowlevel, stats = model(cifti, modes)
        conditions.append(condition.float().cpu())
        latents.append(lowlevel[0].float().cpu())
        cp = stats["cifti_precision"].float()
        mp = stats["modes_precision"].float()
        cifti_precision.append(cp.mean().cpu())
        modes_share.append((mp / (cp + mp)).mean().cpu())
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    diagnostics = {
        "mean_cifti_precision": float(torch.stack(cifti_precision).mean()),
        "mean_modes_precision_share": float(torch.stack(modes_share).mean()),
    }
    return torch.cat(conditions), torch.cat(latents), diagnostics


@torch.inference_mode()
def sample_prior(
    condition: torch.Tensor,
    test_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    steps: int,
    cond_scale: float,
    seed: int,
) -> torch.Tensor:
    prior = load_prior(device)
    outputs = []
    for begin in range(0, len(condition), batch_size):
        end = min(begin + batch_size, len(condition))
        set_seed(seed + int(test_indices[begin]))
        value = condition[begin:end].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = prior.p_sample_loop(
                value.shape,
                text_cond={"text_embed": value},
                cond_scale=cond_scale,
                timesteps=steps,
            )
        outputs.append(prediction.float().cpu())
    del prior
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(outputs)


@torch.inference_mode()
def decode_lowlevel(
    latents: torch.Tensor, device: torch.device, batch_size: int
) -> torch.Tensor:
    vae = load_lowlevel_vae(device)
    output = []
    for begin in range(0, len(latents), batch_size):
        latent = latents[begin : begin + batch_size].to(device=device, dtype=torch.float16)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            image = vae.decode(latent / 0.18215).sample / 2.0 + 0.5
        output.append(
            F.interpolate(image, size=(256, 256), mode="bilinear", align_corners=False)
            .clamp(0, 1)
            .float()
            .cpu()
        )
    del vae
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(output)


@torch.inference_mode()
def reconstruct_semantic(
    prior_tokens: torch.Tensor,
    test_indices: np.ndarray,
    device: torch.device,
    steps: int,
    seed: int,
) -> torch.Tensor:
    configure_vendor()
    import utils as runtime_utils

    engine, vector_suffix = build_unclip_engine(device, steps)
    output = []
    for token, test_index in zip(prior_tokens, test_indices):
        set_seed(seed + int(test_index))
        sample = runtime_utils.unclip_recon(
            token[None].to(device=device, dtype=torch.float16),
            engine,
            vector_suffix,
            num_samples=1,
        )[0]
        output.append(
            F.interpolate(
                sample[None].float(),
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )[0]
            .clamp(0, 1)
            .cpu()
        )
    del engine, vector_suffix
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.stack(output)


@torch.inference_mode()
def openclip_features(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    import open_clip

    checkpoint = WEIGHT_ROOT / "open_clip_vit_l14_openai.safetensors"
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained=str(checkpoint)
    )
    model = model.to(device).eval().requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize(
                (224, 224), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    output = []
    for begin in range(0, len(images), 16):
        value = model.encode_image(preprocess(images[begin : begin + 16]).to(device))
        output.append(value.float().cpu())
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(output)


def corr_features(x: torch.Tensor) -> torch.Tensor:
    x = x.float().flatten(1)
    x = x - x.mean(dim=1, keepdim=True)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    array = (
        tensor.detach().clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().numpy()
    )
    Image.fromarray(array).save(path)


def save_comparison(
    ground_truth: torch.Tensor,
    final: torch.Tensor,
    path: Path,
) -> None:
    tensors = [ground_truth, final]
    labels = ["ground truth", "Ours reconstruction"]
    images = []
    for value in tensors:
        array = value.clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().numpy()
        images.append(Image.fromarray(array).convert("RGB"))
    canvas = Image.new("RGB", (256 * 2, 286), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(zip(images, labels)):
        canvas.paste(image, (i * 256, 30))
        draw.text((i * 256 + 6, 8), label, fill="black")
    canvas.save(path)


def evaluate_and_save(
    data: dict[str, np.ndarray],
    lowlevel: torch.Tensor,
    semantic: torch.Tensor,
    reconstruction: torch.Tensor,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    ground_truth = torch.from_numpy(data["rgb"]).permute(0, 3, 1, 2).float() / 255.0
    all_features = openclip_features(
        torch.cat([ground_truth, reconstruction], dim=0), device
    )
    gt_features, recon_features = all_features.split(len(ground_truth))
    gt_unit = F.normalize(gt_features, dim=1)
    recon_unit = F.normalize(recon_features, dim=1)
    cosine = (gt_unit * recon_unit).sum(1)
    gt_corr = corr_features(gt_features)
    recon_corr = corr_features(recon_features)
    similarity = gt_corr @ recon_corr.T
    diagonal = torch.diag(similarity)
    if len(similarity) > 1:
        per_image_2way = (similarity < diagonal[None, :]).sum(0).float() / (
            len(similarity) - 1
        )
        clip_2way = float(per_image_2way.mean())
    else:
        per_image_2way = torch.full((1,), float("nan"))
        clip_2way = float("nan")

    selected = torch.argsort(cosine, descending=True)[:MAX_SAVED_IMAGES].tolist()
    saved_comparisons: dict[int, str] = {}
    selection_rank = {index: rank + 1 for rank, index in enumerate(selected)}
    for i in selected:
        prefix = f"test_{int(data['test_index'][i]):04d}_stim_{int(data['stim_id'][i])}"
        comparison_path = output / f"{prefix}_ground_truth_vs_ours.png"
        save_comparison(ground_truth[i], reconstruction[i], comparison_path)
        saved_comparisons[i] = str(comparison_path)

    rows = []
    for i in range(len(cosine)):
        prefix = f"test_{int(data['test_index'][i]):04d}_stim_{int(data['stim_id'][i])}"
        rows.append(
            {
                "test_index": int(data["test_index"][i]),
                "source_row": int(data["source_row"][i]),
                "stim_id": int(data["stim_id"][i]),
                "openclip_vitl14_cosine": float(cosine[i]),
                "openclip_vitl14_pearson": float(diagonal[i]),
                "openclip_vitl14_2way": float(per_image_2way[i]),
                "visualization_selection_rank": selection_rank.get(i, ""),
                "comparison_png": display_path(Path(saved_comparisons[i]))
                if i in saved_comparisons else "",
            }
        )
    with (output / "image_level_clip.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "n": len(rows),
        "mean_openclip_vitl14_cosine": float(cosine.mean()),
        "mean_openclip_vitl14_pearson": float(diagonal.mean()),
        "CLIP_2way_original_definition": clip_2way,
        "saved_png_count": len(saved_comparisons),
        "saved_png_limit": MAX_SAVED_IMAGES,
        "saved_test_indices": [int(data["test_index"][i]) for i in selected],
        "display_selection": "top per-image OpenCLIP cosine; qualitative visualization only",
        "image_level_csv": display_path(output / "image_level_clip.csv"),
    }


def run_nsd_reconstruction(
    data_path: Path = DEFAULT_DATA,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    start: int = 0,
    count: int = 10,
    brain_batch_size: int = 4,
    prior_batch_size: int = 4,
    prior_steps: int = 20,
    reconstruction_steps: int = 50,
    cond_scale: float = 1.5,
    seed: int = 42,
) -> dict[str, Any]:
    if not torch.cuda.is_available() and device_name.startswith("cuda"):
        raise RuntimeError("CUDA is required for practical SDXL-unCLIP reconstruction")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    if prior_batch_size != 4:
        raise ValueError(
            "prior_batch_size must remain 4 to preserve the archived seed schedule"
        )
    if start % prior_batch_size != 0:
        raise ValueError(
            "start must be divisible by 4 to preserve the archived seed schedule"
        )
    print(f"[NSD S1] loading {count} test samples from {data_path}", flush=True)
    data = load_test_slice(data_path, start, count)
    set_seed(seed)
    print(f"[NSD S1] brain model: {WEIGHT_ROOT / 'brain_final.pt'}", flush=True)
    condition, latents, poe_diagnostics = predict_brain(
        data, device, brain_batch_size
    )
    print(f"[NSD S1] diffusion prior: {prior_steps} steps", flush=True)
    prior_tokens = sample_prior(
        condition,
        data["test_index"],
        device,
        prior_batch_size,
        prior_steps,
        cond_scale,
        seed,
    )
    print("[NSD S1] decoding low-level reconstruction", flush=True)
    lowlevel = decode_lowlevel(latents, device, brain_batch_size)
    print(
        f"[NSD S1] SDXL-unCLIP semantic reconstruction: {reconstruction_steps} steps",
        flush=True,
    )
    semantic = reconstruct_semantic(
        prior_tokens,
        data["test_index"],
        device,
        reconstruction_steps,
        seed,
    )
    reconstruction = (0.75 * semantic + 0.25 * lowlevel).clamp(0, 1)
    print("[NSD S1] computing image-level OpenCLIP metrics", flush=True)
    metrics = evaluate_and_save(
        data, lowlevel, semantic, reconstruction, output, device
    )
    summary = {
        "task": "NSD Subject-1 within-subject fMRI-to-image reconstruction",
        "protocol": "frozen inference; Shared1000 held-out test rows; no fitting",
        "archived_full_test_CLIP_2way_original_definition": 0.7828989028930664,
        "start": start,
        "count": len(data["test_index"]),
        "test_indices": data["test_index"].tolist(),
        "prior_steps": prior_steps,
        "prior_cond_scale": cond_scale,
        "reconstruction_steps": reconstruction_steps,
        "semantic_weight": 0.75,
        "lowlevel_weight": 0.25,
        "poe_diagnostics": poe_diagnostics,
        "metrics": metrics,
        "output_dir": display_path(output),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[NSD S1] finished; outputs: {output}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of consecutive Shared1000 samples; 0 runs all remaining samples.",
    )
    parser.add_argument("--brain-batch-size", type=int, default=4)
    parser.add_argument("--prior-batch-size", type=int, default=4)
    parser.add_argument("--prior-steps", type=int, default=20)
    parser.add_argument("--reconstruction-steps", type=int, default=50)
    parser.add_argument("--cond-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = run_nsd_reconstruction(
        data_path=args.data,
        output=args.output,
        device_name=args.device,
        start=args.start,
        count=args.count,
        brain_batch_size=args.brain_batch_size,
        prior_batch_size=args.prior_batch_size,
        prior_steps=args.prior_steps,
        reconstruction_steps=args.reconstruction_steps,
        cond_scale=args.cond_scale,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
