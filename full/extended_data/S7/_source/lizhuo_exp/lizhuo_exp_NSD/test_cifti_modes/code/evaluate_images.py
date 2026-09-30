


from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from torchvision import transforms
from torchvision.models import (
    AlexNet_Weights,
    EfficientNet_B1_Weights,
    Inception_V3_Weights,
    alexnet,
    efficientnet_b1,
    inception_v3,
)
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm


DEFAULT_RESULT = Path(
    str(_LOCAL_SOURCE / 'lizhuo_exp/lizhuo_exp_NSD/test_cifti_modes/result')
)


def resize_batch(x: torch.Tensor, size: int) -> torch.Tensor:
    return transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR)(x)


def corr_features(x: torch.Tensor) -> torch.Tensor:
    x = x.float().flatten(1)
    x = x - x.mean(dim=1, keepdim=True)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def extract_features(images, model, preprocess, device, batch_size, feature_layer=None):
    output = []
    for start in tqdm(range(0, len(images), batch_size), leave=False):
        result = model(preprocess(images[start : start + batch_size]).to(device))
        if feature_layer is not None:
            result = result[feature_layer]
        output.append(result.float().flatten(1).cpu())
    return torch.cat(output)


@torch.no_grad()
def two_way_identification(real, reconstructed, model, preprocess, device, feature_layer=None, batch_size=64):
    pred = corr_features(extract_features(reconstructed, model, preprocess, device, batch_size, feature_layer))
    target = corr_features(extract_features(real, model, preprocess, device, batch_size, feature_layer))
    similarity = target @ pred.T
    congruent = torch.diag(similarity)
    return float((similarity < congruent[None, :]).sum(dim=0).float().mean() / (len(real) - 1))


def grayscale(x: torch.Tensor) -> np.ndarray:
    array = x.permute(0, 2, 3, 1).cpu().numpy()
    return array[..., 0] * 0.2125 + array[..., 1] * 0.7154 + array[..., 2] * 0.0721


def ssim_grayscale(real, reconstructed, sigma=1.5):
    c1 = 0.01**2
    c2 = 0.03**2
    ux = gaussian_filter(real, sigma)
    uy = gaussian_filter(reconstructed, sigma)
    uxx = gaussian_filter(real * real, sigma)
    uyy = gaussian_filter(reconstructed * reconstructed, sigma)
    uxy = gaussian_filter(real * reconstructed, sigma)
    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy
    score = ((2 * ux * uy + c1) * (2 * vxy + c2)) / (
        (ux * ux + uy * uy + c1) * (vx + vy + c2)
    )
    return float(np.mean(score))


@torch.no_grad()
def openclip_features(images: torch.Tensor, device: torch.device, batch_size: int = 32) -> torch.Tensor:
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = model.to(device).eval().requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    output = []
    for start in tqdm(range(0, len(images), batch_size), leave=False):
        output.append(model.encode_image(preprocess(images[start : start + batch_size]).to(device)).float().cpu())
    del model
    return torch.cat(output)


def correlation_distance(real: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(
        np.mean([sp.spatial.distance.correlation(real[i], reconstructed[i]) for i in range(len(real))])
    )


def generated_clip_metrics(path: Path) -> dict[str, float]:
    with np.load(path) as value:
        predicted = value["predicted"].astype(np.float32)
        target = value["target"].astype(np.float32)
        generated = value["generated"].astype(np.float32)
    predicted /= np.linalg.norm(predicted, axis=1, keepdims=True) + 1e-8
    target /= np.linalg.norm(target, axis=1, keepdims=True) + 1e-8
    generated /= np.linalg.norm(generated, axis=1, keepdims=True) + 1e-8
    similarity = generated @ target.T
    diagonal = np.diag(similarity)
    rank = np.argsort(-similarity, axis=1)
    positions = np.array([np.where(rank[i] == i)[0][0] for i in range(len(rank))])
    return {
        "GeneratedCLIP_R@1": float((positions < 1).mean()),
        "GeneratedCLIP_R@5": float((positions < 5).mean()),
        "GeneratedCLIP_2way": float((similarity < diagonal[:, None]).sum(axis=1).mean() / (len(rank) - 1)),
        "PredictedTargetCosine": float(np.sum(predicted * target, axis=1).mean()),
        "GeneratedTargetCosine": float(diagonal.mean()),
    }


def evaluate_pair(real: torch.Tensor, reconstructed: torch.Tensor, device: torch.device) -> dict[str, float]:
    metrics = {}
    pixel_real = resize_batch(real, 425).reshape(len(real), -1).numpy()
    pixel_recon = resize_batch(reconstructed, 425).reshape(len(reconstructed), -1).numpy()
    metrics["PixCorr"] = float(
        np.mean([np.corrcoef(pixel_real[i], pixel_recon[i])[0, 1] for i in range(len(real))])
    )
    gray_real = grayscale(resize_batch(real, 425))
    gray_recon = grayscale(resize_batch(reconstructed, 425))
    metrics["SSIM"] = float(
        np.mean([ssim_grayscale(gray_real[i], gray_recon[i]) for i in tqdm(range(len(real)), desc="SSIM")])
    )

    imagenet_256 = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    alex = create_feature_extractor(
        alexnet(weights=AlexNet_Weights.IMAGENET1K_V1), return_nodes=["features.4", "features.11"]
    ).to(device).eval().requires_grad_(False)
    metrics["AlexNet(2)"] = two_way_identification(
        real, reconstructed, alex, imagenet_256, device, "features.4"
    )
    metrics["AlexNet(5)"] = two_way_identification(
        real, reconstructed, alex, imagenet_256, device, "features.11"
    )
    del alex
    if device.type == "cuda":
        torch.cuda.empty_cache()

    inception_preprocess = transforms.Compose(
        [
            transforms.Resize((342, 342), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    inception = create_feature_extractor(
        inception_v3(weights=Inception_V3_Weights.DEFAULT), return_nodes=["avgpool"]
    ).to(device).eval().requires_grad_(False)
    metrics["InceptionV3"] = two_way_identification(
        real, reconstructed, inception, inception_preprocess, device, "avgpool", 32
    )
    del inception
    if device.type == "cuda":
        torch.cuda.empty_cache()

    clip_real = corr_features(openclip_features(real, device))
    clip_recon = corr_features(openclip_features(reconstructed, device))
    clip_similarity = clip_real @ clip_recon.T
    metrics["CLIP"] = float(
        (clip_similarity < torch.diag(clip_similarity)[None, :]).sum(dim=0).float().mean() / (len(real) - 1)
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    effnet = create_feature_extractor(
        efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT), return_nodes=["avgpool"]
    ).to(device).eval().requires_grad_(False)
    eff_preprocess = transforms.Compose(
        [
            transforms.Resize((255, 255), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eff_real = extract_features(real, effnet, eff_preprocess, device, 64, "avgpool").numpy()
    eff_recon = extract_features(reconstructed, effnet, eff_preprocess, device, 64, "avgpool").numpy()
    metrics["EffNet-B"] = correlation_distance(eff_real, eff_recon)
    del effnet
    if device.type == "cuda":
        torch.cuda.empty_cache()

    swav = torch.hub.load(
        "facebookresearch/swav:main", "resnet50", trust_repo=True, skip_validation=True
    )
    swav = create_feature_extractor(swav, return_nodes=["avgpool"]).to(device).eval().requires_grad_(False)
    swav_preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    swav_real = extract_features(real, swav, swav_preprocess, device, 64, "avgpool").numpy()
    swav_recon = extract_features(reconstructed, swav, swav_preprocess, device, 64, "avgpool").numpy()
    metrics["SwAV"] = correlation_distance(swav_real, swav_recon)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--kind",
        choices=[
            "semantic", "blurry", "final", "_", "hybrid_semantic", "hybrid_final",
            "reliable_semantic", "reliable_final",
            "reliable_cifti_only_semantic",
        ],
        default="final",
    )
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    result = Path(args.result_dir)
    cache = result / "cache"
    metrics_dir = result / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    real = torch.load(cache / "ground_truth_100.pt", map_location="cpu", weights_only=False).float()
    recon_name = {
        "semantic": "recon_semantic_100.pt",
        "blurry": "recon_blurry_100.pt",
        "final": "recon_final_100.pt",
        "_": "recon___100.pt",
        "hybrid_semantic": "recon_hybrid_semantic_100.pt",
        "hybrid_final": "recon_hybrid_final_100.pt",
        "reliable_semantic": "recon_reliable_semantic_100.pt",
        "reliable_final": "recon_reliable_final_100.pt",
        "reliable_cifti_only_semantic": "recon_reliable_cifti_only_semantic_100.pt",
    }[args.kind]
    reconstructed = torch.load(cache / recon_name, map_location="cpu", weights_only=False).float()
    metrics = evaluate_pair(real, reconstructed, device)
    if args.kind == "_":
        embedding_file = "__clip_embeddings.npz"
    elif args.kind.startswith("hybrid_"):
        embedding_file = "hybrid_clip_embeddings.npz"
    elif args.kind.startswith("reliable_"):
        embedding_file = (
            "reliable_cifti_only_clip_embeddings.npz"
            if args.kind.startswith("reliable_cifti_only_")
            else "reliable_clip_embeddings.npz"
        )
    else:
        embedding_file = "reconstruction_clip_embeddings.npz"
    metrics.update(generated_clip_metrics(cache / embedding_file))
    payload = {"subject": 1, "count": len(real), "kind": args.kind, "metrics": metrics}
    (metrics_dir / f"image_metrics_{args.kind}.json").write_text(json.dumps(payload, indent=2))
    with (metrics_dir / f"image_metrics_{args.kind}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "Value"])
        writer.writerows(metrics.items())
    for name, value in metrics.items():
        print(f"{name:28s} {value:.6f}")


if __name__ == "__main__":
    main()
