
from __future__ import annotations

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

def resize_batch(x: torch.Tensor, size: int) -> torch.Tensor:
    return transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR)(x)

def corr_features(x: torch.Tensor) -> torch.Tensor:
    x = x.float().flatten(1)
    x = x - x.mean(dim=1, keepdim=True)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)

def extract_features(images, model, preprocess, device, batch_size, feature_layer=None):
    output = []
    for start in tqdm(range(0, len(images), batch_size), leave=False):
        result = model(preprocess(images[start : start + batch_size]).to(device))
        if feature_layer is not None:
            result = result[feature_layer]
        output.append(result.float().flatten(1).cpu())
    return torch.cat(output)

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

def evaluate_pair(real: torch.Tensor, reconstructed: torch.Tensor, device: torch.device) -> dict[str, float]:
    if len(real)<2 or len(real)!=len(reconstructed):
        raise ValueError("Identification metrics require at least two paired images")
    if not torch.isfinite(real).all() or real.min()<0 or real.max()>1:
        raise ValueError("Reference images must be finite floats in [0,1]")
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

    from vendor.swav.loader import load_swav
    swav = load_swav()
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
