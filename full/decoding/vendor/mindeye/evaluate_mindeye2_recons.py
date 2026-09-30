


import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
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


def torch_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().float().clamp(0, 1)
    return transforms.ToPILImage()(x)


def resize_batch(x: torch.Tensor, size: int) -> torch.Tensor:
    return transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR)(x)


def corr_features(x: torch.Tensor) -> torch.Tensor:
    x = x.float().flatten(1)
    x = x - x.mean(dim=1, keepdim=True)
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return x


@torch.no_grad()
def two_way_identification(all_recons, all_images, model, preprocess, feature_layer=None, device="cuda", batch_size=64):
    def extract_features(images):
        feats = []
        for start in tqdm(range(0, len(images), batch_size), leave=False):
            batch = preprocess(images[start : start + batch_size]).to(device)
            out = model(batch)
            if feature_layer is not None:
                out = out[feature_layer]
            feats.append(out.float().flatten(1).cpu())
        return torch.cat(feats, dim=0)

    preds = corr_features(extract_features(all_recons))
    reals = corr_features(extract_features(all_images))
    r = reals @ preds.T
    congruents = torch.diag(r)
    success = r < congruents[None, :]
    return success.sum(dim=0).float().mean().item() / (len(all_images) - 1)


def rgb_to_gray_np(x: torch.Tensor) -> np.ndarray:
    arr = x.permute(0, 2, 3, 1).cpu().numpy()
    return arr[..., 0] * 0.2125 + arr[..., 1] * 0.7154 + arr[..., 2] * 0.0721


def ssim_grayscale(img, rec, sigma=1.5, data_range=1.0):



    k1, k2 = 0.01, 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    ux = gaussian_filter(img, sigma)
    uy = gaussian_filter(rec, sigma)
    uxx = gaussian_filter(img * img, sigma)
    uyy = gaussian_filter(rec * rec, sigma)
    uxy = gaussian_filter(img * rec, sigma)
    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy
    ssim_map = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    return float(np.mean(ssim_map))


@torch.no_grad()
def extract_openclip_features(images, model_name, pretrained, size, device, batch_size=32):
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval().requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    feats = []
    for start in tqdm(range(0, len(images), batch_size), leave=False):
        batch = preprocess(images[start : start + batch_size]).to(device)
        feats.append(model.encode_image(batch).float().cpu())
    return torch.cat(feats, dim=0)


def export_comparisons(images, recons, out_dir: Path, count: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    thumb = 256
    for i in range(min(count, len(images))):
        gt = torch_to_pil(images[i]).resize((thumb, thumb), Image.Resampling.LANCZOS)
        rec = torch_to_pil(recons[i]).resize((thumb, thumb), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb * 2, thumb + 28), "white")
        canvas.paste(gt, (0, 28))
        canvas.paste(rec, (thumb, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), f"{i:04d} original", fill=(0, 0, 0))
        draw.text((thumb + 8, 7), "reconstruction", fill=(0, 0, 0))
        canvas.save(out_dir / f"{i:04d}_original_vs_recon.png")

    grid_cols = 5
    rows = int(np.ceil(min(count, len(images)) / grid_cols))
    cell_w, cell_h = thumb * 2, thumb + 28
    grid = Image.new("RGB", (grid_cols * cell_w, rows * cell_h), "white")
    for i in range(min(count, len(images))):
        panel = Image.open(out_dir / f"{i:04d}_original_vs_recon.png").convert("RGB")
        grid.paste(panel, ((i % grid_cols) * cell_w, (i // grid_cols) * cell_h))
    grid.save(out_dir / "grid_100_original_vs_recon.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_root", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--model_name", default="final_subj01_pretrained_40sess_24bs")
    parser.add_argument("--recons_kind", choices=["enhanced", "raw"], default="enhanced")
    parser.add_argument("--comparison_count", type=int, default=100)
    parser.add_argument("--comparison_subdir", default="recon_100")
    parser.add_argument("--skip_retrieval", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    artifact_root = Path(args.artifact_root)
    result_dir = Path(args.result_dir)
    metrics_dir = result_dir / "metrics"
    recon_dir = result_dir / args.comparison_subdir
    metrics_dir.mkdir(parents=True, exist_ok=True)

    all_images = torch.load(
        artifact_root / "evals" / "all_images.pt", map_location="cpu", weights_only=False
    ).float()
    recons_name = (
        f"{args.model_name}_all_enhancedrecons.pt"
        if args.recons_kind == "enhanced"
        else f"{args.model_name}_all_recons.pt"
    )
    all_recons = torch.load(
        artifact_root / "evals" / args.model_name / recons_name, map_location="cpu", weights_only=False
    ).float()
    all_images = resize_batch(all_images, 256)
    all_recons = resize_batch(all_recons, 256)

    blur_path = artifact_root / "evals" / args.model_name / f"{args.model_name}_all_blurryrecons.pt"
    used_weighted_enhanced = False
    if args.recons_kind == "enhanced" and blur_path.exists():
        blurry = resize_batch(torch.load(blur_path, map_location="cpu", weights_only=False).float(), 256)
        all_recons = all_recons * 0.75 + blurry * 0.25
        used_weighted_enhanced = True

    export_comparisons(all_images, all_recons, recon_dir, args.comparison_count)

    metrics = {}
    pix_images = resize_batch(all_images, 425).reshape(len(all_images), -1)
    pix_recons = resize_batch(all_recons, 425).reshape(len(all_recons), -1)
    metrics["PixCorr"] = float(
        np.mean([np.corrcoef(pix_images[i].numpy(), pix_recons[i].numpy())[0, 1] for i in range(len(all_images))])
    )

    gray_images = rgb_to_gray_np(resize_batch(all_images, 425))
    gray_recons = rgb_to_gray_np(resize_batch(all_recons, 425))
    metrics["SSIM"] = float(
        np.mean([ssim_grayscale(gray_images[i], gray_recons[i]) for i in tqdm(range(len(all_images)), desc="SSIM")])
    )

    alex_model = create_feature_extractor(
        alexnet(weights=AlexNet_Weights.IMAGENET1K_V1), return_nodes=["features.4", "features.11"]
    ).to(device)
    alex_model.eval().requires_grad_(False)
    imagenet_norm_256 = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    metrics["AlexNet(2)"] = two_way_identification(
        all_recons, all_images, alex_model, imagenet_norm_256, "features.4", device
    )
    metrics["AlexNet(5)"] = two_way_identification(
        all_recons, all_images, alex_model, imagenet_norm_256, "features.11", device
    )

    inception_model = create_feature_extractor(
        inception_v3(weights=Inception_V3_Weights.DEFAULT), return_nodes=["avgpool"]
    ).to(device)
    inception_model.eval().requires_grad_(False)
    inception_preprocess = transforms.Compose(
        [
            transforms.Resize((342, 342), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    metrics["InceptionV3"] = two_way_identification(
        all_recons, all_images, inception_model, inception_preprocess, "avgpool", device, batch_size=32
    )

    clip_gt = corr_features(extract_openclip_features(all_images, "ViT-L-14", "openai", 224, device))
    clip_fake = corr_features(extract_openclip_features(all_recons, "ViT-L-14", "openai", 224, device))
    r = clip_gt @ clip_fake.T
    metrics["CLIP"] = float(((r < torch.diag(r)[None, :]).sum(dim=0).float().mean() / (len(all_images) - 1)).item())

    eff_model = create_feature_extractor(
        efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT), return_nodes=["avgpool"]
    ).to(device)
    eff_model.eval().requires_grad_(False)
    eff_preprocess = transforms.Compose(
        [
            transforms.Resize((255, 255), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    with torch.no_grad():
        gt = []
        fake = []
        for start in tqdm(range(0, len(all_images), 64), desc="EffNet", leave=False):
            gt.append(eff_model(eff_preprocess(all_images[start : start + 64]).to(device))["avgpool"].flatten(1).cpu())
            fake.append(eff_model(eff_preprocess(all_recons[start : start + 64]).to(device))["avgpool"].flatten(1).cpu())
    gt = torch.cat(gt).numpy()
    fake = torch.cat(fake).numpy()
    metrics["EffNet-B"] = float(np.array([sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]).mean())

    swav_model = torch.hub.load(
        "facebookresearch/swav:main",
        "resnet50",
        trust_repo=True,
        skip_validation=True,
    )
    swav_model = create_feature_extractor(swav_model, return_nodes=["avgpool"]).to(device)
    swav_model.eval().requires_grad_(False)
    swav_preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    with torch.no_grad():
        gt = []
        fake = []
        for start in tqdm(range(0, len(all_images), 64), desc="SwAV", leave=False):
            gt.append(swav_model(swav_preprocess(all_images[start : start + 64]).to(device))["avgpool"].flatten(1).cpu())
            fake.append(swav_model(swav_preprocess(all_recons[start : start + 64]).to(device))["avgpool"].flatten(1).cpu())
    gt = torch.cat(gt).numpy()
    fake = torch.cat(fake).numpy()
    metrics["SwAV"] = float(np.array([sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]).mean())

    clipvox_path = artifact_root / "evals" / args.model_name / f"{args.model_name}_all_clipvoxels.pt"
    if not args.skip_retrieval and clipvox_path.exists():
        from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

        embedder = FrozenOpenCLIPImageEmbedder(
            arch="ViT-bigG-14",
            version="laion2b_s39b_b160k",
            output_tokens=True,
            only_tokens=True,
        ).to(device)
        all_clipvoxels = torch.load(clipvox_path, map_location="cpu", weights_only=False).float()
        percent_fwds = []
        percent_bwds = []
        rng = np.random.default_rng(42)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == "cuda", dtype=torch.float16):
            for _ in tqdm(range(30), desc="Retrieval"):
                idx = rng.choice(np.arange(len(all_images)), size=300, replace=False)
                emb = embedder(all_images[idx].to(device)).float().reshape(300, -1)
                emb_ = all_clipvoxels[idx].to(device).float().reshape(300, -1)
                emb = nn.functional.normalize(emb, dim=-1)
                emb_ = nn.functional.normalize(emb_, dim=-1)
                fwd = emb_ @ emb.T
                bwd = emb @ emb_.T
                labels = torch.arange(300, device=device)
                percent_fwds.append((fwd.argmax(dim=1) == labels).float().mean().item())
                percent_bwds.append((bwd.argmax(dim=1) == labels).float().mean().item())
        metrics["FwdRetrieval"] = float(np.mean(percent_fwds))
        metrics["BwdRetrieval"] = float(np.mean(percent_bwds))

    df = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
    df.to_csv(metrics_dir / f"{args.model_name}_{args.recons_kind}_metrics.csv", index=False)
    with open(metrics_dir / f"{args.model_name}_{args.recons_kind}_metrics.json", "w") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "recons_kind": args.recons_kind,
                "used_weighted_enhanced": used_weighted_enhanced,
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    print(df.to_string(index=False))
    print(f"saved metrics to {metrics_dir}")
    print(f"saved comparisons to {recon_dir}")


if __name__ == "__main__":
    main()
