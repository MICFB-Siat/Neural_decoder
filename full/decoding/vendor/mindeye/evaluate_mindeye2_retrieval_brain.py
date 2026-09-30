


import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import webdataset as wds
from PIL import Image
from torchmetrics import PearsonCorrCoef
from torchvision import transforms
from tqdm import tqdm

from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
from models import GNet8_Encoder


def num_test_for_subject(subj):
    return 2371 if subj in (3, 6) else 2188 if subj in (4, 8) else 3000


def load_test_voxels_and_indices(artifact_root, subj):
    with h5py.File(artifact_root / f"betas_all_subj0{subj}_fp32_renorm.hdf5", "r") as f:
        betas = torch.from_numpy(f["betas"][:]).float().cpu()

    num_test = num_test_for_subject(subj)
    test_url = str(artifact_root / "wds" / f"subj0{subj}" / "new_test" / "0.tar")
    test_data = (
        wds.WebDataset(test_url, resampled=False, nodesplitter=lambda urls: urls)
        .decode("torch")
        .rename(
            behav="behav.npy",
            past_behav="past_behav.npy",
            future_behav="future_behav.npy",
            olds_behav="olds_behav.npy",
        )
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
    )
    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True
    )
    test_images_idx = []
    test_voxels = None
    for test_i, (behav, _past, _future, _old) in enumerate(test_dl):
        test_voxels = betas[behav[:, 0, 5].cpu().long()]
        test_images_idx = np.append(test_images_idx, behav[:, 0, 0].cpu().numpy())
    assert test_voxels is not None
    assert (test_i + 1) * num_test == len(test_voxels) == len(test_images_idx)
    return test_voxels, test_images_idx.astype(int), betas.shape[-1]


def average_test_voxels(test_voxels, test_images_idx, num_voxels):
    uniq_imgs = np.unique(test_images_idx)
    averaged = torch.zeros((len(uniq_imgs), num_voxels))
    for i, uniq_img in enumerate(uniq_imgs):
        locs = np.where(test_images_idx == uniq_img)[0]
        if len(locs) == 1:
            locs = locs.repeat(3)
        elif len(locs) == 2:
            locs = locs.repeat(2)[:3]
        assert len(locs) == 3
        averaged[i] = torch.mean(test_voxels[None, locs], dim=1)
    return averaged


@torch.no_grad()
def compute_retrieval(artifact_root, model_name, device):
    all_images = torch.load(artifact_root / "evals" / "all_images.pt", map_location="cpu", weights_only=False).float()
    all_clipvoxels = torch.load(
        artifact_root / "evals" / model_name / f"{model_name}_all_clipvoxels.pt",
        map_location="cpu",
        weights_only=False,
    ).float()
    embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    ).to(device)
    percent_fwds = []
    percent_bwds = []
    rng = np.random.default_rng(42)
    with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16):
        for _ in tqdm(range(30), desc="Retrieval"):
            idx = rng.choice(np.arange(len(all_images)), size=300, replace=False)
            emb = embedder(all_images[idx].to(device)).float().reshape(300, -1)
            emb_ = all_clipvoxels[idx].to(device).float().reshape(300, -1)
            emb = nn.functional.normalize(emb, dim=-1)
            emb_ = nn.functional.normalize(emb_, dim=-1)
            labels = torch.arange(300, device=device)
            fwd = emb_ @ emb.T
            bwd = emb @ emb_.T
            percent_fwds.append((fwd.argmax(dim=1) == labels).float().mean().item())
            percent_bwds.append((bwd.argmax(dim=1) == labels).float().mean().item())
    return {
        "FwdRetrieval": float(np.mean(percent_fwds)),
        "BwdRetrieval": float(np.mean(percent_bwds)),
    }


@torch.no_grad()
def compute_brain_corr(artifact_root, model_name, subj, device):
    recons_path = artifact_root / "evals" / model_name / f"{model_name}_all_enhancedrecons.pt"
    all_recons = torch.load(recons_path, map_location="cpu", weights_only=False).float()
    all_recons = transforms.Resize((256, 256))(all_recons)
    blur_path = artifact_root / "evals" / model_name / f"{model_name}_all_blurryrecons.pt"
    if blur_path.exists():
        blurry = transforms.Resize((256, 256))(
            torch.load(blur_path, map_location="cpu", weights_only=False).float()
        )
        all_recons = all_recons * 0.75 + blurry * 0.25

    test_voxels, test_images_idx, num_voxels = load_test_voxels_and_indices(artifact_root, subj)
    test_voxels_averaged = average_test_voxels(test_voxels, test_images_idx, num_voxels)

    with h5py.File(artifact_root / "brain_region_masks.hdf5", "r") as f:
        group = f[f"subj0{subj}"]
        subject_masks = {
            "nsd_general": group["nsd_general"][:],
            "V1": group["V1"][:],
            "V2": group["V2"][:],
            "V3": group["V3"][:],
            "V4": group["V4"][:],
            "higher_vis": group["higher_vis"][:],
        }

    recon_list = [transforms.ToPILImage()(img.detach().cpu()) for img in all_recons]
    gnet = GNet8_Encoder(device=device, subject=subj, model_path=str(artifact_root / "gnet_multisubject.pt"))
    beta_primes = gnet.predict(recon_list)
    pec = PearsonCorrCoef(num_outputs=len(recon_list))

    metrics = {}
    for region, mask in subject_masks.items():
        score = pec(test_voxels_averaged[:, mask].moveaxis(0, 1), beta_primes[:, mask].moveaxis(0, 1))
        metrics[f"Brain Corr. {region}"] = float(torch.mean(score))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_root", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--model_name", default="final_subj01_pretrained_40sess_24bs")
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--skip_retrieval", action="store_true")
    parser.add_argument("--skip_brain_corr", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    artifact_root = Path(args.artifact_root)
    result_dir = Path(args.result_dir)
    metrics_dir = result_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics = {}
    if not args.skip_retrieval:
        metrics.update(compute_retrieval(artifact_root, args.model_name, device))
    if not args.skip_brain_corr:
        metrics.update(compute_brain_corr(artifact_root, args.model_name, args.subj, device))

    df = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
    out_csv = metrics_dir / f"{args.model_name}_retrieval_brain_metrics.csv"
    out_json = metrics_dir / f"{args.model_name}_retrieval_brain_metrics.json"
    df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(df.to_string(index=False))
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
