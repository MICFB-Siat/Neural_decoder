








import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import webdataset as wds
from tqdm import tqdm
from torchvision import transforms

import utils
from models import BrainDiffusionPrior, BrainNetwork, PriorNetwork


class RidgeRegression(torch.nn.Module):
    def __init__(self, input_sizes, out_features):
        super().__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_size, out_features) for input_size in input_sizes]
        )

    def forward(self, x, subj_idx):
        return self.linears[subj_idx](x[:, 0]).unsqueeze(1)


class MindEyeModule(nn.Module):
    def forward(self, x):
        return x


def num_test_for_subject(subj: int) -> int:
    return 2371 if subj in (3, 6) else 2188 if subj in (4, 8) else 3000


def load_test_arrays(data_path: str, subj: int):
    with h5py.File(f"{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5", "r") as f:
        betas = torch.from_numpy(f["betas"][:]).float().cpu()

    num_test = num_test_for_subject(subj)
    test_url = f"{data_path}/wds/subj0{subj}/new_test/0.tar"
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


def build_model(num_voxels: int, hidden_dim: int, n_blocks: int, device: torch.device):
    clip_seq_dim = 256
    clip_emb_dim = 1664

    model = MindEyeModule()
    model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)
    model.backbone = BrainNetwork(
        h=hidden_dim,
        in_dim=hidden_dim,
        seq_len=1,
        clip_size=clip_emb_dim,
        out_dim=clip_emb_dim * clip_seq_dim,
        n_blocks=n_blocks,
    )

    prior_network = PriorNetwork(
        dim=clip_emb_dim,
        depth=6,
        dim_head=52,
        heads=clip_emb_dim // 52,
        causal=False,
        num_tokens=clip_seq_dim,
        learned_query_mode="pos_emb",
    )
    model.diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=clip_emb_dim,
        condition_on_text_encodings=False,
        timesteps=100,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    )
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_root", required=True)
    parser.add_argument("--model_name", default="final_subj01_pretrained_40sess_24bs")
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    utils.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(args.artifact_root, "evals", args.model_name)
    os.makedirs(out_dir, exist_ok=True)
    clip_out = os.path.join(out_dir, f"{args.model_name}_all_clipvoxels.pt")
    blur_out = os.path.join(out_dir, f"{args.model_name}_all_blurryrecons.pt")

    test_voxels, test_images_idx, num_voxels = load_test_arrays(args.artifact_root, args.subj)
    model = build_model(num_voxels, args.hidden_dim, args.n_blocks, device)
    ckpt_path = os.path.join(args.artifact_root, "train_logs", args.model_name, "last.pth")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    model.eval().requires_grad_(False)

    from diffusers import AutoencoderKL

    autoenc = AutoencoderKL(
        down_block_types=[
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ],
        up_block_types=[
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    autoenc.load_state_dict(
        torch.load(
            os.path.join(args.artifact_root, "sd_image_var_autoenc.pth"),
            map_location="cpu",
            weights_only=False,
        )
    )
    autoenc.eval().requires_grad_(False).to(device)

    uniq_images = np.unique(test_images_idx)
    all_clipvoxels = []
    all_blurryrecons = []

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16):
        for start in tqdm(range(0, len(uniq_images), args.batch_size)):
            batch_imgs = uniq_images[start : start + args.batch_size]
            voxels = []
            for uniq_img in batch_imgs:
                locs = np.where(test_images_idx == uniq_img)[0]
                if len(locs) == 1:
                    locs = locs.repeat(3)
                elif len(locs) == 2:
                    locs = locs.repeat(2)[:3]
                assert len(locs) == 3
                voxels.append(test_voxels[locs])
            voxel = torch.stack(voxels, dim=0).to(device)

            clip_voxels = None
            blurry_image_enc = None
            for rep in range(3):
                voxel_ridge = model.ridge(voxel[:, [rep]], 0)
                _backbone0, clip_voxels0, blurry_image_enc0 = model.backbone(voxel_ridge)
                if rep == 0:
                    clip_voxels = clip_voxels0
                    blurry_image_enc = blurry_image_enc0[0]
                else:
                    clip_voxels = clip_voxels + clip_voxels0
                    blurry_image_enc = blurry_image_enc + blurry_image_enc0[0]
            clip_voxels = clip_voxels / 3
            blurry_image_enc = blurry_image_enc / 3

            blurred = (autoenc.decode(blurry_image_enc / 0.18215).sample / 2 + 0.5).clamp(0, 1)
            blurred = transforms.Resize((256, 256))(blurred).float()
            all_clipvoxels.append(clip_voxels.float().cpu())
            all_blurryrecons.append(blurred.cpu())

    all_clipvoxels = torch.cat(all_clipvoxels, dim=0)
    all_blurryrecons = torch.cat(all_blurryrecons, dim=0)
    torch.save(all_clipvoxels, clip_out)
    torch.save(all_blurryrecons, blur_out)
    print(f"saved {clip_out} {tuple(all_clipvoxels.shape)}")
    print(f"saved {blur_out} {tuple(all_blurryrecons.shape)}")


if __name__ == "__main__":
    main()
