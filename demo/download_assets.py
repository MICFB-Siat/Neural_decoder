

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parent
CUSTOM_FILES = [
    "assets/demo_data.h5",
    "assets/MANIFEST.json",
    "assets/nsd_sub01_test.h5",
    "assets/NSD_SUB01_MANIFEST.json",
    "weights/classification_stage2_eeg_tiny.pth",
    "weights/alice_qformer_poe_sublayer.pt",
    "weights/alice_projector.pt",
    "weights/alice_lora/adapter_model.safetensors",
    "weights/alice_lora/adapter_config.json",
    "weights/nsd_sub01/brain_final.pt",
    "weights/nsd_sub01/diffusion_prior.pt",
    "weights/nsd_sub01/sd_image_var_autoenc.pth",
    "weights/nsd_sub01/unclip6_epoch0_step110000.ckpt",
    "weights/nsd_sub01/open_clip_vit_l14_openai.safetensors",
    "assets/spacetop_sub0018/sub-0018_features.h5",
    "assets/spacetop_sub0018/sub-0018_manifest.npz",
    "assets/spacetop_sub0018/sub-0018_reference_prediction.npz",
    "weights/spacetop_sub0018/fold-04_checkpoint.pt",
]
REMOTE_TO_LOCAL = {"assets/demo_data.h5": "assets/inference_data.h5"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default=os.environ.get("EIGEN_HF_REPO", "SSp1ash/Eigenbrain"),
        required=False,
        help="Hugging Face model repository containing this project's assets",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--skip-nsd", action="store_true", help="Skip the large NSD image-reconstruction files")
    parser.add_argument("--cache-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    checkpoint_manifest = json.loads((ROOT / "checkpoint_manifest.json").read_text())
    checkpoint_paths = {item["local_path"]: item["remote_path"]
                        for item in checkpoint_manifest["files"]}

    CUSTOM_FILES.append("weights/nsd_sub01_encoding/model.npz")
    files = [f for f in CUSTOM_FILES if not (args.skip_nsd and f.startswith(("assets/nsd_", "assets/NSD_", "weights/nsd_")))]
    for filename in files:
        destination = args.cache_dir / REMOTE_TO_LOCAL.get(filename, filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        remote_filename = checkpoint_paths.get(filename, filename)
        revision = (checkpoint_manifest["revision"]
                    if filename in checkpoint_paths and args.revision == "main"
                    and args.repo == checkpoint_manifest["repo_id"] else args.revision)
        downloaded = Path(hf_hub_download(args.repo, filename=remote_filename,
                                         revision=revision, local_dir=args.cache_dir))
        if downloaded != destination and downloaded.is_file():
            downloaded.replace(destination)
        print(f"downloaded {filename} -> {destination}")

    models_dir = args.cache_dir / "models"
    snapshot_download("microsoft/Phi-4-mini-instruct", revision=args.revision,
                      local_dir=models_dir / "Phi-4-mini-instruct")
    snapshot_download("BAAI/bge-m3", revision=args.revision,
                      local_dir=models_dir / "bge-m3")
    snapshot_download(
        args.repo,
        revision=args.revision,
        repo_type="model",
        local_dir=args.cache_dir,
        allow_patterns=[
            "assets/nsd_sub01_encoding/**",
        ],
    )
    print(f"base models -> {models_dir}")


if __name__ == "__main__":
    main()
