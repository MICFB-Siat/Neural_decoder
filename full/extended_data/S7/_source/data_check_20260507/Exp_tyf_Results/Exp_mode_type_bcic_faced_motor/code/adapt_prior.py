

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from experiment_config import (
    ADAPT_ARGS,
    CLASSIFIER,
    DATASETS,
    INIT_CHECKPOINT,
    MODE_TYPES,
    PRETRAIN_ROOT,
    assert_classifier_frozen,
    checkpoint_path,
    data_dir,
    sha256_file,
)


def load_classifier_module():
    spec = importlib.util.spec_from_file_location("mode_type_classifier", CLASSIFIER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".partial.{os.getpid()}")
    with src.open("rb") as fi, tmp.open("wb") as fo:
        shutil.copyfileobj(fi, fo, length=16 << 20)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, dst)


def atomic_json(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def completed(ckpt: Path, meta_path: Path, expected: dict) -> bool:
    if not ckpt.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if any(meta.get(k) != v for k, v in expected.items()):
            return False
        saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        return (
            saved.get("dataset") == expected["dataset"]
            and saved.get("mode_type") == expected["mode_type"]
            and saved.get("n_samples") == expected["n_samples"]
            and saved.get("n_modes") == expected["n_modes"]
            and "encoder" in saved
        )
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--mode-type", choices=MODE_TYPES, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=ADAPT_ARGS["epochs"])
    ap.add_argument("--batch-size", type=int, default=ADAPT_ARGS["batch_size"])
    ap.add_argument("--lr", type=float, default=ADAPT_ARGS["lr"])
    ap.add_argument("--seed", type=int, default=ADAPT_ARGS["seed"])
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--subject-limit", type=int, default=None, help="Smoke-test only")
    args = ap.parse_args()

    classifier_sha = assert_classifier_frozen()
    cfg = DATASETS[args.dataset]
    mode_data = (args.data_dir or data_dir(args.dataset, args.mode_type)).resolve()
    output = (args.output_dir or (PRETRAIN_ROOT / args.dataset / args.mode_type)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ckpt = output / f"prior_{args.dataset}_{args.mode_type}_mae{args.epochs}.pt"
    meta_path = ckpt.with_suffix(".json")
    manifest_path = mode_data.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest missing: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    expected = {
        "dataset": args.dataset,
        "mode_type": args.mode_type,
        "n_modes": cfg["n_modes"],
        "n_samples": cfg["n_samples"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "mask_ratio": ADAPT_ARGS["mask_ratio"],
        "val_ratio": ADAPT_ARGS["val_ratio"],
        "classifier_sha256": classifier_sha,
        "data_manifest_sha256": manifest_sha,
        "subject_limit": args.subject_limit,
    }
    if completed(ckpt, meta_path, expected):
        print(f"SKIP complete checkpoint {ckpt}", flush=True)
        return

    log_path = output / f"adapt_{args.dataset}_{args.mode_type}_mae{args.epochs}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    log = logging.getLogger("adapt")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    module = load_classifier_module()



    load_dir = mode_data
    temp_links = None
    if args.subject_limit is not None:
        files = sorted(mode_data.glob("sub-*.h5"))[: args.subject_limit]
        if not files:
            raise FileNotFoundError(f"No H5 files in {mode_data}")
        temp_links = tempfile.TemporaryDirectory(prefix="mode_type_adapt_subjects_", dir="/tmp")
        load_dir = Path(temp_links.name)
        for path in files:
            os.symlink(path, load_dir / path.name)

    dataset = module._load_modes_hetero(str(load_dir), "eeg", device)
    if dataset.T != cfg["n_samples"] or set(dataset.Ks) != {cfg["n_modes"]}:
        raise ValueError(f"Loaded shape mismatch: T={dataset.T}, Ks={dataset.Ks}")

    train_chunks, val_chunks = [], []
    names, eigvals = [], []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for chunk, name, eigval in zip(dataset.chunks, dataset.names, dataset.eigvals):
        order = torch.randperm(len(chunk), generator=generator, device=device)
        n_val = max(1, round(len(chunk) * ADAPT_ARGS["val_ratio"]))
        val_chunks.append(chunk[order[:n_val]])
        train_chunks.append(chunk[order[n_val:]])
        names.append(name)
        eigvals.append(eigval)
    train = module._HeteroKDataset(train_chunks, names, eigvals)
    val = module._HeteroKDataset(val_chunks, names, eigvals)

    init = torch.load(INIT_CHECKPOINT, map_location=device, weights_only=False)
    encoder = module.EEGMEGPriorEigenval3(
        patch_size=module.PATCH_SIZE,
        d_model=init["d_model"],
        n_heads=8,
        n_factor_layers=init.get("n_factor_layers", 2),
        n_time_layers=init.get("n_time_layers", 4),
    ).to(device)
    state = {
        (k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k): v
        for k, v in init["encoder"].items()
    }
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Initialization mismatch missing={missing}, unexpected={unexpected}")
    decoder = module.EEGMEGMAEDecoderEigenval3(
        d_model=init["d_model"],
        d_dec=init["d_model"] // 2,
        n_heads=4,
        n_layers=2,
        patch_size=module.PATCH_SIZE,
    ).to(device)

    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    warmup = max(1, int(args.epochs * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: epoch / warmup if epoch < warmup else 0.5 * (
            1 + np.cos(np.pi * (epoch - warmup) / max(1, args.epochs - warmup))
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(args.seed)
    best, best_epoch = float("inf"), 0
    n_patches = dataset.T // module.PATCH_SIZE
    saved_stage = None
    log.info(
        "Initialized from %s; dataset=%s mode=%s N=%d subjects=%d K=%s T=%d",
        INIT_CHECKPOINT, args.dataset, args.mode_type, dataset.N, len(dataset.chunks), dataset.Ks, dataset.T,
    )
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        train_total = train_n = 0
        for x, _, eigval in train.iter_batches(args.batch_size, rng=rng):
            batch, n_modes, _ = x.shape
            mask = module.brainomni_2d_mask(
                batch, n_modes, n_patches, ADAPT_ARGS["mask_ratio"], device
            )
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                target = module.patchify(x, module.PATCH_SIZE)
                prediction = decoder(encoder.encode_full_grid(x, eigval, mask))
                masked = mask.float().unsqueeze(-1)
                loss = ((prediction.float() - target.float()) ** 2 * masked).sum() / (
                    masked.sum() * module.PATCH_SIZE + 1e-8
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_total += loss.item() * batch
            train_n += batch
        scheduler.step()

        encoder.eval()
        decoder.eval()
        val_total = val_n = 0
        with torch.no_grad():
            for x, _, eigval in val.iter_batches(args.batch_size, shuffle=False):
                batch, n_modes, _ = x.shape
                mask = module.brainomni_2d_mask(
                    batch, n_modes, n_patches, ADAPT_ARGS["mask_ratio"], device
                )
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    target = module.patchify(x, module.PATCH_SIZE)
                    prediction = decoder(encoder.encode_full_grid(x, eigval, mask))
                    masked = mask.float().unsqueeze(-1)
                    loss = ((prediction.float() - target.float()) ** 2 * masked).sum() / (
                        masked.sum() * module.PATCH_SIZE + 1e-8
                    )
                val_total += loss.item() * batch
                val_n += batch
        val_loss = val_total / val_n
        if not np.isfinite(val_loss):
            raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            payload = {
                "encoder": encoder.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "dataset": args.dataset,
                "mode_type": args.mode_type,
                "n_modes": cfg["n_modes"],
                "n_samples": dataset.T,
                "d_model": init["d_model"],
                "n_factor_layers": init.get("n_factor_layers", 2),
                "n_time_layers": init.get("n_time_layers", 4),
                "patch_size": module.PATCH_SIZE,
                "mode_signature": f"{args.mode_type} native lambda + first non-zero spectral gap",
                "adapted_from": str(INIT_CHECKPOINT),
                "classifier_sha256": classifier_sha,
                "data_manifest_sha256": manifest_sha,
            }
            stage_dir = Path(os.environ.get("MODE_TYPE_CKPT_STAGE", "/tmp/mode_type_bcic_faced_motor_ckpt"))
            stage_dir.mkdir(parents=True, exist_ok=True)
            if saved_stage is None:
                saved_stage = stage_dir / f"{args.dataset}_{args.mode_type}_{os.getpid()}.pt"
            torch.save(payload, saved_stage, _use_new_zipfile_serialization=False)
        log.info(
            "epoch %02d/%d train=%.6f val=%.6f best=%.6f@%d",
            epoch, args.epochs, train_total / train_n, val_loss, best, best_epoch,
        )

    if saved_stage is None:
        raise RuntimeError("No finite checkpoint was produced")
    atomic_copy(saved_stage, ckpt)
    saved_stage.unlink(missing_ok=True)
    metadata = {
        **expected,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "best_epoch": best_epoch,
        "best_val_loss": best,
        "n_subjects_loaded": len(dataset.chunks),
        "n_trials_loaded": dataset.N,
        "adaptation_scope": "all target subjects; per-subject 90/10 label-free split",
    }
    atomic_json(meta_path, metadata)
    if not completed(ckpt, meta_path, expected):
        raise RuntimeError(f"Checkpoint post-write validation failed: {ckpt}")
    log.info("DONE checkpoint=%s sha256=%s", ckpt, metadata["checkpoint_sha256"])
    if temp_links is not None:
        temp_links.cleanup()


if __name__ == "__main__":
    main()
