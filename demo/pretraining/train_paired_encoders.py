from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import EEGMEGPriorEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--observation-key", default="language/bci_obs")
    parser.add_argument("--prior-key", default="language/bci_prior")
    parser.add_argument("--output", type=Path, default=Path("pretraining_outputs/paired_encoders.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--d-patch", type=int, default=512)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pair(path: Path, observation_key: str, prior_key: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        observation = np.asarray(handle[observation_key][...], dtype=np.float32)
        prior = np.asarray(handle[prior_key][...], dtype=np.float32)
    if observation.ndim == 2:
        observation = observation[:, None, :]
    if prior.ndim == 2:
        prior = prior[:, None, :]
    if observation.ndim != 3 or prior.ndim != 3:
        raise ValueError("observation and prior must have shape [samples, modes, features]")
    if observation.shape != prior.shape:
        raise ValueError(f"observation/prior shape mismatch: {observation.shape} vs {prior.shape}")
    if observation.shape[-1] % 64 != 0:
        raise ValueError("the final feature dimension must be divisible by patch_size=64")
    if not np.isfinite(observation).all() or not np.isfinite(prior).all():
        raise ValueError("input features contain non-finite values")
    return observation, prior


def standardize(values: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    mean = values[train_indices].mean(axis=(0, 1), keepdims=True)
    std = values[train_indices].std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)
    normalized = ((values - mean) / std).astype(np.float32)
    return normalized, {"mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist()}


class PairedEncoders(nn.Module):
    def __init__(self, d_model: int, d_patch: int, projection_dim: int):
        super().__init__()
        self.observation_encoder = EEGMEGPriorEncoder(d_model=d_model, d_patch=d_patch)
        self.prior_encoder = EEGMEGPriorEncoder(d_model=d_model, d_patch=d_patch)
        self.observation_projection = nn.Linear(d_model, projection_dim)
        self.prior_projection = nn.Linear(d_model, projection_dim)

    def forward(self, observation: torch.Tensor, prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observation = self.observation_encoder(observation).mean(dim=1)
        prior = self.prior_encoder(prior).mean(dim=1)
        observation = F.normalize(self.observation_projection(observation), dim=-1)
        prior = F.normalize(self.prior_projection(prior), dim=-1)
        return observation, prior


def contrastive_loss(observation: torch.Tensor, prior: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = observation @ prior.transpose(0, 1) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels)) / 2


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.temperature <= 0:
        raise ValueError("epochs, batch-size and temperature must be positive")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between 0 and 1")
    set_seed(args.seed)
    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    observation, prior = load_pair(args.data, args.observation_key, args.prior_key)
    n_samples = observation.shape[0]
    if n_samples < 2:
        raise ValueError("at least two paired samples are required")
    generator = np.random.default_rng(args.seed)
    indices = generator.permutation(n_samples)
    n_validation = max(1, int(round(n_samples * args.validation_fraction)))
    validation_indices = indices[:n_validation]
    train_indices = indices[n_validation:]
    if len(train_indices) < 2:
        raise ValueError("training split must contain at least two samples")
    observation, observation_scaler = standardize(observation, train_indices)
    prior, prior_scaler = standardize(prior, train_indices)

    train_set = TensorDataset(torch.from_numpy(observation[train_indices]), torch.from_numpy(prior[train_indices]))
    validation_set = TensorDataset(torch.from_numpy(observation[validation_indices]), torch.from_numpy(prior[validation_indices]))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = PairedEncoders(args.d_model, args.d_patch, args.projection_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_validation = float("inf")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_observation, batch_prior in train_loader:
            batch_observation = batch_observation.to(device)
            batch_prior = batch_prior.to(device)
            predicted_observation, predicted_prior = model(batch_observation, batch_prior)
            loss = contrastive_loss(predicted_observation, predicted_prior, args.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses = []
        with torch.inference_mode():
            for batch_observation, batch_prior in validation_loader:
                predicted_observation, predicted_prior = model(
                    batch_observation.to(device), batch_prior.to(device)
                )
                validation_losses.append(float(
                    contrastive_loss(predicted_observation, predicted_prior, args.temperature).cpu()
                ))
        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} validation_loss={validation_loss:.6f}", flush=True)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "paired-neural-encoders-v1",
        "training_type": "contrastive_pretraining",
        "observation_key": args.observation_key,
        "prior_key": args.prior_key,
        "input_shape": list(observation.shape[1:]),
        "d_model": args.d_model,
        "d_patch": args.d_patch,
        "projection_dim": args.projection_dim,
        "temperature": args.temperature,
        "seed": args.seed,
        "train_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "observation_scaler": observation_scaler,
        "prior_scaler": prior_scaler,
        "model_state": best_state,
        "history": history,
    }
    torch.save(checkpoint, args.output)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps({
        "output": str(args.output),
        "device": str(device),
        "n_samples": n_samples,
        "best_validation_loss": best_validation,
        "epochs": args.epochs,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
