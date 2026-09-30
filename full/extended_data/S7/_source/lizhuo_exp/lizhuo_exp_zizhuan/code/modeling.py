
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        return cls(np.nanmean(x, axis=0), np.nanstd(x, axis=0) + 1e-6)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


class GaussianExpert(nn.Module):


    def __init__(self, d_in: int, d_latent: int = 128, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.mu = nn.Linear(hidden, d_latent)
        self.logvar = nn.Linear(hidden, d_latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.mu(h), self.logvar(h).clamp(-5.0, 5.0)


class GaussianPoE(nn.Module):


    def __init__(self, d_in: int, head_factory, latent: int = 128, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.obs_expert = GaussianExpert(d_in, latent, hidden, dropout)
        self.indiv_expert = GaussianExpert(d_in, latent, hidden, dropout)
        self.obs_head = head_factory(latent)
        self.indiv_head = head_factory(latent)
        self.poe_head = head_factory(latent)

    def forward(self, obs: torch.Tensor, indiv: torch.Tensor) -> dict[str, torch.Tensor]:
        mu_o, lv_o = self.obs_expert(obs)
        mu_i, lv_i = self.indiv_expert(indiv)
        tau_o, tau_i = torch.exp(-lv_o), torch.exp(-lv_i)
        var_p = (1.0 + tau_o + tau_i).reciprocal()
        mu_p = var_p * (tau_o * mu_o + tau_i * mu_i)
        return {
            "obs": self.obs_head(mu_o),
            "indiv": self.indiv_head(mu_i),
            "poe": self.poe_head(mu_p),
            "mu_poe": mu_p,
            "var_poe": var_p,
            "alpha_obs": tau_o / (tau_o + tau_i + 1e-8),
        }


class GaussianSingleExpert(nn.Module):


    def __init__(self, d_in: int, head_factory, latent: int = 128, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.expert = GaussianExpert(d_in, latent, hidden, dropout)
        self.head = head_factory(latent)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.expert(x)
        return {"pred": self.head(mu), "mu": mu, "var": torch.exp(logvar)}


def coral_targets(y: torch.Tensor, n_classes: int) -> torch.Tensor:

    thresholds = torch.arange(1, n_classes, device=y.device, dtype=y.dtype)
    return (y[..., None] > thresholds).float()


def ordinal_expected(logits: torch.Tensor) -> torch.Tensor:

    return 1.0 + torch.sigmoid(logits).sum(dim=-1)


def ordinal_loss(logits: torch.Tensor, y: torch.Tensor, n_classes: int) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, coral_targets(y, n_classes))


def poe_loss(out: dict[str, torch.Tensor], y: torch.Tensor, task_kind: str, aux_weight: float, kl_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    if task_kind == "ordinal":
        main = ordinal_loss(out["poe"], y, 3)
        aux = ordinal_loss(out["obs"], y, 3) + ordinal_loss(out["indiv"], y, 3)
    else:
        main = F.smooth_l1_loss(out["poe"], y)
        aux = F.smooth_l1_loss(out["obs"], y) + F.smooth_l1_loss(out["indiv"], y)
    mu, var = out["mu_poe"], out["var_poe"]
    kl = 0.5 * (mu.square() + var - torch.log(var + 1e-8) - 1.0).mean()
    return main + aux_weight * aux + kl_weight * kl, main


def single_loss(out: dict[str, torch.Tensor], y: torch.Tensor, task_kind: str, kl_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    main = ordinal_loss(out["pred"], y, 3) if task_kind == "ordinal" else F.smooth_l1_loss(out["pred"], y)
    mu, var = out["mu"], out["var"]
    kl = 0.5 * (mu.square() + var - torch.log(var + 1e-8) - 1.0).mean()
    return main + kl_weight * kl, main


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    if len(y) < 3:
        return {"n": int(len(y)), "pearson_r": np.nan, "spearman_rho": np.nan,
                "mae": np.nan, "rmse": np.nan, "r2": np.nan}
    err = pred - y
    denominator = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "pearson_r": float(pearsonr(y, pred).statistic) if y.std() > 1e-12 and pred.std() > 1e-12 else np.nan,
        "spearman_rho": float(spearmanr(y, pred).statistic) if y.std() > 1e-12 and pred.std() > 1e-12 else np.nan,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1.0 - np.sum(err ** 2) / denominator) if denominator > 1e-12 else np.nan,
    }


def fisher_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return float("nan")
    return float(np.tanh(np.arctanh(np.clip(arr, -1 + 1e-7, 1 - 1e-7)).mean()))
