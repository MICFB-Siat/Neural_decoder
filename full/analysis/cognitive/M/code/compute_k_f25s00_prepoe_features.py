








from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
OUT = HERE / "affect45_pearson_spearman" / "f25s00_prepoe_feature_audit"
CODE = (
    HERE.parents[2]
    / "lizhuo_exp_class2"
    / "SKIP"
    / "code"
    / "enc_mod"
)
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE))

import compute_k_l_affect45_pearson_spearman as affect45
import run_loso80_supervised_poe128__ as rsa44
from decode_cross_subject_residual_5fold import ArrayScaler


FOLD = 25
SEED = 0
RATING_SET = "five_happy_sad_afraid_disgusted_engaged"
N_QAP = 50_000
QAP_SEED = 20260725


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def correlations(neural_rdm: np.ndarray, rating_rdm: np.ndarray) -> tuple[float, float]:
    triangle = np.triu_indices(len(rating_rdm), 1)
    return (
        rsa44.finite_corr(neural_rdm[triangle], rating_rdm[triangle], "pearson"),
        rsa44.finite_corr(neural_rdm[triangle], rating_rdm[triangle], "spearman"),
    )


def shared_qap(
    rdms: dict[str, np.ndarray],
    rating_rdm: np.ndarray,
    n_permutations: int,
    seed: int,
) -> dict[str, dict[str, float]]:

    n_subjects = len(rating_rdm)
    triangle = np.triu_indices(n_subjects, 1)
    rating = rating_rdm[triangle].astype(np.float64)
    rating_p = rating - rating.mean()
    rating_p /= np.sqrt(np.sum(rating_p**2))
    rating_s = rankdata(rating, method="average")
    rating_s -= rating_s.mean()
    rating_s /= np.sqrt(np.sum(rating_s**2))

    result = {}
    for name, rdm in rdms.items():
        pearson, spearman = correlations(rdm, rating_rdm)
        result[name] = {
            "pearson_r": pearson,
            "spearman_r": spearman,
            "exceed_pearson": 0,
            "exceed_spearman": 0,
        }

    rng = np.random.default_rng(seed)
    for completed in range(0, n_permutations, 1000):
        count = min(1000, n_permutations - completed)
        orders = np.stack([rng.permutation(n_subjects) for _ in range(count)])
        for name, rdm in rdms.items():
            permuted = rdm[
                orders[:, triangle[0]], orders[:, triangle[1]]
            ].astype(np.float64)
            centered = permuted - permuted.mean(axis=1, keepdims=True)
            centered /= np.sqrt(np.sum(centered**2, axis=1, keepdims=True))
            null_p = centered @ rating_p
            ranks = np.apply_along_axis(rankdata, 1, permuted, method="average")
            ranks -= ranks.mean(axis=1, keepdims=True)
            ranks /= np.sqrt(np.sum(ranks**2, axis=1, keepdims=True))
            null_s = ranks @ rating_s
            result[name]["exceed_pearson"] += int(
                np.sum(np.abs(null_p) >= abs(result[name]["pearson_r"]))
            )
            result[name]["exceed_spearman"] += int(
                np.sum(np.abs(null_s) >= abs(result[name]["spearman_r"]))
            )

    for values in result.values():
        values["pearson_qap_p_two_sided_nominal"] = (
            1 + values.pop("exceed_pearson")
        ) / (n_permutations + 1)
        values["spearman_qap_p_two_sided_nominal"] = (
            1 + values.pop("exceed_spearman")
        ) / (n_permutations + 1)
        values["n_qap_permutations"] = n_permutations
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    subjects = pd.read_csv(rsa44.COHORT_CSV)["subject"].astype(str).tolist()
    videos = [
        (int(row.session), int(row.objective_id))
        for row in pd.read_csv(rsa44.VIDEOS_CSV).itertuples()
    ]
    arrays = rsa44.load_analysis_arrays(subjects, videos)
    rating7 = affect45.load_rating7(subjects, videos)
    five_indices = affect45.SETS[RATING_SET]
    complete = np.all(
        np.isfinite(rating7[:, :, list(five_indices)])
        & (rating7[:, :, list(five_indices)] >= 0),
        axis=2,
    ).all(axis=0)
    if int(complete.sum()) != 30:
        raise AssertionError(f"expected 30 complete videos, got {complete.sum()}")
    center, scale = affect45.load_rating_scaler(five_indices)
    rating = rating7[:, complete][:, :, list(five_indices)]
    rating_rdm = affect45.rating_k_rdm(rating, center, scale)

    checkpoint_path = rsa44.checkpoint_path(FOLD, SEED)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = rsa44.build_model_from_checkpoint(checkpoint, device)
    model.eval()
    scalers = [
        ArrayScaler.from_state(state) for state in checkpoint["feature_scalers"]
    ]
    obs_raw = arrays["obs"].reshape(-1, arrays["obs"].shape[-1])
    indiv_raw = arrays["indiv"].reshape(-1, arrays["indiv"].shape[-1])
    obs = torch.as_tensor(
        scalers[0].transform(obs_raw), dtype=torch.float32, device=device
    )
    indiv = torch.as_tensor(
        scalers[1].transform(indiv_raw), dtype=torch.float32, device=device
    )
    with torch.inference_mode():
        obs_trunk = model.obs_expert.trunk(obs)
        indiv_trunk = model.indiv_expert.trunk(indiv)
        mu_obs = model.obs_expert.mu(obs_trunk)
        logvar_obs = model.obs_expert.logvar(obs_trunk).clamp(-5.0, 5.0)
        mu_indiv = model.indiv_expert.mu(indiv_trunk)
        logvar_indiv = model.indiv_expert.logvar(indiv_trunk).clamp(-5.0, 5.0)
        tau_obs = torch.exp(-logvar_obs)
        tau_indiv = torch.exp(-logvar_indiv)
        var_poe = (1.0 + tau_obs + tau_indiv).reciprocal()
        mu_poe = var_poe * (tau_obs * mu_obs + tau_indiv * mu_indiv)

    base_shape = arrays["obs"].shape[:2]

    def shaped(value: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32).reshape(*base_shape, -1)[
            :, complete
        ]

    features = {
        "Low_raw_obs_input": arrays["Low"][:, complete],
        "Obs_scaled_input": shaped(obs),
        "Obs_trunk_after_MLP": shaped(obs_trunk),
        "Obs_mu_before_PoE": shaped(mu_obs),
        "Indiv_trunk_after_MLP": shaped(indiv_trunk),
        "Indiv_mu_before_PoE": shaped(mu_indiv),
        "Concat_mu_obs_mu_indiv": shaped(torch.cat([mu_obs, mu_indiv], dim=1)),
        "Mean_mu_obs_mu_indiv": shaped((mu_obs + mu_indiv) / 2.0),
        "High_mu_poe": shaped(mu_poe),
    }
    definitions = {
        "Low_raw_obs_input": "original pooled bci_obs; established Low",
        "Obs_scaled_input": "bci_obs after F25/S00 training scaler, before MLP",
        "Obs_trunk_after_MLP": "obs expert 256-D trunk output after LayerNorm-Linear-GELU",
        "Obs_mu_before_PoE": "obs expert 128-D Gaussian mean supplied to PoE",
        "Indiv_trunk_after_MLP": "individual expert 256-D trunk output",
        "Indiv_mu_before_PoE": "individual expert 128-D Gaussian mean supplied to PoE",
        "Concat_mu_obs_mu_indiv": "concatenated 256-D pre-PoE expert means",
        "Mean_mu_obs_mu_indiv": "unweighted mean of the two 128-D expert means",
        "High_mu_poe": "established 128-D precision-weighted PoE mean",
    }
    rdms = {
        name: rsa44.participant_mean_rank_rdm(value)
        for name, value in features.items()
    }
    metrics_output = OUT / "K_F25S00_PREPOE_FEATURES.csv"
    if metrics_output.is_file():
        cached = pd.read_csv(metrics_output)
        if set(cached["representation"]) == set(features):
            frame = cached
            statistics = {
                row.representation: {
                    "pearson_r": float(row.pearson_r),
                    "spearman_r": float(row.spearman_r),
                    "pearson_qap_p_two_sided_nominal": float(
                        row.pearson_qap_p_two_sided_nominal
                    ),
                    "spearman_qap_p_two_sided_nominal": float(
                        row.spearman_qap_p_two_sided_nominal
                    ),
                    "n_qap_permutations": int(row.n_qap_permutations),
                }
                for row in cached.itertuples()
            }
        else:
            raise RuntimeError("cached metric representations are incomplete")
    else:
        statistics = shared_qap(rdms, rating_rdm, N_QAP, QAP_SEED)
        rows = []
        for name, feature in features.items():
            rows.append(
                {
                    "representation": name,
                    "definition": definitions[name],
                    "dimension": int(feature.shape[-1]),
                    **statistics[name],
                }
            )
        frame = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
        frame.to_csv(metrics_output, index=False)
    rdm_output = OUT / "K_F25S00_PREPOE_RDMS.npz"
    with tempfile.NamedTemporaryFile(
        prefix="k_f25s00_prepoe_", suffix=".npz", dir="/tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(
            temporary_path,
            subjects=np.asarray(subjects),
            rating_rdm=rating_rdm,
            **{f"neural_rdm__{name}": rdm for name, rdm in rdms.items()},
        )
        with temporary_path.open("rb") as source, rdm_output.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 << 20)
            target.flush()
            os.fsync(target.fileno())
    finally:
        temporary_path.unlink(missing_ok=True)
    manifest = {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "fold": FOLD,
        "seed": SEED,
        "n_subjects": len(subjects),
        "n_videos": int(complete.sum()),
        "rating_set": RATING_SET,
        "k_protocol": (
            "correlation distance; per-video participant-pair fractional rank; "
            "mean over 30 videos; correlate 136 participant-pair edges"
        ),
        "qap": {
            "kind": "two-sided participant-label permutation",
            "n_permutations": N_QAP,
            "seed": QAP_SEED,
            "shared_permutations_across_representations": True,
            "selection_correction": False,
        },
        "outputs": {
            "metrics": str(metrics_output),
            "rdms": str(rdm_output),
        },
    }
    (OUT / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not np.isclose(
        statistics["High_mu_poe"]["pearson_r"],
        0.3140557105556035,
        atol=1e-10,
    ):
        raise AssertionError(
            "F25/S00 High did not reproduce the established Pearson K-RSA"
        )
    (OUT / "PIPELINE_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
