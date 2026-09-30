


from __future__ import annotations

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

import compute_plot_alignvideo_epochwise_lowlevel_l_rsa as low
from compute_plot_alignvideo_epochwise_obs_expert_prepoe_l_rsa import (
    extract_mu_obs,
    load_checkpoint,
    rating_scalers,
)


EXP = Path(
    "/home/guoyi/nas2/share/Dataset/ds005256-download/experiments/"
    "alignvideo_mae_epochwise_rsa17_two_init_20260724"
)
FIG = EXP / "figures_epochwise_k_l_rsa17"
RUN = EXP / "poe5seed_rsa17_incohort/runs"
FEATURE = EXP / "epochwise_features_incohort17"
RATING_SET = "five_happy_sad_afraid_disgusted_engaged"
RATING_INDICES = list(low.RATING_SETS[RATING_SET])
N_PERMUTATIONS = 50_000
QAP_SEED = 20260725


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values -= values.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(values**2, axis=1, keepdims=True))
    denominator[denominator <= 1e-15] = 1.0
    return values / denominator


def bh_fdr(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def qap_one(task: tuple) -> dict:
    subject, neural_rdms, rating_rdm, selected_seed, random_seed = task
    n_videos = len(rating_rdm)
    triangle = np.triu_indices(n_videos, 1)
    rating = normalize_rows(rating_rdm[triangle][None, :])[0]
    observed_by_seed = normalize_rows(
        np.stack([rdm[triangle] for rdm in neural_rdms])
    ) @ rating
    observed = float(observed_by_seed[selected_seed])
    nominal_exceed = 0
    seed5_exceed = 0
    rng = np.random.default_rng(random_seed)
    for completed in range(0, N_PERMUTATIONS, 1000):
        count = min(1000, N_PERMUTATIONS - completed)
        orders = np.stack(
            [rng.permutation(n_videos) for _ in range(count)]
        )
        null_by_seed = []
        for rdm in neural_rdms:
            permuted = rdm[
                orders[:, triangle[0]],
                orders[:, triangle[1]],
            ]
            null_by_seed.append(normalize_rows(permuted) @ rating)
        null_by_seed = np.stack(null_by_seed)
        nominal_exceed += int(
            np.sum(
                np.abs(null_by_seed[selected_seed]) >= abs(observed)
            )
        )
        seed5_exceed += int(
            np.sum(
                np.max(np.abs(null_by_seed), axis=0) >= abs(observed)
            )
        )
    denominator = N_PERMUTATIONS + 1
    return {
        "subject": subject,
        "lowlevel_r_recomputed": observed,
        "lowlevel_qap_p_nominal": (nominal_exceed + 1) / denominator,
        "lowlevel_qap_p_seed5_adjusted": (
            seed5_exceed + 1
        ) / denominator,
    }


def main() -> None:
    high = pd.read_csv(
        FIG / "L_NONZERO_PEARSON_MAX_EPOCH013_HIGH_LOW_PER_SUBJECT_QAP.csv"
    )[
        [
            "subject",
            "high_pearson_selected_seed",
            "high_pearson_r",
            "high_pearson_qap_p_nominal",
            "high_pearson_qap_p_seed5_adjusted",
            "high_pearson_q_bh17_seed5_adjusted",
        ]
    ].rename(
        columns={
            "high_pearson_selected_seed": "highlevel_seed",
            "high_pearson_r": "highlevel_rsa",
            "high_pearson_qap_p_nominal": "highlevel_qap_p_nominal",
            "high_pearson_qap_p_seed5_adjusted": (
                "highlevel_qap_p_seed5_adjusted"
            ),
            "high_pearson_q_bh17_seed5_adjusted": (
                "highlevel_q_bh17_seed5_adjusted"
            ),
        }
    )
    selected = pd.read_csv(
        FIG / "L_OBS_EXPERT_PREPOE_BEST5_PER_SUBJECT_SOURCE_DATA.csv"
    )
    selected = selected[
        (selected["initialization"] == "crossdataset")
        & (selected["mae_epoch"] == 14)
        & (selected["rating_set"] == RATING_SET)
        & (selected["selection_metric"] == "pearson_r")
    ][["subject", "seed", "selected_value"]].rename(
        columns={
            "seed": "lowlevel_seed",
            "selected_value": "lowlevel_rsa",
        }
    )
    selected = (
        pd.DataFrame({"subject": low.RSA_SUBJECTS})
        .merge(selected, on="subject", validate="one_to_one")
    )

    exact35_frame = pd.read_csv(low.VIDEOS_CSV)
    exact35 = [
        (int(row.session), int(row.objective_id))
        for row in exact35_frame.itertuples()
    ]
    feature_path = FEATURE / "crossdataset/epoch_014.npz"
    data = low.load_feature(feature_path)
    obs30, rating30, valid = low.select_exact30(data, exact35)
    if int(valid.sum()) != 30:
        raise RuntimeError("expected 30 shared videos")

    rdms_by_seed = []
    scaler_reference = None
    for seed in range(5):
        checkpoint = load_checkpoint(
            RUN
            / "crossdataset/epoch_014/gaussian_poe128"
            / f"seed_{seed:02d}/checkpoint.pt"
        )
        scalers = rating_scalers(checkpoint)
        if scaler_reference is None:
            scaler_reference = scalers[RATING_SET]
        else:
            for current, reference in zip(
                scalers[RATING_SET], scaler_reference
            ):
                if not np.array_equal(current, reference):
                    raise RuntimeError("rating scaler differs across seeds")
        mu_obs = extract_mu_obs(obs30, checkpoint)
        rdms_by_seed.append(
            np.stack(
                [
                    squareform(
                        pdist(
                            mu_obs[subject_index],
                            metric="correlation",
                        )
                    )
                    for subject_index in range(17)
                ]
            )
        )
    rdms_by_seed = np.stack(rdms_by_seed)
    assert scaler_reference is not None
    center, scale = scaler_reference

    tasks = []
    for subject_index, subject in enumerate(low.RSA_SUBJECTS):
        rating_rdm = squareform(
            pdist(
                (
                    rating30[subject_index][:, RATING_INDICES] - center
                )
                / scale,
                metric="euclidean",
            )
        )
        lowlevel_seed = int(
            selected.loc[
                selected["subject"] == subject, "lowlevel_seed"
            ].iloc[0]
        )
        tasks.append(
            (
                subject,
                rdms_by_seed[:, subject_index],
                rating_rdm,
                lowlevel_seed,
                QAP_SEED + subject_index,
            )
        )
    with ProcessPoolExecutor(max_workers=8) as executor:
        qap = pd.DataFrame(executor.map(qap_one, tasks))

    output = (
        pd.DataFrame({"subject": low.RSA_SUBJECTS})
        .merge(high, on="subject", validate="one_to_one")
        .merge(selected, on="subject", validate="one_to_one")
        .merge(qap, on="subject", validate="one_to_one")
    )
    if not np.allclose(
        output["lowlevel_rsa"],
        output["lowlevel_r_recomputed"],
        atol=1e-10,
        rtol=1e-8,
    ):
        raise RuntimeError("pre-PoE RSA recomputation mismatch")
    output = output.drop(columns="lowlevel_r_recomputed")
    output["lowlevel_q_bh17_nominal"] = bh_fdr(
        output["lowlevel_qap_p_nominal"]
    )
    output["lowlevel_q_bh17_seed5_adjusted"] = bh_fdr(
        output["lowlevel_qap_p_seed5_adjusted"]
    )
    output["highlevel_significant_q_lt_0_05"] = (
        output["highlevel_q_bh17_seed5_adjusted"] < 0.05
    )
    output["lowlevel_significant_q_lt_0_05"] = (
        output["lowlevel_q_bh17_seed5_adjusted"] < 0.05
    )
    output.insert(1, "initialization", "crossdataset")
    output.insert(2, "rating_set", "five_ratings")
    output.insert(3, "metric", "pearson_rsa")
    output.insert(
        4,
        "highlevel_representation",
        "Gaussian PoE fused latent (128D)",
    )
    output.insert(5, "highlevel_mae_epoch", 13)
    output.insert(
        6,
        "lowlevel_representation",
        "Obs expert post-MLP / pre-PoE (128D)",
    )
    output.insert(7, "lowlevel_mae_epoch", 14)

    destination = (
        FIG / "L_CIRCLED_HIGH_E13_PREPOE_E14_PER_SUBJECT_RSA_QAP.csv"
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv.tmp",
        dir=FIG,
        delete=False,
        encoding="utf-8",
    ) as handle:
        temporary = Path(handle.name)
        output.to_csv(handle, index=False)
    os.replace(temporary, destination)
    print(destination)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
