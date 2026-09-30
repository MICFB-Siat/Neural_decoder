














from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import brainspace.mesh as mesh
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
from lapy import Solver, TriaMesh


DATE = Path(__file__).resolve().parents[1]
REPO = DATE.parent
OUTPUT = DATE / "figures" / "group_individual_spatial_band_subject_boxplot"
CACHE = OUTPUT / "correlation_cache"

SUBJECTS = (
    "sub-18",
    "sub-22",
    "sub-23",
    "sub-24",
    "sub-26",
    "sub-30",
    "sub-31",
    "sub-35",
    "sub-36",
    "sub-37",
)
BANDS = (
    ("1–10", 1, 10),
    ("11–50", 11, 50),
    ("51–200", 51, 200),
    ("201–1000", 201, 1000),
)
SELECTED_RANKS = np.asarray([1, 5, 20, 40, 100, 200, 500, 1000], dtype=int)
N_VERTICES = 32492
N_CORTEX = 29696
N_NONTRIVIAL = 1000

GROUP_SELECTED_DIR = (
    DATE / "generated" / "HCP_S1200" / "mode_1_5_20_40_100_200_500_1000"
)
INDIVIDUAL_SELECTED_DIR = (
    DATE / "generated" / "mode_1_5_20_40_100_200_500_1000"
)
OLD_FULL_ROOT = REPO / "zhf_exp" / "zhf_exp_fmri_eigen" / "data" / "generated"
OLD_SPATIAL_SOURCE = (
    REPO
    / "figures_alice_individual_modes_1000"
    / "revised_spatial"
    / "source_data_spatial_1000.csv"
)

FIGURE_STEM = "Alice_group_individual_spatial_correlation_band_subject_boxplot"
MEAN_STEM = FIGURE_STEM + "_mean_sensitivity"

POINT_COLOR = "#2A9D67"
BOX_FACE = "#D8EEE3"
BOX_EDGE = "#4C8B6C"
SUMMARY_COLOR = "#242424"
CONNECT_COLOR = "#A9BDB3"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_paths(subject: str) -> dict[str, Path]:
    number = subject.split("-", 1)[1]
    selected = INDIVIDUAL_SELECTED_DIR / subject
    raw = DATE / "raw_inputs" / subject / "fsaverage_LR32k"
    low = DATE / "generated" / subject
    return {
        "surface": raw / f"{number}.L.midthickness.32k_fs_LR.surf.gii",
        "mask": raw / f"{number}.L.atlasroi.32k_fs_LR.shape.gii",
        "metadata": selected / f"{subject}_L_selected_eigenmodes_metadata.npz",
        "selected_modes": selected / f"{subject}_L_selected_eigenmodes.npy",
        "eigenvalues": selected / f"{subject}_L_eigenvalues_mode0_to_mode1000.npy",
        "low_modes": low / f"{subject}_L_eigenmodes_201_with_mode0_full.npy",
    }


def load_mask(mask_path: Path) -> np.ndarray:
    if mask_path.suffix == ".txt":
        values = np.loadtxt(mask_path)
    else:
        values = nib.load(str(mask_path)).darrays[0].data
    mask = np.asarray(values).ravel() > 0
    if mask.shape != (N_VERTICES,) or int(mask.sum()) != N_CORTEX:
        raise ValueError(f"Unexpected mask in {mask_path}: {mask.shape}, sum={mask.sum()}")
    return mask


def normalize_columns(modes: np.ndarray) -> np.ndarray:
    values = np.asarray(modes, dtype=np.float32).copy()
    values -= values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(values, axis=0, keepdims=True)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise ValueError("Invalid eigenmode column norm")
    values /= norms
    return values


def load_group_modes() -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    metadata_path = GROUP_SELECTED_DIR / "S1200_L_selected_eigenmodes_metadata.npz"
    selected_path = GROUP_SELECTED_DIR / "S1200_L_selected_eigenmodes.npy"
    with np.load(metadata_path) as metadata:
        keep = np.asarray(metadata["keep_indices"], dtype=int)
        ranks = np.asarray(metadata["selected_mode_indices"], dtype=int)
    if not np.array_equal(ranks, SELECTED_RANKS):
        raise ValueError(f"Unexpected group selected ranks: {ranks}")

    old_path = OLD_FULL_ROOT / "template" / "template_L_eigenmodes_1000_with_mode0_masked.npy"
    old = np.asarray(np.load(old_path, mmap_mode="r")[:, 1:1000], dtype=np.float32)
    selected = np.load(selected_path, mmap_mode="r")
    mode_1000 = np.asarray(selected[keep, -1:], dtype=np.float32)
    combined = np.concatenate([old, mode_1000], axis=1)
    if combined.shape != (N_CORTEX, N_NONTRIVIAL):
        raise ValueError(f"Unexpected combined group shape: {combined.shape}")
    hashes = {
        str(old_path.relative_to(REPO)): sha256(old_path),
        str(selected_path.relative_to(REPO)): sha256(selected_path),
        str(metadata_path.relative_to(REPO)): sha256(metadata_path),
    }
    return normalize_columns(combined), keep, hashes


def same_rank_correlations(group_normalized: np.ndarray, individual_modes: np.ndarray) -> np.ndarray:
    individual_normalized = normalize_columns(individual_modes)
    values = np.abs(np.sum(group_normalized * individual_normalized, axis=0, dtype=np.float64))
    values = np.clip(values, 0.0, 1.0)
    if values.shape != (N_NONTRIVIAL,) or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid correlation vector: {values.shape}")
    return values


def load_retained_subject(subject: str, keep: np.ndarray) -> tuple[np.ndarray, dict]:

    old_path = OLD_FULL_ROOT / subject / f"{subject}_L_eigenmodes_1000_with_mode0_masked.npy"
    selected_path = subject_paths(subject)["selected_modes"]
    old = np.asarray(np.load(old_path, mmap_mode="r")[:, 1:1000], dtype=np.float32)
    selected = np.load(selected_path, mmap_mode="r")
    combined = np.concatenate(
        [old, np.asarray(selected[keep, -1:], dtype=np.float32)], axis=1
    )
    return combined, {
        "provenance": "preserved full modes 1-999 plus preserved selected mode 1000",
        "high_mode_recomputed": False,
        "old_full_sha256": sha256(old_path),
        "selected_sha256": sha256(selected_path),
    }


def recompute_subject(subject: str, keep: np.ndarray) -> tuple[np.ndarray, dict]:

    paths = subject_paths(subject)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    surface_orig = mesh.mesh_io.read_surface(str(paths["surface"]))
    mask = load_mask(paths["mask"])
    if not np.array_equal(np.flatnonzero(mask), keep):
        raise ValueError(f"Cortical mask/order differs for {subject}")
    surface_cut = mesh.mesh_operations.mask_points(surface_orig, mask.astype(np.uint8))
    vertices = np.asarray(surface_cut.Points, dtype=np.float64)
    faces = np.reshape(surface_cut.Polygons, (surface_cut.n_cells, 4))[:, 1:4].astype(np.int64)
    if vertices.shape[0] != N_CORTEX:
        raise ValueError(f"Unexpected cut mesh for {subject}: {vertices.shape}")



    np.random.seed(20260804)
    solver = Solver(TriaMesh(vertices, faces))
    eigenvalues, eigenvectors = solver.eigs(k=1001)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors, dtype=np.float32)
    if eigenvalues.shape != (1001,) or eigenvectors.shape != (N_CORTEX, 1001):
        raise ValueError(
            f"Unexpected eigensolver output for {subject}: {eigenvalues.shape}, {eigenvectors.shape}"
        )

    stored_eigenvalues = np.asarray(np.load(paths["eigenvalues"]), dtype=np.float64)
    eigenvalue_error = float(np.max(np.abs(eigenvalues - stored_eigenvalues)))
    eigenvalue_relative_error = float(
        np.max(
            np.abs(eigenvalues[1:] - stored_eigenvalues[1:])
            / np.maximum(np.abs(stored_eigenvalues[1:]), 1e-30)
        )
    )
    if eigenvalue_relative_error > 5e-6:
        raise ValueError(
            f"Stored-spectrum recovery failed for {subject}: relative error={eigenvalue_relative_error}"
        )

    with np.load(paths["metadata"]) as metadata:
        selected_ranks = np.asarray(metadata["selected_mode_indices"], dtype=int)
    stored_selected = np.asarray(np.load(paths["selected_modes"])[keep], dtype=np.float32)
    recomputed_selected = eigenvectors[:, selected_ranks]
    selected_r = np.clip(np.abs(
        np.sum(
            normalize_columns(stored_selected) * normalize_columns(recomputed_selected),
            axis=0,
            dtype=np.float64,
        )
    ), 0.0, 1.0)



    low_full = np.load(paths["low_modes"], mmap_mode="r")
    low_modes = np.asarray(low_full[keep, 1:201], dtype=np.float32)
    high_modes = eigenvectors[:, 201:1000]
    mode_1000 = stored_selected[:, -1:]
    combined = np.concatenate([low_modes, high_modes, mode_1000], axis=1)
    if combined.shape != (N_CORTEX, N_NONTRIVIAL):
        raise ValueError(f"Unexpected assembled modes for {subject}: {combined.shape}")

    validation = {
        "provenance": "preserved modes 1-200; recomputed modes 201-999; preserved selected mode 1000",
        "high_mode_recomputed": True,
        "eigenvalue_max_absolute_error": eigenvalue_error,
        "eigenvalue_max_relative_error": eigenvalue_relative_error,
        "selected_mode_abs_r_stored_vs_recomputed": {
            str(rank): float(value) for rank, value in zip(selected_ranks, selected_r)
        },
        "surface_sha256": sha256(paths["surface"]),
        "mask_sha256": sha256(paths["mask"]),
        "stored_spectrum_sha256": sha256(paths["eigenvalues"]),
        "stored_low_modes_sha256": sha256(paths["low_modes"]),
        "stored_selected_modes_sha256": sha256(paths["selected_modes"]),
    }
    return combined, validation


def load_or_compute_subject(
    subject: str, group_normalized: np.ndarray, keep: np.ndarray, force: bool
) -> tuple[np.ndarray, dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_stem = CACHE / f"{subject}_same_rank_spatial_r_modes1_to1000"
    cache_values = cache_stem.with_suffix(".npy")
    cache_validation = cache_stem.with_suffix(".json")
    if cache_values.exists() and cache_validation.exists() and not force:
        correlations = np.asarray(np.load(cache_values), dtype=np.float64)
        validation = json.loads(cache_validation.read_text(encoding="utf-8"))
        selected_check = validation.get("selected_mode_abs_r_stored_vs_recomputed", {})
        validation["selected_mode_abs_r_stored_vs_recomputed"] = {
            rank: float(np.clip(value, 0.0, 1.0))
            for rank, value in selected_check.items()
        }
        if correlations.shape != (N_NONTRIVIAL,):
            raise ValueError(f"Bad cache shape in {cache_values}: {correlations.shape}")
        return correlations, validation

    if subject in {"sub-22", "sub-24"}:
        modes, validation = load_retained_subject(subject, keep)
    else:
        modes, validation = recompute_subject(subject, keep)
    correlations = same_rank_correlations(group_normalized, modes)
    del modes




    with tempfile.TemporaryDirectory(prefix="eigenmode_spatial_cache_") as tmp_dir:
        local_values = Path(tmp_dir) / cache_values.name
        local_validation = Path(tmp_dir) / cache_validation.name
        np.save(local_values, correlations)
        local_validation.write_text(
            json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        reloaded = np.load(local_values)
        if not np.array_equal(reloaded, correlations):
            raise ValueError(f"Local cache verification failed for {subject}")
        for local, destination in (
            (local_values, cache_values),
            (local_validation, cache_validation),
        ):
            partial = destination.with_name(destination.name + ".part")
            shutil.copyfile(local, partial)
            partial.replace(destination)
    return correlations, validation


def band_label(rank: int) -> str:
    return next(label for label, lo, hi in BANDS if lo <= rank <= hi)


def summarize(correlations: dict[str, np.ndarray]) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray]:
    mode_rows = []
    subject_rows = []
    medians = np.empty((len(SUBJECTS), len(BANDS)), dtype=float)
    means = np.empty_like(medians)
    for subject_index, subject in enumerate(SUBJECTS):
        values = correlations[subject]
        for rank, value in enumerate(values, start=1):
            mode_rows.append(
                {
                    "dataset": "Alice ds002322",
                    "subject": subject,
                    "hemisphere": "left",
                    "group_template": "HCP S1200",
                    "n_cortical_vertices": N_CORTEX,
                    "mode_rank": rank,
                    "rank_band": band_label(rank),
                    "same_rank_abs_spatial_r": value,
                }
            )
        for band_index, (label, lo, hi) in enumerate(BANDS):
            band_values = values[lo - 1 : hi]
            if len(band_values) != hi - lo + 1:
                raise ValueError(f"Incomplete band {subject}, {label}")
            medians[subject_index, band_index] = np.median(band_values)
            means[subject_index, band_index] = np.mean(band_values)
            subject_rows.append(
                {
                    "dataset": "Alice ds002322",
                    "subject": subject,
                    "hemisphere": "left",
                    "group_template": "HCP S1200",
                    "rank_band": label,
                    "first_mode": lo,
                    "last_mode": hi,
                    "n_modes": hi - lo + 1,
                    "median_same_rank_abs_spatial_r": medians[subject_index, band_index],
                    "mean_same_rank_abs_spatial_r": means[subject_index, band_index],
                    "sd_same_rank_abs_spatial_r": np.std(band_values, ddof=1),
                    "q25_same_rank_abs_spatial_r": np.percentile(band_values, 25),
                    "q75_same_rank_abs_spatial_r": np.percentile(band_values, 75),
                }
            )
    return mode_rows, subject_rows, medians, means


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def draw(matrix: np.ndarray, statistic: str, stem: str) -> None:
    x = np.arange(len(BANDS), dtype=float)
    offsets = np.linspace(-0.13, 0.13, len(SUBJECTS))
    fig, ax = plt.subplots(figsize=(89 / 25.4, 67 / 25.4), facecolor="white")
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.22, top=0.88)

    box = ax.boxplot(
        [matrix[:, index] for index in range(matrix.shape[1])],
        positions=x,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={"color": SUMMARY_COLOR, "linewidth": 1.15},
        boxprops={"facecolor": BOX_FACE, "edgecolor": BOX_EDGE, "linewidth": 0.85},
        whiskerprops={"color": BOX_EDGE, "linewidth": 0.8},
        capprops={"color": BOX_EDGE, "linewidth": 0.8},
        zorder=1,
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.78)

    for subject_index in range(len(SUBJECTS)):
        xs = x + offsets[subject_index]
        ax.plot(xs, matrix[subject_index], color=CONNECT_COLOR, lw=0.5, alpha=0.36, zorder=2)
        ax.scatter(
            xs,
            matrix[subject_index],
            s=22,
            color=POINT_COLOR,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.92,
            zorder=4,
        )

    cohort_median = np.median(matrix, axis=0)
    ax.scatter(
        x,
        cohort_median,
        marker="D",
        s=27,
        color=SUMMARY_COLOR,
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
    )
    for index, value in enumerate(cohort_median):
        ax.annotate(
            f"{value:.2f}",
            (x[index], value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.6,
            fontweight="bold",
            color=SUMMARY_COLOR,
        )

    summary_word = "Median" if statistic == "median" else "Mean"
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.015, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.55, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _, _ in BANDS])
    ax.set_xlabel("Eigenmode rank band")
    ax.set_ylabel(f"{summary_word} group–individual\nspatial correlation  $|r|$")
    ax.set_title("Group–individual eigenmode correspondence", loc="left", fontweight="bold", pad=7)
    ax.text(
        0.01,
        -0.30,
        "Left hemisphere; same-rank comparison; mode 0 excluded",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color="#666666",
    )
    ax.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="none", markersize=4.3,
                markerfacecolor=POINT_COLOR, markeredgecolor="white",
                label=f"Participant (n = {len(SUBJECTS)})",
            ),
            Line2D(
                [0], [0], marker="D", linestyle="none", markersize=4.1,
                markerfacecolor=SUMMARY_COLOR, markeredgecolor="white",
                label="Across-participant median",
            ),
        ],
        loc="upper right",
        fontsize=5.5,
        handletextpad=0.35,
    )

    for extension, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 450}),
        ("tiff", {"dpi": 600}),
    ):
        fig.savefig(OUTPUT / f"{stem}.{extension}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def validate_old_source(correlations: dict[str, np.ndarray]) -> dict:
    if not OLD_SPATIAL_SOURCE.exists():
        return {"status": "not_available"}
    errors = []
    with OLD_SPATIAL_SOURCE.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            subject = row["subject"]
            rank = int(row["reference_rank"])
            errors.append(abs(correlations[subject][rank - 1] - float(row["same_rank_abs_spatial_r"])))
    maximum = float(max(errors))
    if maximum > 5e-6:
        raise ValueError(f"Failed recovery of preserved spatial source: max error={maximum}")
    return {"status": "passed", "n_rows": len(errors), "maximum_absolute_error": maximum}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="recompute even if correlation caches exist")
    parser.add_argument("--only-subject", choices=SUBJECTS, help="compute/cache one subject and stop")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    group, keep, input_hashes = load_group_modes()

    if args.only_subject:
        values, validation = load_or_compute_subject(args.only_subject, group, keep, args.force)
        print(json.dumps({
            "subject": args.only_subject,
            "n_modes": len(values),
            "band_medians": {
                label: float(np.median(values[lo - 1:hi])) for label, lo, hi in BANDS
            },
            "validation": validation,
        }, indent=2))
        return

    correlations = {}
    validations = {}
    for index, subject in enumerate(SUBJECTS, start=1):
        print(f"[{index}/{len(SUBJECTS)}] {subject}", flush=True)
        correlations[subject], validations[subject] = load_or_compute_subject(
            subject, group, keep, args.force
        )

    mode_rows, subject_rows, medians, means = summarize(correlations)
    mode_path = OUTPUT / "source_data_mode_level_n10.csv"
    subject_path = OUTPUT / "source_data_subject_level_n10.csv"
    write_csv(mode_path, mode_rows)
    write_csv(subject_path, subject_rows)
    draw(medians, "median", FIGURE_STEM)
    draw(means, "mean", MEAN_STEM)

    old_recovery = validate_old_source(correlations)
    summary = []
    for band_index, (label, lo, hi) in enumerate(BANDS):
        values = medians[:, band_index]
        summary.append({
            "rank_band": label,
            "first_mode": lo,
            "last_mode": hi,
            "n_modes_per_subject": hi - lo + 1,
            "n_subjects": len(SUBJECTS),
            "across_subject_mean": float(np.mean(values)),
            "across_subject_sd": float(np.std(values, ddof=1)),
            "across_subject_cv": float(np.std(values, ddof=1) / np.mean(values)),
            "across_subject_median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        })
    summary_path = OUTPUT / "source_data_cohort_summary_n10.csv"
    write_csv(summary_path, summary)

    output_names = [
        f"{FIGURE_STEM}.svg", f"{FIGURE_STEM}.pdf", f"{FIGURE_STEM}.png", f"{FIGURE_STEM}.tiff",
        f"{MEAN_STEM}.svg", f"{MEAN_STEM}.pdf", f"{MEAN_STEM}.png", f"{MEAN_STEM}.tiff",
        mode_path.name, subject_path.name, summary_path.name,
    ]
    manifest = {
        "backend": "Python/matplotlib; LaPy 1.3.0 for missing high-rank eigenmodes",
        "dataset": "Alice Datasets EEG-fMRI (ds002322)",
        "subjects": list(SUBJECTS),
        "n_subjects": len(SUBJECTS),
        "hemisphere": "left",
        "group_template": "HCP S1200",
        "n_cortical_vertices": N_CORTEX,
        "bands": [
            {"label": label, "first_mode": lo, "last_mode": hi, "n_modes": hi - lo + 1}
            for label, lo, hi in BANDS
        ],
        "metric": "same-rank absolute Pearson spatial correlation across 29696 matched cortical vertices",
        "primary_subject_summary": "median across modes within each band",
        "statistical_unit": "participant",
        "mode_zero": "excluded",
        "validations": validations,
        "previous_sub22_sub24_source_recovery": old_recovery,
        "group_input_sha256": input_hashes,
        "outputs": output_names,
        "output_sha256": {name: sha256(OUTPUT / name) for name in output_names},
    }
    temporary = OUTPUT / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT / "manifest.json")

    print("Across-subject medians of participant band medians:")
    for row in summary:
        print(f"  {row['rank_band']}: {row['across_subject_median']:.6f}")
    print(f"Primary figure: {OUTPUT / (FIGURE_STEM + '.png')}")


if __name__ == "__main__":
    main()
