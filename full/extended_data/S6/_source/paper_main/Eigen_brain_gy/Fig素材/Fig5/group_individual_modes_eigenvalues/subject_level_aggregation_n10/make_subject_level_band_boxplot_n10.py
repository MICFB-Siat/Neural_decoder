









from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "date").is_dir())
DATE = REPO / "date"

GROUP_EVALS = (
    DATE
    / "generated"
    / "HCP_S1200"
    / "mode_1_5_20_40_100_200_500_1000"
    / "S1200_L_eigenvalues_mode0_to_mode1000.npy"
)
GROUP_META = GROUP_EVALS.with_name("S1200_L_selected_eigenmodes_metadata.npz")
GROUP_SURFACE = (
    DATE
    / "raw_inputs"
    / "HCP_S1200"
    / "template"
    / "S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
)
INDIVIDUAL_ROOT = DATE / "generated" / "mode_1_5_20_40_100_200_500_1000"
INDIVIDUAL_SURFACE_ROOT = DATE / "raw_inputs"

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

PRIMARY_STEM = "Alice_group_individual_eigenvalue_band_subject_boxplot"
SENSITIVITY_STEM = "Alice_group_individual_eigenvalue_band_subject_boxplot_mean_sensitivity"
MODE_SOURCE = HERE / "source_data_mode_level_n10.csv"
SUBJECT_SOURCE = HERE / "source_data_subject_level_n10.csv"
COHORT_SOURCE = HERE / "source_data_cohort_summary_n10.csv"
MANIFEST = HERE / "manifest.json"

POINT_COLOR = "#2C7FB8"
POINT_EDGE = "#FFFFFF"
LINE_COLOR = "#8AA9B8"
BOX_FACE = "#D9EAF2"
BOX_EDGE = "#5B8295"
SUMMARY_COLOR = "#262626"


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


def individual_paths(subject: str) -> tuple[Path, Path, Path]:
    subject_number = subject.split("-", 1)[1]
    folder = INDIVIDUAL_ROOT / subject
    evals = folder / f"{subject}_L_eigenvalues_mode0_to_mode1000.npy"
    meta = folder / f"{subject}_L_selected_eigenmodes_metadata.npz"
    surface = (
        INDIVIDUAL_SURFACE_ROOT
        / subject
        / "fsaverage_LR32k"
        / f"{subject_number}.L.midthickness.32k_fs_LR.surf.gii"
    )
    return evals, meta, surface


def cortical_area(surface_file: Path, meta_file: Path) -> float:

    image = nib.load(str(surface_file))
    xyz = np.asarray(image.darrays[0].data, dtype=np.float64)
    faces = np.asarray(image.darrays[1].data, dtype=np.int64)
    with np.load(meta_file) as metadata:
        keep = np.asarray(metadata["keep_indices"], dtype=np.int64)
    mask = np.zeros(xyz.shape[0], dtype=bool)
    mask[keep] = True
    faces = faces[np.all(mask[faces], axis=1)]
    edge_1 = xyz[faces[:, 1]] - xyz[faces[:, 0]]
    edge_2 = xyz[faces[:, 2]] - xyz[faces[:, 0]]
    area = float(0.5 * np.linalg.norm(np.cross(edge_1, edge_2), axis=1).sum())
    if not np.isfinite(area) or area <= 0:
        raise ValueError(f"Invalid cortical area for {surface_file}: {area}")
    return area


def load_spectrum(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=np.float64)
    if values.shape != (1001,):
        raise ValueError(f"Expected 1001 eigenvalues in {path}; got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite eigenvalue in {path}")
    if np.any(np.diff(values) < -1e-10):
        raise ValueError(f"Eigenvalues are not nondecreasing in {path}")
    modes_1_to_1000 = values[1:1001]
    if np.any(modes_1_to_1000 <= 0):
        raise ValueError(f"Non-positive nonconstant eigenvalue in {path}")
    return modes_1_to_1000


def symmetric_percent_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    return 200.0 * np.abs(a - b) / np.maximum(a + b, 1e-30)


def rank_band(rank: int) -> str:
    return next(label for label, lo, hi in BANDS if lo <= rank <= hi)


def load_mode_level_data() -> tuple[dict, list[dict], dict[str, str]]:
    group_eigenvalues = load_spectrum(GROUP_EVALS)
    group_area = cortical_area(GROUP_SURFACE, GROUP_META)
    group_scaled = group_eigenvalues * group_area

    records: list[dict] = []
    subject_data: dict[str, dict] = {}
    hashes = {
        str(GROUP_EVALS.relative_to(REPO)): sha256(GROUP_EVALS),
        str(GROUP_META.relative_to(REPO)): sha256(GROUP_META),
        str(GROUP_SURFACE.relative_to(REPO)): sha256(GROUP_SURFACE),
    }

    for subject in SUBJECTS:
        evals_path, meta_path, surface_path = individual_paths(subject)
        for path in (evals_path, meta_path, surface_path):
            if not path.exists():
                raise FileNotFoundError(path)
            hashes[str(path.relative_to(REPO))] = sha256(path)

        individual_eigenvalues = load_spectrum(evals_path)
        individual_area = cortical_area(surface_path, meta_path)
        individual_scaled = individual_eigenvalues * individual_area
        difference = symmetric_percent_difference(group_scaled, individual_scaled)
        if np.any((difference < 0) | (difference > 200 + 1e-10)):
            raise ValueError(f"Symmetric difference outside [0, 200] for {subject}")

        subject_data[subject] = {
            "area_mm2": individual_area,
            "difference_percent": difference,
        }
        for index in range(1000):
            records.append(
                {
                    "subject": subject,
                    "hemisphere": "left",
                    "mode_rank": index + 1,
                    "rank_band": rank_band(index + 1),
                    "group_cortical_area_mm2": group_area,
                    "individual_cortical_area_mm2": individual_area,
                    "group_eigenvalue": group_eigenvalues[index],
                    "individual_eigenvalue": individual_eigenvalues[index],
                    "group_area_normalized_eigenvalue": group_scaled[index],
                    "individual_area_normalized_eigenvalue": individual_scaled[index],
                    "symmetric_difference_percent": difference[index],
                }
            )
    return subject_data, records, hashes


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize_subjects(subject_data: dict[str, dict]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows: list[dict] = []
    medians = np.empty((len(SUBJECTS), len(BANDS)), dtype=float)
    means = np.empty_like(medians)
    for subject_index, subject in enumerate(SUBJECTS):
        values = subject_data[subject]["difference_percent"]
        for band_index, (label, lo, hi) in enumerate(BANDS):
            band_values = values[lo - 1 : hi]
            expected = hi - lo + 1
            if band_values.size != expected:
                raise ValueError(f"Incomplete band {subject}, {label}: {band_values.size}")
            medians[subject_index, band_index] = np.median(band_values)
            means[subject_index, band_index] = np.mean(band_values)
            rows.append(
                {
                    "subject": subject,
                    "hemisphere": "left",
                    "rank_band": label,
                    "first_mode": lo,
                    "last_mode": hi,
                    "n_modes": expected,
                    "mean_symmetric_difference_percent": means[subject_index, band_index],
                    "median_symmetric_difference_percent": medians[subject_index, band_index],
                    "sd_symmetric_difference_percent": np.std(band_values, ddof=1),
                    "q25_symmetric_difference_percent": np.percentile(band_values, 25),
                    "q75_symmetric_difference_percent": np.percentile(band_values, 75),
                }
            )
    return rows, medians, means


def cohort_summary(medians: np.ndarray, means: np.ndarray) -> list[dict]:
    rows = []
    for band_index, (label, lo, hi) in enumerate(BANDS):
        participant_values = medians[:, band_index]
        across_mean = float(np.mean(participant_values))
        across_sd = float(np.std(participant_values, ddof=1))
        rows.append(
            {
                "rank_band": label,
                "first_mode": lo,
                "last_mode": hi,
                "n_modes_per_subject": hi - lo + 1,
                "n_subjects": len(SUBJECTS),
                "across_subject_mean_of_subject_medians_percent": across_mean,
                "across_subject_sd_of_subject_medians_percent": across_sd,
                "across_subject_cv_of_subject_medians": across_sd / across_mean,
                "across_subject_median_of_subject_medians_percent": np.median(participant_values),
                "across_subject_q25_of_subject_medians_percent": np.percentile(participant_values, 25),
                "across_subject_q75_of_subject_medians_percent": np.percentile(participant_values, 75),
                "across_subject_min_of_subject_medians_percent": np.min(participant_values),
                "across_subject_max_of_subject_medians_percent": np.max(participant_values),
                "across_subject_mean_of_subject_means_percent": np.mean(means[:, band_index]),
            }
        )
    return rows


def draw_boxplot(matrix: np.ndarray, statistic: str, stem: str) -> None:
    x = np.arange(len(BANDS), dtype=float)
    offsets = np.linspace(-0.135, 0.135, len(SUBJECTS))

    fig, ax = plt.subplots(figsize=(96 / 25.4, 69 / 25.4), facecolor="white")
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.22, top=0.86)

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
        ax.plot(
            x + offsets[subject_index],
            matrix[subject_index],
            color=LINE_COLOR,
            linewidth=0.55,
            alpha=0.36,
            zorder=2,
        )
        ax.scatter(
            x + offsets[subject_index],
            matrix[subject_index],
            s=24,
            color=POINT_COLOR,
            edgecolor=POINT_EDGE,
            linewidth=0.45,
            alpha=0.90,
            zorder=4,
        )

    across_subject_median = np.median(matrix, axis=0)
    ax.scatter(
        x,
        across_subject_median,
        marker="D",
        s=27,
        color=SUMMARY_COLOR,
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
    )
    for band_index, value in enumerate(across_subject_median):
        ax.annotate(
            f"{value:.2f}",
            (x[band_index], value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=SUMMARY_COLOR,
            fontsize=5.6,
            fontweight="bold",
        )

    label_word = "Median" if statistic == "median" else "Mean"
    ax.set_yscale("log")
    lower = max(0.05, float(np.min(matrix)) * 0.62)
    upper = float(np.max(matrix)) * 1.70
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=5))
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax.grid(axis="y", which="major", color="#E6E6E6", linewidth=0.55, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _, _ in BANDS])
    ax.set_xlabel("Eigenmode rank band")
    ax.set_ylabel(
        f"{label_word} area-normalized eigenvalue\ndeviation per participant (%)"
    )
    ax.set_title(
        "Group–individual spectral deviation across eigenmode bands",
        loc="left",
        fontweight="bold",
        pad=7,
    )
    ax.text(
        0.01,
        -0.31,
        "Left hemisphere; mode 0 excluded; logarithmic y-axis",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.6,
        color="#666666",
    )
    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=4.5,
            markerfacecolor=POINT_COLOR, markeredgecolor="white",
            label=f"Participant (n = {len(SUBJECTS)})",
        ),
        Line2D(
            [0], [0], marker="D", linestyle="none", markersize=4.2,
            markerfacecolor=SUMMARY_COLOR, markeredgecolor="white",
            label="Across-participant median",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=5.7, handletextpad=0.35)

    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 450}),
        ("tiff", {"dpi": 600}),
    ):
        fig.savefig(HERE / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def validate_previous_two_subject_result(records: list[dict]) -> dict:
    previous = (
        HERE.parent / "source_data_group_individual_modes_eigenvalues.csv"
    )
    if not previous.exists():
        return {"status": "not_available"}

    current = {
        (row["subject"], row["mode_rank"]): row["symmetric_difference_percent"]
        for row in records
        if row["subject"] in {"sub-22", "sub-24"} and row["mode_rank"] <= 999
    }
    errors = []
    with previous.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["subject"], int(row["mode_rank"]))
            errors.append(
                abs(
                    current[key]
                    - float(row["area_normalized_eigenvalue_difference_percent"])
                )
            )
    maximum_error = float(max(errors))
    if maximum_error > 1e-5:
        raise ValueError(f"Failed previous-result recovery: max error={maximum_error}")
    return {
        "status": "passed",
        "n_compared_rows": len(errors),
        "maximum_absolute_error_percent": maximum_error,
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    subject_data, mode_rows, input_hashes = load_mode_level_data()
    subject_rows, medians, means = summarize_subjects(subject_data)
    cohort_rows = cohort_summary(medians, means)

    write_csv(MODE_SOURCE, mode_rows, list(mode_rows[0]))
    write_csv(SUBJECT_SOURCE, subject_rows, list(subject_rows[0]))
    write_csv(COHORT_SOURCE, cohort_rows, list(cohort_rows[0]))
    draw_boxplot(medians, "median", PRIMARY_STEM)
    draw_boxplot(means, "mean", SENSITIVITY_STEM)

    previous_recovery = validate_previous_two_subject_result(mode_rows)
    manifest = {
        "backend": "Python/matplotlib",
        "dataset": "Alice Datasets EEG-fMRI (ds002322)",
        "group_reference": "HCP S1200 group-average template",
        "hemisphere": "left",
        "subjects": list(SUBJECTS),
        "n_subjects": len(SUBJECTS),
        "bands": [
            {"label": label, "first_mode": lo, "last_mode": hi, "n_modes": hi - lo + 1}
            for label, lo, hi in BANDS
        ],
        "mode_zero": "excluded",
        "area_normalization": "lambda_tilde_k = cortical_area * lambda_k",
        "per_mode_metric": "200 * abs(group_scaled - individual_scaled) / (group_scaled + individual_scaled)",
        "primary_subject_summary": "median across modes within each band",
        "primary_boxplot_unit": "participant",
        "sensitivity_subject_summary": "arithmetic mean across modes within each band",
        "previous_two_subject_recovery": previous_recovery,
        "input_sha256": input_hashes,
        "outputs": [
            f"{PRIMARY_STEM}.svg",
            f"{PRIMARY_STEM}.pdf",
            f"{PRIMARY_STEM}.png",
            f"{PRIMARY_STEM}.tiff",
            f"{SENSITIVITY_STEM}.svg",
            f"{SENSITIVITY_STEM}.pdf",
            f"{SENSITIVITY_STEM}.png",
            f"{SENSITIVITY_STEM}.tiff",
            MODE_SOURCE.name,
            SUBJECT_SOURCE.name,
            COHORT_SOURCE.name,
        ],
    }
    temporary = MANIFEST.with_name(MANIFEST.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)

    print(f"Created primary figure: {HERE / (PRIMARY_STEM + '.png')}")
    print(f"Subjects: {len(SUBJECTS)}; mode-level rows: {len(mode_rows)}; subject-level rows: {len(subject_rows)}")
    print("Across-subject medians of participant band medians:")
    for row in cohort_rows:
        print(
            f"  {row['rank_band']}: "
            f"{row['across_subject_median_of_subject_medians_percent']:.6f}%"
        )


if __name__ == "__main__":
    main()
