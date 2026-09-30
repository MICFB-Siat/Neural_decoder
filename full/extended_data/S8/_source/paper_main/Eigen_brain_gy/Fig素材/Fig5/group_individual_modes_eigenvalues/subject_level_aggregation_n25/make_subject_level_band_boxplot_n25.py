









from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import nibabel as nib
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = next(path for path in HERE.parents if (path / "date").is_dir())
N10_SCRIPT = (
    HERE.parent
    / "subject_level_aggregation_n10"
    / "make_subject_level_band_boxplot_n10.py"
)

SUBJECTS = (
    "sub-18", "sub-22", "sub-23", "sub-24", "sub-26",
    "sub-30", "sub-31", "sub-35", "sub-36", "sub-37",
    "sub-38", "sub-39", "sub-41", "sub-42", "sub-43",
    "sub-44", "sub-45", "sub-46", "sub-47", "sub-48",
    "sub-49", "sub-50", "sub-51", "sub-52", "sub-53",
)

DEFAULT_SOURCE_ROOT = Path(
    "/home/guoyi/nas/ugreen/Language/"
    "8+The_Alice_Datasets_EEGfMRI_version/ds002322-download"
)
SOURCE_VALIDATION = HERE / "source_input_validation_n25.csv"
MEDIAN_WIDE_SOURCE = HERE / "source_data_boxplot_median_wide_n25.csv"
MEAN_WIDE_SOURCE = HERE / "source_data_boxplot_mean_wide_n25.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_source_root() -> Path:
    configured = os.environ.get("ALICE_DATASET_ROOT")
    root = Path(configured) if configured else DEFAULT_SOURCE_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"Alice source root is unavailable: {root}. "
            "Mount it or set ALICE_DATASET_ROOT."
        )
    return root


def source_paths(root: Path, subject: str) -> tuple[Path, Path, Path]:
    number = subject.split("-", 1)[1]
    t1 = root / "derivatives" / "preprocess" / "fmri_hcp" / subject / "T1"
    surface = t1 / f"{number}.L.midthickness.32k_fs_LR.surf.gii"
    mask = t1 / f"{number}.L.atlasroi.32k_fs_LR.shape.gii"
    h5 = root / "preprocess" / "stage3" / f"{subject}.h5"
    return surface, mask, h5


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    shutil.copyfile(source, partial)
    partial.replace(destination)


def stage_array(destination: Path, values: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="alice_n25_eigenvalues_") as tmpdir:
        local = Path(tmpdir) / destination.name
        np.save(local, values)
        reloaded = np.load(local)
        if not np.array_equal(reloaded, values):
            raise ValueError(f"Local array verification failed: {destination}")
        atomic_copy(local, destination)


def stage_metadata(destination: Path, fields: dict[str, object]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="alice_n25_metadata_") as tmpdir:
        local = Path(tmpdir) / destination.name
        np.savez_compressed(local, **fields)
        with np.load(local) as metadata:
            if not np.array_equal(metadata["keep_indices"], fields["keep_indices"]):
                raise ValueError(f"Local metadata verification failed: {destination}")
        atomic_copy(local, destination)


def stage_and_validate_inputs(module) -> list[dict[str, object]]:

    root = resolve_source_root()
    rows: list[dict[str, object]] = []
    for subject in SUBJECTS:
        number = subject.split("-", 1)[1]
        source_surface, source_mask, source_h5 = source_paths(root, subject)
        for path in (source_surface, source_mask, source_h5):
            if not path.exists():
                raise FileNotFoundError(path)

        mask = np.asarray(nib.load(str(source_mask)).darrays[0].data).ravel() > 0
        surface_image = nib.load(str(source_surface))
        n_vertices = int(surface_image.darrays[0].data.shape[0])
        if mask.shape != (32492,) or int(mask.sum()) != 29696 or n_vertices != 32492:
            raise ValueError(
                f"Unexpected mesh/mask for {subject}: vertices={n_vertices}, "
                f"mask_shape={mask.shape}, mask_sum={mask.sum()}"
            )
        keep_indices = np.flatnonzero(mask).astype(np.int32)

        with h5py.File(source_h5, "r") as stream:
            if "fmri_eigenval_indiv" not in stream:
                raise KeyError(f"fmri_eigenval_indiv missing in {source_h5}")
            source_length = int(stream["fmri_eigenval_indiv"].shape[0])
            individual_modes_1_to_1000 = np.asarray(
                stream["fmri_eigenval_indiv"][:1000], dtype=np.float64
            )
        if (
            source_length != 2000
            or individual_modes_1_to_1000.shape != (1000,)
            or not np.all(np.isfinite(individual_modes_1_to_1000))
            or np.any(individual_modes_1_to_1000 <= 0)
            or np.any(np.diff(individual_modes_1_to_1000) < -1e-10)
        ):
            raise ValueError(f"Invalid source eigenvalue spectrum for {subject}")

        evals_destination, metadata_destination, surface_destination = (
            module.individual_paths(subject)
        )
        mask_destination = (
            surface_destination.parent / f"{number}.L.atlasroi.32k_fs_LR.shape.gii"
        )
        if surface_destination.exists():
            surface_hash_match = sha256(surface_destination) == sha256(source_surface)
            if not surface_hash_match:
                raise ValueError(f"Staged/source surface mismatch for {subject}")
        else:
            atomic_copy(source_surface, surface_destination)
            surface_hash_match = True
        if mask_destination.exists():
            mask_hash_match = sha256(mask_destination) == sha256(source_mask)
            if not mask_hash_match:
                raise ValueError(f"Staged/source mask mismatch for {subject}")
        else:
            atomic_copy(source_mask, mask_destination)
            mask_hash_match = True

        spectrum_with_mode_zero = np.concatenate(
            [np.zeros(1, dtype=np.float64), individual_modes_1_to_1000]
        )
        if evals_destination.exists():
            staged_spectrum = np.asarray(np.load(evals_destination), dtype=np.float64)
            if staged_spectrum.shape != (1001,):
                raise ValueError(f"Bad staged spectrum shape for {subject}")
            relative_error = float(
                np.max(
                    np.abs(staged_spectrum[1:] - individual_modes_1_to_1000)
                    / individual_modes_1_to_1000
                )
            )
            if relative_error > 1e-6:
                raise ValueError(
                    f"Source/staged eigenvalue mismatch for {subject}: {relative_error}"
                )
        else:
            stage_array(evals_destination, spectrum_with_mode_zero)
            relative_error = 0.0

        if metadata_destination.exists():
            with np.load(metadata_destination) as metadata:
                staged_keep = np.asarray(metadata["keep_indices"], dtype=np.int32)
            if not np.array_equal(staged_keep, keep_indices):
                raise ValueError(f"Staged/source cortical mask mismatch for {subject}")
        else:
            stage_metadata(
                metadata_destination,
                {
                    "subject": np.asarray(subject),
                    "surface": np.asarray(str(source_surface)),
                    "mask": np.asarray(str(source_mask)),
                    "source_h5": np.asarray(str(source_h5)),
                    "source_h5_dataset": np.asarray("fmri_eigenval_indiv[0:1000]"),
                    "keep_indices": keep_indices,
                    "n_vertices_total": np.asarray(n_vertices, dtype=np.int32),
                    "n_vertices_masked": np.asarray(keep_indices.size, dtype=np.int32),
                    "eigensolver_k": np.asarray(1001, dtype=np.int32),
                    "selected_mode_indices": np.asarray(
                        [1, 5, 20, 40, 100, 200, 500, 1000], dtype=np.int32
                    )
                },
            )

        rows.append(
            {
                "subject": subject,
                "source_h5_dataset": "fmri_eigenval_indiv[0:1000]",
                "source_h5_dataset_length_both_hemispheres": source_length,
                "selected_left_hemisphere_values": 1000,
                "source_mesh_vertices": n_vertices,
                "source_mask_vertices": int(mask.sum()),
                "surface_sha256_match": surface_hash_match,
                "mask_sha256_match": mask_hash_match,
                "max_relative_error_source_vs_staged_modes1_to1000": relative_error,
            }
        )

    temporary = SOURCE_VALIDATION.with_name(SOURCE_VALIDATION.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(SOURCE_VALIDATION)
    return rows


def load_accepted_implementation():
    spec = importlib.util.spec_from_file_location("alice_eigenvalue_n10", N10_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load accepted implementation: {N10_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_n25(module) -> None:
    module.HERE = HERE
    module.SUBJECTS = SUBJECTS
    module.PRIMARY_STEM = "Alice_group_individual_eigenvalue_band_subject_boxplot_n25"
    module.SENSITIVITY_STEM = (
        "Alice_group_individual_eigenvalue_band_subject_boxplot_n25_mean_sensitivity"
    )
    module.MODE_SOURCE = HERE / "source_data_mode_level_n25.csv"
    module.SUBJECT_SOURCE = HERE / "source_data_subject_level_n25.csv"
    module.COHORT_SOURCE = HERE / "source_data_cohort_summary_n25.csv"
    module.MANIFEST = HERE / "manifest.json"


def audit_inputs(module) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subject in SUBJECTS:
        eigenvalues, metadata, surface = module.individual_paths(subject)
        required = {
            "eigenvalues_mode0_to_mode1000": eigenvalues,
            "selected_eigenmodes_metadata": metadata,
            "midthickness_fsLR32k": surface,
        }
        row: dict[str, object] = {
            "subject": subject,
            "all_required_inputs_present": all(path.exists() for path in required.values()),
        }
        for label, path in required.items():
            row[f"{label}_present"] = path.exists()
            row[f"{label}_path"] = str(path.relative_to(REPO))
        rows.append(row)

    output = HERE / "input_audit_n25.csv"
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    return rows


def write_boxplot_wide_sources(subject_source: Path) -> None:

    with subject_source.open(encoding="utf-8") as stream:
        long_rows = list(csv.DictReader(stream))
    band_labels = ["1–10", "11–50", "51–200", "201–1000"]
    output_labels = ["1-10", "11-50", "51-200", "201-1000"]
    by_subject = {
        subject: {
            row["rank_band"]: row
            for row in long_rows
            if row["subject"] == subject
        }
        for subject in SUBJECTS
    }
    for statistic, destination in (
        ("median_symmetric_difference_percent", MEDIAN_WIDE_SOURCE),
        ("mean_symmetric_difference_percent", MEAN_WIDE_SOURCE),
    ):
        rows = []
        for subject in SUBJECTS:
            if set(by_subject[subject]) != set(band_labels):
                raise ValueError(f"Incomplete band data for {subject}")
            row = {"subject": subject}
            for source_label, output_label in zip(band_labels, output_labels):
                row[output_label] = by_subject[subject][source_label][statistic]
            rows.append(row)
        temporary = destination.with_name(destination.name + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["subject", *output_labels]
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    module = load_accepted_implementation()
    configure_n25(module)
    validation_rows = stage_and_validate_inputs(module)
    rows = audit_inputs(module)
    missing = [row["subject"] for row in rows if not row["all_required_inputs_present"]]
    if missing:
        raise FileNotFoundError(
            "n=25 input audit failed; missing accepted surface/eigenvalue inputs for: "
            + ", ".join(str(subject) for subject in missing)
            + f". See {HERE / 'input_audit_n25.csv'}"
        )
    module.main()
    write_boxplot_wide_sources(module.SUBJECT_SOURCE)
    manifest = json.loads(module.MANIFEST.read_text(encoding="utf-8"))
    manifest["source_eigenvalue_dataset"] = "fmri_eigenval_indiv[0:1000]"
    pass
    manifest["source_validation_file"] = SOURCE_VALIDATION.name
    manifest["maximum_relative_error_source_vs_preexisting_n10_spectra"] = max(
        float(row["max_relative_error_source_vs_staged_modes1_to1000"])
        for row in validation_rows
    )
    manifest["outputs"].append(SOURCE_VALIDATION.name)
    manifest["outputs"].extend(
        [MEDIAN_WIDE_SOURCE.name, MEAN_WIDE_SOURCE.name]
    )
    manifest["outputs"].append("QA_REPORT.md")
    temporary = module.MANIFEST.with_name(module.MANIFEST.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(module.MANIFEST)


if __name__ == "__main__":
    main()
