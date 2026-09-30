












from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(str(_LOCAL_SOURCE / ''))
DEFAULT_STAGE2 = ROOT / "lizhuo_exp/lizhuo_exp_data/SKIP/preprocess/stage2"
DEFAULT_BIDS = Path("/home/guoyi/nas/ugreen/NoLanguageData/ds005256-download")
DEFAULT_FMRI = DEFAULT_BIDS / "derivatives/preprocess/fmri_hcp"
DEFAULT_OUT = ROOT / "lizhuo_exp/lizhuo_exp_data/SKIP/enc_trained_all/residual_trial_manifests"

TASKS = ("alignvideo", "faces", "narratives", "shortvideo")
ALIGN_TARGETS = ("relevance", "happy", "sad", "afraid", "disgusted", "warm", "engaged")
FACE_TARGETS = ("intensity", "sex", "age")
NARRATIVE_TARGETS = (
    "feeling_valence", "feeling_intensity",
    "expectation_valence", "expectation_intensity",
)
SHORT_TARGETS = ("similarity", "likeability", "mentalizing")
TARGETS = {
    "alignvideo": ALIGN_TARGETS,
    "faces": FACE_TARGETS,
    "narratives": NARRATIVE_TARGETS,
    "shortvideo": SHORT_TARGETS,
}
EXPRESSION_MAP = {v: i for i, v in enumerate(
    ("anger", "disgust", "fear", "happy", "pain", "pleasure", "sad", "surprise"))}
SEX_MAP = {"female": 0, "male": 1}
RACE_MAP = {"African": 0, "EA": 1, "WC": 2}
AGE_MAP = {"young": 0, "old": 1}
NARRATIVE_X0 = 960.0
NARRATIVE_Y0 = 707.0
NARRATIVE_RADIUS = 500.0


def read_events(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def as_float(value: str | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_ses_run(name: str) -> tuple[int, int]:
    ses = re.search(r"ses-(\d+)", name)
    run = re.search(r"run-(\d+)", name)
    if not ses or not run:
        raise ValueError(f"Cannot parse session/run: {name}")
    return int(ses.group(1)), int(run.group(1))


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=np.str_)


def semantic_narrative(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    out = np.full(raw.shape, np.nan, dtype=np.float32)
    for src, dst in ((0, 0), (2, 2)):
        dx = raw[:, src] - NARRATIVE_X0
        dy = raw[:, src + 1] - NARRATIVE_Y0
        out[:, dst] = dx / NARRATIVE_RADIUS
        out[:, dst + 1] = np.sqrt(dx * dx + dy * dy) / NARRATIVE_RADIUS
    return out


def source_runs(fmri_root: Path, subject: str, task: str) -> list[tuple[int, int, Path]]:
    cift = fmri_root / subject / "fmri/cift"
    paths = sorted(cift.glob(f"ses-*_task-{task}_run-*_*.dtseries.nii"))
    runs = [(parse_ses_run(path.name)[0], parse_ses_run(path.name)[1], path) for path in paths]
    runs.sort(key=lambda item: (item[0], item[1]))
    if not runs:
        raise FileNotFoundError(f"No CIFTI runs for {subject} {task}: {cift}")
    return runs


def event_path(bids_root: Path, subject: str, task: str, ses: int, run: int) -> Path:
    path = (bids_root / subject / f"ses-{ses:02d}/func" /
            f"{subject}_ses-{ses:02d}_task-{task}_acq-mb8_run-{run:02d}_events.tsv")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def reconstruct_align(bids_root: Path, fmri_root: Path, subject: str) -> dict[str, np.ndarray]:
    rows_out: list[dict] = []
    for ses, run, _ in source_runs(fmri_root, subject, "alignvideo"):
        path = event_path(bids_root, subject, "alignvideo", ses, run)
        rows = read_events(path)
        for i, row in enumerate(rows):
            if row.get("trial_type") != "video":
                continue
            label = np.full(len(ALIGN_TARGETS), -1.0, dtype=np.float32)
            for later in rows[i + 1:]:
                if later.get("trial_type") == "video":
                    break
                trial_type = later.get("trial_type", "")
                if trial_type.startswith("rating_"):
                    name = trial_type.removeprefix("rating_")
                    value = as_float(later.get("response_value"))
                    if name in ALIGN_TARGETS and value is not None:
                        label[ALIGN_TARGETS.index(name)] = value
            if not np.any(label >= 0):
                continue
            rows_out.append({
                "stim_file": row.get("stim_file", ""),
                "session": ses,
                "run": run,
                "event_file": str(path),
                "event_row": i,
                "label": label,
            })
    return {
        "stim_file": np.asarray([row["stim_file"] for row in rows_out], dtype=np.str_),
        "session": np.asarray([row["session"] for row in rows_out], dtype=np.int64),
        "run": np.asarray([row["run"] for row in rows_out], dtype=np.int64),
        "event_file": np.asarray([row["event_file"] for row in rows_out], dtype=np.str_),
        "event_row": np.asarray([row["event_row"] for row in rows_out], dtype=np.int64),
        "target_id": np.full(len(rows_out), -1, dtype=np.int64),
        "target_values": np.asarray([row["label"] for row in rows_out], dtype=np.float32),
    }


def reconstruct_from_source_roots(
    task: str,
    subject: str,
    bids_roots: list[Path],
    fmri_roots: list[Path],
) -> dict[str, np.ndarray]:

    if len(bids_roots) != len(fmri_roots):
        raise ValueError("--bids-roots and --fmri-roots must contain the same number of paths")
    errors: list[str] = []
    for bids_root, fmri_root in zip(bids_roots, fmri_roots):
        try:
            if task == "alignvideo":
                return reconstruct_align(bids_root, fmri_root, subject)
            if task == "faces":
                return reconstruct_faces(bids_root, fmri_root, subject)
            raise ValueError(f"source reconstruction is unsupported for task={task}")
        except FileNotFoundError as exc:
            errors.append(f"bids={bids_root} fmri={fmri_root}: {exc}")
    raise FileNotFoundError(
        f"No matching {task} source root for {subject}; tried: " + " | ".join(errors)
    )


def reconstruct_faces(bids_root: Path, fmri_root: Path, subject: str) -> dict[str, np.ndarray]:
    rows_out: list[dict] = []
    for ses, run, _ in source_runs(fmri_root, subject, "faces"):
        path = event_path(bids_root, subject, "faces", ses, run)
        rows = read_events(path)
        for i, row in enumerate(rows):
            if row.get("trial_type") != "face":
                continue
            value = None
            for later in rows[i + 1:min(i + 4, len(rows))]:
                if later.get("trial_type") == "rating":
                    value = as_float(later.get("response_value"))
                    break
            if value is None:
                continue
            rating_type = row.get("rating_type", "")
            if rating_type not in FACE_TARGETS:
                raise ValueError(f"{path}:{i + 2}: unknown rating_type={rating_type!r}")
            objective = np.asarray([
                EXPRESSION_MAP.get(row.get("expression", ""), -1),
                SEX_MAP.get(row.get("sex", ""), -1),
                RACE_MAP.get(row.get("race", ""), -1),
                AGE_MAP.get(row.get("age", ""), -1),
            ], dtype=np.int64)
            target_values = np.full(len(FACE_TARGETS), np.nan, dtype=np.float32)
            target_values[FACE_TARGETS.index(rating_type)] = value
            rows_out.append({
                "stim_file": row.get("stim_file", ""),
                "session": ses,
                "run": run,
                "event_file": str(path),
                "event_row": i,
                "rating_type": rating_type,
                "target_id": FACE_TARGETS.index(rating_type),
                "target_values": target_values,
                "objective": objective,
            })
    return {
        "stim_file": np.asarray([row["stim_file"] for row in rows_out], dtype=np.str_),
        "session": np.asarray([row["session"] for row in rows_out], dtype=np.int64),
        "run": np.asarray([row["run"] for row in rows_out], dtype=np.int64),
        "event_file": np.asarray([row["event_file"] for row in rows_out], dtype=np.str_),
        "event_row": np.asarray([row["event_row"] for row in rows_out], dtype=np.int64),
        "rating_type": np.asarray([row["rating_type"] for row in rows_out], dtype=np.str_),
        "target_id": np.asarray([row["target_id"] for row in rows_out], dtype=np.int64),
        "target_values": np.asarray([row["target_values"] for row in rows_out], dtype=np.float32),
        "objective": np.asarray([row["objective"] for row in rows_out], dtype=np.int64),
    }


def direct_manifest(h5: h5py.File, task: str) -> dict[str, np.ndarray]:
    n = len(h5["labels_subjective"])
    if task == "narratives":
        raw = np.asarray(h5["labels_subjective"], dtype=np.float32)
        return {
            "stim_file": decode_strings(h5["stim_file"][...]),
            "session": np.asarray(h5["session"], dtype=np.int64),
            "run": np.asarray(h5["run"], dtype=np.int64),
            "target_id": np.full(n, -1, dtype=np.int64),
            "target_values": semantic_narrative(raw),
            "raw_labels": raw,
            "story_id": np.asarray(h5["story_id"], dtype=np.int64),
            "situation_index": np.asarray(h5["situation_index"], dtype=np.int64),
        }
    condition = decode_strings(h5["condition"][...])
    unknown = sorted(set(condition) - set(SHORT_TARGETS))
    if unknown:
        raise ValueError(f"Unknown shortvideo conditions: {unknown}")
    target_id = np.asarray([SHORT_TARGETS.index(value) for value in condition], dtype=np.int64)
    labels = np.asarray(h5["labels_subjective"], dtype=np.float32).reshape(-1)
    target_values = np.full((n, len(SHORT_TARGETS)), np.nan, dtype=np.float32)
    target_values[np.arange(n), target_id] = labels
    result = {
        "stim_file": decode_strings(h5["stim_file"][...]),
        "session": np.asarray(h5["session"], dtype=np.int64),
        "run": np.asarray(h5["run"], dtype=np.int64),
        "condition": condition,
        "target_id": target_id,
        "target_values": target_values,
        "raw_labels": labels,
    }
    if "video_id" in h5:
        values = h5["video_id"][...]
        result["video_id"] = (
            decode_strings(values) if values.dtype.kind in "SO"
            else np.asarray(values, dtype=np.int64)
        )
    if "mentalizing_level" in h5:
        result["mentalizing_level"] = decode_strings(h5["mentalizing_level"][...])
    return result


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape {actual.shape} != {expected.shape}")
    if np.issubdtype(actual.dtype, np.floating) or np.issubdtype(expected.dtype, np.floating):
        equal = np.allclose(actual, expected, rtol=0.0, atol=1e-6, equal_nan=True)
    else:
        equal = np.array_equal(actual, expected)
    if not equal:
        mismatch = np.argwhere(~np.isclose(actual, expected, rtol=0.0, atol=1e-6, equal_nan=True))
        first = tuple(mismatch[0]) if len(mismatch) else "unknown"
        raise AssertionError(f"{name}: first mismatch at {first}")


def validate_against_h5(task: str, h5: h5py.File, manifest: dict[str, np.ndarray]) -> None:
    n = len(h5["labels_subjective"])
    for key, value in manifest.items():
        if len(value) != n:
            raise AssertionError(f"{task}:{key}: manifest n={len(value)} != h5 n={n}")
    if np.any(manifest["stim_file"] == ""):
        raise AssertionError(f"{task}: empty stim_file")
    assert_close("session", manifest["session"], np.asarray(h5["session"], dtype=np.int64))
    if task == "alignvideo":
        assert_close("align labels", manifest["target_values"],
                     np.asarray(h5["labels_subjective"], dtype=np.float32))
    elif task == "faces":
        labels = np.asarray(h5["labels_subjective"], dtype=np.float32)
        selected = manifest["target_values"][np.arange(n), manifest["target_id"]]
        assert_close("face labels", selected, labels)
        assert_close("face objective", manifest["objective"],
                     np.asarray(h5["labels_objective"], dtype=np.int64))
    elif task == "narratives":
        assert_close("narrative raw labels", manifest["raw_labels"],
                     np.asarray(h5["labels_subjective"], dtype=np.float32))
    else:
        labels = np.asarray(h5["labels_subjective"], dtype=np.float32).reshape(-1)
        selected = manifest["target_values"][np.arange(n), manifest["target_id"]]
        assert_close("shortvideo labels", selected, labels)


def digest_manifest(manifest: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(manifest):
        value = np.asarray(manifest[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        if value.dtype.kind in "USO":
            digest.update("\0".join(value.astype(str).reshape(-1)).encode())
        else:
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def atomic_npz_save(path: Path, payload: dict[str, np.ndarray]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.stem + "_", suffix=".npz", dir="/tmp") as tmp:
        np.savez_compressed(tmp.name, **payload)
        destination = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        shutil.copyfile(tmp.name, destination)
        os.replace(destination, path)


def write_outer_folds(
    task: str, subjects: list[str], output: Path, seed: int, outer_cv: str
) -> list[dict]:
    if outer_cv == "loso":
        folds = [
            {
                "fold": fold_index,
                "train_subjects": [value for value in sorted(subjects) if value != test_subject],
                "test_subjects": [test_subject],
            }
            for fold_index, test_subject in enumerate(sorted(subjects))
        ]
        payload = {
            "task": task,
            "split_unit": "subject",
            "outer_cv": "leave-one-subject-out",
            "n_splits": len(subjects),
            "shuffle_seed": None,
            "subjects": sorted(subjects),
            "folds": folds,
        }
        (output / task / "outer_folds.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return folds

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(sorted(subjects), dtype=object)[rng.permutation(len(subjects))]
    tests = np.array_split(shuffled, 5)
    folds = []
    for fold_index, test in enumerate(tests):
        test_subjects = [str(value) for value in test]
        train_subjects = [value for value in sorted(subjects) if value not in test_subjects]
        folds.append({
            "fold": fold_index,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
        })
    observed = [subject for fold in folds for subject in fold["test_subjects"]]
    if sorted(observed) != sorted(subjects) or len(observed) != len(set(observed)):
        raise AssertionError(f"{task}: invalid outer folds")
    payload = {
        "task": task,
        "split_unit": "subject",
        "n_splits": 5,
        "shuffle_seed": seed,
        "subjects": sorted(subjects),
        "folds": folds,
    }
    (output / task / "outer_folds.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-root", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument(
        "--stage2-roots", default="",
        help="Comma-separated stage2 roots to merge by subject; overrides --stage2-root.",
    )
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS)
    parser.add_argument("--fmri-root", type=Path, default=DEFAULT_FMRI)
    parser.add_argument(
        "--bids-roots", default="",
        help="comma-separated BIDS roots tried in order; overrides --bids-root",
    )
    parser.add_argument(
        "--fmri-roots", default="",
        help="comma-separated HCP derivative roots paired with --bids-roots",
    )
    parser.add_argument(
        "--subject-selection-json", type=Path, default=None,
        help="optional JSON object mapping each task to the exact allowed subject list",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--fold-seed", type=int, default=20260715)
    parser.add_argument("--outer-cv", choices=("five-fold", "loso"), default="five-fold")
    args = parser.parse_args()

    stage2_roots = (
        [Path(value.strip()) for value in args.stage2_roots.split(",") if value.strip()]
        if args.stage2_roots else [args.stage2_root]
    )
    bids_roots = (
        [Path(value.strip()) for value in args.bids_roots.split(",") if value.strip()]
        if args.bids_roots else [args.bids_root]
    )
    fmri_roots = (
        [Path(value.strip()) for value in args.fmri_roots.split(",") if value.strip()]
        if args.fmri_roots else [args.fmri_root]
    )
    if len(bids_roots) != len(fmri_roots):
        raise ValueError("--bids-roots and --fmri-roots must have equal lengths")
    selections: dict[str, list[str]] = {}
    if args.subject_selection_json is not None:
        raw_selection = json.loads(args.subject_selection_json.read_text(encoding="utf-8"))
        if not isinstance(raw_selection, dict):
            raise ValueError("subject selection JSON must be an object")
        for task, values in raw_selection.items():
            if task not in TASKS or not isinstance(values, list):
                raise ValueError(f"invalid subject selection entry: {task!r}")
            normalized = [str(value) for value in values]
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"duplicate subject in selection for {task}")
            selections[task] = normalized

    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    if not tasks or any(task not in TASKS for task in tasks):
        raise ValueError(f"tasks must be a subset of {TASKS}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    qc_rows = []
    for task in tasks:
        task_out = args.output_root / task
        task_out.mkdir(parents=True, exist_ok=True)
        by_subject: dict[str, Path] = {}
        for stage2_root in stage2_roots:
            for h5_path in sorted((stage2_root / task).glob("sub-*.h5")):
                if h5_path.stem in by_subject:
                    raise RuntimeError(
                        f"{task}: duplicate {h5_path.stem} in "
                        f"{by_subject[h5_path.stem]} and {h5_path}"
                    )
                by_subject[h5_path.stem] = h5_path
        if task in selections:
            requested = selections[task]
            missing = sorted(set(requested) - set(by_subject))
            if missing:
                raise FileNotFoundError(f"{task}: selected subjects missing from stage2: {missing}")
            by_subject = {subject: by_subject[subject] for subject in requested}
        h5_paths = [by_subject[subject] for subject in sorted(by_subject)]
        if len(h5_paths) < 5:
            raise RuntimeError(f"{task}: found only {len(h5_paths)} stage-2 subjects")
        subjects = []
        for h5_path in h5_paths:
            subject = h5_path.stem
            with h5py.File(h5_path, "r") as h5:
                if task == "alignvideo":
                    manifest = reconstruct_from_source_roots(
                        task, subject, bids_roots, fmri_roots
                    )
                elif task == "faces":
                    manifest = reconstruct_from_source_roots(
                        task, subject, bids_roots, fmri_roots
                    )
                else:
                    manifest = direct_manifest(h5, task)
                validate_against_h5(task, h5, manifest)
            manifest["trial_index"] = np.arange(len(manifest["stim_file"]), dtype=np.int64)
            manifest["subject"] = np.full(len(manifest["stim_file"]), subject, dtype=np.str_)
            digest = digest_manifest(manifest)
            atomic_npz_save(task_out / f"{subject}.npz", manifest)
            subjects.append(subject)
            qc_rows.append({
                "task": task,
                "subject": subject,
                "n_trials": len(manifest["stim_file"]),
                "n_stimuli": len(set(manifest["stim_file"].tolist())),
                "targets": ",".join(TARGETS[task]),
                "sha256": digest,
                "status": "exact_match",
            })
        folds = write_outer_folds(
            task, subjects, args.output_root, args.fold_seed, args.outer_cv
        )
        print(f"[OK] {task}: subjects={len(subjects)} trials={sum(r['n_trials'] for r in qc_rows if r['task'] == task)} "
              f"test_sizes={[len(f['test_subjects']) for f in folds]}", flush=True)

    qc_path = args.output_root / "MANIFEST_QC.csv"
    with qc_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(qc_rows[0]))
        writer.writeheader()
        writer.writerows(qc_rows)
    summary = {
        "protocol": (
            "leave-one-subject-out; every subject is one outer test fold"
            if args.outer_cv == "loso" else
            "five-fold subject-grouped CV; every subject is outer test exactly once"
        ),
        "outer_cv": args.outer_cv,
        "residual_reference": "outer-training-subject mean for each exact stim_file and target",
        "tasks": tasks,
        "fold_seed": args.fold_seed,
        "stage2_roots": [str(path) for path in stage2_roots],
        "qc_csv": str(qc_path),
        "n_subjects": {task: sum(row["task"] == task for row in qc_rows) for task in tasks},
    }
    (args.output_root / "MANIFEST_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
