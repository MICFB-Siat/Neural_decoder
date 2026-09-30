









from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


NOISE = {"noise_03": 0.3, "noise_05": 0.5, "noise_10": 1.0}
DROP = {"drop_01": 0.1, "drop_03": 0.3, "drop_05": 0.5}
REDUCE = {"red_50": 0.5, "red_25": 0.25, "red_10": 0.1}
ALL_TAGS = list(NOISE) + list(DROP) + list(REDUCE)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_dump_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def npy_dump_atomic(value: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("wb") as f:
        np.save(f, value)
    os.replace(tmp, path)


def build_basis(d_path: Path, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(d_path).astype(np.float64)
    u, s, _ = np.linalg.svd(d, full_matrices=False)
    k = int((s / s[0] >= ratio).sum())
    if k < 1:
        raise RuntimeError(f"No singular component survived ratio={ratio:g}: {d_path}")
    return u[:, :k].astype(np.float32), s[:k]


def stratified_keep(labels: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    chunks = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        n = min(len(idx), max(1, int(round(frac * len(idx)))))
        chunks.append(rng.choice(idx, size=n, replace=False))
    return np.sort(np.concatenate(chunks)).astype(np.int64)


def class_counts(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def write_h5(
    source: h5py.File,
    destination: Path,
    raw: np.ndarray,
    modes: np.ndarray,
    label_key: str,
    keep_idx: np.ndarray | None,
    attrs: dict[str, Any],
) -> None:
    tmp = destination.with_suffix(destination.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    labels = source[label_key][:]
    if keep_idx is not None:
        labels = labels[keep_idx]
    with h5py.File(tmp, "w") as out:
        out.create_dataset("eeg_raw", data=np.asarray(raw, dtype=np.float32))
        out.create_dataset("eeg_modes", data=np.asarray(modes, dtype=np.float32))
        out.create_dataset("eeg_pos", data=source["eeg_pos"][:])
        out.create_dataset("eeg_sensor_type", data=source["eeg_sensor_type"][:])
        out.create_dataset("labels", data=labels.astype(np.int64, copy=False))
        if label_key != "labels":
            out.create_dataset(label_key, data=labels.astype(np.int64, copy=False))
            if "labels" in source:
                labels_original = source["labels"][:]
                if keep_idx is not None:
                    labels_original = labels_original[keep_idx]
                out.create_dataset("labels9", data=labels_original.astype(np.int64, copy=False))
        for key, value in source.attrs.items():
            out.attrs[key] = value
        for key, value in attrs.items():
            out.attrs[key] = value
    os.replace(tmp, destination)


def condition_complete(dst: Path, subjects: list[Path]) -> bool:
    if not all((dst / name).is_file() for name in ("complete.flag", "manifest.json", "psi.npy")):
        return False
    return all((dst / p.name).is_file() for p in subjects)


def build_condition(
    dataset: str,
    cfg: dict[str, Any],
    tag: str,
    root: Path,
    seed: int,
    mirror_root: Path | None,
) -> dict[str, Any]:
    src_dir = Path(cfg["source_h5_dir"])
    subjects = sorted(src_dir.glob("sub-*.h5"))
    if len(subjects) != int(cfg["expected_subjects"]):
        raise RuntimeError(
            f"{dataset}: expected {cfg['expected_subjects']} subjects, found {len(subjects)}"
        )
    dst = root / dataset / tag
    dst.mkdir(parents=True, exist_ok=True)
    if condition_complete(dst, subjects):
        print(f"[skip] {dataset}/{tag}: complete cache exists", flush=True)
        with (dst / "manifest.json").open(encoding="utf-8") as f:
            return json.load(f)

    for stale in dst.glob("*.partial"):
        stale.unlink()
    (dst / "complete.flag").unlink(missing_ok=True)

    d_path = Path(cfg["d_path"])
    psi, singular = build_basis(d_path, float(cfg["ratio"]))
    npy_dump_atomic(psi, dst / "psi.npy")
    if mirror_root is not None:
        npy_dump_atomic(psi, mirror_root / dataset / "psi.npy")
    label_key = str(cfg["source_label_key"])
    rng = np.random.default_rng(seed)
    subject_rows: list[dict[str, Any]] = []

    print(
        f"[build] {dataset}/{tag}: n={len(subjects)} C={psi.shape[0]} "
        f"K={psi.shape[1]} ratio={cfg['ratio']}",
        flush=True,
    )
    for ordinal, source_path in enumerate(subjects, start=1):
        output_path = dst / source_path.name
        with h5py.File(source_path, "r") as source:
            raw_clean = source["eeg_raw"][:].astype(np.float32)
            labels_clean = source[label_key][:].astype(np.int64)
            n_trials, n_channels, n_samples = raw_clean.shape
            if n_channels != psi.shape[0] or n_samples != int(cfg["n_samples"]):
                raise ValueError(
                    f"{source_path.name}: raw={raw_clean.shape}, expected C={psi.shape[0]} "
                    f"T={cfg['n_samples']}"
                )

            keep_idx = None
            row: dict[str, Any] = {
                "subject": source_path.stem,
                "source_size": source_path.stat().st_size,
                "source_mtime_ns": source_path.stat().st_mtime_ns,
                "n_source": int(n_trials),
                "source_class_counts": class_counts(labels_clean),
            }
            attrs: dict[str, Any] = {
                "uncertainty_dataset": dataset,
                "uncertainty_tag": tag,
                "uncertainty_seed": int(seed),
                "source_label_key": label_key,
                "num_classes": int(cfg["n_classes"]),
                "eeg_modes_K": int(psi.shape[1]),
                "eeg_modes_ratio": float(cfg["ratio"]),
                "eeg_modes_method": "svd_truncated_sensor_basis_reprojected",
            }

            if tag in NOISE:
                alpha = NOISE[tag]
                sigma = raw_clean.std(axis=(0, 2), keepdims=True)
                noise = rng.standard_normal(raw_clean.shape).astype(np.float32)
                raw_out = raw_clean + noise * (alpha * sigma)
                attrs.update(
                    perturbation="noise_per_channel_sigma",
                    noise_alpha=float(alpha),
                )
            elif tag in DROP:
                frac = DROP[tag]
                xyz = source["eeg_pos"][:, :3].astype(np.float64)
                n_drop = int(round(frac * n_channels))
                dropped = np.sort(rng.choice(n_channels, size=n_drop, replace=False))
                kept = np.setdiff1d(np.arange(n_channels), dropped)
                raw_out = raw_clean.copy()
                for channel in dropped:
                    distances = np.linalg.norm(xyz[kept] - xyz[channel][None, :], axis=1)
                    neighbours = np.argsort(distances)[:4]
                    weights = 1.0 / (distances[neighbours] + 1e-6)
                    weights /= weights.sum()
                    raw_out[:, channel, :] = (
                        weights[None, :, None] * raw_clean[:, kept[neighbours], :]
                    ).sum(axis=1)
                row["dropped_channels"] = [int(x) for x in dropped]
                attrs.update(
                    perturbation="dropchan_invdist_nn_k4",
                    drop_frac=float(frac),
                    n_drop=int(n_drop),
                )
            elif tag in REDUCE:
                frac = REDUCE[tag]
                keep_idx = stratified_keep(labels_clean, frac, rng)
                raw_out = raw_clean[keep_idx]
                row["keep_idx"] = [int(x) for x in keep_idx]
                attrs.update(
                    perturbation="stratified_per_subject",
                    reduction_frac=float(frac),
                    n_kept=int(len(keep_idx)),
                )
            else:
                raise ValueError(f"Unknown perturbation tag: {tag}")

            modes_out = np.matmul(psi.T[None, :, :], raw_out).astype(np.float32)
            write_h5(source, output_path, raw_out, modes_out, label_key, keep_idx, attrs)
            out_labels = labels_clean if keep_idx is None else labels_clean[keep_idx]
            row.update(
                n_output=int(len(out_labels)),
                output_class_counts=class_counts(out_labels),
                output_size=output_path.stat().st_size,
                finite_raw=bool(np.isfinite(raw_out).all()),
                finite_modes=bool(np.isfinite(modes_out).all()),
            )
            subject_rows.append(row)
        if ordinal == 1 or ordinal % 20 == 0 or ordinal == len(subjects):
            print(f"  {dataset}/{tag}: {ordinal}/{len(subjects)}", flush=True)

    min_class = min(
        min(row["output_class_counts"].values()) for row in subject_rows
    )
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "tag": tag,
        "cache_dir": str(dst),
        "source_h5_dir": str(src_dir),
        "source_label_key": label_key,
        "output_label_key": "labels",
        "n_classes": int(cfg["n_classes"]),
        "seed": int(seed),
        "d_path": str(d_path),
        "d_sha256": sha256(d_path),
        "psi_path": str(dst / "psi.npy"),
        "psi_sha256": sha256(dst / "psi.npy"),
        "ratio": float(cfg["ratio"]),
        "K": int(psi.shape[1]),
        "singular_values": [float(x) for x in singular],
        "n_subjects": len(subject_rows),
        "min_output_class_count_per_subject": int(min_class),
        "fivefold_class_complete": bool(min_class >= 5),
        "subjects": subject_rows,
    }
    json_dump_atomic(manifest, dst / "manifest.json")
    if mirror_root is not None:
        json_dump_atomic(manifest, mirror_root / dataset / f"{tag}.json")
    (dst / "complete.flag").touch()
    print(
        f"[done] {dataset}/{tag}: {len(subject_rows)} subjects; "
        f"fivefold_class_complete={min_class >= 5}",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest-mirror", type=Path)
    parser.add_argument("--datasets", nargs="+", default=["BCIC", "MOTOR", "FACED"])
    parser.add_argument("--tags", nargs="+", default=ALL_TAGS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        configs = json.load(f)
    bad_tags = sorted(set(args.tags) - set(ALL_TAGS))
    if bad_tags:
        raise ValueError(f"Unknown tags: {bad_tags}")
    for dataset in args.datasets:
        if dataset not in configs:
            raise KeyError(f"Unknown dataset: {dataset}")
        for tag in args.tags:
            build_condition(
                dataset,
                configs[dataset],
                tag,
                args.cache_root,
                args.seed,
                args.manifest_mirror,
            )


if __name__ == "__main__":
    main()
