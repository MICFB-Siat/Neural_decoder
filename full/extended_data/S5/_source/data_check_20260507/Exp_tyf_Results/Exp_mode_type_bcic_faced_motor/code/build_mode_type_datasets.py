









from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

from experiment_config import DATASETS, DATA_ROOT, MODE_LABELS, MODE_TYPES

try:
    from scipy.special import sph_harm_y as _sph_harm_y

    def complex_sh(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return _sph_harm_y(l, m, theta, phi)
except ImportError:
    from scipy.special import sph_harm as _sph_harm

    def complex_sh(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return _sph_harm(m, l, phi, theta)


def canonicalize_columns(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64).copy()
    for j in range(u.shape[1]):
        i = int(np.argmax(np.abs(u[:, j])))
        if u[i, j] < 0:
            u[:, j] *= -1
    return u


def graph_laplacian_basis(xyz: np.ndarray, k: int, k_nn: int = 6):
    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    nn = np.argsort(dist, axis=1)[:, : min(k_nn, len(xyz) - 1)]
    mask = np.zeros_like(dist, dtype=bool)
    mask[np.arange(len(xyz))[:, None], nn] = True
    mask |= mask.T
    sigma = float(np.mean(dist[mask]))
    adjacency = np.zeros_like(dist)
    adjacency[mask] = np.exp(-(dist[mask] ** 2) / (2.0 * sigma**2))
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
    lap = np.eye(len(xyz)) - inv_sqrt @ adjacency @ inv_sqrt
    eigval, eigvec = np.linalg.eigh(lap)
    order = np.argsort(eigval)
    return canonicalize_columns(eigvec[:, order[:k]]), eigval[order[:k]]


def real_spherical_harmonic(l: int, m: int, theta: np.ndarray, phi: np.ndarray):
    if m < 0:
        return np.sqrt(2.0) * (-1) ** m * complex_sh(l, -m, theta, phi).imag
    if m == 0:
        return complex_sh(l, 0, theta, phi).real
    return np.sqrt(2.0) * (-1) ** m * complex_sh(l, m, theta, phi).real


def spherical_harmonic_basis(xyz: np.ndarray, k: int):
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Electrode coordinates contain a zero-length position vector")
    unit = xyz / norms
    theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(unit[:, 1], unit[:, 0]), 2.0 * np.pi)
    lm = []
    l = 0
    while len(lm) < k:
        lm.extend((l, m) for m in range(-l, l + 1))
        l += 1
    lm = lm[:k]
    y = np.column_stack([real_spherical_harmonic(l, m, theta, phi) for l, m in lm])
    q, _ = np.linalg.qr(y, mode="reduced")
    if q.shape[1] < k:
        raise ValueError(f"Spherical-harmonic design rank {q.shape[1]} < requested K={k}")
    native = np.asarray([l * (l + 1) for l, _ in lm], dtype=np.float64)
    return canonicalize_columns(q[:, :k]), native, lm


def dct_basis(n_sensors: int, k: int):
    row = np.arange(n_sensors, dtype=np.float64)[:, None]
    freq = np.arange(k, dtype=np.float64)[None, :]
    u = np.sqrt(2.0 / n_sensors) * np.cos(np.pi * (row + 0.5) * freq / n_sensors)
    u[:, 0] = 1.0 / np.sqrt(n_sensors)
    native = 2.0 - 2.0 * np.cos(np.pi * np.arange(k) / n_sensors)
    return canonicalize_columns(u), native


def shifted_position_eigenvalues(native: np.ndarray):
    native = np.asarray(native, dtype=np.float64)
    positive_gap = native[native > native[0] + 1e-12]
    if not len(positive_gap):
        raise ValueError("No non-zero spectral gap found")
    gap = float(positive_gap[0] - native[0])
    shifted = native + gap
    if np.any(shifted <= 0) or not np.isfinite(shifted).all():
        raise ValueError(f"Invalid shifted eigenvalues: min={shifted.min()}")
    return shifted, gap


def sha256_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def atomic_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".partial.{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    with src.open("rb") as fi, tmp.open("wb") as fo:
        shutil.copyfileobj(fi, fo, length=16 << 20)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, dst)


def valid_output(path: Path, mode_type: str, shape: tuple[int, int, int]) -> bool:
    try:
        with h5py.File(path, "r") as f:
            return (
                f["eeg_modes"].shape == shape
                and f["eeg_mode_eigval"].shape == (shape[1],)
                and f.attrs.get("mode_type", "") == mode_type
                and np.isfinite(f["eeg_modes"][0, :, :8]).all()
            )
    except Exception:
        return False


def preflight(dataset: str, subject_limit: int | None = None):
    cfg = DATASETS[dataset]
    subjects = sorted(Path(cfg["source"]).glob("sub-*.h5"))
    if len(subjects) != cfg["expected_subjects"]:
        raise RuntimeError(
            f"{dataset}: expected {cfg['expected_subjects']} subjects, found {len(subjects)}"
        )
    if subject_limit is not None:
        subjects = subjects[:subject_limit]
    reference_pos = None
    total_trials = 0
    label_counts = np.zeros(cfg["n_classes"], dtype=np.int64)
    inventory = []
    for path in subjects:
        with h5py.File(path, "r") as f:
            required = {"eeg_raw", "eeg_pos", "eeg_sensor_type", cfg["label_key"]}
            missing = required - set(f.keys())
            if missing:
                raise KeyError(f"{path}: missing {sorted(missing)}")
            shape = tuple(int(x) for x in f["eeg_raw"].shape)
            pos = f["eeg_pos"][:].astype(np.float64)
            labels = f[cfg["label_key"]][:].astype(np.int64)
            if shape[1:] != (cfg["n_sensors"], cfg["n_samples"]):
                raise ValueError(f"{path}: eeg_raw {shape}, expected (*,{cfg['n_sensors']},{cfg['n_samples']})")
            if labels.shape != (shape[0],) or labels.min() < 0 or labels.max() >= cfg["n_classes"]:
                raise ValueError(f"{path}: invalid {cfg['label_key']} shape/range")
            if reference_pos is None:
                reference_pos = pos
            elif pos.shape != reference_pos.shape or not np.allclose(pos, reference_pos, atol=1e-6, rtol=0):
                raise ValueError(f"{path}: montage differs from first subject")
            total_trials += shape[0]
            label_counts += np.bincount(labels, minlength=cfg["n_classes"])
            inventory.append({"subject": path.stem, "trials": shape[0], "source": str(path.resolve())})
    return subjects, reference_pos, total_trials, label_counts, inventory


def build_dataset(dataset: str, output_root: Path, subject_limit: int | None, k_nn: int, block_trials: int):
    cfg = DATASETS[dataset]
    subjects, pos, total_trials, label_counts, inventory = preflight(dataset, subject_limit)
    xyz = pos[:, -3:] if pos.shape[1] >= 6 else pos[:, :3]
    if xyz.shape != (cfg["n_sensors"], 3):
        raise ValueError(f"{dataset}: electrode xyz shape {xyz.shape}")

    k = cfg["n_modes"]
    dct_u, dct_ev = dct_basis(cfg["n_sensors"], k)
    sh_u, sh_ev, lm = spherical_harmonic_basis(xyz, k)
    lap_u, lap_ev = graph_laplacian_basis(xyz, k, k_nn)
    bases = {
        "fourier_dct": (dct_u, dct_ev),
        "spherical_harmonic": (sh_u, sh_ev),
        "laplacian": (lap_u, lap_ev),
    }

    manifest = {
        "dataset": dataset,
        "source": str(Path(cfg["source"]).resolve()),
        "label_key": cfg["label_key"],
        "n_classes": cfg["n_classes"],
        "n_subjects": len(subjects),
        "n_trials": total_trials,
        "n_sensors": cfg["n_sensors"],
        "n_modes": k,
        "n_samples": cfg["n_samples"],
        "k_rule": "min(30, n_sensors)",
        "label_counts": label_counts.tolist(),
        "projection": "eeg_modes = basis.T @ eeg_raw",
        "position_encoding": "native basis eigenvalues + first non-zero spectral gap",
        "spherical_harmonic_order": [[l, m] for l, m in lm],
        "subjects": inventory,
        "bases": {},
    }

    out_dataset = output_root / dataset
    for mode_type, (basis, native) in bases.items():
        err = float(np.linalg.norm(basis.T @ basis - np.eye(k), ord="fro"))
        if basis.shape != (cfg["n_sensors"], k) or err > 1e-8 or not np.isfinite(basis).all():
            raise ValueError(f"{dataset}/{mode_type}: invalid basis shape={basis.shape}, orth_err={err}")
        shifted, gap = shifted_position_eigenvalues(native)
        mode_dir = out_dataset / mode_type
        mode_dir.mkdir(parents=True, exist_ok=True)
        np.save(mode_dir / f"basis_K{k}.npy", basis.astype(np.float32))
        np.save(mode_dir / f"native_eigval_K{k}.npy", native.astype(np.float32))
        np.save(mode_dir / f"position_eigval_K{k}.npy", shifted.astype(np.float32))
        manifest["bases"][mode_type] = {
            "label": MODE_LABELS[mode_type],
            "shape": list(basis.shape),
            "orthogonality_fro_error": err,
            "basis_sha256": sha256_array(basis.astype(np.float32)),
            "native_eigenvalue_range": [float(native.min()), float(native.max())],
            "position_eigenvalue_range": [float(shifted.min()), float(shifted.max())],
            "spectral_gap_shift": gap,
        }

    stage_parent = Path(os.environ.get("MODE_TYPE_STAGE", "/tmp/mode_type_bcic_faced_motor_build"))
    stage_parent.mkdir(parents=True, exist_ok=True)
    for subject_index, src in enumerate(subjects, 1):
        with h5py.File(src, "r") as fs:
            n_trials = int(fs["eeg_raw"].shape[0])
            shape = (n_trials, k, cfg["n_samples"])
            pending = [m for m in MODE_TYPES if not valid_output(out_dataset / m / src.name, m, shape)]
            if not pending:
                print(f"{dataset} {src.stem}: already complete", flush=True)
                continue
            with tempfile.TemporaryDirectory(prefix=f"{dataset}_{src.stem}_", dir=stage_parent) as td:
                staged = {}
                handles = {}
                try:
                    for mode_type in pending:
                        path = Path(td) / f"{mode_type}_{src.name}"
                        fd = h5py.File(path, "w")
                        handles[mode_type] = fd
                        staged[mode_type] = path
                        fd.create_dataset(
                            "eeg_modes", shape=shape, dtype="f4", compression="lzf",
                            chunks=(min(block_trials, n_trials), k, min(512, cfg["n_samples"])),
                        )
                        fd.create_dataset(
                            "eeg_mode_eigval",
                            data=np.load(out_dataset / mode_type / f"position_eigval_K{k}.npy"),
                        )
                        for key in ("eeg_raw", "eeg_pos", "eeg_sensor_type", "labels", "label2"):
                            if key in fs:
                                fd[key] = h5py.ExternalLink(str(src.resolve()), f"/{key}")
                        fd.attrs["dataset"] = dataset
                        fd.attrs["mode_type"] = mode_type
                        fd.attrs["eeg_modes_K"] = k
                        fd.attrs["source_h5"] = str(src.resolve())
                        fd.attrs["projection"] = "basis.T @ eeg_raw"
                        fd.attrs["position_encoding"] = "native eigenvalues + first non-zero spectral gap"
                    for start in range(0, n_trials, block_trials):
                        stop = min(start + block_trials, n_trials)
                        raw = fs["eeg_raw"][start:stop].astype(np.float32, copy=False)
                        for mode_type in pending:
                            basis = bases[mode_type][0].astype(np.float32, copy=False)
                            handles[mode_type]["eeg_modes"][start:stop] = np.einsum(
                                "ck,nct->nkt", basis, raw, optimize=True
                            )
                    for fd in handles.values():
                        fd.flush()
                        fd.close()
                    handles.clear()
                    for mode_type in pending:
                        dst = out_dataset / mode_type / src.name
                        atomic_copy(staged[mode_type], dst)
                        if not valid_output(dst, mode_type, shape):
                            raise RuntimeError(f"Post-copy validation failed: {dst}")
                finally:
                    for fd in handles.values():
                        fd.close()
        print(f"{dataset} {src.stem}: built {len(pending)} bases ({subject_index}/{len(subjects)})", flush=True)


    with h5py.File(subjects[0], "r") as fs:
        raw = fs["eeg_raw"][0].astype(np.float32)
    for mode_type, (basis, _) in bases.items():
        with h5py.File(out_dataset / mode_type / subjects[0].name, "r") as f:
            observed = f["eeg_modes"][0]
        expected = basis.astype(np.float32).T @ raw
        max_abs = float(np.max(np.abs(observed - expected)))
        manifest["bases"][mode_type]["projection_check_max_abs_error"] = max_abs
        if not np.allclose(observed, expected, atol=2e-5, rtol=2e-5):
            raise RuntimeError(f"{dataset}/{mode_type}: projection check failed, max_abs={max_abs}")

    atomic_json(out_dataset / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("dataset", "n_subjects", "n_trials", "n_sensors", "n_modes")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    ap.add_argument("--output-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--subject-limit", type=int, default=None, help="Smoke-test only")
    ap.add_argument("--k-nn", type=int, default=6)
    ap.add_argument("--block-trials", type=int, default=8)
    args = ap.parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        build_dataset(dataset, args.output_root.resolve(), args.subject_limit, args.k_nn, args.block_trials)


if __name__ == "__main__":
    main()
