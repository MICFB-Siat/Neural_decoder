






from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
try:
    from scipy.special import sph_harm_y as _sph_harm_y

    def complex_sh(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return _sph_harm_y(l, m, theta, phi)
except ImportError:
    from scipy.special import sph_harm as _sph_harm

    def complex_sh(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return _sph_harm(m, l, phi, theta)

MODE_TYPES = ("laplacian", "spherical_harmonic", "fourier_dct")
K = 30


def canonicalize_columns(u: np.ndarray) -> np.ndarray:

    u = np.asarray(u, dtype=np.float64).copy()
    for j in range(u.shape[1]):
        i = int(np.argmax(np.abs(u[:, j])))
        if u[i, j] < 0:
            u[:, j] *= -1
    return u


def graph_laplacian_basis(xyz: np.ndarray, k_nn: int = 6) -> tuple[np.ndarray, np.ndarray]:
    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    nn = np.argsort(dist, axis=1)[:, :k_nn]
    mask = np.zeros_like(dist, dtype=bool)
    mask[np.arange(len(xyz))[:, None], nn] = True
    mask |= mask.T
    sigma = float(np.mean(dist[mask]))
    a = np.zeros_like(dist)
    a[mask] = np.exp(-(dist[mask] ** 2) / (2.0 * sigma**2))
    degree = a.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
    lap = np.eye(len(xyz)) - d_inv_sqrt @ a @ d_inv_sqrt
    eigval, eigvec = np.linalg.eigh(lap)
    order = np.argsort(eigval)
    return canonicalize_columns(eigvec[:, order[:K]]), eigval[order[:K]]


def real_spherical_harmonic(l: int, m: int, theta: np.ndarray,
                            phi: np.ndarray) -> np.ndarray:

    if m < 0:
        return np.sqrt(2.0) * (-1) ** m * complex_sh(l, -m, theta, phi).imag
    if m == 0:
        return complex_sh(l, 0, theta, phi).real
    return np.sqrt(2.0) * (-1) ** m * complex_sh(l, m, theta, phi).real


def spherical_harmonic_basis(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    unit = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(unit[:, 1], unit[:, 0]), 2.0 * np.pi)
    lm = [(l, m) for l in range(6) for m in range(-l, l + 1)][:K]
    y = np.column_stack([real_spherical_harmonic(l, m, theta, phi) for l, m in lm])

    q, _ = np.linalg.qr(y, mode="reduced")
    q = canonicalize_columns(q[:, :K])
    eigval = np.asarray([l * (l + 1) for l, _ in lm], dtype=np.float64)
    return q, eigval, lm


def dct_basis(n: int) -> tuple[np.ndarray, np.ndarray]:
    row = np.arange(n, dtype=np.float64)[:, None]
    freq = np.arange(K, dtype=np.float64)[None, :]
    u = np.sqrt(2.0 / n) * np.cos(np.pi * (row + 0.5) * freq / n)
    u[:, 0] = 1.0 / np.sqrt(n)
    eigval = 2.0 - 2.0 * np.cos(np.pi * np.arange(K) / n)
    return canonicalize_columns(u), eigval


def validate_basis(name: str, u: np.ndarray, n_sensor: int) -> dict:
    if u.shape != (n_sensor, K):
        raise ValueError(f"{name}: expected {(n_sensor, K)}, got {u.shape}")
    err = float(np.linalg.norm(u.T @ u - np.eye(K), ord="fro"))
    if not np.isfinite(u).all() or err > 1e-8:
        raise ValueError(f"{name}: invalid/non-orthogonal basis, Frobenius error={err:g}")
    return {"shape": list(u.shape), "orthogonality_fro_error": err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--k-nn", type=int, default=6)
    args = ap.parse_args()

    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    subjects = sorted(source.glob("sub-*.h5"))
    if not subjects:
        raise FileNotFoundError(f"No sub-*.h5 in {source}")
    with h5py.File(subjects[0], "r") as f:
        pos = f["eeg_pos"][:].astype(np.float64)
        n_sensor = int(f["eeg_raw"].shape[1])
    xyz = pos[:, -3:] if pos.shape[1] >= 6 else pos[:, :3]
    if xyz.shape != (n_sensor, 3):
        raise ValueError(f"Expected sensor xyz {(n_sensor, 3)}, got {xyz.shape}")

    lap_u, lap_ev = graph_laplacian_basis(xyz, args.k_nn)
    sh_u, sh_ev, lm = spherical_harmonic_basis(xyz)
    dct_u, dct_ev = dct_basis(n_sensor)
    bases = {
        "laplacian": (lap_u, lap_ev),
        "spherical_harmonic": (sh_u, sh_ev),
        "fourier_dct": (dct_u, dct_ev),
    }

    original_d = np.load(args.template).astype(np.float64)
    if original_d.shape[0] != n_sensor:
        raise ValueError(f"Template sensors={original_d.shape[0]}, H5 sensors={n_sensor}")
    _, s, vt = np.linalg.svd(original_d, full_matrices=False)
    if len(s) < K:
        raise ValueError(f"Template rank dimension {len(s)} < K={K}")



    fs32k = Path(args.template).resolve().parent.parent / "fs32k"
    half_lh = (original_d.shape[1] + 1) // 2
    half_rh = original_d.shape[1] // 2
    lam_cortex = np.concatenate([
        np.load(fs32k / "fsLR_32k_lh_eval_1024.npy")[:half_lh],
        np.load(fs32k / "fsLR_32k_rh_eval_1024.npy")[:half_rh],
    ]).astype(np.float64)
    np.save(output / "lam_cortex_K48.npy", lam_cortex)

    manifest = {
        "source": str(source), "template": str(Path(args.template).resolve()),
        "n_subjects": len(subjects), "n_sensor": n_sensor, "K": K,
        "fairness": "left basis varies; original singular values/right singular vectors fixed",
        "lam_cortex_path": str(output / "lam_cortex_K48.npy"),
        "spherical_harmonic_order": [[l, m] for l, m in lm], "bases": {},
    }
    for mode_type, (basis, native_ev) in bases.items():
        info = validate_basis(mode_type, basis, n_sensor)
        mode_dir = output / mode_type
        mode_dir.mkdir(exist_ok=True)
        np.save(mode_dir / "basis_K30.npy", basis.astype(np.float32))
        np.save(mode_dir / "native_eigenvalues_K30.npy", native_ev.astype(np.float32))

        d_type = (basis * s[:K]) @ vt[:K]
        np.save(mode_dir / "D_K30.npy", d_type.astype(np.float64))
        info["native_eigenvalue_range"] = [float(native_ev.min()), float(native_ev.max())]
        manifest["bases"][mode_type] = info

        for src in subjects:
            dst = mode_dir / src.name
            if dst.exists():
                with h5py.File(dst, "r") as f:
                    if f["eeg_modes"].shape[1] == K and f.attrs.get("mode_type", "") == mode_type:
                        continue
                dst.unlink()
            with h5py.File(src, "r") as fs:
                raw = fs["eeg_raw"][:].astype(np.float32)
                modes = np.matmul(basis.T.astype(np.float32)[None, :, :], raw)
            with h5py.File(dst, "w") as fd:
                fd.create_dataset("eeg_modes", data=modes, compression="lzf")
                for key in ("eeg_raw", "eeg_pos", "eeg_sensor_type", "labels"):
                    fd[key] = h5py.ExternalLink(str(src), f"/{key}")
                fd.attrs["mode_type"] = mode_type
                fd.attrs["eeg_modes_K"] = K
                fd.attrs["source_h5"] = str(src)
                fd.attrs["template"] = str(Path(args.template).resolve())
        print(f"{mode_type}: built {len(subjects)} subjects", flush=True)

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
