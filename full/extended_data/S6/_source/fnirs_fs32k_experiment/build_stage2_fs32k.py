


















from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(str(_LOCAL_SOURCE / ''))
FSLR_MODE_ROOT = PROJECT_ROOT / "guoyi_exp/data/eigenmode_test/fs32k"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    src_stage2: Path
    out_stage2: Path
    data_my: Path


DATASETS = {
    "dubois": DatasetConfig(
        name="dubois",
        root=Path('/media/wsqlab/nas2/Dataset/Reliability-Dubois2024'),
        src_stage2=Path('/media/wsqlab/nas2/Dataset/Reliability-Dubois2024/preprocess/stage2'),
        out_stage2=Path('/media/wsqlab/nas2/Dataset/Reliability-Dubois2024/preprocess/stage2_fs32k'),
        data_my=Path('/media/wsqlab/nas2/Dataset/Reliability-Dubois2024/data_my'),
    ),
    "cocktail": DatasetConfig(
        name="cocktail",
        root=Path('/media/wsqlab/nas2/Dataset/wholehead-cocktail-party-fnirs'),
        src_stage2=Path('/media/wsqlab/nas2/Dataset/wholehead-cocktail-party-fnirs/preprocess/stage2'),
        out_stage2=Path('/media/wsqlab/nas2/Dataset/wholehead-cocktail-party-fnirs/preprocess/stage2_fs32k'),
        data_my=Path('/media/wsqlab/nas2/Dataset/wholehead-cocktail-party-fnirs/data_my'),
    ),
}


def read_ascii_vtk_points(path: Path) -> np.ndarray:

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        tokens = fh.read().split()
    try:
        i = tokens.index("POINTS")
    except ValueError as exc:
        raise ValueError(f"POINTS section not found in {path}") from exc
    n = int(tokens[i + 1])
    vals = np.asarray(tokens[i + 3 : i + 3 + 3 * n], dtype=np.float64)
    if vals.size != 3 * n:
        raise ValueError(f"POINTS section truncated in {path}: expected {3*n}, got {vals.size}")
    return vals.reshape(n, 3)


def load_fs32k_basis(data_my: Path, k_total: int) -> tuple[np.ndarray, np.ndarray, dict]:

    k_left = k_total // 2
    k_right = k_total - k_left

    lh_xyz = read_ascii_vtk_points(data_my / "fsLR_32k_midthickness-lh.vtk")
    rh_xyz = read_ascii_vtk_points(data_my / "fsLR_32k_midthickness-rh.vtk")
    lh_mask = np.loadtxt(data_my / "fsLR_32k_cortex-lh_mask.txt").astype(bool).ravel()
    rh_mask = np.loadtxt(data_my / "fsLR_32k_cortex-rh_mask.txt").astype(bool).ravel()

    lh_modes = np.load(FSLR_MODE_ROOT / "fsLR_32k_lh_emode_1024.npy", mmap_mode="r")
    rh_modes = np.load(FSLR_MODE_ROOT / "fsLR_32k_rh_emode_1024.npy", mmap_mode="r")
    if k_left > lh_modes.shape[1] or k_right > rh_modes.shape[1]:
        raise ValueError(f"Requested k_total={k_total}, but only 1024 modes/hemi are cached")

    coords = np.concatenate([lh_xyz[lh_mask], rh_xyz[rh_mask]], axis=0).astype(np.float64)
    n_lh = int(lh_mask.sum())
    n_rh = int(rh_mask.sum())

    phi = np.zeros((n_lh + n_rh, k_total), dtype=np.float32)
    phi[:n_lh, :k_left] = np.asarray(lh_modes[lh_mask, :k_left], dtype=np.float32)
    phi[n_lh:, k_left:] = np.asarray(rh_modes[rh_mask, :k_right], dtype=np.float32)

    meta = {
        "k_left": int(k_left),
        "k_right": int(k_right),
        "n_lh_vertices": n_lh,
        "n_rh_vertices": n_rh,
        "fs32k_mode_root": str(FSLR_MODE_ROOT),
    }
    return coords, phi, meta


def align_midpoints_to_cortex(midpoints: np.ndarray, cortex_coords: np.ndarray) -> np.ndarray:

    mid = np.asarray(midpoints, dtype=np.float64)
    cortex = np.asarray(cortex_coords, dtype=np.float64)
    cm = mid.mean(axis=0)
    cc = cortex.mean(axis=0)
    rm = np.median(np.linalg.norm(mid - cm, axis=1))
    rc = np.median(np.linalg.norm(cortex - cc, axis=1))
    if rm <= 0 or rc <= 0:
        raise ValueError("Cannot align degenerate midpoint/cortex cloud")
    return (mid - cm) * (rc / rm) + cc


def build_sensor_template(
    midpoints: np.ndarray,
    cortex_coords: np.ndarray,
    phi_fs32k: np.ndarray,
    k_nn: int,
    sigma_mm: float,
) -> tuple[np.ndarray, dict]:

    aligned = align_midpoints_to_cortex(midpoints, cortex_coords)
    tree = cKDTree(cortex_coords)
    k_query = min(int(k_nn), cortex_coords.shape[0])
    dist, idx = tree.query(aligned, k=k_query, workers=-1)
    if k_query == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = np.exp(-0.5 * (dist / float(sigma_mm)) ** 2).astype(np.float64)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    n_ch = midpoints.shape[0]
    k_total = phi_fs32k.shape[1]
    d_template = np.empty((n_ch, k_total), dtype=np.float32)
    for i in range(n_ch):
        d_template[i] = weights[i] @ phi_fs32k[idx[i]].astype(np.float64)

    meta = {
        "k_nn": int(k_query),
        "sigma_mm": float(sigma_mm),
        "distance_mean_mm": float(dist.mean()),
        "distance_median_mm": float(np.median(dist)),
        "distance_p95_mm": float(np.percentile(dist, 95)),
        "aligned_midpoint_min": aligned.min(axis=0).tolist(),
        "aligned_midpoint_max": aligned.max(axis=0).tolist(),
    }
    return d_template, meta


def build_psi_from_template(d_template: np.ndarray, svd_ratio: float) -> tuple[np.ndarray, np.ndarray, int]:

    u, s, _ = np.linalg.svd(np.asarray(d_template, dtype=np.float64), full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        raise ValueError("Degenerate D template: no positive singular values")
    k_eff = int((s / s[0] >= float(svd_ratio)).sum())
    k_eff = max(1, k_eff)
    psi = u[:, :k_eff].astype(np.float32)
    return psi, s.astype(np.float64), k_eff


def project_trials(psi: np.ndarray, trials_data: np.ndarray, chunk: int = 20000) -> np.ndarray:

    nt, n_ch, n_time = trials_data.shape
    merged = np.transpose(trials_data, (1, 0, 2)).reshape(n_ch, -1)
    coeff = np.empty((psi.shape[1], merged.shape[1]), dtype=np.float32)
    psi_t = np.asarray(psi.T, dtype=np.float64)
    for start in range(0, merged.shape[1], chunk):
        end = min(start + chunk, merged.shape[1])
        coeff[:, start:end] = (psi_t @ merged[:, start:end].astype(np.float64)).astype(np.float32)
    return coeff.reshape(psi.shape[1], nt, n_time).transpose(1, 0, 2)


def copy_base_h5(src_path: Path, dst_path: Path, attrs_extra: dict) -> None:

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for key in [
            "labels",
            "session",
            "fnirs_pos",
            "fnirs_sensor_type",
            "fnirs_hbo_raw",
            "fnirs_hbr_raw",
        ]:
            src.copy(key, dst)
        for key, val in src.attrs.items():
            dst.attrs[key] = val
        for key, val in attrs_extra.items():
            dst.attrs[key] = val


def process_dataset(cfg: DatasetConfig, args: argparse.Namespace) -> dict:
    h5_files = sorted(cfg.src_stage2.glob("sub-*.h5"))
    if args.subjects:
        wanted = set(args.subjects)
        h5_files = [p for p in h5_files if p.stem in wanted]
    if args.max_subjects:
        h5_files = h5_files[: args.max_subjects]
    if not h5_files:
        raise FileNotFoundError(f"No source H5 files found in {cfg.src_stage2}")

    cfg.out_stage2.mkdir(parents=True, exist_ok=True)
    cache_dir = cfg.out_stage2 / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_files[0], "r") as f:
        midpoints = np.asarray(f["fnirs_pos"][:, :3], dtype=np.float64)
        n_ch = int(midpoints.shape[0])
        old_k = int(f.attrs.get("k_modes", np.floor(n_ch * 0.8)))

    k_total = int(args.k_total or old_k)
    cache_tag = f"K{k_total}_knn{args.k_nn}_sig{args.sigma_mm:g}_svd{args.svd_ratio:g}"
    psi_path = cache_dir / f"fs32k_sensor_psi_{cache_tag}.npy"
    meta_path = cache_dir / f"fs32k_sensor_meta_{cache_tag}.json"
    s_path = cache_dir / f"fs32k_sensor_singular_{cache_tag}.npy"
    d_path = cache_dir / f"fs32k_D_template_{cache_tag}.npy"

    if psi_path.exists() and meta_path.exists() and not args.recompute_template:
        psi = np.load(psi_path)
        singular = np.load(s_path)
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        print(f"[cache] {cfg.name}: psi={psi.shape} from {psi_path}", flush=True)
    else:
        cortex_coords, phi_fs32k, phi_meta = load_fs32k_basis(cfg.data_my, k_total)
        d_template, map_meta = build_sensor_template(
            midpoints=midpoints,
            cortex_coords=cortex_coords,
            phi_fs32k=phi_fs32k,
            k_nn=args.k_nn,
            sigma_mm=args.sigma_mm,
        )
        psi, singular, k_eff = build_psi_from_template(d_template, args.svd_ratio)
        meta = {
            "dataset": cfg.name,
            "source_stage2": str(cfg.src_stage2),
            "out_stage2": str(cfg.out_stage2),
            "n_channels": n_ch,
            "k_total_cortical": k_total,
            "k_eff_sensor": int(k_eff),
            "svd_ratio": float(args.svd_ratio),
            "method": "D = W(Phi_fsLR32k), Psi = left singular vectors of D, coeff = Psi.T X",
            **phi_meta,
            **map_meta,
        }
        np.save(psi_path, psi)
        np.save(s_path, singular)
        if args.save_d_template:
            np.save(d_path, d_template.astype(np.float32))
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        print(
            f"[build] {cfg.name}: D={d_template.shape} psi={psi.shape} "
            f"S0={singular[0]:.4g} Seff/S0={singular[psi.shape[1]-1]/singular[0]:.3g}",
            flush=True,
        )

    attrs_extra = {
        "stage2_variant": "fs32k_fNIRS",
        "fs32k_k_total_cortical": int(meta["k_total_cortical"]),
        "fs32k_k_eff_sensor": int(psi.shape[1]),
        "fs32k_svd_ratio": float(args.svd_ratio),
        "fs32k_mapping": "gaussian SD-midpoint to fsLR32k midthickness",
        "fs32k_template_cache": str(psi_path),
    }

    n_done = 0
    for src_path in h5_files:
        dst_path = cfg.out_stage2 / src_path.name
        if dst_path.exists() and not args.overwrite:
            with h5py.File(dst_path, "r") as f:
                if "fnirs_hbo_fs32k_modes" in f and "fnirs_hbr_fs32k_modes" in f:
                    print(f"[skip] {cfg.name}/{src_path.stem}: exists", flush=True)
                    continue
        tmp_path = dst_path.with_suffix(".tmp.h5")
        if tmp_path.exists():
            tmp_path.unlink()
        copy_base_h5(src_path, tmp_path, attrs_extra)
        with h5py.File(tmp_path, "a") as f:
            hbo = np.asarray(f["fnirs_hbo_raw"], dtype=np.float32)
            hbr = np.asarray(f["fnirs_hbr_raw"], dtype=np.float32)
            hbo_modes = project_trials(psi, hbo, chunk=args.chunk)
            hbr_modes = project_trials(psi, hbr, chunk=args.chunk)
            f.create_dataset("fnirs_hbo_fs32k_modes", data=hbo_modes, dtype=np.float32)
            f.create_dataset("fnirs_hbr_fs32k_modes", data=hbr_modes, dtype=np.float32)
            f.attrs["k_modes"] = int(psi.shape[1])
            f.attrs["n_channels"] = int(psi.shape[0])
        shutil.move(str(tmp_path), str(dst_path))
        n_done += 1
        print(
            f"[done] {cfg.name}/{src_path.stem}: "
            f"hbo_modes={hbo_modes.shape} hbr_modes={hbr_modes.shape}",
            flush=True,
        )

    return {
        "dataset": cfg.name,
        "out_stage2": str(cfg.out_stage2),
        "n_subjects_processed": n_done,
        "psi_shape": list(psi.shape),
        "meta_path": str(meta_path),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["dubois", "cocktail"], choices=sorted(DATASETS))
    ap.add_argument("--subjects", nargs="*", default=None, help="Optional subject IDs, e.g. sub-01")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--k-total", type=int, default=None, help="Cortical mode count; default=old stage2 k_modes")
    ap.add_argument("--k-nn", type=int, default=96)
    ap.add_argument("--sigma-mm", type=float, default=25.0)
    ap.add_argument("--svd-ratio", type=float, default=1e-3)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--recompute-template", action="store_true")
    ap.add_argument("--save-d-template", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for name in args.datasets:
        results.append(process_dataset(DATASETS[name], args))
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
