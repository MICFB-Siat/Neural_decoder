





















from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fnirs_fs32k_experiment.build_stage2_fs32k import (
    DATASETS,
    FSLR_MODE_ROOT,
    build_sensor_template,
    load_fs32k_basis,
)


DEFAULT_OUT_SUFFIX = "stage2_fs32k_poe"


def _as_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def load_stage2_meta(stage2_dir: Path) -> dict:
    meta_files = sorted((stage2_dir / "cache").glob("fs32k_sensor_meta_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No fs32k_sensor_meta_*.json found under {stage2_dir / 'cache'}")
    if len(meta_files) > 1:
        raise RuntimeError(
            f"Multiple template metadata files found under {stage2_dir / 'cache'}; "
            "rerun with a clean cache or keep only the intended template."
        )
    with meta_files[0].open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["_meta_path"] = str(meta_files[0])
    return meta


def load_group_lam_cortex(k_left: int, k_right: int) -> np.ndarray:
    lh = np.load(FSLR_MODE_ROOT / "fsLR_32k_lh_eval_1024.npy")[:k_left].astype(np.float64)
    rh = np.load(FSLR_MODE_ROOT / "fsLR_32k_rh_eval_1024.npy")[:k_right].astype(np.float64)
    return np.concatenate([lh, rh], axis=0)


def compute_eigval_from_d(d_template: np.ndarray, lam_cortex: np.ndarray, ratio: float) -> tuple[np.ndarray, int]:
    d_np = np.asarray(d_template, dtype=np.float64)
    if d_np.ndim != 2:
        raise ValueError(f"D template must be 2D, got {d_np.shape}")
    if lam_cortex.shape != (d_np.shape[1],):
        raise ValueError(f"lambda shape {lam_cortex.shape} does not match D columns {d_np.shape[1]}")
    _, singular, vt = np.linalg.svd(d_np, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0:
        raise ValueError("Degenerate D template: no positive singular values")
    k_eff = int((singular / singular[0] >= float(ratio)).sum())
    k_eff = max(1, k_eff)
    v = vt.T
    eigval = (v[:, :k_eff] ** 2 * lam_cortex[:, None]).sum(axis=0).astype(np.float32)
    return eigval, k_eff


def compute_or_load_single_chromophore_eigval(cfg, stage2_dir: Path, meta: dict, args) -> np.ndarray:
    cache_dir = stage2_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = (
        f"K{int(meta['k_total_cortical'])}_knn{int(meta['k_nn'])}"
        f"_sig{float(meta['sigma_mm']):g}_svd{float(meta['svd_ratio']):g}"
    )
    eig_path = cache_dir / f"fs32k_sensor_mode_eigval_{cache_tag}.npy"
    if eig_path.exists() and not args.recompute_eigval:
        eigval = np.load(eig_path).astype(np.float32)
        print(f"[cache] {cfg.name}: eigval={eigval.shape} from {eig_path}", flush=True)
        return eigval

    h5_files = sorted(stage2_dir.glob("sub-*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No stage2_fs32k H5 files found in {stage2_dir}")
    with h5py.File(h5_files[0], "r") as f:
        midpoints = np.asarray(f["fnirs_pos"][:, :3], dtype=np.float64)

    k_total = int(meta["k_total_cortical"])
    k_left = int(meta.get("k_left", k_total // 2))
    k_right = int(meta.get("k_right", k_total - k_left))
    cortex_coords, phi_fs32k, _ = load_fs32k_basis(cfg.data_my, k_total)
    d_template, _ = build_sensor_template(
        midpoints=midpoints,
        cortex_coords=cortex_coords,
        phi_fs32k=phi_fs32k,
        k_nn=int(meta["k_nn"]),
        sigma_mm=float(meta["sigma_mm"]),
    )
    lam = load_group_lam_cortex(k_left, k_right)
    eigval, k_eff = compute_eigval_from_d(d_template, lam, ratio=float(meta["svd_ratio"]))
    expected_k = int(meta["k_eff_sensor"])
    if k_eff != expected_k:
        raise RuntimeError(f"Computed K_eff={k_eff}, but stage2 metadata says {expected_k}")
    np.save(eig_path, eigval)
    print(f"[build] {cfg.name}: eigval={eigval.shape} saved to {eig_path}", flush=True)
    return eigval


def resolve_n_samples(original_t: int, patch_size: int, policy: str, target: int | None) -> int:
    if target is not None:
        if target <= 0:
            raise ValueError("--n-samples must be positive")
        if target % patch_size != 0:
            raise ValueError(f"--n-samples={target} is not divisible by --patch-size={patch_size}")
        if policy == "exact" and target != original_t:
            raise ValueError(f"--time-policy exact requires target {target} == original T {original_t}")
        return int(target)
    if policy == "exact":
        if original_t % patch_size != 0:
            raise ValueError(f"Original T={original_t} is not divisible by patch size {patch_size}")
        return int(original_t)
    if policy == "crop":
        out_t = (original_t // patch_size) * patch_size
        if out_t <= 0:
            raise ValueError(f"Cannot crop T={original_t} to a positive multiple of {patch_size}")
        return int(out_t)
    if policy == "pad":
        return int(((original_t + patch_size - 1) // patch_size) * patch_size)
    raise ValueError(f"Unknown time policy: {policy}")


def fit_time(x: np.ndarray, n_samples: int) -> np.ndarray:
    t = x.shape[-1]
    if t == n_samples:
        return np.asarray(x, dtype=np.float32)
    if t > n_samples:
        return np.asarray(x[..., :n_samples], dtype=np.float32)
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, n_samples - t)
    return np.pad(np.asarray(x, dtype=np.float32), pad_width, mode="constant")


def create_array_dataset(f: h5py.File, key: str, data: np.ndarray, compression: str) -> None:
    kwargs = {}
    if compression != "none" and np.asarray(data).ndim >= 2 and np.asarray(data).dtype.kind in "fiu":
        kwargs["compression"] = compression
        kwargs["shuffle"] = True
    f.create_dataset(key, data=data, **kwargs)


def select_subjects(stage2_dir: Path, subjects: list[str] | None, max_subjects: int | None) -> list[Path]:
    h5_files = sorted(stage2_dir.glob("sub-*.h5"))
    if subjects:
        wanted = set(subjects)
        h5_files = [p for p in h5_files if p.stem in wanted]
    if max_subjects is not None:
        h5_files = h5_files[:max_subjects]
    if not h5_files:
        raise FileNotFoundError(f"No matching H5 files found in {stage2_dir}")
    return h5_files


def build_one_subject(
    src_path: Path,
    dst_path: Path,
    eigval_single: np.ndarray,
    chromophores: str,
    n_samples: int,
    time_policy: str,
    patch_size: int,
    compression: str,
    overwrite: bool,
) -> dict:
    if dst_path.exists() and not overwrite:
        with h5py.File(dst_path, "r") as f:
            return {
                "subject": dst_path.stem,
                "status": "skip",
                "raw_shape": list(f["fnirs_raw"].shape),
                "modes_shape": list(f["fnirs_modes"].shape),
            }

    tmp_path = dst_path.with_suffix(".tmp.h5")
    if tmp_path.exists():
        tmp_path.unlink()

    with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        labels = np.asarray(src["labels"])
        sessions = np.asarray(src["session"]) if "session" in src else None
        pos = np.asarray(src["fnirs_pos"], dtype=np.float32)
        stype = np.asarray(src["fnirs_sensor_type"], dtype=np.int64)

        raw_parts = []
        mode_parts = []
        pos_parts = []
        stype_parts = []
        eig_parts = []
        used = []
        if chromophores in {"hbo", "both"}:
            raw_parts.append(fit_time(np.asarray(src["fnirs_hbo_raw"]), n_samples))
            mode_parts.append(fit_time(np.asarray(src["fnirs_hbo_fs32k_modes"]), n_samples))
            pos_parts.append(pos)
            stype_parts.append(stype)
            eig_parts.append(eigval_single)
            used.append("hbo")
        if chromophores in {"hbr", "both"}:
            raw_parts.append(fit_time(np.asarray(src["fnirs_hbr_raw"]), n_samples))
            mode_parts.append(fit_time(np.asarray(src["fnirs_hbr_fs32k_modes"]), n_samples))
            pos_parts.append(pos)
            stype_parts.append(stype)
            eig_parts.append(eigval_single)
            used.append("hbr")

        raw = np.concatenate(raw_parts, axis=1).astype(np.float32)
        modes = np.concatenate(mode_parts, axis=1).astype(np.float32)
        pos_out = np.concatenate(pos_parts, axis=0).astype(np.float32)
        stype_out = np.concatenate(stype_parts, axis=0).astype(np.int64)
        eigval_out = np.concatenate(eig_parts, axis=0).astype(np.float32)

        create_array_dataset(dst, "labels", labels, "none")
        if sessions is not None:
            create_array_dataset(dst, "session", sessions, "none")
        create_array_dataset(dst, "fnirs_raw", raw, compression)
        create_array_dataset(dst, "fnirs_modes", modes, compression)
        create_array_dataset(dst, "fnirs_pos", pos_out, compression)
        create_array_dataset(dst, "fnirs_sensor_type", stype_out, "none")
        create_array_dataset(dst, "fnirs_mode_eigval", eigval_out, "none")

        for key, val in src.attrs.items():
            dst.attrs[key] = val
        dst.attrs["poe_ready"] = True
        dst.attrs["poe_source_h5"] = str(src_path)
        dst.attrs["poe_chromophores"] = "+".join(used)
        dst.attrs["poe_time_policy"] = time_policy
        dst.attrs["poe_n_samples"] = int(n_samples)
        dst.attrs["poe_patch_size"] = int(patch_size)
        dst.attrs["poe_raw_channels"] = int(raw.shape[1])
        dst.attrs["poe_mode_count"] = int(modes.shape[1])
        dst.attrs["poe_decoder_modality"] = "fnirs"

    shutil.move(str(tmp_path), str(dst_path))
    return {
        "subject": dst_path.stem,
        "status": "done",
        "raw_shape": list(raw.shape),
        "modes_shape": list(modes.shape),
        "n_samples": int(n_samples),
    }


def process_dataset(name: str, args) -> dict:
    cfg = DATASETS[name]
    if args.src_dir and len(args.datasets) != 1:
        raise ValueError("--src-dir can only be used with one dataset")
    stage2_dir = Path(args.src_dir) if args.src_dir else cfg.out_stage2
    if args.out_dir:
        if len(args.datasets) != 1:
            raise ValueError("--out-dir can only be used with one dataset")
        out_dir = Path(args.out_dir)
    else:
        out_dir = cfg.root / "preprocess" / args.out_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_stage2_meta(stage2_dir)
    eigval_single = compute_or_load_single_chromophore_eigval(cfg, stage2_dir, meta, args)
    h5_files = select_subjects(stage2_dir, args.subjects, args.max_subjects)

    with h5py.File(h5_files[0], "r") as f:
        original_t = int(f["fnirs_hbo_raw"].shape[-1])
    n_samples = resolve_n_samples(original_t, args.patch_size, args.time_policy, args.n_samples)

    results = []
    for src_path in h5_files:
        dst_path = out_dir / src_path.name
        results.append(
            build_one_subject(
                src_path=src_path,
                dst_path=dst_path,
                eigval_single=eigval_single,
                chromophores=args.chromophores,
                n_samples=n_samples,
                time_policy=args.time_policy,
                patch_size=args.patch_size,
                compression=args.compression,
                overwrite=args.overwrite,
            )
        )
        item = results[-1]
        print(
            f"[{item['status']}] {name}/{item['subject']}: "
            f"raw={tuple(item['raw_shape'])} modes={tuple(item['modes_shape'])}",
            flush=True,
        )

    summary = {
        "dataset": name,
        "stage2_fs32k": str(stage2_dir),
        "out_dir": str(out_dir),
        "template_meta": meta["_meta_path"],
        "chromophores": args.chromophores,
        "time_policy": args.time_policy,
        "patch_size": int(args.patch_size),
        "n_samples": int(n_samples),
        "single_chromophore_eigval_shape": list(eigval_single.shape),
        "n_subjects": len(results),
        "subjects": results,
    }
    with (out_dir / "prep_fnirs_poe_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=_as_jsonable)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["dubois", "cocktail"], choices=sorted(DATASETS))
    ap.add_argument("--src-dir", type=str, default=None,
                    help="Override source stage2_fs32k directory; only valid for one dataset.")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Override output directory; only valid for one dataset.")
    ap.add_argument("--out-suffix", type=str, default=DEFAULT_OUT_SUFFIX,
                    help="Output folder under each dataset's preprocess directory.")
    ap.add_argument("--subjects", nargs="*", default=None, help="Optional subject IDs, e.g. sub-01")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--chromophores", choices=["both", "hbo", "hbr"], default="both")
    ap.add_argument("--time-policy", choices=["crop", "pad", "exact"], default="crop",
                    help="The prior encoder uses PATCH_SIZE=64, so T must be divisible by --patch-size.")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Optional explicit output T; must be divisible by --patch-size.")
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--compression", choices=["lzf", "gzip", "none"], default="lzf")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--recompute-eigval", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [process_dataset(name, args) for name in args.datasets]
    print(json.dumps(summaries, indent=2, ensure_ascii=False, default=_as_jsonable), flush=True)


if __name__ == "__main__":
    main()
