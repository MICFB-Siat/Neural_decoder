







from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from experiment_config import CONFIRMED_KS, DATASETS, DATA_ROOT, FS32K


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_lambda(n_cortex: int, output: Path) -> None:
    lh = np.load(FS32K / "fsLR_32k_lh_eval_1024.npy")
    rh = np.load(FS32K / "fsLR_32k_rh_eval_1024.npy")
    n_lh, n_rh = (n_cortex + 1) // 2, n_cortex // 2
    lam = np.concatenate([lh[:n_lh], rh[:n_rh]]).astype(np.float32)
    if lam.shape != (n_cortex,):
        raise RuntimeError(f"lambda shape {lam.shape} != ({n_cortex},)")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, lam)


def valid_existing(path: Path, k: int, n_trials: int, n_samples: int,
                   label_key: str) -> bool:
    try:
        with h5py.File(path, "r") as f:
            return (
                f["eeg_modes"].shape == (n_trials, k, n_samples)
                and f[label_key].shape[0] == n_trials
                and int(f.attrs["eeg_modes_K"]) == k
                and f.attrs["eeg_modes_method"] == "fixed_k_nested_svd"
            )
    except (OSError, KeyError, ValueError):
        return False


def build_subject(src: Path, dst: Path, psi: np.ndarray, k: int,
                  template: Path, label_key: str, block_trials: int) -> dict:
    with h5py.File(src, "r") as fs:
        raw = fs["eeg_raw"]
        n_trials, n_sensors, n_samples = raw.shape
        if n_sensors != psi.shape[0]:
            raise ValueError(
                f"{src}: raw sensors={n_sensors}, basis sensors={psi.shape[0]}"
            )
        if valid_existing(dst, k, n_trials, n_samples, label_key):
            return {"subject": src.stem, "shape": [n_trials, k, n_samples], "status": "reused"}

        dst.parent.mkdir(parents=True, exist_ok=True)



        fd_tmp, tmp_name = tempfile.mkstemp(
            prefix=f".{src.stem}_K{k}_", suffix=".partial", dir=dst.parent
        )
        os.close(fd_tmp)
        local_tmp = Path(tmp_name)
        try:
            with h5py.File(local_tmp, "w") as fd:
                modes = fd.create_dataset(
                    "eeg_modes",
                    shape=(n_trials, k, n_samples),
                    dtype=np.float32,
                    compression="lzf",
                )
                for start in range(0, n_trials, block_trials):
                    stop = min(start + block_trials, n_trials)
                    x = raw[start:stop].astype(np.float32, copy=False)
                    modes[start:stop] = np.matmul(psi.T[None, :, :], x)

                for name in ("eeg_raw", "eeg_pos", "eeg_sensor_type", "labels", "label2"):
                    if name in fs:
                        fd[name] = h5py.ExternalLink(str(src.resolve()), f"/{name}")
                for key, value in fs.attrs.items():
                    fd.attrs[f"source_{key}"] = value
                fd.attrs["eeg_modes_method"] = "fixed_k_nested_svd"
                fd.attrs["eeg_modes_K"] = k
                fd.attrs["source_h5"] = str(src.resolve())
                fd.attrs["template"] = str(template.resolve())

            with h5py.File(local_tmp, "r") as check:
                if check["eeg_modes"].shape != (n_trials, k, n_samples):
                    raise RuntimeError(f"local staging validation failed: {local_tmp}")
                if check[label_key].shape[0] != n_trials:
                    raise RuntimeError(f"external label validation failed: {local_tmp}")

                check["eeg_modes"][0:1, :, 0:1]
                check[label_key][0:1]
            os.replace(local_tmp, dst)
        finally:
            local_tmp.unlink(missing_ok=True)
    return {"subject": src.stem, "shape": [n_trials, k, n_samples], "status": "built"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    ap.add_argument("--ks", type=int, nargs="+", default=None)
    ap.add_argument("--block-trials", type=int, default=8)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    source, template = Path(cfg["source"]), Path(cfg["template"])
    if not source.is_dir() or not template.is_file():
        raise FileNotFoundError(f"source={source} template={template}")
    subjects = sorted(source.glob("sub-*.h5"))
    if not subjects:
        raise FileNotFoundError(f"no sub-*.h5 in {source}")

    D = np.load(template).astype(np.float64)
    U, s, vt = np.linalg.svd(D, full_matrices=False)
    rank_1e12 = int((s / s[0] >= 1e-12).sum())
    ks = tuple(dict.fromkeys(args.ks or CONFIRMED_KS[args.dataset]))
    invalid = [k for k in ks if k < 1 or k > rank_1e12]
    if invalid:
        raise ValueError(
            f"{args.dataset}: requested K={invalid} exceeds nonzero template rank "
            f"{rank_1e12}; D shape={D.shape}. Zero padding is deliberately forbidden."
        )

    data_root = DATA_ROOT / args.dataset
    data_root.mkdir(parents=True, exist_ok=True)
    lam_path = data_root / f"fsLR_32k_group_eval_{D.shape[1]}.npy"
    build_lambda(D.shape[1], lam_path)

    manifest = {
        "dataset": args.dataset,
        "source": str(source.resolve()),
        "template": str(template.resolve()),
        "template_sha256": sha256(template),
        "template_shape": list(D.shape),
        "template_rank_1e12": rank_1e12,
        "requested_ks": list(ks),
        "subjects": [],
        "lambda_path": str(lam_path.resolve()),
        "lambda_sha256": sha256(lam_path),
    }
    for k in ks:
        kdir = data_root / f"K{k}"
        kdir.mkdir(parents=True, exist_ok=True)
        psi = U[:, :k].astype(np.float32)
        d_k = (U[:, :k] * s[:k]) @ vt[:k]
        np.save(kdir / f"psi_K{k}.npy", psi)
        np.save(kdir / f"D_rank{k}.npy", d_k.astype(np.float64))
        for src in subjects:
            entry = build_subject(
                src, kdir / src.name, psi, k, template, cfg["label_key"],
                args.block_trials
            )
            entry["K"] = k
            manifest["subjects"].append(entry)
            print(f"{args.dataset} K={k} {src.stem}: {entry['status']} {entry['shape']}", flush=True)

    (data_root / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"completed {args.dataset}: K={list(ks)}, subjects={len(subjects)}", flush=True)


if __name__ == "__main__":
    main()
