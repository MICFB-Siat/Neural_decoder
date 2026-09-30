

from pathlib import Path
import argparse
import json
import h5py
import numpy as np

HERE = Path(__file__).resolve()
EXP = HERE.parents[1]
BASE = HERE.parents[2]
SRC = BASE / "data" / "fourier_dct"
OUT = EXP / "data"


def main():
    global SRC, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode-type", default="fourier_dct", choices=["fourier_dct", "spherical_harmonic", "laplacian"])
    ap.add_argument("--experiment-dir", default=str(EXP))
    args = ap.parse_args()
    exp = Path(args.experiment_dir).resolve()
    SRC = BASE / "data" / args.mode_type
    OUT = exp / "data"
    OUT.mkdir(parents=True, exist_ok=True)
    basis = np.load(SRC / "basis_K30.npy").astype(np.float32)
    native = np.load(SRC / "native_eigenvalues_K30.npy").astype(np.float32)
    if basis.shape != (60, 30) or native.shape != (30,):
        raise ValueError(f"Unexpected Fourier shapes: basis={basis.shape}, lambda={native.shape}")
    orth_err = float(np.max(np.abs(basis.T @ basis - np.eye(30))))
    if orth_err > 1e-5 or np.any(np.diff(native) < 0):
        raise ValueError(f"Invalid Fourier basis: orth_err={orth_err}, ordered={np.all(np.diff(native)>=0)}")




    eigval = native + native[1]
    np.save(OUT / "native_eigval_K30.npy", eigval)
    np.save(OUT / "basis_K30.npy", basis)

    files = sorted(SRC.glob("sub-*.h5"))
    if len(files) != 16:
        raise RuntimeError(f"Expected 16 subjects, found {len(files)}")
    total = 0
    for src in files:
        dst = OUT / src.name
        with h5py.File(src, "r") as f:
            required = {"eeg_modes", "eeg_raw", "eeg_pos", "eeg_sensor_type", "labels"}
            if required - set(f):
                raise KeyError(f"{src}: missing {required-set(f)}")
            if f["eeg_modes"].shape[1:] != (30, 2560):
                raise ValueError(f"{src}: bad eeg_modes shape {f['eeg_modes'].shape}")
            total += f["eeg_modes"].shape[0]
        with h5py.File(dst, "w") as f:
            for key in sorted(required):
                f[key] = h5py.ExternalLink(str(src.resolve()), f"/{key}")
            f.create_dataset("eeg_mode_eigval", data=eigval)
            f.attrs["mode_type"] = args.mode_type
            f.attrs["position_encoding"] = "native basis eigenvalues + first non-zero spectral gap"
            f.attrs["source_h5"] = str(src.resolve())
    manifest = {
        "mode_type": args.mode_type, "subjects": len(files), "trials": total,
        "n_channels": 60, "n_modes": 30, "n_samples": 2560,
        "native_lambda_source": str(SRC / "native_eigenvalues_K30.npy"),
        "position_eigval": "native_lambda + native_lambda[1]",
        "position_reason": "avoid log(0) while preserving spectral ordering and gaps",
        "basis_orthogonality_max_error": orth_err,
        "eigval_min": float(eigval.min()), "eigval_max": float(eigval.max()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
