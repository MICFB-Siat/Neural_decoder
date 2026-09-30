








from pathlib import Path
import argparse
import h5py
import numpy as np

KS = (1, 3, 5, 10, 20, 40, 48)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    D = np.load(args.template).astype(np.float64)
    U, s, vt = np.linalg.svd(D, full_matrices=False)
    if len(s) < max(KS):
        raise ValueError(f"Template only provides {len(s)} modes; need {max(KS)}")

    for k in KS:
        kdir = output / f"K{k}"
        kdir.mkdir(exist_ok=True)
        psi = U[:, :k].astype(np.float32)
        np.save(kdir / f"psi_K{k}.npy", psi)


        d_k = (U[:, :k] * s[:k]) @ vt[:k]
        np.save(kdir / f"D_rank{k}.npy", d_k.astype(np.float64))

        for src in sorted(source.glob("sub-*.h5")):
            dst = kdir / src.name
            if dst.exists():
                with h5py.File(dst, "r") as f:
                    if f["eeg_modes"].shape[1] == k:
                        continue
                dst.unlink()
            with h5py.File(src, "r") as fs:
                raw = fs["eeg_raw"][:].astype(np.float32)
                modes = np.matmul(psi.T[None, :, :], raw)
            with h5py.File(dst, "w") as fd:
                fd.create_dataset("eeg_modes", data=modes, compression="lzf")
                for name in ("eeg_raw", "eeg_pos", "eeg_sensor_type", "labels"):
                    fd[name] = h5py.ExternalLink(str(src), f"/{name}")
                fd.attrs["eeg_modes_method"] = "fixed_k_nested_svd"
                fd.attrs["eeg_modes_K"] = k
                fd.attrs["source_h5"] = str(src)
                fd.attrs["template"] = str(Path(args.template).resolve())
        print(f"K={k}: built {len(list(kdir.glob('sub-*.h5')))} subjects")


if __name__ == "__main__":
    main()
