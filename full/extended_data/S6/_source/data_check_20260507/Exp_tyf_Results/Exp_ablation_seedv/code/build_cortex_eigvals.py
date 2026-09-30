

from pathlib import Path
import argparse
import numpy as np
import pyvista as pv
from lapy import Solver, TriaMesh


def mesh_with_mask(mesh_path, mask_path):
    mesh = pv.read(mesh_path)
    xyz = np.asarray(mesh.points, dtype=np.float64)
    tri = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    mask = np.loadtxt(mask_path).astype(bool)
    keep = np.flatnonzero(mask)
    valid = np.all(mask[tri], axis=1)
    remap = np.full(len(mask), -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    return xyz[keep], remap[tri[valid]]


def hemisphere_eigvals(data, hemi, n=24):
    xyz, tri = mesh_with_mask(
        data / f"fsLR_32k_midthickness-{hemi}.vtk",
        data / f"fsLR_32k_cortex-{hemi}_mask.txt",
    )
    vals, _ = Solver(TriaMesh(xyz, tri)).eigs(k=n + 1, sigma=0.0)

    return np.asarray(vals[1:n + 1], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seedv-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    data = Path(args.seedv_root).resolve() / "data"
    lh = hemisphere_eigvals(data, "lh")
    rh = hemisphere_eigvals(data, "rh")
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.concatenate([lh, rh]).astype(np.float32))
    np.save(out.with_name("cortex_eigval_lh24.npy"), lh.astype(np.float32))
    np.save(out.with_name("cortex_eigval_rh24.npy"), rh.astype(np.float32))
    print(f"saved {out}: shape=(48,), range={min(lh.min(),rh.min()):.6g}..{max(lh.max(),rh.max()):.6g}")


if __name__ == "__main__":
    main()
