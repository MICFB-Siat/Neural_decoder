
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


REQUIRED = {
    "classification": {"eeg_modes": (3,), "label": (1,), "source_index": (1,)},
    "language": {"bci_obs": (3,), "bci_prior": (3,), "bge_target": (3,),
                 "text": (1,), "subject_index": (1,), "source_index": (1,)},
}

NSD_SHAPES = {
    "cifti": (1000, 4096),
    "modes": (1000, 2000),
    "rgb": (1000, 256, 256, 3),
    "test_index": (1000,),
    "source_row": (1000,),
    "stim_id": (1000,),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("assets/inference_data.h5"))
    parser.add_argument("--manifest", type=Path, default=Path("assets/MANIFEST.json"))
    parser.add_argument("--nsd-data", type=Path, default=Path("assets/nsd_sub01_test.h5"))
    parser.add_argument("--nsd-manifest", type=Path, default=Path("assets/NSD_SUB01_MANIFEST.json"))
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    with h5py.File(args.data, "r") as h5:
        if h5.attrs.get("schema_version", "") != "1.0":
            raise ValueError("Unsupported or missing schema_version")
        for group_name, fields in REQUIRED.items():
            if group_name not in h5:
                raise KeyError(f"Missing group: {group_name}")
            group = h5[group_name]
            n = None
            for field, ndim in fields.items():
                if field not in group:
                    raise KeyError(f"Missing dataset: /{group_name}/{field}")
                array = group[field]
                if array.ndim != ndim[0]:
                    raise ValueError(f"/{group_name}/{field}: expected {ndim[0]} dimensions, got {array.ndim}")
                n = array.shape[0] if n is None else n
                if array.shape[0] != n:
                    raise ValueError(f"/{group_name}: sample counts are not aligned")
                if np.issubdtype(array.dtype, np.number) and not np.isfinite(array[...]).all():
                    raise ValueError(f"/{group_name}/{field} contains non-finite values")
            print(f"/{group_name}: n={n}")

    if args.manifest.is_file():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "1.0":
            raise ValueError("Unsupported non-image manifest schema")
        print("manifest: metadata readable")

    if not args.nsd_data.is_file():
        raise FileNotFoundError(args.nsd_data)
    with h5py.File(args.nsd_data, "r") as h5:
        if h5.attrs.get("schema_version", "") != "nsd-sub01-test-v1":
            raise ValueError("Unsupported or missing NSD schema_version")
        if h5.attrs.get("split", "") != "Shared1000":
            raise ValueError("NSD test split must be Shared1000")
        for name, expected in NSD_SHAPES.items():
            if name not in h5 or h5[name].shape != expected:
                actual = None if name not in h5 else h5[name].shape
                raise ValueError(f"/{name}: expected {expected}, got {actual}")
        if not np.array_equal(h5["test_index"][...], np.arange(1000)):
            raise ValueError("NSD test_index must be exactly 0..999")
        for name in ("cifti", "modes"):
            if not np.isfinite(h5[name][...]).all():
                raise ValueError(f"/{name} contains non-finite values")
        print("NSD Subject-1: n=1000, CIFTI=4096, modes=2000, RGB=256x256")

    if not args.nsd_manifest.is_file():
        raise FileNotFoundError(args.nsd_manifest)
    nsd_manifest = json.loads(args.nsd_manifest.read_text(encoding="utf-8"))
    files = nsd_manifest.get("files", {})
    for rel, record in files.items():
        path = args.nsd_manifest.parent.parent / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"Size mismatch: {rel}")
    print(f"NSD manifest: verified {len(files)} file sizes")
    print("OK: all frozen inference assets are internally consistent")


if __name__ == "__main__":
    main()
