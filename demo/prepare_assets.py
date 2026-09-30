






from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np


DEST = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path(os.environ.get("EIGEN_SOURCE_ROOT", "."))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_weight(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def as_text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--force", action="store_true", help="replace assets/inference_data.h5 and copied weights")
    args = parser.parse_args()
    root = args.source_root.resolve()
    assets, weights = DEST / "assets", DEST / "weights"
    assets.mkdir(exist_ok=True); weights.mkdir(exist_ok=True)
    h5_out = assets / "inference_data.h5"
    tmp_h5 = Path("/tmp") / f"eigen_brain_decoding_assets_{os.getpid()}.h5"
    if h5_out.exists() and not args.force:
        raise FileExistsError(f"{h5_out} already exists; use --force to rebuild it")

    class_ckpt = Path(os.environ.get("EIGEN_CLASS_CKPT", root / "checkpoints/classification.pth"))
    language_dir = Path(os.environ.get("EIGEN_LANGUAGE_DIR", root / "language"))
    for source in [class_ckpt, language_dir / "qformer_poe_sublayer.pt", language_dir / "projector.pt",
                   language_dir / "lora/adapter_model.safetensors", language_dir / "lora/adapter_config.json"]:
        if not source.is_file():
            raise FileNotFoundError(source)
    copy_weight(class_ckpt, weights / "classification_stage2_eeg_tiny.pth")
    copy_weight(language_dir / "qformer_poe_sublayer.pt", weights / "alice_qformer_poe_sublayer.pt")
    copy_weight(language_dir / "projector.pt", weights / "alice_projector.pt")
    copy_weight(language_dir / "lora/adapter_model.safetensors", weights / "alice_lora/adapter_model.safetensors")
    copy_weight(language_dir / "lora/adapter_config.json", weights / "alice_lora/adapter_config.json")

    seed_h5 = Path(os.environ.get("EIGEN_SEED_H5", root / "data/seed_stage2.h5"))
    alice_h5 = Path(os.environ.get("EIGEN_LANGUAGE_H5", root / "data/language.h5"))
    alice_bge_h5 = Path(os.environ.get("EIGEN_LANGUAGE_BGE_H5", root / "data/language_bge.h5"))
    for source in [seed_h5, alice_h5, alice_bge_h5]:
        if not source.is_file():
            raise FileNotFoundError(source)

    with h5py.File(seed_h5, "r") as h5:
        all_labels = h5["labels"][...].astype(np.int64)
        class_indices = np.sort(np.concatenate([np.flatnonzero(all_labels == cls)[:2] for cls in np.unique(all_labels)]))
        eeg = h5["eeg_modes"][class_indices].astype(np.float32)
        eeg_labels = all_labels[class_indices]

    language_indices = np.arange(58, 63, dtype=np.int64)
    with h5py.File(alice_h5, "r") as h5, h5py.File(alice_bge_h5, "r") as labels_h5:
        obs = h5["bci_obs"][language_indices].astype(np.float16)
        prior = h5["bci_prior"][language_indices].astype(np.float16)
        text = h5["labels"][language_indices]
        bge = labels_h5["bge_emb"][language_indices].astype(np.float16)
        bge_text = labels_h5["labels"][language_indices]
        if [as_text(x) for x in text] != [as_text(x) for x in bge_text]:
            raise ValueError("Alice BCI labels and BGE labels do not align")

    text_dtype = h5py.string_dtype(encoding="utf-8")
    if tmp_h5.exists():
        tmp_h5.unlink()
    with h5py.File(tmp_h5, "w") as h5:
        h5.attrs["schema_version"] = "1.0"
        h5.attrs["bundle_type"] = "fixed frozen-checkpoint inference package"
        pass
        group = h5.create_group("classification")
        group.create_dataset("eeg_modes", data=eeg, compression="gzip", compression_opts=4)
        group.create_dataset("label", data=eeg_labels)
        group.create_dataset("source_index", data=class_indices)
        group.attrs["source_subject"] = "SEED-V sub-01"
        group.attrs["class_labels"] = "archived numeric class IDs; names were not verified in this bundle"
        group = h5.create_group("language")
        group.create_dataset("bci_obs", data=obs, compression="gzip", compression_opts=4)
        group.create_dataset("bci_prior", data=prior, compression="gzip", compression_opts=4)
        group.create_dataset("bge_target", data=bge, compression="gzip", compression_opts=4)
        group.create_dataset("text", data=np.asarray([as_text(x) for x in text], dtype=text_dtype), dtype=text_dtype)
        group.create_dataset("subject_index", data=np.zeros(len(language_indices), dtype=np.int64))
        group.create_dataset("source_index", data=language_indices)
        group.attrs["source_subject"] = "Alice sub-18 (checkpoint subject index 0)"
        group.attrs["target_space"] = "20 x 1024 BGE-M3 token embeddings"
    shutil.move(str(tmp_h5), h5_out)

    sources = {str(path): sha256(path) for path in [class_ckpt, language_dir / "qformer_poe_sublayer.pt",
               language_dir / "projector.pt", language_dir / "lora/adapter_model.safetensors"]}
    bundle_files = [h5_out, weights / "classification_stage2_eeg_tiny.pth", weights / "alice_qformer_poe_sublayer.pt",
                    weights / "alice_projector.pt", weights / "alice_lora/adapter_model.safetensors",
                    weights / "alice_lora/adapter_config.json"]
    manifest = {
        "schema_version": "1.0",
        "created_by": "prepare_assets.py",
        "source_sha256": sources,
        "bundle_sha256": {str(path.relative_to(DEST)): sha256(path) for path in bundle_files},
        "selection": {"classification_source_indices": class_indices.tolist(),
                      "language_source_indices": language_indices.tolist()},
        "external_base_models_not_copied": {
            "Phi-4-mini-instruct": "microsoft/Phi-4-mini-instruct",
            "BGE-M3": "BAAI/bge-m3"},
        "limits": ["Frozen-checkpoint inference on the packaged samples.",
                   "NSD Subject-1 reconstruction assets have a separate manifest and are not regenerated here.",
                   "Language alignment checkpoint originated from an archived training protocol; do not use as held-out generalisation evidence."]}
    (assets / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created {h5_out}")
    print(f"Created frozen weights in {weights}")


if __name__ == "__main__":
    main()
