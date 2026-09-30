
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


WORK_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = WORK_ROOT / "configs" / "datasets.json"





OUTPUT_ROOT = Path(os.environ.get("ZIZHUAN_OUTPUT_ROOT", "/var/tmp/eigen_brain_decoding_zizhuan")).resolve()
LOCAL_INDIV_EIGEN_CACHE = Path(
    os.environ.get(
        "ZIZHUAN_LOCAL_INDIV_EIGEN_CACHE",
        "/var/tmp/eigen_brain_decoding_zizhuan/stage2_fmri_indiv_cache",
    )
).resolve()


@dataclass(frozen=True)
class Unit:
    dataset: str
    subject: str
    path: str
    n_trials: int
    timepoints: int


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def dataset_names(config: dict[str, Any]) -> list[str]:
    return list(config["datasets"].keys())


def stage2_path(config: dict[str, Any], dataset: str) -> Path:
    path = Path(config["datasets"][dataset]["stage2_dir"])
    if not path.is_dir():
        raise FileNotFoundError(f"未找到 {dataset} Stage2: {path}")
    return path


def discover_units(config: dict[str, Any], datasets: list[str] | None = None) -> list[Unit]:
    datasets = datasets or dataset_names(config)
    result: list[Unit] = []
    for dataset in datasets:
        for path in sorted(stage2_path(config, dataset).glob("sub-*.h5")):
            with h5py.File(path, "r") as handle:
                if not all(key in handle for key in ("fmri_cift", "fmri_modes", "fmri_modes_indiv")):
                    raise KeyError(f"{path}: 缺少 Stage2 fMRI 字段")
                n, _, t = handle["fmri_cift"].shape
            result.append(Unit(dataset, path.stem, str(path), int(n), int(t)))
    if not result:
        raise RuntimeError("没有找到任何 Stage2 H5 文件")
    return result


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def load_indiv_eigenvalues(config: dict[str, Any], dataset: str, subject: str, n_modes: int = 2000) -> np.ndarray:







    candidates = (
        stage2_path(config, dataset) / "cache" / "fmri_indiv" / subject,
        LOCAL_INDIV_EIGEN_CACHE / dataset / subject,
    )
    half = n_modes // 2
    for cache in candidates:
        left = cache / "L_evals_1000.npy"
        right = cache / "R_evals_1000.npy"
        if left.is_file() and right.is_file():
            break
    else:
        attempted = "; ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"{dataset}/{subject} 缺少个体本征值缓存: {attempted}")
    values = np.concatenate((np.load(left)[:half], np.load(right)[:half])).astype(np.float32)
    if values.shape != (n_modes,) or not np.isfinite(values).all():
        raise ValueError(f"{dataset}/{subject} 个体本征值异常: {values.shape}")
    return values


def zscore_trials_time(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True) + 1e-8
    return ((x - mean) / std).astype(np.float32)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_manifest(
    config: dict[str, Any], datasets: list[str] | None = None, units: list[Unit] | None = None,
) -> dict[str, Any]:
    units = list(units) if units is not None else discover_units(config, datasets)



    unit_records = []
    for unit in units:
        stat = Path(unit.path).stat()
        unit_records.append(asdict(unit) | {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "units": unit_records,
        "warm_starts": {
            name: {"path": path, "sha256": sha256(path)}
            for name, path in config["warm_starts"].items()
        },
    }


def write_json(path: str | Path, obj: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, target)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_to(path: str | Path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = target.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        handle.write(line + "\n")
        handle.flush()

    return handle, log


def experiment_paths(kind: str) -> dict[str, Path]:
    root = WORK_ROOT
    paths = {
        "root": root,
        "code": root / "code",
        "features": OUTPUT_ROOT / "features",
        "checkpoints": OUTPUT_ROOT / "checkpoints",
        "logs": root / "logs",
        "results": root / "results",
        "splits": root / "splits",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    paths["kind"] = paths["checkpoints"] / kind
    paths["kind"].mkdir(parents=True, exist_ok=True)
    return paths
