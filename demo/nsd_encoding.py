

















from __future__ import annotations

import argparse
import gc
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import sparse
import torch


ROOT = Path(__file__).resolve().parent
ASSET_ROOT = ROOT / "assets/nsd_sub01_encoding"
WEIGHT_ROOT = ROOT / "weights/nsd_sub01_encoding"

DEFAULT_DATA = ASSET_ROOT / "sub01_shared1000_encoding.h5"
DEFAULT_EIGENBASIS = ASSET_ROOT / "group_eigenbasis_2000.npz"
DEFAULT_MAPPING = ASSET_ROOT / "cifti_mapping_59412.npz"
DEFAULT_FILESTORE = ASSET_ROOT / "pycortex_db"
DEFAULT_WEIGHTS = WEIGHT_ROOT / "model.npz"
DEFAULT_OUTPUT = ROOT / "outputs/nsd_sub01_encoding/sub01_encoding_flatmap.png"

N_CIFTI = 59412
N_SURFACE_HEMI = 32492
N_MODES_PER_HEMI = 1000
DISPLAY_THRESHOLD = 0.10
DISPLAY_ALPHA = 0.50
DISPLAY_ITERATIONS = 2
DISPLAY_GAIN = 1.055
DISPLAY_VMIN = 0.0
DISPLAY_VMAX = 0.6
EXPECTED_MEAN_R = 0.12337327003479004

REQUIRED_MODEL_KEYS = {
    "semantic_mu",
    "semantic_sd",
    "modes_mu",
    "modes_sd",
    "cift_mu",
    "cift_sd",
    "geometry_weight",
    "caption_to_poe_x_mean",
    "caption_to_poe_y_mean",
    "caption_to_poe_coef",
    "poe_to_cifti_x_mean",
    "poe_to_cifti_y_mean",
    "poe_to_cifti_coef",
    "poe_to_modes_x_mean",
    "poe_to_modes_y_mean",
    "poe_to_modes_coef",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--eigenbasis", type=Path, default=DEFAULT_EIGENBASIS)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--filestore", type=Path, default=DEFAULT_FILESTORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--vertex-chunk", type=int, default=4096)
    return parser.parse_args()


def require_inputs(args: argparse.Namespace) -> None:
    required = [
        args.data,
        args.weights,
        args.eigenbasis,
        args.mapping,
        args.filestore / "fsLR32k/surfaces/fiducial_lh.gii",
        args.filestore / "fsLR32k/surfaces/fiducial_rh.gii",
        args.filestore / "fsLR32k/surfaces/flat_lh.gii",
        args.filestore / "fsLR32k/surfaces/flat_rh.gii",
        args.filestore / "fsLR32k/surface-info/curvature.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing NSD encoding test assets:\n" + "\n".join(missing))
    if args.vertex_chunk < 1:
        raise ValueError("--vertex-chunk must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")


def load_test_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("schema_version", "") != "nsd-sub01-encoding-test-v1":
            raise ValueError("Unsupported NSD encoding test schema")
        if h5.attrs.get("subject", "") != "sub-01":
            raise ValueError("This frozen test bundle must be sub-01")
        if h5.attrs.get("split", "") != "Shared1000":
            raise ValueError("This frozen test bundle must use Shared1000")
        semantic = np.asarray(h5["semantic_bge_m3"][:], dtype=np.float32)
        true_fmri = np.asarray(h5["fmri_cift"][:], dtype=np.float32)
        stim_id = np.asarray(h5["stim_id"][:], dtype=np.int64)
        test_index = np.asarray(h5["test_index"][:], dtype=np.int64)
        shared = np.asarray(h5["is_shared1000"][:], dtype=bool)
    if semantic.shape != (1000, 1024):
        raise ValueError(f"Expected semantic shape (1000, 1024), got {semantic.shape}")
    if true_fmri.shape != (1000, N_CIFTI):
        raise ValueError(f"Expected fMRI shape (1000, {N_CIFTI}), got {true_fmri.shape}")
    if stim_id.shape != (1000,):
        raise ValueError(f"Expected 1000 stim_id values, got {stim_id.shape}")
    if not np.array_equal(test_index, np.arange(1000)):
        raise ValueError("test_index must be exactly 0..999")
    if not np.all(shared):
        raise ValueError("Every packaged row must be a Shared1000 test row")
    if not np.isfinite(semantic).all() or not np.isfinite(true_fmri).all():
        raise ValueError("Non-finite semantic or fMRI values in test data")
    return semantic, true_fmri, stim_id


@torch.inference_mode()
def ridge_predict(
    x: np.ndarray,
    x_mean: np.ndarray,
    coef: np.ndarray,
    y_mean: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x_tensor = torch.as_tensor(x - x_mean, dtype=torch.float32, device=device)
    coef_tensor = torch.as_tensor(coef, dtype=torch.float32, device=device)
    output = x_tensor @ coef_tensor
    output += torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    return output.cpu().numpy().astype(np.float32, copy=False)


def column_corr(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    true_centered = true - true.mean(axis=0, keepdims=True)
    numerator = np.sum(pred_centered * true_centered, axis=0)
    denominator = (
        np.linalg.norm(pred_centered, axis=0)
        * np.linalg.norm(true_centered, axis=0)
        + 1e-8
    )
    return (numerator / denominator).astype(np.float32)


@torch.inference_mode()
def infer_vertex_correlations(
    semantic: np.ndarray,
    true_raw: np.ndarray,
    weight_path: Path,
    eigenbasis_path: Path,
    device_name: str,
    vertex_chunk: int,
) -> np.ndarray:
    device = torch.device(device_name)
    with np.load(weight_path) as model, np.load(eigenbasis_path) as eigenbasis:
        missing_keys = sorted(REQUIRED_MODEL_KEYS - set(model.files))
        if missing_keys:
            raise KeyError(f"model.npz missing keys: {missing_keys}")
        geometry_weight = float(model["geometry_weight"])
        if not np.isclose(geometry_weight, 0.10, atol=1e-7):
            raise ValueError(f"Expected geometry_weight=0.10, got {geometry_weight}")

        x = ((semantic - model["semantic_mu"]) / model["semantic_sd"]).astype(np.float32)
        poe = ridge_predict(
            x,
            model["caption_to_poe_x_mean"],
            model["caption_to_poe_coef"],
            model["caption_to_poe_y_mean"],
            device,
        )
        poe /= np.linalg.norm(poe, axis=1, keepdims=True) + 1e-8
        poe_aug = np.concatenate(
            [poe, np.sign(poe) * np.sqrt(np.abs(poe) + 1e-8)], axis=1
        ).astype(np.float32)

        modes_std = ridge_predict(
            poe_aug,
            model["poe_to_modes_x_mean"],
            model["poe_to_modes_coef"],
            model["poe_to_modes_y_mean"],
            device,
        )
        modes_raw = modes_std * model["modes_sd"] + model["modes_mu"]

        direct_x = np.asarray(poe_aug - model["poe_to_cifti_x_mean"], dtype=np.float32)
        direct_coef = np.asarray(model["poe_to_cifti_coef"], dtype=np.float32)
        direct_y_mean = np.asarray(model["poe_to_cifti_y_mean"], dtype=np.float32)
        cift_mu = np.asarray(model["cift_mu"], dtype=np.float32)
        cift_sd = np.asarray(model["cift_sd"], dtype=np.float32)
        basis_left = np.asarray(eigenbasis["mL"], dtype=np.float32)
        basis_right = np.asarray(eigenbasis["mR"], dtype=np.float32)

        if direct_coef.shape != (1536, N_CIFTI):
            raise ValueError(f"Unexpected direct ridge shape: {direct_coef.shape}")
        n_left = int(basis_left.shape[0])
        if n_left + basis_right.shape[0] != N_CIFTI:
            raise ValueError("Eigenbasis does not cover all 59,412 CIFTI points")

        true = ((true_raw - cift_mu) / cift_sd).astype(np.float32)
        correlations = np.empty(N_CIFTI, dtype=np.float32)
        direct_x_tensor = torch.as_tensor(direct_x, dtype=torch.float32, device=device)
        left_modes_tensor = torch.as_tensor(
            modes_raw[:, :N_MODES_PER_HEMI], dtype=torch.float32, device=device
        )
        right_modes_tensor = torch.as_tensor(
            modes_raw[:, N_MODES_PER_HEMI:], dtype=torch.float32, device=device
        )

        for lo in range(0, N_CIFTI, vertex_chunk):
            hi = min(N_CIFTI, lo + vertex_chunk)
            direct = direct_x_tensor @ torch.as_tensor(
                direct_coef[:, lo:hi], dtype=torch.float32, device=device
            )
            direct += torch.as_tensor(
                direct_y_mean[lo:hi], dtype=torch.float32, device=device
            )

            if hi <= n_left:
                geometry_raw = left_modes_tensor @ torch.as_tensor(
                    basis_left[lo:hi].T, dtype=torch.float32, device=device
                )
            elif lo >= n_left:
                geometry_raw = right_modes_tensor @ torch.as_tensor(
                    basis_right[lo - n_left : hi - n_left].T,
                    dtype=torch.float32,
                    device=device,
                )
            else:
                left_geometry = left_modes_tensor @ torch.as_tensor(
                    basis_left[lo:n_left].T, dtype=torch.float32, device=device
                )
                right_geometry = right_modes_tensor @ torch.as_tensor(
                    basis_right[: hi - n_left].T, dtype=torch.float32, device=device
                )
                geometry_raw = torch.cat([left_geometry, right_geometry], dim=1)

            geometry = (
                geometry_raw
                - torch.as_tensor(cift_mu[lo:hi], dtype=torch.float32, device=device)
            ) / torch.as_tensor(cift_sd[lo:hi], dtype=torch.float32, device=device)
            pred = (
                (1.0 - geometry_weight) * direct + geometry_weight * geometry
            ).cpu().numpy()
            correlations[lo:hi] = column_corr(pred, true[:, lo:hi])
            print(f"[encoding] vertices {lo:05d}:{hi:05d}/{N_CIFTI}", flush=True)

    del true_raw, true, direct_coef, basis_left, basis_right
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if abs(float(correlations.mean()) - EXPECTED_MEAN_R) > 1e-5:
        raise RuntimeError(
            f"Unexpected sub-01 mean r: {correlations.mean():.9f}; "
            f"expected approximately {EXPECTED_MEAN_R:.9f}"
        )
    return correlations


def to_full_surface(cifti_values: np.ndarray, mapping_path: Path) -> np.ndarray:
    with np.load(mapping_path) as mapping:
        left_vertices = np.asarray(mapping["lh_cifti_vertices"], dtype=np.int64)
        right_vertices = np.asarray(mapping["rh_cifti_vertices"], dtype=np.int64)
    if cifti_values.shape != (left_vertices.size + right_vertices.size,):
        raise ValueError("CIFTI vector and surface mapping do not match")
    left = np.full(N_SURFACE_HEMI, np.nan, dtype=np.float32)
    right = np.full(N_SURFACE_HEMI, np.nan, dtype=np.float32)
    left[left_vertices] = cifti_values[: left_vertices.size]
    right[right_vertices] = cifti_values[left_vertices.size :]
    return np.concatenate([left, right])


def adjacency(surface_path: Path) -> sparse.csr_matrix:
    surface = nib.load(str(surface_path))
    vertices = np.asarray(surface.darrays[0].data)
    faces = np.asarray(surface.darrays[1].data, dtype=np.int64)
    rows = np.concatenate(
        [faces[:, 0], faces[:, 1], faces[:, 1], faces[:, 2], faces[:, 2], faces[:, 0]]
    )
    columns = np.concatenate(
        [faces[:, 1], faces[:, 0], faces[:, 2], faces[:, 1], faces[:, 0], faces[:, 2]]
    )
    matrix = sparse.coo_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, columns)),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()
    return matrix


def smooth_hemi(values: np.ndarray, matrix: sparse.csr_matrix) -> np.ndarray:
    valid = np.isfinite(values)
    smoothed = np.nan_to_num(values, nan=0.0).astype(np.float64)
    weights = valid.astype(np.float64)
    for _ in range(DISPLAY_ITERATIONS):
        denominator = matrix @ weights
        neighbor_mean = np.divide(
            matrix @ smoothed,
            denominator,
            out=smoothed.copy(),
            where=denominator > 0,
        )
        smoothed = (1.0 - DISPLAY_ALPHA) * smoothed + DISPLAY_ALPHA * neighbor_mean
    smoothed *= DISPLAY_GAIN
    smoothed[~valid] = np.nan
    return smoothed.astype(np.float32)


def display_map(correlations: np.ndarray, mapping: Path, filestore: Path) -> np.ndarray:
    full = to_full_surface(correlations, mapping)
    surfaces = filestore / "fsLR32k/surfaces"
    left = smooth_hemi(full[:N_SURFACE_HEMI], adjacency(surfaces / "fiducial_lh.gii"))
    right = smooth_hemi(full[N_SURFACE_HEMI:], adjacency(surfaces / "fiducial_rh.gii"))
    result = np.concatenate([left, right]).astype(np.float32)
    result[~np.isfinite(result) | (result < DISPLAY_THRESHOLD)] = np.nan
    return result


def render_flatmap(values: np.ndarray, filestore: Path, output: Path) -> None:
    import cortex

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    cortex.options.config.set("basic", "filestore", str(filestore.resolve()))
    cortex.db.filestore = str(filestore.resolve())
    cmap_name = "nsd_encoding_positive"
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        cmap_name, ["#000000", "#B00000", "#FF0000", "#FF8C00", "#FFFF00"]
    )
    if cmap_name not in mpl.colormaps:
        mpl.colormaps.register(cmap, name=cmap_name)
    vertex = cortex.Vertex(
        values,
        subject="fsLR32k",
        cmap=cmap_name,
        vmin=DISPLAY_VMIN,
        vmax=DISPLAY_VMAX,
    )
    figure = cortex.quickflat.make_figure(
        vertex,
        with_curvature=True,
        with_sulci=False,
        with_labels=False,
        with_rois=False,
        with_borders=False,
        with_colorbar=False,
        height=1800,
        dpi=300,
        recache=False,
        curvature_brightness=0.55,
        curvature_contrast=0.35,
        curvature_threshold=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        transparent=False,
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    require_inputs(args)
    semantic, true_fmri, stim_id = load_test_data(args.data)
    correlations = infer_vertex_correlations(
        semantic,
        true_fmri,
        args.weights,
        args.eigenbasis,
        args.device,
        args.vertex_chunk,
    )
    values = display_map(correlations, args.mapping, args.filestore)
    render_flatmap(values, args.filestore, args.output)
    print(f"subject=sub-01 split=Shared1000 n_images={len(stim_id)}")
    print(f"n_cifti={len(correlations)} mean_r={float(correlations.mean()):.9f}")
    print(
        "display=r>=0.10, alpha=0.50, iterations=2, gain=1.055, "
        "vmin=0.0, vmax=0.6"
    )
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
