from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np

try:
    import pyvista as pv
    from PIL import Image
    from matplotlib.colors import LinearSegmentedColormap
    from lapy import Solver, TriaMesh
except ImportError:
    pv = None
    Image = None
    LinearSegmentedColormap = None
    Solver = None
    TriaMesh = None


CODE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_DIR.parent
DATA_DIR = PACKAGE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw_inputs"
GENERATED_DIR = DATA_DIR / "generated"
REFERENCE_DIR = DATA_DIR / "reference_old"
FIGURES_DIR = DATA_DIR / "figures"

SUB22_RAW_DIR = RAW_DIR / "sub-22"
SUB24_RAW_DIR = RAW_DIR / "sub-24"
TEMPLATE_RAW_DIR = RAW_DIR / "template"

SUB22_GEN_DIR = GENERATED_DIR / "sub-22"
SUB24_GEN_DIR = GENERATED_DIR / "sub-24"
TEMPLATE_GEN_DIR = GENERATED_DIR / "template"

SUB22_SURF = SUB22_RAW_DIR / "22.L.midthickness.32k_fs_LR.surf.gii"
SUB22_MASK = SUB22_RAW_DIR / "22.L.atlasroi.32k_fs_LR.shape.gii"
SUB24_SURF = SUB24_RAW_DIR / "24.L.midthickness.32k_fs_LR.surf.gii"
SUB24_MASK = SUB24_RAW_DIR / "24.L.atlasroi.32k_fs_LR.shape.gii"
TEMPLATE_RENDER_SURF = TEMPLATE_RAW_DIR / "S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
TEMPLATE_VTK = TEMPLATE_RAW_DIR / "fsLR_32k_midthickness-lh.vtk"
TEMPLATE_MASK = TEMPLATE_RAW_DIR / "fsLR_32k_cortex-lh_mask.txt"

SUB22_MASKED_MODES = SUB22_GEN_DIR / "sub-22_L_eigenmodes_1000_with_mode0_masked.npy"
SUB22_EVALS = SUB22_GEN_DIR / "sub-22_L_eigenmodes_1000_with_mode0_evals.npy"
SUB22_META = SUB22_GEN_DIR / "sub-22_L_eigenmodes_1000_with_mode0_meta.npz"
SUB22_FULL = SUB22_GEN_DIR / "sub-22_L_eigenmodes_1000_with_mode0_full32492.npy"

SUB24_MASKED_MODES = SUB24_GEN_DIR / "sub-24_L_eigenmodes_1000_with_mode0_masked.npy"
SUB24_EVALS = SUB24_GEN_DIR / "sub-24_L_eigenmodes_1000_with_mode0_evals.npy"
SUB24_META = SUB24_GEN_DIR / "sub-24_L_eigenmodes_1000_with_mode0_meta.npz"
SUB24_FULL = SUB24_GEN_DIR / "sub-24_L_eigenmodes_1000_with_mode0_full32492.npy"

TEMPLATE_MASKED_MODES = TEMPLATE_GEN_DIR / "template_L_eigenmodes_1000_with_mode0_masked.npy"
TEMPLATE_EVALS = TEMPLATE_GEN_DIR / "template_L_eigenmodes_1000_with_mode0_evals.npy"
TEMPLATE_META = TEMPLATE_GEN_DIR / "template_L_eigenmodes_1000_with_mode0_meta.npz"
TEMPLATE_FULL = TEMPLATE_GEN_DIR / "template_L_eigenmodes_1000_with_mode0_full32492.npy"

FINAL_GRID = FIGURES_DIR / "模板_被试1_被试2_mode0_4_3x5_mode0全蓝.png"


def ensure_dependencies() -> None:
    if Solver is None or TriaMesh is None:
        raise SystemExit("缺少依赖，请安装: pip install lapy nibabel numpy pyvista pillow matplotlib")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_gifti_surface(surf_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    surf = nib.load(str(surf_file))
    coords = np.asarray(surf.darrays[0].data, dtype=np.float64)
    faces = np.asarray(surf.darrays[1].data, dtype=np.int32)
    return coords, faces


def load_gifti_mask(mask_file: Path) -> np.ndarray:
    mask = np.asarray(nib.load(str(mask_file)).darrays[0].data).ravel() > 0
    if not np.any(mask):
        raise ValueError(f"mask 全为 0: {mask_file}")
    return mask


def load_vtk_surface(vtk_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    if pv is None:
        raise SystemExit("缺少 pyvista，请安装: pip install pyvista")
    mesh = pv.read(str(vtk_file))
    coords = np.asarray(mesh.points, dtype=np.float64)
    faces = np.asarray(mesh.faces.reshape(-1, 4)[:, 1:], dtype=np.int32)
    return coords, faces


def load_text_mask(mask_file: Path) -> np.ndarray:
    mask = np.asarray(np.loadtxt(str(mask_file)) > 0, dtype=bool)
    if not np.any(mask):
        raise ValueError(f"mask 全为 0: {mask_file}")
    return mask


def mask_surface(
    coords: np.ndarray,
    faces: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mask.shape[0] != coords.shape[0]:
        raise ValueError("mask 顶点数与表面不一致")
    keep_idx = np.where(mask)[0]
    keep_faces = np.all(mask[faces], axis=1)
    faces_kept = faces[keep_faces]
    remap = np.full(coords.shape[0], -1, dtype=np.int64)
    remap[keep_idx] = np.arange(keep_idx.size, dtype=np.int64)
    coords_kept = coords[keep_idx]
    faces_remap = remap[faces_kept]
    return coords_kept, faces_remap, keep_idx


def compute_lb_modes_with_mode0(
    coords: np.ndarray,
    faces: np.ndarray,
    n_modes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ensure_dependencies()
    mesh = TriaMesh(coords, faces)
    solver = Solver(mesh)
    evals_all, evecs_all = solver.eigs(k=n_modes)
    evals = np.asarray(evals_all[:n_modes], dtype=np.float64)
    evecs = np.asarray(evecs_all[:, :n_modes], dtype=np.float32)
    return evals, evecs


def expand_to_full(masked_modes: np.ndarray, keep_idx: np.ndarray, n_vertices_total: int) -> np.ndarray:
    full = np.zeros((n_vertices_total, masked_modes.shape[1]), dtype=np.float32)
    full[keep_idx, :] = masked_modes
    return full


def load_full_matrix(file_path: Path) -> np.ndarray:
    matrix = np.load(str(file_path))
    return matrix.astype(np.float32, copy=False)


def export_shape_gii(mode_values: np.ndarray, out_file: Path) -> None:
    darray = nib.gifti.GiftiDataArray(mode_values.astype(np.float32), intent="NIFTI_INTENT_SHAPE")
    gii = nib.GiftiImage(darrays=[darray])
    nib.save(gii, str(out_file))


def build_colormap():
    if LinearSegmentedColormap is None:
        raise SystemExit("缺少 matplotlib，请安装: pip install matplotlib")
    return LinearSegmentedColormap.from_list(
        "mode_map",
        ["#08306b", "#2171b5", "#41b6c4", "#fff7bc", "#fdae61", "#f03b20", "#bd0026"],
        N=256,
    )


def load_surface_as_polydata(surf_file: Path):
    if pv is None:
        raise SystemExit("缺少 pyvista，请安装: pip install pyvista")
    vertices, faces = load_gifti_surface(surf_file)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32), faces]).ravel()
    poly = pv.PolyData(vertices.astype(np.float32), faces_pv)
    poly.compute_normals(
        cell_normals=False,
        point_normals=True,
        inplace=True,
        auto_orient_normals=True,
        split_vertices=False,
    )
    return poly


def mode_scale(mode_values: np.ndarray, quantile: float = 0.995) -> float:
    scale = float(np.quantile(np.abs(mode_values), quantile))
    return scale if scale > 0 else 1.0


def align_modes_to_reference(matrix: np.ndarray, reference: np.ndarray, roi_indices: np.ndarray) -> np.ndarray:
    aligned = matrix.copy()
    roi = np.asarray(roi_indices, dtype=np.int64)
    for i in range(min(aligned.shape[1], reference.shape[1])):
        a = reference[roi, i]
        b = aligned[roi, i]
        if i == 0:
            if np.sign(b.mean()) != np.sign(a.mean()):
                aligned[:, i] *= -1.0
            continue
        a0 = a - a.mean()
        b0 = b - b.mean()
        denom = np.linalg.norm(a0) * np.linalg.norm(b0)
        if denom < 1e-12:
            if np.sign(b.mean()) != np.sign(a.mean()):
                aligned[:, i] *= -1.0
            continue
        corr = float((a0 @ b0) / denom)
        if corr < 0:
            aligned[:, i] *= -1.0
    return aligned


def force_mode0_blue(matrix: np.ndarray) -> np.ndarray:
    forced = matrix.copy()
    if forced[:, 0].mean() > 0:
        forced[:, 0] *= -1.0
    return forced
