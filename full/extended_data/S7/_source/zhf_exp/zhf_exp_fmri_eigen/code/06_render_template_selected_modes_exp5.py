import argparse
import os
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import nibabel as nib
import numpy as np
import pyvista as pv
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_MODES = [25, 74, 123, 172, 342, 391, 440, 610, 659, 829, 878, 926]
CODE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_DIR.parent
DATA_DIR = PACKAGE_ROOT / "data"
TEMPLATE_FULL = DATA_DIR / "generated" / "template" / "template_L_eigenmodes_1000_with_mode0_full32492.npy"
TEMPLATE_RENDER_SURF = (
    DATA_DIR / "raw_inputs" / "template" / "S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
)
DEFAULT_OUT_DIR = PACKAGE_ROOT.parent.parent / "guoyi_exp" / "Exp5" / "Eigen_mode_cache"
DEFAULT_WINDOW_SIZE = (2200, 1600)
OUTPUT_DPI = (600, 600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render selected template fsLR 32k left-hemisphere eigenmodes as separate PNG files."
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        type=int,
        default=DEFAULT_MODES,
        help="Mode indices to render. These follow the mode0-based column numbering used by this package.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for output PNG files.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=DEFAULT_WINDOW_SIZE[0],
        help="PyVista screenshot width in pixels.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=DEFAULT_WINDOW_SIZE[1],
        help="PyVista screenshot height in pixels.",
    )
    return parser.parse_args()


def prepare_plotter(window_size):
    plotter = pv.Plotter(off_screen=True, window_size=window_size, lighting="light kit")
    plotter.set_background("white")
    plotter.enable_anti_aliasing("msaa")
    return plotter


def build_colormap():
    return LinearSegmentedColormap.from_list(
        "mode_map",
        ["#08306b", "#2171b5", "#41b6c4", "#fff7bc", "#fdae61", "#f03b20", "#bd0026"],
        N=256,
    )


def load_full_matrix(file_path: Path) -> np.ndarray:
    return np.load(str(file_path)).astype(np.float32, copy=False)


def load_surface_as_polydata(surf_file: Path):
    surf = nib.load(str(surf_file))
    vertices = np.asarray(surf.darrays[0].data, dtype=np.float32)
    faces = np.asarray(surf.darrays[1].data, dtype=np.int32)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32), faces]).ravel()
    poly = pv.PolyData(vertices, faces_pv)
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


def set_left_lateral_camera(plotter, mesh) -> None:
    center = np.array(mesh.center)
    bounds = np.array(mesh.bounds, dtype=np.float32)
    size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    position = center + np.array([-2.8 * size, 0.0, 0.10 * size], dtype=np.float32)
    focal = center + np.array([0.0, 0.0, -0.02 * size], dtype=np.float32)
    plotter.camera.position = position.tolist()
    plotter.camera.focal_point = focal.tolist()
    plotter.camera.up = (0.0, 0.0, 1.0)
    plotter.camera.parallel_projection = False
    plotter.camera.view_angle = 18.0


def render_mode(base_mesh, mode_values, cmap, out_file: Path, window_size) -> None:
    mesh = base_mesh.copy(deep=True)
    mesh.point_data.clear()
    mesh["mode_values"] = mode_values.astype(np.float32, copy=False)
    scale = mode_scale(mode_values)

    plotter = prepare_plotter(window_size)
    plotter.add_mesh(
        mesh,
        scalars="mode_values",
        cmap=cmap,
        clim=(-scale, scale),
        lighting=True,
        smooth_shading=True,
        show_scalar_bar=False,
        ambient=0.22,
        diffuse=0.75,
        specular=0.14,
        specular_power=18.0,
        interpolate_before_map=True,
    )
    set_left_lateral_camera(plotter, mesh)
    plotter.camera.zoom(1.18)
    img = plotter.screenshot(return_img=True, transparent_background=False)
    plotter.close()

    Image.fromarray(img).save(out_file, dpi=OUTPUT_DPI)


def main() -> None:
    args = parse_args()
    modes = list(dict.fromkeys(args.modes))
    if any(mode < 0 for mode in modes):
        raise ValueError(f"Mode indices must be non-negative: {modes}")

    matrix = load_full_matrix(TEMPLATE_FULL)
    max_mode = max(modes)
    if max_mode >= matrix.shape[1]:
        raise ValueError(f"Requested mode {max_mode}, but only {matrix.shape[1]} modes are available.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mesh = load_surface_as_polydata(TEMPLATE_RENDER_SURF)
    cmap = build_colormap()
    window_size = (args.window_width, args.window_height)

    for mode_idx in modes:
        out_file = args.out_dir / f"template_fs32k_L_mode_{mode_idx:03d}.png"
        render_mode(mesh, matrix[:, mode_idx], cmap, out_file, window_size)
        print(f"saved {out_file}")


if __name__ == "__main__":
    main()
