from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

from eigenmode_common import (
    FINAL_GRID,
    SUB22_FULL,
    SUB22_SURF,
    SUB24_FULL,
    SUB24_SURF,
    TEMPLATE_FULL,
    TEMPLATE_META,
    TEMPLATE_RENDER_SURF,
    align_modes_to_reference,
    build_colormap,
    force_mode0_blue,
    load_full_matrix,
    load_surface_as_polydata,
    mode_scale,
)


N_MODES = 5
CELL_W = 620
CELL_H = 430
TITLE_H = 36
ROW_LABEL_W = 150
MARGIN_X = 16
MARGIN_Y = 16
TOP_PAD = 56
BG = (255, 255, 255)
OUTPUT_DPI = (600, 600)
RESAMPLING_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def prepare_plotter(window_size=(1000, 760)):
    plotter = pv.Plotter(off_screen=True, window_size=window_size, lighting="light kit")
    plotter.set_background("white")
    plotter.enable_anti_aliasing("msaa")
    return plotter


def set_left_lateral_camera(plotter, mesh):
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


def render_mode_tile(base_mesh, mode_values, cmap):
    mesh = base_mesh.copy(deep=True)
    mesh.point_data.clear()
    mesh["mode_values"] = mode_values
    scale = mode_scale(mode_values)
    plotter = prepare_plotter()
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

    pil_img = Image.fromarray(img)
    pil_img = pil_img.crop(pil_img.getbbox())
    pil_img.thumbnail((CELL_W - 20, CELL_H - TITLE_H - 16), RESAMPLING_LANCZOS)
    return pil_img


def render_row(label, surf_file: Path, matrix: np.ndarray):
    mesh = load_surface_as_polydata(surf_file)
    cmap = build_colormap()
    tiles = []
    for i in range(N_MODES):
        tiles.append(render_mode_tile(mesh, matrix[:, i], cmap))
    return label, tiles


def build_grid(rows, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(rows)
    n_cols = N_MODES
    canvas_w = ROW_LABEL_W + n_cols * CELL_W + (n_cols + 1) * MARGIN_X
    canvas_h = TOP_PAD + n_rows * CELL_H + (n_rows + 1) * MARGIN_Y
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for col in range(n_cols):
        x0 = ROW_LABEL_W + MARGIN_X + col * CELL_W
        draw.text((x0 + 12, 12), f"mode {col}", fill=(0, 0, 0), font=font)

    for row_idx, (row_name, tiles) in enumerate(rows):
        y0 = TOP_PAD + MARGIN_Y + row_idx * CELL_H
        draw.text((16, y0 + CELL_H // 2 - 8), row_name, fill=(0, 0, 0), font=font)
        for col_idx, tile in enumerate(tiles):
            x0 = ROW_LABEL_W + MARGIN_X + col_idx * CELL_W
            img_x = x0 + (CELL_W - tile.width) // 2
            img_y = y0 + TITLE_H + (CELL_H - TITLE_H - tile.height) // 2
            canvas.paste(tile, (img_x, img_y))

    canvas.save(output_file, dpi=OUTPUT_DPI)


def main() -> None:
    template_matrix = load_full_matrix(TEMPLATE_FULL)[:, :N_MODES]
    sub22_matrix = load_full_matrix(SUB22_FULL)[:, :N_MODES]
    sub24_matrix = load_full_matrix(SUB24_FULL)[:, :N_MODES]
    template_roi = np.asarray(np.load(TEMPLATE_META)["keep_indices"], dtype=np.int64)

    sub22_matrix = align_modes_to_reference(sub22_matrix, template_matrix, template_roi)
    sub24_matrix = align_modes_to_reference(sub24_matrix, template_matrix, template_roi)

    template_matrix = force_mode0_blue(template_matrix)
    sub22_matrix = force_mode0_blue(sub22_matrix)
    sub24_matrix = force_mode0_blue(sub24_matrix)

    rows = [
        render_row("Template", TEMPLATE_RENDER_SURF, template_matrix),
        render_row("sub-22", SUB22_SURF, sub22_matrix),
        render_row("sub-24", SUB24_SURF, sub24_matrix),
    ]
    build_grid(rows, FINAL_GRID)
    print(f"🎉 已输出: {FINAL_GRID}")


if __name__ == "__main__":
    main()
