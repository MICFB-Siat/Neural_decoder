import numpy as np

from eigenmode_common import (
    TEMPLATE_EVALS,
    TEMPLATE_FULL,
    TEMPLATE_GEN_DIR,
    TEMPLATE_MASK,
    TEMPLATE_MASKED_MODES,
    TEMPLATE_META,
    TEMPLATE_VTK,
    compute_lb_modes_with_mode0,
    ensure_dir,
    expand_to_full,
    load_text_mask,
    load_vtk_surface,
    mask_surface,
)


N_MODES = 1000


def main() -> None:
    ensure_dir(TEMPLATE_GEN_DIR)

    coords, faces = load_vtk_surface(TEMPLATE_VTK)
    mask = load_text_mask(TEMPLATE_MASK)
    coords_m, faces_m, keep_idx = mask_surface(coords, faces, mask)

    print(f"🚀 template 原始顶点数: {coords.shape[0]}")
    print(f"🚀 template mask 后顶点数: {coords_m.shape[0]}")
    evals, masked_modes = compute_lb_modes_with_mode0(coords_m, faces_m, N_MODES)
    full_modes = expand_to_full(masked_modes, keep_idx, coords.shape[0])

    np.save(TEMPLATE_MASKED_MODES, masked_modes)
    np.save(TEMPLATE_EVALS, evals)
    np.save(TEMPLATE_FULL, full_modes)
    np.savez(
        TEMPLATE_META,
        surf=str(TEMPLATE_VTK),
        mask=str(TEMPLATE_MASK),
        keep_indices=keep_idx.astype(np.int32),
        n_vertices_total=np.int32(coords.shape[0]),
        n_vertices_masked=np.int32(coords_m.shape[0]),
        n_modes=np.int32(N_MODES),
    )
    print(f"🎉 已输出: {TEMPLATE_FULL}")


if __name__ == "__main__":
    main()
