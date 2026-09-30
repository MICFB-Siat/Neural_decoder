import numpy as np

from eigenmode_common import (
    SUB22_EVALS,
    SUB22_FULL,
    SUB22_GEN_DIR,
    SUB22_MASK,
    SUB22_MASKED_MODES,
    SUB22_META,
    SUB22_SURF,
    compute_lb_modes_with_mode0,
    ensure_dir,
    expand_to_full,
    load_gifti_mask,
    load_gifti_surface,
    mask_surface,
)


N_MODES = 1000


def main() -> None:
    ensure_dir(SUB22_GEN_DIR)

    coords, faces = load_gifti_surface(SUB22_SURF)
    mask = load_gifti_mask(SUB22_MASK)
    coords_m, faces_m, keep_idx = mask_surface(coords, faces, mask)

    print(f"🚀 sub-22 原始顶点数: {coords.shape[0]}")
    print(f"🚀 sub-22 mask 后顶点数: {coords_m.shape[0]}")
    evals, masked_modes = compute_lb_modes_with_mode0(coords_m, faces_m, N_MODES)
    full_modes = expand_to_full(masked_modes, keep_idx, coords.shape[0])

    np.save(SUB22_MASKED_MODES, masked_modes)
    np.save(SUB22_EVALS, evals)
    np.save(SUB22_FULL, full_modes)
    np.savez(
        SUB22_META,
        surf=str(SUB22_SURF),
        mask=str(SUB22_MASK),
        keep_indices=keep_idx.astype(np.int32),
        n_vertices_total=np.int32(coords.shape[0]),
        n_vertices_masked=np.int32(coords_m.shape[0]),
        n_modes=np.int32(N_MODES),
    )
    print(f"🎉 已输出: {SUB22_FULL}")


if __name__ == "__main__":
    main()
