import numpy as np

from eigenmode_common import (
    SUB24_EVALS,
    SUB24_FULL,
    SUB24_GEN_DIR,
    SUB24_MASK,
    SUB24_MASKED_MODES,
    SUB24_META,
    SUB24_SURF,
    compute_lb_modes_with_mode0,
    ensure_dir,
    expand_to_full,
    load_gifti_mask,
    load_gifti_surface,
    mask_surface,
)


N_MODES = 1000


def main() -> None:
    ensure_dir(SUB24_GEN_DIR)

    coords, faces = load_gifti_surface(SUB24_SURF)
    mask = load_gifti_mask(SUB24_MASK)
    coords_m, faces_m, keep_idx = mask_surface(coords, faces, mask)

    print(f"🚀 sub-24 原始顶点数: {coords.shape[0]}")
    print(f"🚀 sub-24 mask 后顶点数: {coords_m.shape[0]}")
    evals, masked_modes = compute_lb_modes_with_mode0(coords_m, faces_m, N_MODES)
    full_modes = expand_to_full(masked_modes, keep_idx, coords.shape[0])

    np.save(SUB24_MASKED_MODES, masked_modes)
    np.save(SUB24_EVALS, evals)
    np.save(SUB24_FULL, full_modes)
    np.savez(
        SUB24_META,
        surf=str(SUB24_SURF),
        mask=str(SUB24_MASK),
        keep_indices=keep_idx.astype(np.int32),
        n_vertices_total=np.int32(coords.shape[0]),
        n_vertices_masked=np.int32(coords_m.shape[0]),
        n_modes=np.int32(N_MODES),
    )
    print(f"🎉 已输出: {SUB24_FULL}")


if __name__ == "__main__":
    main()
