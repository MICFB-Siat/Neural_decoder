














from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any

import mne
import nibabel as nib
import numpy as np
import pyvista as pv
from lapy import Solver, TriaMesh
from scipy.spatial import cKDTree


Array = np.ndarray


@dataclass
class SurfaceModes:
    evals: Array
    modes_masked: Array
    coords_masked: Array
    keep_indices: Array


@dataclass
class SrcModes:
    evals: Array
    phi_src: Array
    k_left: int
    k_right: int


def _ensure_3d(name: str, x: Array | None) -> Array | None:
    if x is None:
        return None
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"{name} 必须是 3 维 ndarray，当前 shape={x.shape}")
    return x


def _split_modes(k_modes: int) -> tuple[int, int]:
    k_left = int(k_modes) // 2
    k_right = int(k_modes) - k_left
    return k_left, k_right


def _glob_first(base_dir: str, patterns: list[str]) -> str:
    for pat in patterns:
        found = sorted(glob.glob(os.path.join(base_dir, pat)))
        if found:
            return found[0]
    raise FileNotFoundError(f"在 {base_dir} 未找到匹配文件，patterns={patterns}")


def _load_surface_any(surface_path: str) -> tuple[Array, Array]:
    ext = os.path.splitext(surface_path)[1].lower()
    if ext == ".vtk":
        mesh = pv.read(surface_path)
        coords = np.asarray(mesh.points, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
        return coords, faces
    if ext == ".gii":
        img = nib.load(surface_path)
        if len(img.darrays) < 2:
            raise ValueError(f"表面文件缺少坐标/三角面数据: {surface_path}")
        coords = np.asarray(img.darrays[0].data, dtype=np.float64)
        faces = np.asarray(img.darrays[1].data, dtype=np.int64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"顶点坐标维度错误: {surface_path}, shape={coords.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"三角面维度错误: {surface_path}, shape={faces.shape}")
        return coords, faces
    raise ValueError(f"不支持的表面格式: {surface_path}")


def _load_mask_any(mask_path: str) -> Array:
    ext = os.path.splitext(mask_path)[1].lower()
    if ext == ".txt":
        mask = np.loadtxt(mask_path)
    elif ext == ".gii":
        mask = nib.load(mask_path).darrays[0].data
    else:
        raise ValueError(f"不支持的 mask 格式: {mask_path}")
    mask = np.asarray(mask).ravel() > 0
    if not np.any(mask):
        raise ValueError(f"mask 全为 0: {mask_path}")
    return mask


def _mask_surface(coords: Array, faces: Array, mask: Array) -> tuple[Array, Array, Array]:
    if mask.shape[0] != coords.shape[0]:
        raise ValueError(f"mask 顶点数与表面不一致: mask={mask.shape[0]}, surface={coords.shape[0]}")
    keep_idx = np.where(mask)[0]
    keep_faces = np.all(mask[faces], axis=1)
    faces_kept = faces[keep_faces]
    remap = np.full(coords.shape[0], -1, dtype=np.int64)
    remap[keep_idx] = np.arange(keep_idx.size, dtype=np.int64)
    faces_remap = remap[faces_kept]
    coords_kept = coords[keep_idx]
    return coords_kept, faces_remap, keep_idx


def _compute_lb_modes(coords: Array, faces: Array, n_modes: int) -> tuple[Array, Array]:
    max_modes = coords.shape[0] - 1
    used_modes = min(int(n_modes), int(max_modes))
    if used_modes < 1:
        raise ValueError("可用本征模数为 0")
    mesh = TriaMesh(coords, faces)
    solver = Solver(mesh)
    evals_all, evecs_all = solver.eigs(k=used_modes + 1, sigma=0.0)
    evals = np.asarray(evals_all[1 : used_modes + 1], dtype=np.float64)
    evecs = np.asarray(evecs_all[:, 1 : used_modes + 1], dtype=np.float64)
    return evals, evecs


def _match_unit_scale(surface_coords: Array, src_coords: Array) -> Array:
    src_norm = np.median(np.linalg.norm(src_coords, axis=1))
    surf_norm = np.median(np.linalg.norm(surface_coords, axis=1))
    if src_norm <= 0 or surf_norm <= 0:
        return surface_coords
    ratio = surf_norm / src_norm
    if ratio > 100:
        return surface_coords / 1000.0
    if ratio < 0.01:
        return surface_coords * 1000.0
    return surface_coords


def _find_template_surfaces(template_dir: str) -> dict[str, str]:
    if not os.path.isdir(template_dir):
        raise FileNotFoundError(f"群体模板路径不存在: {template_dir}")
    return {
        "lh_surf": _glob_first(template_dir, ["*midthickness-lh.vtk", "*.L.midthickness*.surf.gii", "*lh*.vtk"]),
        "rh_surf": _glob_first(template_dir, ["*midthickness-rh.vtk", "*.R.midthickness*.surf.gii", "*rh*.vtk"]),
        "lh_mask": _glob_first(template_dir, ["*cortex-lh_mask.txt", "*.L.atlasroi*.shape.gii", "*lh*mask*.txt"]),
        "rh_mask": _glob_first(template_dir, ["*cortex-rh_mask.txt", "*.R.atlasroi*.shape.gii", "*rh*mask*.txt"]),
    }


def _find_individual_surfaces(individual_t1_dir: str) -> dict[str, str]:
    if not os.path.isdir(individual_t1_dir):
        raise FileNotFoundError(f"个体 T1 路径不存在: {individual_t1_dir}")
    return {
        "lh_surf": _glob_first(individual_t1_dir, ["*.L.midthickness.32k_fs_LR.surf.gii", "*L*midthickness*.surf.gii"]),
        "rh_surf": _glob_first(individual_t1_dir, ["*.R.midthickness.32k_fs_LR.surf.gii", "*R*midthickness*.surf.gii"]),
        "lh_mask": _glob_first(individual_t1_dir, ["*.L.atlasroi.32k_fs_LR.shape.gii", "*L*atlasroi*.shape.gii", "*L*mask*.gii"]),
        "rh_mask": _glob_first(individual_t1_dir, ["*.R.atlasroi.32k_fs_LR.shape.gii", "*R*atlasroi*.shape.gii", "*R*mask*.gii"]),
    }


def _compute_surface_modes_from_files(lh_surf: str, rh_surf: str, lh_mask: str, rh_mask: str, k_modes: int) -> tuple[SurfaceModes, SurfaceModes]:
    k_left, k_right = _split_modes(k_modes)

    lh_coords, lh_faces = _load_surface_any(lh_surf)
    rh_coords, rh_faces = _load_surface_any(rh_surf)
    lh_mask_arr = _load_mask_any(lh_mask)
    rh_mask_arr = _load_mask_any(rh_mask)

    lh_coords_m, lh_faces_m, lh_keep = _mask_surface(lh_coords, lh_faces, lh_mask_arr)
    rh_coords_m, rh_faces_m, rh_keep = _mask_surface(rh_coords, rh_faces, rh_mask_arr)

    lh_evals, lh_modes = _compute_lb_modes(lh_coords_m, lh_faces_m, k_left)
    rh_evals, rh_modes = _compute_lb_modes(rh_coords_m, rh_faces_m, k_right)

    left = SurfaceModes(evals=lh_evals, modes_masked=lh_modes, coords_masked=lh_coords_m, keep_indices=lh_keep)
    right = SurfaceModes(evals=rh_evals, modes_masked=rh_modes, coords_masked=rh_coords_m, keep_indices=rh_keep)
    return left, right


def _surface_modes_to_fmri_phi(left: SurfaceModes, right: SurfaceModes) -> tuple[Array, Array]:
    phi = np.concatenate([left.modes_masked, right.modes_masked], axis=0)
    evals = np.concatenate([left.evals, right.evals], axis=0)
    return phi.astype(np.float64), evals.astype(np.float64)


def _surface_modes_to_src_phi(left: SurfaceModes, right: SurfaceModes, src) -> SrcModes:
    lh_src = np.asarray(src[0]["rr"][src[0]["vertno"]], dtype=np.float64)
    rh_src = np.asarray(src[1]["rr"][src[1]["vertno"]], dtype=np.float64)

    lh_coords_aligned = _match_unit_scale(left.coords_masked, lh_src)
    rh_coords_aligned = _match_unit_scale(right.coords_masked, rh_src)

    lh_idx = cKDTree(lh_coords_aligned).query(lh_src, k=1)[1]
    rh_idx = cKDTree(rh_coords_aligned).query(rh_src, k=1)[1]

    lh_src_modes = left.modes_masked[lh_idx]
    rh_src_modes = right.modes_masked[rh_idx]
    n_left = lh_src_modes.shape[0]
    k_left = lh_src_modes.shape[1]
    k_right = rh_src_modes.shape[1]
    phi_src = np.zeros((n_left + rh_src_modes.shape[0], k_left + k_right), dtype=np.float64)
    phi_src[:n_left, :k_left] = lh_src_modes
    phi_src[n_left:, k_left:] = rh_src_modes
    evals = np.concatenate([left.evals, right.evals], axis=0)
    return SrcModes(evals=evals.astype(np.float64), phi_src=phi_src, k_left=k_left, k_right=k_right)


def _setup_fsaverage_forward_resources(spacing: str = "ico4"):
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = fs_dir.parent
    src = mne.setup_source_space(
        subject="fsaverage",
        spacing=spacing,
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose="error",
    )
    conductivity = (0.3, 0.006, 0.3)
    model = mne.make_bem_model(
        subject="fsaverage",
        ico=4,
        conductivity=conductivity,
        subjects_dir=subjects_dir,
        verbose="error",
    )
    bem = mne.make_bem_solution(model, verbose="error")
    return src, bem


def _build_default_eeg_info(n_channels: int, sfreq: float, eeg_channel_names: list[str] | None = None) -> mne.Info:
    if eeg_channel_names is None:
        montage128 = mne.channels.make_standard_montage("biosemi128")
        names = list(montage128.ch_names)
        if n_channels <= len(names):
            ch_names = names[:n_channels]
        else:
            ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    else:
        if len(eeg_channel_names) != n_channels:
            raise ValueError(f"eeg_channel_names 数量应为 {n_channels}，当前为 {len(eeg_channel_names)}")
        ch_names = list(eeg_channel_names)

    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types=["eeg"] * n_channels)
    if eeg_channel_names is None and n_channels <= 128:
        info.set_montage(mne.channels.make_standard_montage("biosemi128"), on_missing="ignore")
    else:
        info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    return info


def _make_forward(info: mne.Info, src, bem, eeg: bool, meg: bool):
    fwd = mne.make_forward_solution(
        info=info,
        trans="fsaverage",
        src=src,
        bem=bem,
        eeg=eeg,
        meg=meg,
        mindist=5.0,
        verbose="error",
    )
    return mne.convert_forward_solution(
        fwd,
        surf_ori=True,
        force_fixed=True,
        use_cps=True,
        verbose="error",
    )


def _align_leadfield_and_data(fwd_fixed, data_bct: Array, info: mne.Info) -> tuple[Array, Array]:
    leadfield = np.asarray(fwd_fixed["sol"]["data"], dtype=np.float64)
    fwd_names = list(fwd_fixed["sol"]["row_names"])
    info_name_to_idx = {name: i for i, name in enumerate(info.ch_names)}
    common = [name for name in fwd_names if name in info_name_to_idx]
    if len(common) < 8:
        raise ValueError(f"可匹配前向通道过少: {len(common)}")
    fwd_idx = [fwd_names.index(n) for n in common]
    data_idx = [info_name_to_idx[n] for n in common]
    l_used = leadfield[fwd_idx, :]
    y_used = data_bct[:, data_idx, :]
    return l_used, y_used


def _solve_batch_coefficients(design: Array, data_bct: Array) -> Array:

    pinv_d = np.linalg.pinv(design)
    return np.einsum("ks,bst->bkt", pinv_d, data_bct, optimize=True).astype(np.float32)


def _sensor_eigenmodes(data_bct: Array, k: int) -> tuple[Array, Array]:
    b, c, t = data_bct.shape
    x = np.transpose(data_bct, (1, 0, 2)).reshape(c, b * t).astype(np.float64, copy=False)
    cov = (x @ x.T) / max(1, x.shape[1])
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    k_use = min(k, c)
    return vals[:k_use].astype(np.float32), vecs[:, :k_use].astype(np.float32)


def _load_first_fif_info(meg_fif_dir: str) -> mne.Info:
    if not os.path.isdir(meg_fif_dir):
        raise FileNotFoundError(f"meg_fif_dir 不存在: {meg_fif_dir}")
    fif_files = sorted(glob.glob(os.path.join(meg_fif_dir, "**", "*.fif"), recursive=True))
    if not fif_files:
        raise FileNotFoundError(f"在 meg_fif_dir 中未找到 .fif 文件: {meg_fif_dir}")
    raw = mne.io.read_raw_fif(fif_files[0], preload=False, verbose="error")
    return raw.info


def _adapt_meg_info_for_data(meg_info: mne.Info, n_channels: int) -> mne.Info:

    meg_picks = mne.pick_types(meg_info, meg=True, ref_meg=True, eeg=False, exclude=[])
    if len(meg_picks) == n_channels:
        return mne.pick_info(meg_info, sel=meg_picks)
    if len(meg_info["ch_names"]) == n_channels:
        return meg_info
    if len(meg_picks) > n_channels:
        return mne.pick_info(meg_info, sel=meg_picks[:n_channels])
    if len(meg_info["ch_names"]) > n_channels:
        return mne.pick_info(meg_info, sel=list(range(n_channels)))
    raise ValueError(
        f"无法将 meg_info 适配到 meg_data 通道数: info_channels={len(meg_info['ch_names'])}, data_channels={n_channels}"
    )


def run_geometric_eigenmodes(
    *,
    template_root: str,
    eeg_data: Array | None = None,
    meg_data: Array | None = None,
    fmri_data: Array | None = None,
    individual_t1_root: str | None = None,
    eeg_mode_ratio: float = 0.8,
    meg_mode_ratio: float = 0.8,
    fmri_num_modes: int = 2000,
    sfreq: float = 256.0,
    eeg_channel_names: list[str] | None = None,
    meg_fif_dir: str | None = None,
) -> dict[str, Any]:




















    eeg_data = _ensure_3d("eeg_data", eeg_data)
    meg_data = _ensure_3d("meg_data", meg_data)
    fmri_data = _ensure_3d("fmri_data", fmri_data)
    if eeg_data is None and meg_data is None and fmri_data is None:
        raise ValueError("至少需要传入 eeg_data / meg_data / fmri_data 中的一个")
    if not template_root:
        raise ValueError("template_root 为必传参数")

    template_paths = _find_template_surfaces(os.path.abspath(template_root))
    src, bem = _setup_fsaverage_forward_resources(spacing="ico4")
    has_individual = individual_t1_root is not None
    indiv_paths = _find_individual_surfaces(os.path.abspath(individual_t1_root)) if has_individual else None

    results: dict[str, Any] = {"group": {}, "individual": {} if has_individual else None}

    fmri_template_cache: dict[int, tuple[Array, Array]] = {}
    src_template_cache: dict[int, SrcModes] = {}
    src_individual_cache: dict[int, SrcModes] = {}

    def get_fmri_template_modes(k: int) -> tuple[Array, Array]:
        if k not in fmri_template_cache:
            left, right = _compute_surface_modes_from_files(
                template_paths["lh_surf"],
                template_paths["rh_surf"],
                template_paths["lh_mask"],
                template_paths["rh_mask"],
                k,
            )
            fmri_template_cache[k] = _surface_modes_to_fmri_phi(left, right)
        return fmri_template_cache[k]

    def get_src_template_modes(k: int) -> SrcModes:
        if k not in src_template_cache:
            left, right = _compute_surface_modes_from_files(
                template_paths["lh_surf"],
                template_paths["rh_surf"],
                template_paths["lh_mask"],
                template_paths["rh_mask"],
                k,
            )
            src_template_cache[k] = _surface_modes_to_src_phi(left, right, src)
        return src_template_cache[k]

    def get_src_individual_modes(k: int) -> SrcModes:
        if not has_individual or indiv_paths is None:
            raise ValueError("未提供 individual_t1_root，无法计算个体几何本征模")
        if k not in src_individual_cache:
            left, right = _compute_surface_modes_from_files(
                indiv_paths["lh_surf"],
                indiv_paths["rh_surf"],
                indiv_paths["lh_mask"],
                indiv_paths["rh_mask"],
                k,
            )
            src_individual_cache[k] = _surface_modes_to_src_phi(left, right, src)
        return src_individual_cache[k]


    if fmri_data is not None:
        b, v, t = fmri_data.shape
        fmri_k = int(fmri_num_modes)
        if fmri_k < 1:
            raise ValueError("fmri_num_modes 必须 >= 1")

        phi_group, evals_group = get_fmri_template_modes(fmri_k)
        if v != phi_group.shape[0]:
            raise ValueError(
                f"fMRI 顶点数与模板模式不一致: data={v}, template_masked_vertices={phi_group.shape[0]}"
            )
        coeff_group = _solve_batch_coefficients(phi_group, fmri_data.astype(np.float64, copy=False))
        results["group"]["fmri"] = {
            "coefficients": coeff_group,
            "k_modes": int(phi_group.shape[1]),
            "evals": evals_group.astype(np.float32),
        }

        if has_individual and indiv_paths is not None:
            left_i, right_i = _compute_surface_modes_from_files(
                indiv_paths["lh_surf"],
                indiv_paths["rh_surf"],
                indiv_paths["lh_mask"],
                indiv_paths["rh_mask"],
                fmri_k,
            )
            phi_ind, evals_ind = _surface_modes_to_fmri_phi(left_i, right_i)
            if v != phi_ind.shape[0]:
                raise ValueError(
                    f"fMRI 顶点数与个体模式不一致: data={v}, individual_masked_vertices={phi_ind.shape[0]}"
                )
            coeff_ind = _solve_batch_coefficients(phi_ind, fmri_data.astype(np.float64, copy=False))
            results["individual"]["fmri"] = {
                "coefficients": coeff_ind,
                "modes": phi_ind.astype(np.float32),
                "evals": evals_ind.astype(np.float32),
            }


    if eeg_data is not None:
        b, c, t = eeg_data.shape
        eeg_k = max(1, int(np.floor(c * float(eeg_mode_ratio))))

        eeg_info = _build_default_eeg_info(c, sfreq=sfreq, eeg_channel_names=eeg_channel_names)
        fwd_eeg = _make_forward(eeg_info, src=src, bem=bem, eeg=True, meg=False)
        l_eeg, y_eeg = _align_leadfield_and_data(fwd_eeg, eeg_data.astype(np.float64, copy=False), eeg_info)

        src_group = get_src_template_modes(eeg_k)
        d_group = l_eeg @ src_group.phi_src
        coeff_group = _solve_batch_coefficients(d_group, y_eeg)
        results["group"]["eeg"] = {
            "coefficients": coeff_group,
            "k_modes": int(src_group.phi_src.shape[1]),
        }

        if has_individual:
            src_ind = get_src_individual_modes(eeg_k)
            d_ind = l_eeg @ src_ind.phi_src
            coeff_ind = _solve_batch_coefficients(d_ind, y_eeg)
            sensor_vals, sensor_modes = _sensor_eigenmodes(y_eeg, k=src_ind.phi_src.shape[1])
            results["individual"]["eeg"] = {
                "coefficients": coeff_ind,
                "modes": src_ind.phi_src.astype(np.float32),
                "evals": src_ind.evals.astype(np.float32),
                "sensor_evals": sensor_vals,
                "sensor_modes": sensor_modes,
            }


    if meg_data is not None:
        b, c, t = meg_data.shape
        meg_k = max(1, int(np.floor(c * float(meg_mode_ratio))))
        if meg_fif_dir is None:
            raise ValueError("传入 meg_data 时，需要提供 meg_fif_dir")
        meg_info = _load_first_fif_info(os.path.abspath(meg_fif_dir))
        meg_info = _adapt_meg_info_for_data(meg_info, n_channels=c)

        fwd_meg = _make_forward(meg_info, src=src, bem=bem, eeg=False, meg=True)
        l_meg, y_meg = _align_leadfield_and_data(fwd_meg, meg_data.astype(np.float64, copy=False), meg_info)

        src_group = get_src_template_modes(meg_k)
        d_group = l_meg @ src_group.phi_src
        coeff_group = _solve_batch_coefficients(d_group, y_meg)
        results["group"]["meg"] = {
            "coefficients": coeff_group,
            "k_modes": int(src_group.phi_src.shape[1]),
        }

        if has_individual:
            src_ind = get_src_individual_modes(meg_k)
            d_ind = l_meg @ src_ind.phi_src
            coeff_ind = _solve_batch_coefficients(d_ind, y_meg)
            sensor_vals, sensor_modes = _sensor_eigenmodes(y_meg, k=src_ind.phi_src.shape[1])
            results["individual"]["meg"] = {
                "coefficients": coeff_ind,
                "modes": src_ind.phi_src.astype(np.float32),
                "evals": src_ind.evals.astype(np.float32),
                "sensor_evals": sensor_vals,
                "sensor_modes": sensor_modes,
            }

    return results
