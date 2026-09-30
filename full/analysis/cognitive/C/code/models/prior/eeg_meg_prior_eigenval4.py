














































from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from .eeg_meg_prior_eigenval3 import (
    EEGMEGPriorEigenval3,
    EEGMEGMAEDecoderEigenval3,
)





import os as _os
_FS32K_LOCAL = _os.path.normpath(_os.path.join(
    _os.path.dirname(__file__), "..", "..",
    "guoyi_exp", "data", "eigenmode_test", "fs32k"))
_FS32K_FALLBACK = "/media/wsqlab/data/tyf/ugreen/eigenmode_test/fs32k"
_FS32K_DIR = _FS32K_LOCAL if _os.path.isdir(_FS32K_LOCAL) else _FS32K_FALLBACK
DEFAULT_LH_EVAL = _os.path.join(_FS32K_DIR, "fsLR_32k_lh_eval_1024.npy")
DEFAULT_RH_EVAL = _os.path.join(_FS32K_DIR, "fsLR_32k_rh_eval_1024.npy")


def _load_group_lam_cortex(K_cortex: int,
                           lh_path: str = DEFAULT_LH_EVAL,
                           rh_path: str = DEFAULT_RH_EVAL) -> np.ndarray:

    half_lh = (K_cortex + 1) // 2
    half_rh = K_cortex // 2
    lh = np.load(lh_path)[:half_lh].astype(np.float64)
    rh = np.load(rh_path)[:half_rh].astype(np.float64)
    return np.concatenate([lh, rh])


def compute_eigval_from_D(D,
                          lam_cortex=None,
                          ratio: float = 1e-3,
                          lh_path: str = DEFAULT_LH_EVAL,
                          rh_path: str = DEFAULT_RH_EVAL):

















    D_np = (D.detach().cpu().numpy() if isinstance(D, torch.Tensor) else np.asarray(D)).astype(np.float64)
    if D_np.ndim != 2:
        raise ValueError(f"D must be 2D (n_sensor, K_cortex); got shape {D_np.shape}")
    n_sensor, K_cortex = D_np.shape

    if lam_cortex is None:
        lam = _load_group_lam_cortex(K_cortex, lh_path, rh_path)
    else:
        lam = (lam_cortex.detach().cpu().numpy()
               if isinstance(lam_cortex, torch.Tensor)
               else np.asarray(lam_cortex)).astype(np.float64)
        if lam.shape != (K_cortex,):
            raise ValueError(
                f"lam_cortex shape {lam.shape} != D cortex dim ({K_cortex},)"
            )

    U, S, Vt = np.linalg.svd(D_np, full_matrices=False)
    K_eff = int((S / S[0] >= ratio).sum())
    V = Vt.T
    eigval = (V[:, :K_eff] ** 2 * lam[:, None]).sum(axis=0).astype(np.float32)
    return eigval, K_eff


class EEGMEGPriorEigenval4(EEGMEGPriorEigenval3):











    def __init__(self, *args,
                 lh_eval_path: str = DEFAULT_LH_EVAL,
                 rh_eval_path: str = DEFAULT_RH_EVAL,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._lh_eval_path = lh_eval_path
        self._rh_eval_path = rh_eval_path


        self._eigval_cache_key = None
        self._eigval_cache_val = None




    def resolve_eigval(self,
                       eigval=None,
                       D=None,
                       lam_cortex=None,
                       ratio: float = 1e-3,
                       device=None,
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:




        if eigval is not None:
            ev = eigval if isinstance(eigval, torch.Tensor) else torch.as_tensor(eigval)
            return ev.to(device=device, dtype=dtype) if device is not None else ev.to(dtype=dtype)

        if D is None:
            raise ValueError(
                "EEGMEGPriorEigenval4: must provide either `eigval` (pre-computed) "
                "or `D` (sensor × cortex Φ matrix)."
            )



        D_key = (D.data_ptr(), tuple(D.shape)) if isinstance(D, torch.Tensor) else (id(D), tuple(np.shape(D)))
        if lam_cortex is None:
            lam_key = ("group", self._lh_eval_path, self._rh_eval_path)
        elif isinstance(lam_cortex, torch.Tensor):
            lam_key = ("indiv-t", lam_cortex.data_ptr(), tuple(lam_cortex.shape))
        else:
            lam_key = ("indiv-n", id(lam_cortex), tuple(np.shape(lam_cortex)))
        cache_key = (D_key, lam_key, float(ratio))

        if cache_key != self._eigval_cache_key or self._eigval_cache_val is None:
            ev_np, _ = compute_eigval_from_D(
                D, lam_cortex, ratio, self._lh_eval_path, self._rh_eval_path
            )
            self._eigval_cache_key = cache_key
            self._eigval_cache_val = torch.from_numpy(ev_np)

        ev = self._eigval_cache_val
        return ev.to(device=device, dtype=dtype) if device is not None else ev.to(dtype=dtype)

    def clear_eigval_cache(self):

        self._eigval_cache_key = None
        self._eigval_cache_val = None




    def encode_full_grid(self, x: torch.Tensor, *,
                         eigval=None, D=None, lam_cortex=None,
                         ratio: float = 1e-3,
                         loss_mask: torch.Tensor | None = None) -> torch.Tensor:
        ev = self.resolve_eigval(eigval=eigval, D=D, lam_cortex=lam_cortex,
                                  ratio=ratio, device=x.device, dtype=x.dtype)
        return super().encode_full_grid(x, ev, loss_mask)

    def encode_pooled(self, x: torch.Tensor, *,
                      eigval=None, D=None, lam_cortex=None,
                      ratio: float = 1e-3,
                      loss_mask: torch.Tensor | None = None) -> torch.Tensor:
        ev = self.resolve_eigval(eigval=eigval, D=D, lam_cortex=lam_cortex,
                                  ratio=ratio, device=x.device, dtype=x.dtype)
        return super().encode_pooled(x, ev, loss_mask)

    def forward(self, x: torch.Tensor, *,
                eigval=None, D=None, lam_cortex=None,
                ratio: float = 1e-3,
                loss_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode_pooled(x, eigval=eigval, D=D, lam_cortex=lam_cortex,
                                   ratio=ratio, loss_mask=loss_mask)



EEGMEGMAEDecoderEigenval4 = EEGMEGMAEDecoderEigenval3
