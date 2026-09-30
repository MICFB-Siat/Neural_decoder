






















from __future__ import annotations
from typing import Optional, Union
import numpy as np

ArrayLike = Union[np.ndarray, str]


def _load_phi(D: ArrayLike) -> np.ndarray:
    if isinstance(D, str):
        return np.load(D).astype(np.float64)
    return np.asarray(D, dtype=np.float64)


def build_psi(D: ArrayLike,
              ratio: float = 1e-2,
              K_max: Optional[int] = None,
              return_Vt: bool = False):















    Phi = _load_phi(D)
    if Phi.ndim != 2:
        raise ValueError(f"D must be 2-D (n_sensor × K), got shape {Phi.shape}")

    U, S, Vt = np.linalg.svd(Phi, full_matrices=False)
    K_eff = int((S / S[0] >= ratio).sum())
    if K_max is not None:
        K_eff = min(K_eff, K_max)
    if K_eff == 0:
        raise RuntimeError("ratio too large — no components survived truncation")

    Psi = U[:, :K_eff].astype(np.float32)
    if return_Vt:
        return Psi, K_eff, S[:K_eff], Vt[:K_eff, :]
    return Psi, K_eff, S[:K_eff]


def project_to_modes(eeg: np.ndarray, Psi: np.ndarray) -> np.ndarray:









    if eeg.shape[-2] != Psi.shape[0]:
        raise ValueError(
            f"sensor dim mismatch: eeg has {eeg.shape[-2]} sensors, "
            f"Ψ expects {Psi.shape[0]}")
    PsiT = Psi.T.astype(np.float32)
    return np.matmul(PsiT, eeg.astype(np.float32))


def reconstruct_from_modes(coef: np.ndarray, Psi: np.ndarray) -> np.ndarray:

    return np.matmul(Psi.astype(np.float32), coef.astype(np.float32))




def eval_reconstruction(eeg: np.ndarray,
                        coef: np.ndarray,
                        Psi: np.ndarray,
                        verbose: bool = True) -> dict:



















    y     = eeg.astype(np.float32)
    y_hat = reconstruct_from_modes(coef, Psi)
    res   = y - y_hat


    num = float((res ** 2).sum())
    den = float((y ** 2).sum() + 1e-12)
    ve_global = 1.0 - num / den



    y_ch  = np.moveaxis(y,   -2, 0).reshape(y.shape[-2], -1)
    yh_ch = np.moveaxis(y_hat, -2, 0).reshape(y.shape[-2], -1)
    res_ch = y_ch - yh_ch
    var_y  = (y_ch  ** 2).sum(axis=1) + 1e-12
    var_r  = (res_ch ** 2).sum(axis=1)
    ve_ch  = 1.0 - var_r / var_y


    yc  = y_ch  - y_ch.mean(axis=1, keepdims=True)
    yhc = yh_ch - yh_ch.mean(axis=1, keepdims=True)
    num_c = (yc * yhc).sum(axis=1)
    den_c = np.sqrt((yc ** 2).sum(axis=1) * (yhc ** 2).sum(axis=1)) + 1e-12
    corr_ch = num_c / den_c

    rmse = float(np.sqrt(num / y.size))


    G = Psi.astype(np.float64).T @ Psi.astype(np.float64)
    orth_err = float(np.linalg.norm(G - np.eye(G.shape[0]), ord='fro'))
    cond = float(np.linalg.cond(Psi))

    out = {
        "K_eff":              int(Psi.shape[1]),
        "n_sensor":           int(Psi.shape[0]),
        "var_explained_global": float(ve_global),
        "var_explained_mean":   float(ve_ch.mean()),
        "var_explained_min":    float(ve_ch.min()),
        "corr_mean":            float(corr_ch.mean()),
        "corr_min":             float(corr_ch.min()),
        "rmse_global":          rmse,
        "cond_basis":           cond,
        "orth_err":             orth_err,
    }
    if verbose:
        print(f"\n[eval_reconstruction]  K'={out['K_eff']}/{out['n_sensor']} sensors")
        print(f"  variance explained (global): {out['var_explained_global']*100:6.2f}%")
        print(f"  variance explained (mean ch): {out['var_explained_mean']*100:6.2f}%   "
              f"(worst ch: {out['var_explained_min']*100:6.2f}%)")
        print(f"  Pearson corr (mean ch):       {out['corr_mean']:+.4f}    "
              f"(worst ch: {out['corr_min']:+.4f})")
        print(f"  RMSE (global):                {out['rmse_global']:.4g}")
        print(f"  basis: cond={out['cond_basis']:.3f}  ‖ΨᵀΨ-I‖_F={out['orth_err']:.2e}")
    return out




def compute_modes(eeg: np.ndarray,
                  D: ArrayLike,
                  ratio: float = 1e-3,
                  K_max: Optional[int] = None,
                  evaluate: bool = False) -> dict:












    Psi, K_eff, sigma = build_psi(D, ratio=ratio, K_max=K_max)
    coef = project_to_modes(eeg, Psi)
    out = {"Psi": Psi, "coef": coef, "K_eff": K_eff, "sigma": sigma}
    if evaluate:
        out["report"] = eval_reconstruction(eeg, coef, Psi)
    return out


if __name__ == "__main__":

    rng = np.random.default_rng(0)
    n_sensor, K, T, N = 60, 48, 2560, 4
    D_fake = rng.standard_normal((n_sensor, K)).astype(np.float32)
    eeg_fake = rng.standard_normal((N, n_sensor, T)).astype(np.float32)
    res = compute_modes(eeg_fake, D_fake, ratio=1e-2, evaluate=True)
    print(f"\nshapes — Psi: {res['Psi'].shape}  coef: {res['coef'].shape}")
