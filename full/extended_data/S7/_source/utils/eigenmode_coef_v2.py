



















from __future__ import annotations
from typing import Optional

import numpy as np

from utils.eigenmode_coef import (
    ArrayLike,
    build_psi,
    project_to_modes,
    reconstruct_from_modes,
    eval_reconstruction,
)




def compute_effective_eigenvalues(lambda_cortex: np.ndarray,
                                  Vt: np.ndarray,
                                  sigma: Optional[np.ndarray] = None,
                                  squared: bool = True) -> np.ndarray:






















    lambda_cortex = np.asarray(lambda_cortex, dtype=np.float64)
    Vt = np.asarray(Vt, dtype=np.float64)
    if lambda_cortex.ndim != 1 or Vt.ndim != 2:
        raise ValueError("lambda_cortex must be 1-D and Vt must be 2-D")
    if Vt.shape[1] != lambda_cortex.shape[0]:
        raise ValueError(
            f"Vt has {Vt.shape[1]} columns but lambda_cortex has {lambda_cortex.shape[0]} entries")

    if squared:
        lambda_eff = (Vt ** 2) @ lambda_cortex
    else:
        lambda_eff = Vt @ lambda_cortex
        if sigma is not None:
            lambda_eff = np.asarray(sigma, dtype=np.float64) * lambda_eff
    return lambda_eff




def compute_modes_v2(eeg: np.ndarray,
                     D: ArrayLike,
                     lambda_cortex: Optional[np.ndarray] = None,
                     ratio: float = 1e-3,
                     K_max: Optional[int] = None,
                     squared: bool = True,
                     evaluate: bool = False) -> dict:




























    Psi, K_eff, sigma, Vt = build_psi(D, ratio=ratio, K_max=K_max, return_Vt=True)
    modes = project_to_modes(eeg, Psi)

    eigval = None
    if lambda_cortex is not None:
        eigval = compute_effective_eigenvalues(
            lambda_cortex, Vt, sigma=sigma, squared=squared,
        ).astype(np.float32)

    out = {
        "Psi":    Psi,
        "modes":  modes,
        "eigval": eigval,
        "sigma":  sigma,
        "K_eff":  K_eff,
        "Vt":     Vt,
    }
    if evaluate:
        out["report"] = eval_reconstruction(eeg, modes, Psi)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_sensor, K_cortex, T, N = 60, 48, 2560, 4
    D_fake   = rng.standard_normal((n_sensor, K_cortex)).astype(np.float32)
    eeg_fake = rng.standard_normal((N, n_sensor, T)).astype(np.float32)
    lam_fake = np.sort(rng.uniform(0, 500, K_cortex))

    res = compute_modes_v2(
        eeg_fake, D_fake,
        lambda_cortex=lam_fake,
        ratio=1e-2,
        evaluate=True,
    )
    print(f"\n[V2]  Psi={res['Psi'].shape}  modes={res['modes'].shape}  "
          f"eigval={res['eigval'].shape}  K_eff={res['K_eff']}")
    print(f"      eigval range=[{res['eigval'].min():.3g} .. {res['eigval'].max():.3g}]")


    res_lin = compute_modes_v2(
        eeg_fake, D_fake,
        lambda_cortex=lam_fake,
        ratio=1e-2,
        squared=False,
    )
    print(f"[V2 linear+σ]  eigval range=[{res_lin['eigval'].min():.3g} .. "
          f"{res_lin['eigval'].max():.3g}]")
