import numpy as np
from scipy import linalg
from sklearn.linear_model import LinearRegression


def calc_eigendecomposition(data, eigenvectors, method='matrix'):
























































    if data.ndim == 1:
        data = data.reshape(-1, 1)

    M, P = data.shape
    _, N = eigenvectors.shape


    if eigenvectors.shape[0] != M:
        raise ValueError(
            f"First dimension of data ({M}) must match first dimension of eigenvectors ({eigenvectors.shape[0]})")

    if method == 'matrix':


        coeffs = linalg.solve(
            eigenvectors.T @ eigenvectors,
            eigenvectors.T @ data,
            assume_a='pos'
        )

    elif method == 'matrix_separate':

        coeffs = np.zeros((N, P))
        gram_matrix = eigenvectors.T @ eigenvectors

        for p in range(P):
            coeffs[:, p] = linalg.solve(
                gram_matrix,
                eigenvectors.T @ data[:, p],
                assume_a='pos'
            )

    elif method == 'regression':

        coeffs = np.zeros((N, P))
        model = LinearRegression(fit_intercept=False)

        for p in range(P):
            model.fit(eigenvectors, data[:, p])
            coeffs[:, p] = model.coef_

    elif method == 'lstsq':

        coeffs, residuals, rank, s = np.linalg.lstsq(eigenvectors, data, rcond=None)

    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'matrix', 'matrix_separate', 'regression', or 'lstsq'")


    if P == 1:
        coeffs = coeffs.ravel()

    return coeffs


def reconstruct_from_eigenmodes(eigenvectors, coeffs):















    return eigenvectors @ coeffs



if __name__ == "__main__":

    num_vertices = 1000
    num_modes = 50
    num_timepoints = 100


    eigenmodes = np.random.randn(num_vertices, num_modes)
    eigenmodes, _ = np.linalg.qr(eigenmodes)


    print("Example 1: Reconstructing spatial map")
    true_coeffs_spatial = np.random.randn(num_modes)
    brain_activity_spatial = eigenmodes @ true_coeffs_spatial
    brain_activity_spatial += 0.1 * np.random.randn(num_vertices)


    fitted_coeffs = calc_eigendecomposition(brain_activity_spatial, eigenmodes, method='matrix')
    reconstructed = reconstruct_from_eigenmodes(eigenmodes, fitted_coeffs)


    corr = np.corrcoef(brain_activity_spatial, reconstructed)[0, 1]
    print(f"Reconstruction correlation: {corr:.4f}")


    print("\nExample 2: Reconstructing time series")
    true_coeffs_temporal = np.random.randn(num_modes, num_timepoints)
    brain_activity_temporal = eigenmodes @ true_coeffs_temporal
    brain_activity_temporal += 0.1 * np.random.randn(num_vertices, num_timepoints)


    fitted_coeffs_temporal = calc_eigendecomposition(brain_activity_temporal, eigenmodes, method='matrix')
    reconstructed_temporal = reconstruct_from_eigenmodes(eigenmodes, fitted_coeffs_temporal)


    corrs = [np.corrcoef(brain_activity_temporal[:, t], reconstructed_temporal[:, t])[0, 1]
             for t in range(num_timepoints)]
    print(f"Mean reconstruction correlation across time: {np.mean(corrs):.4f}")


    print("\nExample 3: Comparing methods")
    methods = ['matrix', 'matrix_separate', 'lstsq', 'regression']
    for method in methods:
        coeffs = calc_eigendecomposition(brain_activity_spatial, eigenmodes, method=method)
        recon = reconstruct_from_eigenmodes(eigenmodes, coeffs)
        corr = np.corrcoef(brain_activity_spatial, recon)[0, 1]
        print(f"{method:20s}: correlation = {corr:.6f}")
