import numpy as np

def cv_r2(y: np.ndarray, predictors: np.ndarray, trial: np.ndarray, n_fold: int=5, seed: int=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yc = y - y.mean(0)
    uniq = np.unique(trial)
    fold_map = {t: i % n_fold for i, t in enumerate(rng.permutation(uniq))}
    fold = np.array([fold_map[t] for t in trial])
    sse = np.zeros(y.shape[1], dtype=np.float64)
    sst = np.zeros(y.shape[1], dtype=np.float64)
    for f in range(n_fold):
        train, test = (fold != f, fold == f)
        a = predictors[train].T @ predictors[train] + np.eye(predictors.shape[1])
        w = np.linalg.solve(a, predictors[train].T @ yc[train])
        resid = yc[test] - predictors[test] @ w
        sse += (resid ** 2).sum(0)
        sst += (yc[test] ** 2).sum(0)
    return 1.0 - sse / (sst + 1e-12)

def semantic_maps(cift_mean: np.ndarray, modes_coeff: np.ndarray, bge: np.ndarray, n_pc: int, pl2: np.ndarray, pr2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = bge - bge.mean(0)
    _, _, vt = np.linalg.svd(b, full_matrices=False)
    pc = b @ vt[:n_pc].T
    pc = (pc - pc.mean(0)) / (pc.std(0) + 1e-08)
    trial = np.arange(cift_mean.shape[0])
    r2_vertex = np.clip(cv_r2(cift_mean, pc, trial), 0, None)
    low = np.sqrt(r2_vertex)
    r2_mode = np.clip(cv_r2(modes_coeff, pc, trial), 0, None)
    high_l = pl2 @ r2_mode[:1000]
    high_r = pr2 @ r2_mode[1000:]
    high = np.sqrt(np.concatenate([high_l, high_r]))
    return (low.astype(np.float32), high.astype(np.float32))

def local_mask(centroids: np.ndarray, k: int=40) -> np.ndarray:
    distances = np.linalg.norm(centroids[:, None] - centroids[None], axis=2)
    mask = np.zeros_like(distances, dtype=bool)
    for j in range(len(distances)):
        mask[np.argsort(distances[j])[:k], j] = True
    return mask

def fc_from_cortical_ts(cort_lh: np.ndarray, parcel_matrix: np.ndarray) -> np.ndarray:
    parcel_ts = parcel_matrix @ cort_lh
    parcel_ts = (parcel_ts - parcel_ts.mean(1, keepdims=True)) / (parcel_ts.std(1, keepdims=True) + 1e-08)
    return parcel_ts @ parcel_ts.T / parcel_ts.shape[1]

def actual_and_predicted(vertex_map: np.ndarray, fc_local: np.ndarray, parcel_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    map_z = (vertex_map - vertex_map.mean()) / (vertex_map.std() + 1e-08)
    actual = parcel_matrix @ map_z
    actual = (actual - actual.mean()) / (actual.std() + 1e-08)
    predicted = fc_local.T @ actual
    return (actual.astype(np.float32), predicted.astype(np.float32))

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])
