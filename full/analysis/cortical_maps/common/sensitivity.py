import numpy as np
N_PC = 20
N_FOLD = 5
RIDGE = 1.0

def seed_r2(y_centered: np.ndarray, p: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(p))
    fold = np.empty(len(p), np.int8)
    fold[permutation] = np.arange(len(p)) % N_FOLD
    sse = np.zeros(y_centered.shape[1], np.float64)
    sst = np.zeros(y_centered.shape[1], np.float64)
    for fold_index in range(N_FOLD):
        train = fold != fold_index
        test = fold == fold_index
        x_train = p[train]
        operator = np.linalg.solve(x_train.T @ x_train + RIDGE * np.eye(N_PC), x_train.T).astype(np.float32)
        weights = operator @ y_centered[train]
        residual = y_centered[test] - p[test] @ weights
        sse += np.square(residual, dtype=np.float64).sum(axis=0)
        sst += np.square(y_centered[test], dtype=np.float64).sum(axis=0)
    return (1.0 - sse / (sst + 1e-12)).astype(np.float32)

def fold_operators(p: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(p))
    fold = np.empty(len(p), np.int8)
    fold[perm] = np.arange(len(p)) % N_FOLD
    operators = []
    for k in range(N_FOLD):
        tr, te = (fold != k, fold == k)
        xtr = p[tr]
        h = np.linalg.solve(xtr.T @ xtr + RIDGE * np.eye(p.shape[1]), xtr.T).astype(np.float32)
        operators.append((tr, te, h))
    return (fold, operators)

def cv_r2_matrix(y: np.ndarray, p: np.ndarray, operators) -> np.ndarray:
    yc = y.astype(np.float32, copy=False) - y.mean(axis=0, dtype=np.float64).astype(np.float32)
    sse = np.zeros(y.shape[1], np.float64)
    sst = np.zeros(y.shape[1], np.float64)
    for tr, te, h in operators:
        w = h @ yc[tr]
        resid = yc[te] - p[te] @ w
        sse += np.square(resid, dtype=np.float64).sum(axis=0)
        sst += np.square(yc[te], dtype=np.float64).sum(axis=0)
    return (1.0 - sse / (sst + 1e-12)).astype(np.float32)
