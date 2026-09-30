from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata
from scipy.spatial.distance import pdist
import itertools

def column_corr(pred: np.ndarray, true: np.ndarray, chunk: int=8192, eps: float=1e-08) -> np.ndarray:
    out = np.zeros(pred.shape[1], dtype=np.float32)
    for lo in range(0, pred.shape[1], chunk):
        hi = min(pred.shape[1], lo + chunk)
        p = pred[:, lo:hi] - pred[:, lo:hi].mean(axis=0, keepdims=True)
        t = true[:, lo:hi] - true[:, lo:hi].mean(axis=0, keepdims=True)
        out[lo:hi] = (p * t).sum(axis=0) / (np.linalg.norm(p, axis=0) * np.linalg.norm(t, axis=0) + eps)
    return out

def unpatch_cift(patches: np.ndarray) -> np.ndarray:
    patches = patches[:, 0]
    lh = patches[:, :4, :7424].reshape(len(patches), 4 * 7424)
    rh = patches[:, 4:, :7429].reshape(len(patches), 4 * 7429)
    return np.concatenate([lh, rh], axis=1).astype(np.float32)

def ridge_affine(z: dict[str, np.ndarray], prefix: str) -> tuple[np.ndarray, np.ndarray]:
    coef = np.asarray(z[f'{prefix}_coef'], np.float32)
    xm = np.asarray(z[f'{prefix}_x_mean'], np.float32)
    ym = np.asarray(z[f'{prefix}_y_mean'], np.float32)
    return (coef, (ym - xm @ coef).astype(np.float32))

class DirectResidualMLP(torch.nn.Module):

    def __init__(self, input_dim: int=1280, hidden: int=512):
        super().__init__()
        self.norm = torch.nn.LayerNorm(input_dim)
        self.fc1 = torch.nn.Linear(input_dim, hidden)
        self.out_weight = torch.nn.Embedding(N_VERTICES, hidden, sparse=True)
        self.out_bias = torch.nn.Embedding(N_VERTICES, 1, sparse=True)
        torch.nn.init.zeros_(self.out_weight.weight)
        torch.nn.init.zeros_(self.out_bias.weight)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.fc1(self.norm(x)))

    def residual(self, hidden: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
        weight = self.out_weight(vertices)
        bias = self.out_bias(vertices).squeeze(-1)
        return hidden @ weight.T + bias

def direct_predict_vertices(model: DirectResidualMLP, x: np.ndarray, base_coef: torch.Tensor, base_bias: torch.Tensor, vertices: np.ndarray, device: torch.device, batch_size: int=256) -> np.ndarray:
    v = torch.from_numpy(np.asarray(vertices, np.int64)).to(device)
    out = np.empty((len(x), len(vertices)), np.float32)
    model.eval()
    with torch.no_grad():
        for lo in range(0, len(x), batch_size):
            hi = min(lo + batch_size, len(x))
            xb = torch.from_numpy(x[lo:hi]).to(device)
            base = xb @ base_coef[:, v] + base_bias[v]
            pred = base + model.residual(model.hidden(xb), v)
            out[lo:hi] = pred.cpu().numpy()
    return out

def predict_full_direct(model, x, base_coef, base_bias, device, vertex_chunk=2048):
    out = np.empty((len(x), N_VERTICES), np.float32)
    for lo in range(0, N_VERTICES, vertex_chunk):
        hi = min(lo + vertex_chunk, N_VERTICES)
        out[:, lo:hi] = direct_predict_vertices(model, x, base_coef, base_bias, np.arange(lo, hi), device)
    return out
N_VERTICES = 59412

def project_hemi(x, modes, device, row_chunk=128):
    m = torch.from_numpy(np.asarray(modes, np.float32)).to(device)
    gram = m.T @ m
    ridge = float(gram.diagonal().mean()) * 1e-07
    gram.diagonal().add_(ridge)
    chol = torch.linalg.cholesky(gram)
    out = np.empty((len(x), modes.shape[1]), np.float32)
    for lo in range(0, len(x), row_chunk):
        hi = min(lo + row_chunk, len(x))
        xb = torch.from_numpy(np.asarray(x[lo:hi], np.float32)).to(device)
        rhs = m.T @ xb.T
        out[lo:hi] = torch.cholesky_solve(rhs, chol).T.cpu().numpy()
    del m, gram, chol
    torch.cuda.empty_cache()
    return out

def unit_rank(x):
    x = rankdata(x).astype(np.float64)
    x -= x.mean()
    norm = np.linalg.norm(x)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError('Non-finite or constant RDM')
    return x / norm

def rdm(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite representation')
    d = pdist(x, metric='correlation')
    if not np.isfinite(d).all():
        raise ValueError('Constant spatial pattern in RDM')
    return d

def bh(p):
    p = np.asarray(p)
    idx = np.argsort(p)
    q = np.minimum.accumulate((p[idx] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q)
    out[idx] = np.minimum(q, 1)
    return out

def rank_gpu(values):
    import torch
    val, order = torch.sort(values, dim=1)
    i = torch.arange(values.shape[1], device=values.device).expand_as(order)
    first = torch.ones_like(val, dtype=torch.bool)
    last = torch.ones_like(first)
    first[:, 1:] = val[:, 1:] != val[:, :-1]
    last[:, :-1] = val[:, :-1] != val[:, 1:]
    starts = torch.where(first, i, 0).cummax(dim=1).values
    ends = torch.where(last, i, values.shape[1] - 1).flip([1]).cummin(dim=1).values.flip([1])
    rank = torch.empty_like(values)
    rank.scatter_(1, order, (starts.float() + ends.float()) / 2)
    rank -= (values.shape[1] - 1) / 2
    return rank / torch.linalg.vector_norm(rank, dim=1, keepdim=True)

def batch_rsa(brain, refs, centers, indices, indptr):
    import torch
    neighbors = [indices[indptr[v]:indptr[v + 1]] for v in centers]
    size = max(map(len, neighbors))
    ids = np.zeros((len(centers), size), dtype=np.int64)
    mask = np.zeros_like(ids, dtype=np.float32)
    for j, n in enumerate(neighbors):
        ids[j, :len(n)] = n
        mask[j, :len(n)] = 1
    ids = torch.as_tensor(ids, device=brain.device)
    mask = torch.as_tensor(mask, device=brain.device)[:, None, :]
    x = brain[:, ids].permute(1, 0, 2)
    x = (x - (x * mask).sum(2, keepdim=True) / mask.sum(2, keepdim=True)) * mask
    norm = torch.linalg.vector_norm(x, dim=2, keepdim=True)
    if torch.any(norm == 0):
        raise ValueError('Constant measured spatial pattern; explicit exclusion policy needed')
    x = x / norm
    corr = torch.bmm(x, x.transpose(1, 2))
    tri = torch.triu_indices(brain.shape[0], brain.shape[0], 1, device=brain.device)
    dis = 1 - corr[:, tri[0], tri[1]]
    ranks = rank_gpu(dis)
    return (ranks @ refs.T).cpu().numpy()

def p_signflip(x):
    signs = np.array(list(itertools.product((-1, 1), repeat=8)))
    out = np.empty(x.shape[1])
    for lo in range(0, x.shape[1], 4096):
        a = x[:, lo:lo + 4096].astype(float)
        null = abs(signs @ a / 8)
        out[lo:lo + 4096] = (null >= abs(a.mean(0))[None, :] - 1e-14).mean(0)
    return out

class Bridge(nn.Module):

    def __init__(self, nx, ny, mean, width=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nx, width), nn.GELU(), nn.Dropout(0.1), nn.Linear(width, ny))
        self.register_buffer('target_mean', mean)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x) + self.target_mean

def _band_engagement(absmaps, mask, lo, hi):

    def vals(am):
        return am[mask] if mask is not None else am
    scores = np.array([float(np.nanmean(np.abs(vals(am)))) for am in absmaps])
    valid = scores > 0
    rf = np.zeros(len(scores))
    if valid.sum() > 1:
        rf[valid] = np.argsort(np.argsort(scores[valid])) / (valid.sum() - 1)
    elif valid.sum() == 1:
        rf[valid] = 1.0
    K = lo + (hi - lo) * rf
    eng, thr = ([], [])
    for i, am in enumerate(absmaps):
        v = np.abs(vals(am))
        fp = float(np.mean(v > 0) * 100)
        if fp < 1 or not valid[i]:
            eng.append(0.0)
            thr.append(1000000000.0)
            continue
        k = min(K[i], fp)
        t = float(np.nanpercentile(v, 100 - k))
        if t <= 0:
            t = float(np.min(v[v > 0]))
        thr.append(max(t, 1e-12))
        eng.append(float(np.mean(v >= t) * 100))
    return (np.array(eng), thr)

def cv_r2(Y, P, trial, n_fold=5, seed=0):
    rng = np.random.default_rng(seed)
    Yc = Y - Y.mean(0)
    uniq = np.unique(trial)
    fmap = {t: i % n_fold for i, t in enumerate(rng.permutation(uniq))}
    fold = np.array([fmap[t] for t in trial])
    sse = np.zeros(Y.shape[1])
    sst = np.zeros(Y.shape[1])
    for f in range(n_fold):
        tr, te = (fold != f, fold == f)
        A = P[tr].T @ P[tr] + 1.0 * np.eye(P.shape[1])
        W = np.linalg.solve(A, P[tr].T @ Yc[tr])
        resid = Yc[te] - P[te] @ W
        sse += (resid ** 2).sum(0)
        sst += (Yc[te] ** 2).sum(0)
    return 1.0 - sse / (sst + 1e-12)

def subject_maps(cift_mean, modes_coeff, bge, trial, n_pc=40):
    B = bge - bge.mean(0)
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    Pc = B @ Vt[:n_pc].T
    Pc = (Pc - Pc.mean(0)) / (Pc.std(0) + 1e-08)
    r2_vx = np.clip(cv_r2(cift_mean, Pc, trial), 0, None)
    obs = np.sqrt(r2_vx)
    r2_md = np.clip(cv_r2(modes_coeff, Pc, trial), 0, None)
    rl, rr = (r2_md[:1000], r2_md[1000:])
    enl = PLc ** 2 @ rl / (PLc ** 2).sum(1).clip(1e-12)
    enr = PRc ** 2 @ rr / (PRc ** 2).sum(1).clip(1e-12)
    post = np.sqrt(np.concatenate([enl, enr]))
    return (obs, post)
