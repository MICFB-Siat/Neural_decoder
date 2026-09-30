






from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EigenmodeResult:
    scores: np.ndarray
    eigenvalues: np.ndarray
    components: np.ndarray


def _top_eigh(matrix: np.ndarray, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1][:n_modes]
    return values[order].astype(np.float32), vectors[:, order].astype(np.float32)


def compute_sensor_eigenmodes(
    data: np.ndarray, n_modes: int, center: bool = True
) -> EigenmodeResult:





    values = np.asarray(data, dtype=np.float64)
    if values.ndim == 2:
        flat = values
        leading = None
    elif values.ndim == 3:
        flat = values.transpose(1, 0, 2).reshape(values.shape[1], -1)
        leading = values.shape[0]
    else:
        raise ValueError("sensor data must have shape [channels,time] or [trials,channels,time]")
    if flat.shape[1] < 2:
        raise ValueError("at least two observations are required")
    flat = flat - flat.mean(axis=1, keepdims=True) if center else flat
    covariance = flat @ flat.T / (flat.shape[1] - 1)
    eigenvalues, components = _top_eigh(covariance, min(n_modes, flat.shape[0]))
    projected = np.asarray(data, dtype=np.float64)
    if center:
        projected = projected - projected.mean(axis=-1, keepdims=True)
    scores = np.einsum("mc,tcq->tmq", components.T, projected) if projected.ndim == 3 else components.T @ projected
    return EigenmodeResult(scores=scores.astype(np.float32), eigenvalues=eigenvalues,
                           components=components)


def compute_eeg_eigenmodes(data: np.ndarray, n_modes: int) -> EigenmodeResult:

    return compute_sensor_eigenmodes(data, n_modes)


def compute_meg_eigenmodes(data: np.ndarray, n_modes: int) -> EigenmodeResult:

    return compute_sensor_eigenmodes(data, n_modes)


def compute_fmri_eigenmodes(
    data: np.ndarray, adjacency: np.ndarray, n_modes: int, normalized: bool = True
) -> EigenmodeResult:






    signals = np.asarray(data, dtype=np.float64)
    graph = np.asarray(adjacency, dtype=np.float64)
    if signals.ndim != 2 or graph.ndim != 2 or graph.shape[0] != graph.shape[1] != signals.shape[1]:
        raise ValueError("fMRI data must be [observations,vertices] and adjacency must be square")
    degree = graph.sum(axis=1)
    laplacian = np.diag(degree) - graph
    if normalized:
        inv_sqrt = np.zeros_like(degree)
        nonzero = degree > 0
        inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
        laplacian = inv_sqrt[:, None] * laplacian * inv_sqrt[None, :]
    values, vectors = np.linalg.eigh(laplacian)
    start = 1 if len(values) > 1 and abs(values[0]) < 1e-8 else 0
    stop = min(start + n_modes, len(values))
    eigenvalues = values[start:stop].astype(np.float32)
    components = vectors[:, start:stop].astype(np.float32)
    centered = signals - signals.mean(axis=0, keepdims=True)
    scores = centered @ components
    return EigenmodeResult(scores=scores.astype(np.float32), eigenvalues=eigenvalues,
                           components=components)


def compute_svd_modes(data: np.ndarray, n_modes: int, center: bool = True) -> EigenmodeResult:






    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("SVD data must have shape [observations,features]")
    centered = values - values.mean(axis=0, keepdims=True) if center else values
    u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    k = min(n_modes, singular.shape[0])
    scores = u[:, :k] * singular[:k]
    eigenvalues = (singular[:k] ** 2 / max(values.shape[0] - 1, 1)).astype(np.float32)
    return EigenmodeResult(scores=scores.astype(np.float32), eigenvalues=eigenvalues,
                           components=vh[:k].T.astype(np.float32))


def eigenvalue_positional_encoding(
    eigenvalues: np.ndarray, dimension: int, base: float = 10000.0
) -> np.ndarray:

    values = np.asarray(eigenvalues, dtype=np.float32).reshape(-1)
    if dimension < 1:
        raise ValueError("dimension must be positive")
    scale = (values - values.mean()) / (values.std() + 1e-6)
    frequencies = np.exp(-np.log(base) * np.arange(0, dimension, 2) / dimension)
    angles = scale[:, None] * frequencies[None, :]
    encoding = np.zeros((len(values), dimension), dtype=np.float32)
    encoding[:, 0::2] = np.sin(angles).astype(np.float32)
    encoding[:, 1::2] = np.cos(angles[:, :encoding[:, 1::2].shape[1]]).astype(np.float32)
    return encoding
