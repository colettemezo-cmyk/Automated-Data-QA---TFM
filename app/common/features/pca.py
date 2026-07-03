"""Frequency-weighted first principal component (no ONNX dependency)."""

from __future__ import annotations

import numpy as np


def _weighted_cov_eigh(
    unique_vectors: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, eigenvalues ascending, eigenvectors columns)."""
    counts_f = counts.astype(np.float64)
    total = counts_f.sum()
    dim = unique_vectors.shape[1]
    if total == 0:
        z = np.zeros(dim, dtype=np.float32)
        return z, np.zeros(dim, dtype=np.float64), np.eye(dim, dtype=np.float64)
    mean = (counts_f[:, None] * unique_vectors).sum(axis=0) / total
    centered = unique_vectors - mean
    cov = (counts_f[:, None] * centered).T @ centered / total
    eigvals, eigvecs = np.linalg.eigh(cov)
    return mean.astype(np.float32), eigvals, eigvecs


def explained_variance_ratios(eigvals: np.ndarray) -> np.ndarray:
    """Fraction of total variance per component (descending eigenvalue order)."""
    eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
    total = float(eigvals.sum())
    if total == 0:
        return np.zeros_like(eigvals)
    return eigvals / total


def first_pc_axis(unique_vectors: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted first PC axis and mean for embedding vectors.

    Returns (axis, mean) such that the per-row PC1 score is
    `(row_vector - mean) @ axis`.
    """
    mean, eigvals, eigvecs = _weighted_cov_eigh(unique_vectors, counts)
    if eigvals.sum() == 0:
        dim = unique_vectors.shape[1]
        return np.zeros(dim, dtype=np.float32), mean
    axis = eigvecs[:, -1].astype(np.float32)
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    return axis, mean


def first_pc_fit_stats(
    unique_vectors: np.ndarray, counts: np.ndarray, *, top_k: int = 5
) -> dict:
    """Axis/mean plus explained-variance ratios for classifier-style PC1."""
    mean, eigvals, eigvecs = _weighted_cov_eigh(unique_vectors, counts)
    ratios = explained_variance_ratios(eigvals)
    dim = unique_vectors.shape[1]
    if eigvals.sum() == 0:
        axis = np.zeros(dim, dtype=np.float32)
    else:
        axis = eigvecs[:, -1].astype(np.float32)
        if axis[int(np.argmax(np.abs(axis)))] < 0:
            axis = -axis
    k = min(top_k, len(ratios))
    return {
        "axis": axis,
        "mean": mean,
        "pc1_explained_variance_ratio": float(ratios[0]) if len(ratios) else 0.0,
        "top_explained_variance_ratios": [float(x) for x in ratios[:k]],
        "top_k_cumulative_explained_variance": float(ratios[:k].sum()) if k else 0.0,
        "n_unique_vectors": int(len(unique_vectors)),
        "n_weighted_rows": int(counts.sum()),
        "embedding_dim": int(dim),
    }


def _sign_normalise(axes: np.ndarray) -> np.ndarray:
    """Flip each axis so its max-|value| coordinate is positive.

    Matches `first_pc_axis`'s sign convention, applied per-axis. Keeps the
    sign of PC1 stable across reruns / random_state, which is what makes the
    serialised axis files diffable.
    """
    if axes.size == 0:
        return axes
    for i in range(axes.shape[0]):
        a = axes[i]
        pivot = int(np.argmax(np.abs(a)))
        if a[pivot] < 0:
            axes[i] = -a
    return axes


def top_k_pc_axes(
    unique_vectors: np.ndarray,
    counts: np.ndarray,
    *,
    variance_target: float | None = None,
    fixed_k: int | None = None,
    k_max: int | None = None,
    k_min: int = 1,
) -> dict:
    """Weighted top-K principal components.

    Exactly one of `variance_target` and `fixed_k` must be set:
      * `variance_target` (adaptive) — pick the smallest K such that the
        cumulative explained variance is >= the target, clamped to
        `[k_min, min(k_max or rank, rank)]`.
      * `fixed_k` (constant) — use exactly that many components, clamped
        to `[1, min(rank, dim)]`.

    Returns a dict with:
      * `axes`  : (k, dim) float32, each ROW is a PC axis (descending by eigenvalue).
                  Project a row vector with `(vec - mean) @ axes.T`.
      * `mean`  : (dim,) float32.
      * `explained_variance_ratios` : (k,) float32, descending.
      * `cumulative_explained_variance` : float, sum of the K ratios actually kept.
      * `k`     : int, the K actually chosen (post-clamp).
      * `k_requested` : int | None, K that would meet variance_target without clamp
                        (None when fixed_k mode).
      * `variance_target_met` : bool — false when capped below the requested K.
      * `embedding_dim` : int.
      * `n_unique_vectors` : int.
      * `n_weighted_rows` : int.
    """
    if (variance_target is None) == (fixed_k is None):
        raise ValueError(
            "top_k_pc_axes requires exactly one of `variance_target` or `fixed_k`."
        )

    mean, eigvals, eigvecs = _weighted_cov_eigh(unique_vectors, counts)
    ratios = explained_variance_ratios(eigvals)
    dim = int(unique_vectors.shape[1]) if unique_vectors.size else int(mean.shape[0])

    # Effective rank (positive eigenvalues only). Anything beyond this is
    # numerical noise; do not waste K on it.
    rank = int(np.count_nonzero(eigvals > 0))
    if rank == 0:
        return {
            "axes": np.zeros((0, dim), dtype=np.float32),
            "mean": mean,
            "explained_variance_ratios": np.zeros(0, dtype=np.float32),
            "cumulative_explained_variance": 0.0,
            "k": 0,
            "k_requested": 0,
            "variance_target_met": False,
            "embedding_dim": dim,
            "n_unique_vectors": int(len(unique_vectors)),
            "n_weighted_rows": int(counts.sum()),
        }

    hard_cap = rank if k_max is None else min(int(k_max), rank)
    hard_cap = max(hard_cap, 1)

    if fixed_k is not None:
        k_requested = int(fixed_k)
        k = max(k_min, min(k_requested, hard_cap))
        variance_target_met = True  # n/a for fixed mode
    else:
        cum = np.cumsum(ratios)
        # `searchsorted` returns the first index where cum >= target; +1 gives count.
        k_requested = int(np.searchsorted(cum, float(variance_target)) + 1)
        k_requested = max(k_min, min(k_requested, rank))
        k = min(k_requested, hard_cap)
        achieved = float(cum[k - 1]) if k > 0 else 0.0
        variance_target_met = achieved + 1e-9 >= float(variance_target)

    # eigvecs columns are ascending; reverse + slice for descending top-K.
    axes_cols = eigvecs[:, ::-1][:, :k]  # (dim, k)
    axes = axes_cols.T.astype(np.float32).copy()  # (k, dim), row-major projection
    axes = _sign_normalise(axes)

    kept_ratios = ratios[:k].astype(np.float32)
    return {
        "axes": axes,
        "mean": mean,
        "explained_variance_ratios": kept_ratios,
        "cumulative_explained_variance": float(kept_ratios.sum()),
        "k": int(k),
        "k_requested": int(k_requested),
        "variance_target_met": bool(variance_target_met),
        "embedding_dim": dim,
        "n_unique_vectors": int(len(unique_vectors)),
        "n_weighted_rows": int(counts.sum()),
    }
