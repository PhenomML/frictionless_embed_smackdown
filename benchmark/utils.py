
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float32]
ArrayI = NDArray[np.int64]

try:
    import torch
except ImportError:
    torch = None

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def subsample_rows(
    x: ArrayF,
    y: Optional[ArrayI],
    max_points: Optional[int],
    seed: int,
    stratified: bool = True,
) -> tuple[ArrayF, Optional[ArrayI], Optional[ArrayI]]:
    n = x.shape[0]
    if max_points is None or n <= max_points:
        return x, y, None

    rng = np.random.default_rng(seed)

    if stratified and y is not None and np.issubdtype(y.dtype, np.integer):

        unique, inverse = np.unique(y, return_inverse=True)
        n_classes = len(unique)
        counts = np.bincount(inverse, minlength=n_classes)

        prop = max_points * counts / counts.sum()
        n_per_class = np.clip(np.round(prop).astype(int), 1, counts)
        total = n_per_class.sum()

        diff = max_points - total
        if diff != 0:
            c = int(np.argmax(counts))
            n_per_class[c] = np.clip(n_per_class[c] + diff, 0, counts[c])
        idx_list = []
        for c in range(n_classes):
            class_idx = np.where(inverse == c)[0]
            n_take = min(int(n_per_class[c]), len(class_idx))
            if n_take > 0:
                chosen = rng.choice(len(class_idx), size=n_take, replace=False)
                idx_list.extend(class_idx[chosen].tolist())
        idx = np.sort(np.array(idx_list))
    else:
        idx = rng.choice(n, size=max_points, replace=False)
        idx = np.sort(idx)

    x_sub = x[idx].astype(np.float32, copy=False)
    if y is None:
        y_sub = None
    elif np.issubdtype(y.dtype, np.integer):
        y_sub = y[idx].astype(np.int64, copy=False)
    else:
        y_sub = y[idx].astype(np.float32, copy=False)
    return x_sub, y_sub, idx.astype(np.int64)

def to_float32(x: NDArray) -> ArrayF:
    return np.asarray(x, dtype=np.float32)

def remove_outliers_by_norm(
    x: ArrayF,
    y: Optional[ArrayI],
    percentile: float = 99.9,
) -> tuple[ArrayF, Optional[ArrayI], int]:
    norms = np.linalg.norm(x, axis=1)
    thresh = np.percentile(norms, percentile)
    keep = norms <= thresh
    n_removed = int((~keep).sum())
    x_clean = x[keep].astype(np.float32, copy=False)
    y_clean = y[keep] if y is not None else None
    return x_clean, y_clean, n_removed

def verify_coil20_labels(y: NDArray) -> tuple[bool, str]:
    if y is None or len(y) == 0:
        return False, "no labels"
    unique, counts = np.unique(y, return_counts=True)
    n_classes = len(unique)
    if n_classes != 20:
        return False, f"expected 20 classes, got {n_classes}"
    if unique.min() < 0 or unique.max() > 19:
        return False, f"labels outside 0-19: min={unique.min()}, max={unique.max()}"
    n = len(y)
    if n == 1440 and not np.all(counts == 72):
        return False, f"expected 72 views/object at n=1440, got {counts[:3]}..."
    return True, f"OK: {n_classes} object IDs (0-19), n={n}"
