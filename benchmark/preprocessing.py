
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

from .utils import to_float32

ArrayF = NDArray[np.float32]

def pca_reduce(x: ArrayF, n_components: int, seed: int) -> ArrayF:
    if x.shape[1] <= n_components:
        return x
    pca = PCA(n_components=n_components, random_state=seed)
    return to_float32(pca.fit_transform(x))

def svd_reduce_sparse(x, n_components: int, seed: int) -> ArrayF:
    if x.shape[1] <= n_components:
        return to_float32(x.toarray())
    svd_model = TruncatedSVD(n_components=n_components, random_state=seed)
    z = svd_model.fit_transform(x)
    return to_float32(z)

def standardize(x: ArrayF) -> ArrayF:
    from scipy.sparse import issparse
    use_mean = not issparse(x)
    scaler = StandardScaler(with_mean=use_mean, with_std=True)
    return to_float32(scaler.fit_transform(x))
