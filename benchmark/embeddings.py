
from __future__ import annotations

import warnings
from typing import Dict

import numpy as np

warnings.filterwarnings("ignore", message="n_jobs value .* overridden", category=UserWarning)
warnings.filterwarnings("ignore", message="Graph is not fully connected", category=UserWarning)
warnings.filterwarnings("ignore", message="zero-centering .* densifies", category=UserWarning)
from numpy.typing import NDArray
from scipy.linalg import eigh, svd
from sklearn.manifold import TSNE

from .config import EmbedConfig
from .utils import to_float32

ArrayF = NDArray[np.float32]

def embed_mds(x: ArrayF, cfg: EmbedConfig) -> ArrayF:
    n, d = x.shape

    x_centered = x - np.mean(x, axis=0, keepdims=True)

    gram = x_centered @ x_centered.T

    D_sq = np.add.outer(np.diag(gram), np.diag(gram)) - 2 * gram
    np.fill_diagonal(D_sq, 0)
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * (H @ D_sq @ H)

    evals, evecs = eigh(B)
    evals = np.real(evals)
    evecs = np.real(evecs)

    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    pos = evals > 1e-10
    evals = evals[pos]
    evecs = evecs[:, pos]
    k = min(2, len(evals))
    z = evecs[:, :k] * np.sqrt(evals[:k])
    return to_float32(z)

def embed_tsne(x: ArrayF, cfg: EmbedConfig) -> ArrayF:
    tsne = TSNE(
        n_components=2, perplexity=cfg.tsne_perplexity,
        learning_rate=cfg.tsne_learning_rate, max_iter=cfg.tsne_n_iter,
        init="pca", random_state=cfg.random_state, method="barnes_hut",
    )
    z = tsne.fit_transform(x)
    return to_float32(z)

def embed_umap(x: ArrayF, cfg: EmbedConfig) -> ArrayF:

    try:
        import umap
    except ImportError as exc:
        raise ImportError("Install umap-learn: pip install umap-learn") from exc

    reducer = umap.UMAP(
        n_components=2, n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist, random_state=cfg.random_state, metric="euclidean",
    )
    z = reducer.fit_transform(x)
    return to_float32(z)

def center(z: ArrayF) -> ArrayF:
    return to_float32(z - np.mean(z, axis=0, keepdims=True))

def normalize_embedding(z: ArrayF) -> ArrayF:
    z0 = center(z)
    rms = np.sqrt(np.mean(z0 ** 2))
    if rms < 1e-10:
        return z0
    return to_float32(z0 / rms)

def procrustes_align_to_reference(z: ArrayF, z_ref: ArrayF, allow_reflection: bool = True) -> ArrayF:
    z0 = center(z)
    y0 = center(z_ref)
    c = z0.T @ y0
    u, _, vt = svd(c)
    r = u @ vt
    if not allow_reflection and np.linalg.det(r) < 0:
        u = u.copy()
        u[:, -1] *= -1.0
        r = u @ vt
    return to_float32(z0 @ r)

def compute_all_embeddings(x50: ArrayF, cfg: EmbedConfig) -> Dict[str, ArrayF]:
    z_mds_raw = embed_mds(x50, cfg)
    z_tsne_raw = embed_tsne(x50, cfg)
    z_umap_raw = embed_umap(x50, cfg)
    z_mds = normalize_embedding(z_mds_raw)
    z_tsne = normalize_embedding(z_tsne_raw)
    z_umap = normalize_embedding(z_umap_raw)
    z_tsne_al = procrustes_align_to_reference(z_tsne, z_mds, allow_reflection=True)
    z_umap_al = procrustes_align_to_reference(z_umap, z_mds, allow_reflection=True)
    return {
        "mds_raw": z_mds_raw, "tsne_raw": z_tsne_raw, "umap_raw": z_umap_raw,
        "mds": z_mds, "tsne": z_tsne, "umap": z_umap,
        "tsne_aligned": z_tsne_al, "umap_aligned": z_umap_al,
    }
