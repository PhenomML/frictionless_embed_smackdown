from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.config import EmbedConfig, get_default_paths
from benchmark.embeddings import compute_all_embeddings
from benchmark.figure_io import save_figure_png_pdf
from benchmark.loaders import load_dataset
from benchmark.utils import subsample_rows, to_float32


datasets = [
    ("mnist", "MNIST"),
    ("uci_har", "UCI HAR"),
    ("cifar10", "CIFAR-10"),
    ("ag_news", "AG News"),
    ("olivetti_faces", "Olivetti Faces"),
    ("20newsgroups", "20 Newsgroups"),
]


def _class_cmap(n_classes: int):
    if n_classes <= 10:
        return plt.cm.tab10
    if n_classes <= 20:
        return plt.cm.tab20
    stack = np.vstack([plt.cm.tab20(np.linspace(0, 1, 20)), plt.cm.tab20b(np.linspace(0, 1, 20)), plt.cm.tab20c(np.linspace(0, 1, 20))])
    return mcolors.ListedColormap(stack[:n_classes])


def plot_embedding_gallery(root: Path, figures_dir: Path, tables_dir: Path) -> Path:
    paths = get_default_paths()
    cfg = EmbedConfig(random_state=123, pca_dim=50, tsne_perplexity=30.0, tsne_learning_rate=200.0, tsne_n_iter=1000, umap_n_neighbors=15, umap_min_dist=0.1, max_points=3000)
    rows = []
    for dataset_id, display_name in datasets:
        x, y, meta = load_dataset(dataset_id, paths["data_root"], paths["corpus_dir"], pca_dim=cfg.pca_dim, subsample=None, use_corpus_when_available=True, seed=cfg.random_state, use_standardize=True)
        x, y, _ = subsample_rows(x, y, cfg.max_points, cfg.random_state)
        rows.append((dataset_id, display_name, compute_all_embeddings(to_float32(x), cfg), y, meta))

    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 2.2 * len(rows)), constrained_layout=True)
    for j, title in enumerate(["MDS", "t-SNE (aligned to MDS)", "UMAP (aligned to MDS)"]):
        axes[0, j].set_title(title, fontsize=11)
    for i, (dataset_id, display_name, embs, y, _) in enumerate(rows):
        panels = [embs["mds"], embs["tsne_aligned"], embs["umap_aligned"]]
        if y is not None and len(y) == len(panels[0]):
            y_arr = np.asarray(y)
            if np.issubdtype(y_arr.dtype, np.number):
                cvals = y_arr
                cmap = "viridis" if np.issubdtype(y_arr.dtype, np.floating) else _class_cmap(len(np.unique(y_arr)))
            else:
                cvals, uniques = pd.factorize(y_arr)
                cmap = _class_cmap(len(uniques))
        else:
            cvals = None
            cmap = None
        for j, z in enumerate(panels):
            ax = axes[i, j]
            if cvals is None:
                ax.scatter(z[:, 0], z[:, 1], s=2, alpha=0.75, rasterized=True)
            else:
                ax.scatter(z[:, 0], z[:, 1], c=cvals, s=2, alpha=0.75, cmap=cmap, rasterized=True)
            if j == 0:
                ax.set_ylabel(display_name, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

    meta_rows = [
        {
            "dataset_id": dataset_id,
            "dataset_name": display_name,
            "display_random_state": cfg.random_state,
            "max_points": cfg.max_points,
            "tsne_perplexity": cfg.tsne_perplexity,
            "umap_n_neighbors": cfg.umap_n_neighbors,
            "alignment_reference": "MDS",
            "alignment_note": "t-SNE and UMAP aligned to MDS for display only",
        }
        for dataset_id, display_name, _, _, _ in rows
    ]
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(meta_rows).to_csv(tables_dir / "fig1_embedding_gallery_metadata.csv", index=False)
    return save_figure_png_pdf(fig, figures_dir, "fig1_embedding_gallery_main.png")
