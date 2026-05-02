#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import numpy as np

from benchmark.utils import ensure_dir, to_float32

def _fetch_via_census(max_cells: int | None = 5000, seed: int = 123) -> "tuple":
    try:
        import cellxgene_census
    except ImportError:
        raise ImportError(
            "Install cellxgene-census: pip install cellxgene-census\n"
            "See https://chanzuckerberg.github.io/cellxgene-census/"
        )

    import scanpy as sc

    print("Opening CELLxGENE Census...")
    census = cellxgene_census.open_soma()
    try:

        print("Querying Mus musculus retina...")
        adata = cellxgene_census.get_anndata(
            census=census,
            organism="Mus musculus",
            obs_value_filter="tissue == 'retina' and is_primary_data == True",
            obs_column_names=["tissue", "cell_type", "assay"],
        )
    finally:
        census.close()

    if adata.n_obs == 0:

        census = cellxgene_census.open_soma()
        try:
            adata = cellxgene_census.get_anndata(
                census=census,
                organism="Mus musculus",
                obs_value_filter="tissue_general == 'retina' and is_primary_data == True",
                obs_column_names=["tissue", "tissue_general", "cell_type"],
            )
        finally:
            census.close()

    if adata.n_obs == 0:
        raise RuntimeError(
            "No mouse retina cells found in Census. "
            "Try browsing https://cellxgene.cziscience.com/datasets and filtering for retina, "
            "then download h5ad manually and run: python scripts/build_mouse_retina_from_h5ad.py <path>"
        )

    print(f"Fetched {adata.n_obs} cells, {adata.n_vars} genes")

    if max_cells is not None and adata.n_obs > max_cells:
        np.random.seed(seed)
        idx = np.random.choice(adata.n_obs, max_cells, replace=False)
        adata = adata[idx].copy()
        print(f"Subsampled to {max_cells} cells")

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.scale(adata, max_value=10)
    n_neighbors = 15
    sc.tl.pca(adata, svd_solver="arpack", n_comps=50)
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=n_neighbors, random_state=seed)
    sc.tl.leiden(
        adata, resolution=1.0, random_state=seed,
        flavor="igraph", n_iterations=2, directed=False,
    )

    y = np.array(adata.obs["leiden"].astype(int), dtype=np.int64)
    x = to_float32(adata.obsm["X_pca"])
    n_clusters = len(np.unique(y))

    meta = {
        "name": "Mouse Retina",
        "modality": "scRNA-seq",
        "source": "CELLxGENE Census (Mus musculus, tissue=retina)",
        "label_type": "derived",
        "label_source": "leiden_on_pca50",
        "neighbors_params": {"n_neighbors": n_neighbors, "use_rep": "X_pca", "random_state": seed},
        "leiden_params": {"resolution": 1.0, "random_state": seed, "flavor": "igraph", "n_iterations": 2, "directed": False},
        "class_names": [f"cluster_{i}" for i in range(n_clusters)],
        "build_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    return x, y, meta

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Fetch mouse retina data and build corpus")
    parser.add_argument("--max-cells", type=int, default=5000, help="Max cells (subsample if larger)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    corpus_dir = root / "corpus" / "mouse_retina"
    ensure_dir(corpus_dir)

    x, y, meta = _fetch_via_census(max_cells=args.max_cells, seed=args.seed)

    np.save(corpus_dir / "X_pca50.npy", x)
    np.save(corpus_dir / "y.npy", y)
    meta["original_dim"] = 2000
    meta["preprocessing"] = ["normalize_total(1e4)", "log1p", "HVG(2000)", "zscore", "PCA->50"]
    with open(corpus_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Built {corpus_dir}")
    print(f"  X_pca50: {x.shape}, y: {y.shape}, n_clusters: {len(np.unique(y))}")
    print("Done.")

if __name__ == "__main__":
    main()
