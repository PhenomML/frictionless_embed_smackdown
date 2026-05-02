#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import numpy as np
import scanpy as sc

from benchmark.utils import ensure_dir, to_float32

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build mouse_retina corpus from h5ad")
    parser.add_argument("h5ad_path", type=Path, help="Path to mouse retina h5ad file")
    parser.add_argument("--max-cells", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    h5ad = Path(args.h5ad_path)
    if not h5ad.exists():
        print(f"File not found: {h5ad}")
        sys.exit(1)

    corpus_dir = root / "corpus" / "mouse_retina"
    ensure_dir(corpus_dir)

    print(f"Loading {h5ad}...")
    adata = sc.read_h5ad(h5ad)

    if adata.n_obs > args.max_cells:
        np.random.seed(args.seed)
        idx = np.random.choice(adata.n_obs, args.max_cells, replace=False)
        adata = adata[idx].copy()
        print(f"Subsampled to {args.max_cells} cells")

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.scale(adata, max_value=10)
    n_neighbors = 15
    sc.tl.pca(adata, svd_solver="arpack", n_comps=50)
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=n_neighbors, random_state=args.seed)
    sc.tl.leiden(
        adata, resolution=1.0, random_state=args.seed,
        flavor="igraph", n_iterations=2, directed=False,
    )

    y = np.array(adata.obs["leiden"].astype(int), dtype=np.int64)
    x = to_float32(adata.obsm["X_pca"])
    n_clusters = len(np.unique(y))

    np.save(corpus_dir / "X_pca50.npy", x)
    np.save(corpus_dir / "y.npy", y)
    meta = {
        "name": "Mouse Retina",
        "modality": "scRNA-seq",
        "source": str(h5ad),
        "label_type": "derived",
        "label_source": "leiden_on_pca50",
        "neighbors_params": {"n_neighbors": n_neighbors, "use_rep": "X_pca", "random_state": args.seed},
        "leiden_params": {"resolution": 1.0, "random_state": args.seed, "flavor": "igraph", "n_iterations": 2, "directed": False},
        "class_names": [f"cluster_{i}" for i in range(n_clusters)],
        "build_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    with open(corpus_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Built {corpus_dir}: X {x.shape}, y {y.shape}, n_clusters={n_clusters}")

if __name__ == "__main__":
    main()
