#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from benchmark.loaders import load_pbmc3k
from benchmark.utils import ensure_dir

leiden_resolution = 1.0
leiden_seed = 123
leiden_params = {
    "resolution": leiden_resolution,
    "random_state": leiden_seed,
    "flavor": "igraph",
    "n_iterations": 2,
    "directed": False,
}

def build_pbmc3k_corpus(data_root: Path, corpus_dir: Path) -> None:
    corpus_dir = Path(corpus_dir)
    ensure_dir(corpus_dir)

    x, y, _ = load_pbmc3k(data_root, leiden_resolution=leiden_resolution, seed=leiden_seed)

    np.save(corpus_dir / "X_pca50.npy", x)
    np.save(corpus_dir / "y.npy", y)

    n_clusters = len(np.unique(y))
    meta = {
        "name": "PBMC-3k",
        "modality": "scRNA-seq",
        "original_dim": 2000,
        "preprocessing": [
            "normalize_total(1e4)",
            "log1p",
            "HVG(2000)",
            "zscore",
            "PCA->50",
        ],
        "source": "10xgenomics pbmc3k v1.1.0",
        "label_type": "derived",
        "label_source": "leiden_on_pca50",
        "leiden_params": leiden_params,
        "class_names": [f"cluster_{i}" for i in range(n_clusters)],
        "build_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    with open(corpus_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Built {corpus_dir}")
    print(f"  X_pca50: {x.shape}, y: {y.shape}, n_clusters: {n_clusters}")

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build scRNA corpus with frozen Leiden labels")
    parser.add_argument(
        "--datasets",
        default="pbmc3k",
        help="Comma-separated: pbmc3k",
    )
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument("--corpus-dir", type=Path, default=root / "corpus")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    corpus_dir = Path(args.corpus_dir)
    datasets = [s.strip() for s in args.datasets.split(",")]

    for ds in datasets:
        if ds == "pbmc3k":
            build_pbmc3k_corpus(data_root, corpus_dir / "pbmc3k")
        else:
            print(f"Unknown dataset: {ds}")
            sys.exit(1)

    print("Done.")

if __name__ == "__main__":
    main()
