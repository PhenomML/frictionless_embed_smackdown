#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from benchmark.fixed_k0_hyperparam_sweep import (
    run_fixed_k0_frontier_experiment,
    save_fixed_k0_results,
)
from benchmark.paper_datasets import paper_dataset_id_order

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Hyperparameter climbing at known nominal class count K0 "
            "(fixed-K0 frontier experiment)"
        )
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(paper_dataset_id_order),
        help="Dataset IDs with known nominal class count K0.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["tsne", "umap"],
        choices=["tsne", "umap"],
        help="Embedding methods to sweep.",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["ari", "ami"],
        choices=["ari", "ami", "jaccard"],
        help="Partition similarity metrics M for T_M and V_M.",
    )
    p.add_argument(
        "--include-jaccard",
        action="store_true",
        help="Append jaccard to --metrics if not already present.",
    )
    p.add_argument("--b", type=int, default=200, help="Number of perturbation replicates.")
    p.add_argument(
        "--subsample-frac",
        type=float,
        default=0.8,
        help="Subsample fraction per replicate.",
    )
    p.add_argument("--max-pairs", type=int, default=2000, help="Sampled replicate pairs.")
    p.add_argument("--run-null", action="store_true", help="Run null calibration.")
    p.add_argument(
        "--tau",
        type=int,
        default=200,
        help="Number of null draws when --run-null is enabled.",
    )
    p.add_argument("--random-state", type=int, default=123, help="Random seed.")
    p.add_argument("--max-points", type=int, default=3000, help="Max points after subsampling.")
    p.add_argument("--min-pairs-m", type=int, default=1000, help="Min eligible pair target.")
    p.add_argument(
        "--tsne-perplexities",
        nargs="+",
        type=float,
        default=None,
        help="Override t-SNE perplexity grid.",
    )
    p.add_argument(
        "--umap-neighbors",
        nargs="+",
        type=int,
        default=None,
        help="Override UMAP n_neighbors grid.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "fixed_k0_hyperparam_sweep",
        help="Output directory.",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = list(args.metrics)
    if args.include_jaccard and "jaccard" not in metrics:
        metrics.append("jaccard")

    grid_by_method: dict[str, list[dict]] = {}
    if args.tsne_perplexities is not None:
        grid_by_method["tsne"] = [
            {"tsne_perplexity": float(p)} for p in args.tsne_perplexities
        ]
    if args.umap_neighbors is not None:
        grid_by_method["umap"] = [
            {"umap_n_neighbors": int(n)} for n in args.umap_neighbors
        ]

    print(
        "Running fixed-K0 sweep:\n"
        f"  datasets={args.datasets}\n"
        f"  methods={args.methods}\n"
        f"  metrics={metrics}\n"
        f"  b={args.b}, subsample_frac={args.subsample_frac}, max_pairs={args.max_pairs}\n"
        f"  run_null={args.run_null}, random_state={args.random_state}"
    )

    results = run_fixed_k0_frontier_experiment(
        datasets=args.datasets,
        methods=args.methods,
        metrics=metrics,
        b=args.b,
        subsample_frac=args.subsample_frac,
        max_pairs=args.max_pairs,
        run_null=args.run_null,
        tau=args.tau,
        random_state=args.random_state,
        max_points=args.max_points,
        min_pairs_m=args.min_pairs_m,
        hyperparameter_grid_by_method=(grid_by_method or None),
        partial_output_dir=args.output_dir,
        partial_every_n_rows=1,
    )

    save_fixed_k0_results(results, args.output_dir)
    print(f"Saved results to: {args.output_dir}")
    print(f"Rows: {len(results)}")

if __name__ == "__main__":
    main()

