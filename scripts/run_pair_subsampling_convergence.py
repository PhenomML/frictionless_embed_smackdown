#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

import pandas as pd

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from benchmark.replicability import run_pair_subsampling_sensitivity
from benchmark.paper_datasets import paper_dataset_id_order


def _default_k_grid(dataset_id: str) -> list[int]:
    if dataset_id in {"mnist", "cifar10"}:
        return [2, 3, 5, 8, 10, 15, 20, 25, 30, 40]
    if dataset_id in {"ag_news", "uci_har"}:
        return [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    return [2, 3, 5, 8, 10, 15, 20]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pair-subsampling convergence checks.")
    parser.add_argument("--datasets", nargs="+", default=list(paper_dataset_id_order))
    parser.add_argument("--method", type=str, default="umap", choices=["tsne", "umap"])
    parser.add_argument("--metric", type=str, default="ami", choices=["ari", "ami", "jaccard"])
    parser.add_argument("--m-grid", type=int, nargs="+", default=[250, 500, 1000, 2000])
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--b", type=int, default=120)
    parser.add_argument("--pairs-pool-size", type=int, default=5000)
    parser.add_argument("--delta-threshold", type=float, default=0.02)
    parser.add_argument("--max-points", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results_standardized" / "controls",
    )
    args = parser.parse_args()

    summary_rows: list[dict] = []
    curve_rows: list[dict] = []
    run_meta = {
        "scenario_type": "pair_subsampling_convergence",
        "b": int(args.b),
        "R": int(args.R),
        "pairs_pool_size": int(args.pairs_pool_size),
        "delta_threshold": float(args.delta_threshold),
        "max_points": int(args.max_points) if args.max_points is not None else None,
        "seed": int(args.seed),
    }

    for dataset_id in args.datasets:
        out = run_pair_subsampling_sensitivity(
            dataset_id=dataset_id,
            method=args.method,
            k_range=_default_k_grid(dataset_id),
            b=args.b,
            pairs_pool_size=args.pairs_pool_size,
            M_grid=args.m_grid,
            R=args.R,
            base_seed=args.seed,
            max_points=args.max_points,
            delta_threshold=args.delta_threshold,
            metric=args.metric,
        )
        for m in out["M_grid"]:
            m_key = str(m)
            summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "method": args.method,
                    "metric": args.metric,
                    "M_pairs": int(m),
                    "delta_max_abs_median": float(out["delta_by_M"][m_key]),
                    "delta_pass": bool(out["delta_pass"][m_key]),
                    "mean_eligible_pairs": float(out["mean_eligible_count_by_M"][m_key]),
                    "m_min_fixed": int(out["m_min_fixed"]),
                    "k_range_json": json.dumps(out["k_range"]),
                    "n_k": int(len(out["k_range"])),
                    "n_replicates": int(args.R),
                    **run_meta,
                }
            )
            curves_m = out["rep_curves_by_M"][m_key]
            for r_idx, curve in enumerate(curves_m):
                for k_idx, k in enumerate(out["k_range"]):
                    curve_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "method": args.method,
                            "metric": args.metric,
                            "M_pairs": int(m),
                            "replicate_id": int(r_idx),
                            "K": int(k),
                            "T_mean": float(curve[k_idx]),
                            **run_meta,
                        }
                    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "pair_subsampling_summary.csv"
    curves_path = out_dir / "pair_subsampling_curves.csv"
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["dataset_id", "method", "metric", "M_pairs"]
    )
    curves_df = pd.DataFrame(curve_rows).sort_values(
        ["dataset_id", "method", "metric", "M_pairs", "replicate_id", "K"]
    )
    summary_df.to_csv(summary_path, index=False)
    curves_df.to_csv(curves_path, index=False)
    print(f"Saved: {summary_path}")
    print(f"Saved: {curves_path}")


if __name__ == "__main__":
    main()
