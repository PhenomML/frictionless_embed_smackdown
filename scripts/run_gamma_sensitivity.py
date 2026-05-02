#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from benchmark.config import EmbedConfig, get_default_paths
from benchmark.embeddings import embed_tsne, embed_umap
from benchmark.loaders import load_dataset
from benchmark.paper_datasets import paper_dataset_id_order
from benchmark.replicability import k_sweep_replicability
from benchmark.utils import subsample_rows, to_float32

dataset_k_grid = {
    "ag_news": [2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
    "mnist": [2, 3, 5, 8, 10, 15, 20, 25, 30, 40],
    "uci_har": [2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
    "cifar10": [2, 3, 5, 8, 10, 15, 20, 25, 30, 40],
}


def _embed(method: str, x: np.ndarray, cfg: EmbedConfig) -> np.ndarray:
    if method == "tsne":
        return embed_tsne(x, cfg)
    if method == "umap":
        return embed_umap(x, cfg)
    raise ValueError(f"Unsupported method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run perturbation-strength sensitivity over gamma.")
    parser.add_argument("--datasets", nargs="+", default=list(paper_dataset_id_order))
    parser.add_argument("--methods", nargs="+", default=["tsne", "umap"])
    parser.add_argument("--metric", type=str, default="ami", choices=["ari", "ami", "jaccard"])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--b", type=int, default=80)
    parser.add_argument("--pairs-per-k", type=int, default=1200)
    parser.add_argument("--n-null", type=int, default=200)
    parser.add_argument("--max-points", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results_standardized" / "controls",
    )
    args = parser.parse_args()

    paths = get_default_paths()
    cfg = EmbedConfig(max_points=args.max_points, random_state=args.seed)
    run_meta = {
        "b": int(args.b),
        "pairs_per_k": int(args.pairs_per_k),
        "n_null": int(args.n_null),
        "max_points": int(args.max_points) if args.max_points is not None else None,
        "seed": int(args.seed),
    }

    curve_rows: list[dict] = []
    summary_rows: list[dict] = []
    for dataset_id in args.datasets:
        x, y, meta = load_dataset(
            dataset_id,
            paths["data_root"],
            paths["corpus_dir"],
            subsample=None,
            use_corpus_when_available=True,
            seed=args.seed,
        )
        if args.max_points is not None and x.shape[0] > args.max_points:
            strat = meta.get("label_type") in {"ground_truth", "derived"} and y is not None
            x, y, _ = subsample_rows(x, y, args.max_points, args.seed, stratified=strat)
        x = to_float32(x)
        k_range = dataset_k_grid.get(dataset_id, [2, 3, 5, 8, 10, 15, 20])

        for method in args.methods:
            z = _embed(method, x, cfg)
            for gamma in args.gammas:
                result = k_sweep_replicability(
                    z=z,
                    y=y,
                    meta=meta,
                    k_range=k_range,
                    b=args.b,
                    fraction=float(gamma),
                    base_seed=args.seed,
                    pairs_per_k=args.pairs_per_k,
                    n_null=args.n_null,
                    alpha=0.05,
                    z_star=2.0,
                    metric=args.metric,
                )
                current_rows: list[dict] = []
                for i, k in enumerate(result["k_range"]):
                    t = result["rep_mean"][i]
                    t_std = result["rep_std"][i]
                    q95 = result["null_q95"][i]
                    current_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "method": method,
                            "metric": args.metric,
                            "scenario_type": "gamma_sensitivity",
                            "gamma": float(gamma),
                            "K": int(k),
                            "T_mean": t,
                            "T_std": t_std,
                            "q95_null": q95,
                            "gap_T_minus_q95": float(t - q95) if pd.notna(t) and pd.notna(q95) else np.nan,
                            "p_value": result["p_value"][i],
                            "K_BH": result.get("hat_k_bh"),
                            **run_meta,
                        }
                    )
                curve_rows.extend(current_rows)

                cdf = pd.DataFrame(current_rows)
                summary_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "method": method,
                        "metric": args.metric,
                        "scenario_type": "gamma_sensitivity",
                        "gamma": float(gamma),
                        "K_BH": result.get("hat_k_bh"),
                        "max_gap_T_minus_q95": float(cdf["gap_T_minus_q95"].max()),
                        "mean_gap_T_minus_q95": float(cdf["gap_T_minus_q95"].mean()),
                        "median_p_value": float(pd.to_numeric(cdf["p_value"], errors="coerce").median()),
                        "frac_p_le_0p05": float((pd.to_numeric(cdf["p_value"], errors="coerce") <= 0.05).mean()),
                        **run_meta,
                    }
                )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_path = out_dir / "gamma_sensitivity_curves.csv"
    summary_path = out_dir / "gamma_sensitivity_summary.csv"
    pd.DataFrame(curve_rows).to_csv(curves_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved: {curves_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
