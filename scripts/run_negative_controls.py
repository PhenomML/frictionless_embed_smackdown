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

default_k_grid = [2, 3, 5, 8, 10, 15, 20]


def _embed(method: str, x: np.ndarray, cfg: EmbedConfig) -> np.ndarray:
    if method == "tsne":
        return embed_tsne(x, cfg)
    if method == "umap":
        return embed_umap(x, cfg)
    raise ValueError(f"Unsupported method: {method}")


def _feature_shuffle(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x2 = np.array(x, copy=True)
    for j in range(x2.shape[1]):
        x2[:, j] = rng.permutation(x2[:, j])
    return x2


def _rows(
    dataset_id: str,
    method: str,
    scenario: str,
    scenario_type: str,
    result: dict,
    run_meta: dict,
) -> list[dict]:
    out: list[dict] = []
    for i, k in enumerate(result["k_range"]):
        p = result["p_value"][i]
        t = result["rep_mean"][i]
        t_std = result["rep_std"][i]
        q95 = result["null_q95"][i]
        out.append(
            {
                "dataset_id": dataset_id,
                "method": method,
                "metric": result.get("metric", "ami"),
                "scenario": scenario,
                "scenario_type": scenario_type,
                "K": int(k),
                "T_mean": t,
                "T_std": t_std,
                "q95_null": q95,
                "p_value": p,
                "p_le_0p05": bool(pd.notna(p) and float(p) <= 0.05),
                "gap_T_minus_q95": float(t - q95) if pd.notna(t) and pd.notna(q95) else np.nan,
                "K_BH": result.get("hat_k_bh"),
                **run_meta,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compact negative controls.")
    parser.add_argument("--datasets", nargs="+", default=list(paper_dataset_id_order))
    parser.add_argument("--method", type=str, default="umap", choices=["tsne", "umap"])
    parser.add_argument("--metric", type=str, default="ami", choices=["ari", "ami", "jaccard"])
    parser.add_argument("--k-list", type=int, nargs="+", default=default_k_grid)
    parser.add_argument("--b", type=int, default=80)
    parser.add_argument("--fraction", type=float, default=0.8)
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

    cfg = EmbedConfig(max_points=args.max_points, random_state=args.seed)
    paths = get_default_paths()
    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    run_meta = {
        "b": int(args.b),
        "fraction": float(args.fraction),
        "pairs_per_k": int(args.pairs_per_k),
        "n_null": int(args.n_null),
        "max_points": int(args.max_points) if args.max_points is not None else None,
        "seed": int(args.seed),
    }

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

        z_rand2d = np.random.default_rng(args.seed + 111).standard_normal((x.shape[0], 2)).astype(np.float32)
        z_shuf = _embed(args.method, _feature_shuffle(x, args.seed + 222), cfg)

        common = dict(
            y=y,
            meta=meta,
            k_range=args.k_list,
            b=args.b,
            fraction=args.fraction,
            base_seed=args.seed,
            pairs_per_k=args.pairs_per_k,
            n_null=args.n_null,
            alpha=0.05,
            z_star=2.0,
            metric=args.metric,
        )
        results = {
            "rand2d_embedding": k_sweep_replicability(z=z_rand2d, **common),
            "shuffled_features_embedding": k_sweep_replicability(z=z_shuf, **common),
        }
        for scenario, result in results.items():
            rows = _rows(
                dataset_id,
                args.method,
                scenario,
                "negative_control",
                result,
                run_meta,
            )
            all_rows.extend(rows)
            sdf = pd.DataFrame(rows)
            summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "method": args.method,
                    "metric": args.metric,
                    "scenario": scenario,
                    "scenario_type": "negative_control",
                    "uncorrected_rejections_at_alpha_0p05": int(sdf["p_le_0p05"].sum()),
                    "frac_p_le_0p05": float((pd.to_numeric(sdf["p_value"], errors="coerce") <= 0.05).mean()),
                    "K_BH": result.get("hat_k_bh"),
                    "max_gap_T_minus_q95": float(sdf["gap_T_minus_q95"].max()),
                    "mean_gap_T_minus_q95": float(sdf["gap_T_minus_q95"].mean()),
                    **run_meta,
                }
            )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_path = out_dir / "negative_control_curves.csv"
    summary_path = out_dir / "negative_control_summary.csv"
    pd.DataFrame(all_rows).to_csv(curves_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved: {curves_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
