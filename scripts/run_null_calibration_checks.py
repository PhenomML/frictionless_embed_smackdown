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
from benchmark.replicability import k_sweep_replicability
from benchmark.utils import subsample_rows, to_float32

default_k_grid = [2, 3, 5, 8, 10, 15, 20]


def _embed(method: str, x: np.ndarray, cfg: EmbedConfig) -> np.ndarray:
    if method == "tsne":
        return embed_tsne(x, cfg)
    if method == "umap":
        return embed_umap(x, cfg)
    raise ValueError(f"Unsupported method: {method}")


def _curve_rows(
    dataset_id: str,
    method: str,
    scenario: str,
    scenario_type: str,
    result: dict,
    run_meta: dict,
) -> list[dict]:
    rows: list[dict] = []
    k_range = result["k_range"]
    for i, k in enumerate(k_range):
        p = result["p_value"][i]
        rows.append(
            {
                "dataset_id": dataset_id,
                "method": method,
                "scenario": scenario,
                "scenario_type": scenario_type,
                "metric": result.get("metric", "ami"),
                "K": int(k),
                "T_mean": result["rep_mean"][i],
                "T_std": result["rep_std"][i],
                "q95_null": result["null_q95"][i],
                "p_value": p,
                "p_le_0p05": bool(pd.notna(p) and float(p) <= 0.05),
                "K_BH": result.get("hat_k_bh"),
                **run_meta,
            }
        )
    return rows


def _summary_row(
    dataset_id: str,
    method: str,
    scenario: str,
    scenario_type: str,
    result: dict,
    run_meta: dict,
) -> dict:
    df = pd.DataFrame(_curve_rows(dataset_id, method, scenario, scenario_type, result, run_meta))
    df["gap_T_minus_q95"] = pd.to_numeric(df["T_mean"], errors="coerce") - pd.to_numeric(
        df["q95_null"], errors="coerce"
    )
    return {
        "dataset_id": dataset_id,
        "method": method,
        "metric": result.get("metric", "ami"),
        "scenario": scenario,
        "scenario_type": scenario_type,
        "n_k": int(len(df)),
        "uncorrected_rejections_at_alpha_0p05": int(df["p_le_0p05"].sum()),
        "K_BH": result.get("hat_k_bh"),
        "median_p_value": float(pd.to_numeric(df["p_value"], errors="coerce").median()),
        "frac_p_le_0p05": float((pd.to_numeric(df["p_value"], errors="coerce") <= 0.05).mean()),
        "max_T_minus_q95": float(df["gap_T_minus_q95"].max()),
        **run_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run null calibration sanity checks.")
    parser.add_argument("--dataset", type=str, default="mnist")
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

    paths = get_default_paths()
    cfg = EmbedConfig(max_points=args.max_points, random_state=args.seed)
    x, y, meta = load_dataset(
        args.dataset,
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

    z_real = _embed(args.method, x, cfg)
    rng = np.random.default_rng(args.seed + 901)
    z_rand2d = rng.standard_normal((x.shape[0], 2)).astype(np.float32)

    rep_kwargs = dict(
        z=z_real,
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
    run_meta = {
        "fraction": float(args.fraction),
        "b": int(args.b),
        "pairs_per_k": int(args.pairs_per_k),
        "n_null": int(args.n_null),
        "max_points": int(args.max_points) if args.max_points is not None else None,
        "seed": int(args.seed),
    }
    real_result = k_sweep_replicability(**rep_kwargs)
    rand_result = k_sweep_replicability(**{**rep_kwargs, "z": z_rand2d, "y": y})

    curves = []
    summaries = []
    for scenario, scenario_type, res in [
        ("real_embedding_real_labels", "reference", real_result),
        ("rand2d_embedding_real_labels", "negative_control", rand_result),
    ]:
        curves.extend(_curve_rows(args.dataset, args.method, scenario, scenario_type, res, run_meta))
        summaries.append(_summary_row(args.dataset, args.method, scenario, scenario_type, res, run_meta))

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_path = out_dir / "null_calibration_curves.csv"
    summary_path = out_dir / "null_calibration_summary.csv"
    pd.DataFrame(curves).to_csv(curves_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Saved: {curves_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
