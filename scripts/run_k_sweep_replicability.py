#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os

if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
import csv
import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from benchmark.config import EmbedConfig, get_default_paths
from benchmark.loaders import load_dataset
from benchmark.paper_datasets import paper_dataset_id_order
from benchmark.replicability import draw_subsample_indices, run_single_config
from benchmark.replicability_config import K_sweep_config
from benchmark.utils import subsample_rows

dataset_k_grids: dict[str, list[int]] = {
    "s_curve": [2, 3, 5, 8, 10, 15, 20],
    "20newsgroups": [2, 3, 5, 8, 10, 15, 20, 25, 30],
    "ag_news": [2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
    "dbpedia_14": [2, 3, 5, 8, 10, 14, 20, 30, 40],
    "olivetti_faces": [2, 3, 5, 8, 10, 15, 20, 30, 40],
    "mnist": [2, 3, 5, 8, 10, 15, 20, 25, 30, 40],
    "uci_har": [2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
    "cifar10": [2, 3, 5, 8, 10, 15, 20, 25, 30, 40],
}

def run_sanity_check() -> None:
    paths = get_default_paths()
    cfg = EmbedConfig(max_points=3000)
    rng = np.random.default_rng(123)

    for dataset_id in ["olivetti_faces", "20newsgroups"]:
        x, y, meta = load_dataset(
            dataset_id,
            paths["data_root"],
            paths["corpus_dir"],
            subsample=None,
            use_corpus_when_available=True,
            seed=123,
        )
        if cfg.max_points is not None and x.shape[0] > cfg.max_points:
            strat = meta.get("label_type") in {"ground_truth", "derived"} and y is not None
            x, y, _ = subsample_rows(x, y, cfg.max_points, 123, stratified=strat)
        n = x.shape[0]
        stratified = meta.get("label_type") in {"ground_truth", "derived"} and y is not None
        idx = draw_subsample_indices(n, 0.8, rng, stratified=stratified, y=y)
        y_sub = y[idx]
        unique, counts = np.unique(y_sub, return_counts=True)
        print(f"\n{dataset_id} (stratified={stratified}, n={n}, n_take={len(idx)}):")
        print(f"  class counts in y[idx]: {dict(zip(unique.tolist(), counts.tolist()))}")
        print(f"  total: {counts.sum()}")

def main() -> None:
    parser = argparse.ArgumentParser(description="K-sweep replicability benchmark")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(paper_dataset_id_order),
        help="Dataset IDs",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["tsne", "umap"],
        help="Embedding methods",
    )
    parser.add_argument("--k-min", type=int, default=2, help="Min K (used with --k-max if --k-list not set)")
    parser.add_argument("--k-max", type=int, default=10, help="Max K (used with --k-min if --k-list not set)")
    parser.add_argument("--k-list", type=int, nargs="+", default=None,
                        help="Explicit K grid (overrides k-min/k-max and dataset-specific grid)")
    parser.add_argument("--b", type=int, default=200, help="Number of replicates")
    parser.add_argument("--pairs-per-k", type=int, default=2000, help="Sampled pairs per K")
    parser.add_argument("--tau", type=float, default=0.8,
                        help="Appendix only: tau line for legacy sensitivity outputs (not used for selection)")
    parser.add_argument("--consec", type=int, default=2,
                        help="Appendix only: consecutive-m tau rule (0 disables)")
    parser.add_argument("--baseline", action="store_true", help="Include rand2d baseline (Z ~ N(0,I)) per dataset")
    parser.add_argument("--max-points", type=int, default=3000, help="Max points (subsample)")
    parser.add_argument("--n-null", type=int, default=200, help="Monte Carlo null draws (0 to skip null)")
    parser.add_argument("--alpha", type=float, default=0.05, help="FDR level for BH selection rule")
    parser.add_argument("--z-star", type=float, default=2.0, help="Z-score threshold for effect-size rule")
    parser.add_argument("--min-pairs-M", type=int, default=1000,
                        help="Target: retain >= M eligible pairs when learning m_min (Option A)")
    parser.add_argument("--no-learn-min-overlap", action="store_true",
                        help="Use fixed min_overlap instead of learning m_min from pairs")
    parser.add_argument("--enable-tau-sweep", action="store_true",
                        help="Appendix only: compute tau-sweep outputs (hat_k_tau, hat_k_tau075/080/085)")
    parser.add_argument("--metric", type=str, default="ari", choices=["ari", "jaccard", "ami"],
                        help="Partition similarity metric for pairwise replicability (ari, jaccard, or ami)")
    parser.add_argument("--store-pairwise-metric-vec", action="store_true",
                        help="Store raw pairwise metric vectors per K for violin plot (increases JSON size)")
    parser.add_argument("--store-ari-vec", action="store_true",
                        help="Deprecated alias for --store-pairwise-metric-vec")
    parser.add_argument("--knife-edge-delta", type=float, default=0.02,
                        help="Knife-edge threshold: Rep(K_hat) within delta of null boundary")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=123, help="Base seed")
    parser.add_argument("--sanity-check", action="store_true", help="Print stratified class counts and exit")
    args = parser.parse_args()

    if args.sanity_check:
        run_sanity_check()
        return

    if args.k_list is not None:
        k_range = args.k_list
    else:
        k_range = list(range(args.k_min, args.k_max + 1))
    paths = get_default_paths()

    out_dir = args.output_dir or (root / "outputs" / f"replicability_benchmark_{args.metric}")
    out_dir.mkdir(parents=True, exist_ok=True)

    embed_cfg = EmbedConfig(
        random_state=args.seed,
        max_points=args.max_points,
    )
    store_pairwise_metric_vec = args.store_pairwise_metric_vec or args.store_ari_vec
    ksweep_cfg = K_sweep_config(
        k_range=k_range,
        b=args.b,
        fraction=0.8,
        pairs_per_k=args.pairs_per_k,
        learn_min_overlap=not args.no_learn_min_overlap,
        min_pairs_M=args.min_pairs_M,
        enable_tau_sweep=args.enable_tau_sweep,
        knife_edge_delta=args.knife_edge_delta,
        tau=args.tau,
        consec_m=args.consec if args.consec > 0 else None,
        base_seed=args.seed,
        n_null=args.n_null,
        alpha=args.alpha,
        z_star=args.z_star,
        metric=args.metric,
        store_ari_vec_by_k=args.store_ari_vec,
        store_pairwise_metric_vec_by_k=store_pairwise_metric_vec,
    )

    methods = list(args.methods)
    if args.baseline:
        methods = methods + ["rand2d"]

    results: list[dict] = []
    for dataset_id in args.datasets:
        k_range_use = dataset_k_grids.get(dataset_id, k_range) if args.k_list is None else k_range
        ksweep_cfg_ds = K_sweep_config(
            k_range=k_range_use,
            b=ksweep_cfg.b,
            fraction=ksweep_cfg.fraction,
            pairs_per_k=ksweep_cfg.pairs_per_k,
            min_overlap=ksweep_cfg.min_overlap,
            min_overlap_default=ksweep_cfg.min_overlap_default,
            learn_min_overlap=ksweep_cfg.learn_min_overlap,
            min_pairs_M=ksweep_cfg.min_pairs_M,
            enable_tau_sweep=ksweep_cfg.enable_tau_sweep,
            knife_edge_delta=ksweep_cfg.knife_edge_delta,
            tau=ksweep_cfg.tau,
            consec_m=ksweep_cfg.consec_m,
            tau_grid=ksweep_cfg.tau_grid,
            report_kmax_cap_hit=ksweep_cfg.report_kmax_cap_hit,
            s_curve_kmax=ksweep_cfg.s_curve_kmax,
            base_seed=ksweep_cfg.base_seed,
            n_init=ksweep_cfg.n_init,
            max_iter=ksweep_cfg.max_iter,
            n_null=ksweep_cfg.n_null,
            alpha=ksweep_cfg.alpha,
            z_star=ksweep_cfg.z_star,
            metric=ksweep_cfg.metric,
            store_ari_vec_by_k=ksweep_cfg.store_ari_vec_by_k,
            store_pairwise_metric_vec_by_k=ksweep_cfg.store_pairwise_metric_vec_by_k,
        )
        for method in methods:
            print(f"Running {dataset_id} / {method}...")
            try:
                r = run_single_config(
                    dataset_id, method, paths, embed_cfg, ksweep_cfg_ds
                )
                results.append(r)
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "dataset_id": dataset_id,
                    "method": method,
                    "error": str(e),
                })

    json_path = out_dir / "replicability_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {json_path}")

    csv_path = out_dir / "replicability_summary.csv"
    rows = []
    empty_row_fields = {
        "dataset_id": "", "method": "", "n_points": "",
        "hat_k": "", "hat_k_bh": "", "hat_k_z": "", "hat_k_q95": "", "hat_k_q975": "", "hat_k_tau": "",
        "bh_found": "", "no_significant_k": "",
        "k_min_p": "", "p_min_p": "", "rep_at_k_min_p": "",
        "k_max_z": "", "z_max_z": "",
        "m_min": "", "c_implied": "",
        "pairs_total_possible": "", "pairs_sampled": "", "eligible_pairs_count": "",
        "min_pairs_M_requested": "", "min_pairs_M_used": "",
        "overlap_median": "", "overlap_iqr": "",
        "hat_k_consec": "", "cap_hit": "", "knife_edge": "",
        "rep_at_hat_bh": "", "p_at_hat_bh": "", "z_at_hat_bh": "",
        "pairs_used_at_hat_k": "",
        "k_eff_at_hat": "", "min_size_at_hat": "", "n_small_2_at_hat": "",
        "hat_k_tau075": "", "hat_k_tau080": "", "hat_k_tau085": "",
        "degenerate_at_hat": "", "error": "",
    }
    for r in results:
        if "error" in r:
            rows.append({
                **empty_row_fields,
                "dataset_id": r["dataset_id"],
                "method": r["method"],
                "error": r["error"],
            })
        else:
            hat_k = r.get("hat_k")
            hat_k_bh = r.get("hat_k_bh")
            hat_k_z = r.get("hat_k_z")
            enable_tau = r.get("meta", {}).get("enable_tau_sweep", False)
            hat_k_tau = r.get("hat_k_tau") if enable_tau else None
            hat_k_consec = r.get("hat_k_consec") if enable_tau else None
            pairs_used = r.get("pairs_used", [])
            k_range_r = r.get("k_range", [])
            pairs_at_hat = ""
            if hat_k is not None and hat_k in k_range_r:
                ki = k_range_r.index(hat_k)
                pairs_at_hat = pairs_used[ki] if ki < len(pairs_used) else ""

            meta_r = r.get("meta", {})
            eligible_count = r.get("eligible_pairs_count", meta_r.get("eligible_pairs_count")) or 0
            min_pairs_used_over_k = min(pairs_used) if pairs_used else 0
            if eligible_count > 0 and min_pairs_used_over_k < 0.9 * eligible_count:
                print(
                    f"  WARNING {r['dataset_id']}/{r['method']}: "
                    f"min pairs_used_over_k={min_pairs_used_over_k} < 0.9 * eligible={eligible_count}"
                )
            rows.append({
                "dataset_id": r["dataset_id"],
                "method": r["method"],
                "n_points": r.get("n_points", ""),
                "hat_k": hat_k if hat_k is not None else "",
                "hat_k_bh": hat_k_bh if hat_k_bh is not None else "",
                "hat_k_z": hat_k_z if hat_k_z is not None else "",
                "hat_k_q95": r.get("hat_k_q95") if r.get("hat_k_q95") is not None else "",
                "hat_k_q975": r.get("hat_k_q975") if r.get("hat_k_q975") is not None else "",
                "hat_k_tau": hat_k_tau if hat_k_tau is not None else "",
                "bh_found": r.get("bh_found", False),
                "no_significant_k": r.get("no_significant_k", False),
                "k_min_p": r.get("k_min_p", ""),
                "p_min_p": r.get("p_min_p", ""),
                "rep_at_k_min_p": r.get("rep_at_k_min_p", ""),
                "k_max_z": r.get("k_max_z", ""),
                "z_max_z": r.get("z_max_z", ""),
                "m_min": r.get("m_min", r.get("meta", {}).get("m_min", "")),
                "c_implied": r.get("c_implied", r.get("meta", {}).get("c_implied", "")),
                "pairs_total_possible": r.get("pairs_total_possible", r.get("meta", {}).get("pairs_total_possible", "")),
                "pairs_sampled": r.get("pairs_sampled", r.get("meta", {}).get("pairs_sampled", "")),
                "eligible_pairs_count": r.get("eligible_pairs_count", r.get("meta", {}).get("eligible_pairs_count", "")),
                "min_pairs_M_requested": r.get("min_pairs_M_requested", r.get("meta", {}).get("min_pairs_M_requested", "")),
                "min_pairs_M_used": r.get("min_pairs_M_used", r.get("meta", {}).get("min_pairs_M_used", "")),
                "overlap_median": r.get("overlap_median", r.get("meta", {}).get("overlap_median", "")),
                "overlap_iqr": r.get("overlap_iqr", r.get("meta", {}).get("overlap_iqr", "")),
                "hat_k_consec": hat_k_consec if hat_k_consec is not None else "",
                "cap_hit": r.get("cap_hit", False),
                "knife_edge": r.get("knife_edge", False),
                "rep_at_hat_bh": r.get("rep_at_hat_bh", ""),
                "p_at_hat_bh": r.get("p_at_hat_bh", ""),
                "z_at_hat_bh": r.get("z_at_hat_bh", ""),
                "pairs_used_at_hat_k": pairs_at_hat,
                "k_eff_at_hat": r.get("k_eff_at_hat", ""),
                "min_size_at_hat": r.get("min_size_at_hat", ""),
                "n_small_2_at_hat": r.get("n_small_2_at_hat", ""),
                "hat_k_tau075": r.get("hat_k_tau075", "") if enable_tau else "",
                "hat_k_tau080": r.get("hat_k_tau080", "") if enable_tau else "",
                "hat_k_tau085": r.get("hat_k_tau085", "") if enable_tau else "",
                "degenerate_at_hat": r.get("degenerate_at_hat", False),
                "error": "",
            })
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {csv_path}")

    curves_path = out_dir / "replicability_curves.csv"
    curve_rows = []
    fieldnames = ["dataset_id", "method", "metric", "k", "rep_mean", "rep_std", "q10", "pairs_used",
                  "null_mu0", "null_sigma0", "z_score", "p_value", "null_q025", "null_q975", "null_q95",
                  "k_eff_mean", "min_size_mean", "n_small_2_mean",
                  "metric_vs_gt_mean", "metric_vs_gt_std", "ari_vs_gt_mean", "ari_vs_gt_std", "ami_vs_gt_mean", "ami_vs_gt_std"]

    def _safe_get(lst, idx):
        if lst and idx < len(lst):
            v = lst[idx]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return v
        return ""

    for r in results:
        if "error" in r:
            continue
        for ki, k in enumerate(r.get("k_range", [])):
            row = {
                "dataset_id": r["dataset_id"],
                "method": r["method"],
                "metric": r.get("metric", "ari"),
                "k": k,
                "rep_mean": _safe_get(r.get("rep_mean"), ki),
                "rep_std": _safe_get(r.get("rep_std"), ki),
                "q10": _safe_get(r.get("q10"), ki),
                "pairs_used": _safe_get(r.get("pairs_used"), ki),
                "null_mu0": _safe_get(r.get("null_mu0"), ki),
                "null_sigma0": _safe_get(r.get("null_sigma0"), ki),
                "z_score": _safe_get(r.get("z_score"), ki),
                "p_value": _safe_get(r.get("p_value"), ki),
                "null_q025": _safe_get(r.get("null_q025"), ki),
                "null_q975": _safe_get(r.get("null_q975"), ki),
                "null_q95": _safe_get(r.get("null_q95"), ki),
                "k_eff_mean": _safe_get(r.get("k_eff_mean"), ki),
                "min_size_mean": _safe_get(r.get("min_size_mean"), ki),
                "n_small_2_mean": _safe_get(r.get("n_small_2_mean"), ki),
                "metric_vs_gt_mean": _safe_get(r.get("metric_vs_gt_mean"), ki),
                "metric_vs_gt_std": _safe_get(r.get("metric_vs_gt_std"), ki),
                "ari_vs_gt_mean": _safe_get(r.get("ari_vs_gt_mean"), ki),
                "ari_vs_gt_std": _safe_get(r.get("ari_vs_gt_std"), ki),
                "ami_vs_gt_mean": _safe_get(r.get("ami_vs_gt_mean"), ki),
                "ami_vs_gt_std": _safe_get(r.get("ami_vs_gt_std"), ki),
            }
            curve_rows.append(row)
    if curve_rows:
        with open(curves_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(curve_rows)
        print(f"Wrote {curves_path}")

if __name__ == "__main__":
    main()
