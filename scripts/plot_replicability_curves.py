#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

root = Path(__file__).resolve().parent.parent

def load_from_json(
    path: Path, dataset: str | None, method: str | None
) -> tuple[pd.DataFrame, list[dict]]:
    with open(path) as f:
        results = json.load(f)
    rows = []
    meta_list = []
    for r in results:
        if "error" in r:
            continue
        if dataset and r.get("dataset_id") != dataset:
            continue
        if method and r.get("method") != method:
            continue
        meta_list.append(r)
        for ki, k in enumerate(r.get("k_range", [])):
            def _g(key):
                arr = r.get(key)
                if arr and ki < len(arr):
                    return arr[ki]
                return None
            row = {
                "dataset_id": r["dataset_id"],
                "method": r["method"],
                "metric": r.get("metric", "ari"),
                "k": k,
                "rep_mean": _g("rep_mean"),
                "rep_std": _g("rep_std") or 0,
                "null_q025": _g("null_q025"),
                "null_q975": _g("null_q975"),
                "metric_vs_gt_mean": _g("metric_vs_gt_mean") or _g("ari_vs_gt_mean"),
                "metric_vs_gt_std": _g("metric_vs_gt_std") or _g("ari_vs_gt_std") or 0,
            }
            rows.append(row)
    return pd.DataFrame(rows), meta_list

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Rep(K) + ARI-vs-GT dual-axis")
    canonical_json = root / "outputs" / "replicability_benchmark" / "replicability_results.json"
    parser.add_argument("--curves", type=Path, default=None, help="replicability_curves.csv path")
    parser.add_argument("--json", type=Path, default=canonical_json,
                        help="replicability_results.json path (default: outputs/replicability_benchmark/)")
    parser.add_argument("--dataset", type=str, default=None, help="Filter dataset (e.g. olivetti_faces)")
    parser.add_argument("--method", type=str, default=None, help="Filter method (e.g. umap)")
    parser.add_argument("--all-methods", action="store_true",
                        help="Plot Rep(K) for all methods (tsne, umap, rand2d) on same axes; requires --dataset")
    parser.add_argument("--tau", type=float, default=0.8, help="Threshold line (legacy)")
    parser.add_argument("--out", type=Path, default=None, help="Output figure path")
    args = parser.parse_args()

    meta_list: list[dict] = []
    if args.curves:
        df = pd.read_csv(args.curves)
        if args.dataset:
            df = df[df["dataset_id"] == args.dataset]
        if not args.all_methods and args.method:
            df = df[df["method"] == args.method]
    elif args.json.exists():
        df, meta_list = load_from_json(args.json, args.dataset, None if args.all_methods else args.method)
    else:
        print(f"Artifact not found: {args.json}. Run the benchmark first (notebook cell 2).", file=sys.stderr)
        sys.exit(1)
    if args.dataset:
        df = df[df["dataset_id"] == args.dataset]
    if not args.all_methods and args.method:
        df = df[df["method"] == args.method]
    df = df.sort_values(["method", "k"]).reset_index(drop=True)
    if df.empty:
        print("No rows match filters", file=sys.stderr)
        sys.exit(1)

    dataset_id = df["dataset_id"].iloc[0]
    methods = df["method"].unique().tolist()

    if args.all_methods:
        if not args.dataset:
            print("--all-methods requires --dataset", file=sys.stderr)
            sys.exit(1)
        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        ax1.set_xlabel(r"$K$")
        ax1.set_ylabel(r"$\mathrm{Rep}(K)$")
        colors = {"tsne": "C0", "umap": "C1", "rand2d": "gray"}
        for m in methods:
            dm = df[df["method"] == m]
            k = dm["k"].values
            rep_mean = dm["rep_mean"].fillna(0).values
            rep_std = dm["rep_std"].fillna(0).values
            c = colors.get(m, "C2")
            ax1.plot(k, rep_mean, "o-", color=c, label=f"{m} (mean)", markersize=4)
            rep_lo = np.maximum(0.0, rep_mean - rep_std)
            rep_hi = np.minimum(1.0, rep_mean + rep_std)
            ax1.fill_between(k, rep_lo, rep_hi, color=c, alpha=0.2,
                             label="pairwise dispersion (+/-1 SD across replicate-pairs)" if m == methods[0] else None)
        ax1.axhline(args.tau, color="gray", linestyle="--", alpha=0.6, label=r"$\tau$=" + str(args.tau))
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc="upper right", fontsize=7)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"{dataset_id}: Rep(K) comparison")
        fig.tight_layout()
        out = args.out or (root / "outputs" / "replicability_benchmark" / f"{dataset_id}_all_methods_curves.png")
    else:
        method = df["method"].iloc[0]
        df1 = df[df["method"] == method]
        k = df1["k"].values
        rep_mean = df1["rep_mean"].fillna(0).values
        rep_std = df1["rep_std"].fillna(0).values
        metric_name = meta_list[0].get("metric", "ari") if meta_list else "ari"
        metric_display = meta_list[0].get("metric_display_name", metric_name.upper()) if meta_list else metric_name.upper()
        has_gt = ("metric_vs_gt_mean" in df.columns and not df1["metric_vs_gt_mean"].isna().all()) or (
            "ari_vs_gt_mean" in df.columns and not df1["ari_vs_gt_mean"].isna().all()
        )
        metric_vs_gt_mean = (
            df1["metric_vs_gt_mean"].fillna(0).values
            if "metric_vs_gt_mean" in df.columns
            else (df1["ari_vs_gt_mean"].fillna(0).values if "ari_vs_gt_mean" in df.columns else None)
        )
        metric_vs_gt_std = (
            df1["metric_vs_gt_std"].fillna(0).values
            if "metric_vs_gt_std" in df.columns
            else (df1["ari_vs_gt_std"].fillna(0).values if "ari_vs_gt_std" in df.columns else None)
        )
        if not has_gt:
            metric_vs_gt_mean = metric_vs_gt_std = None
        has_null = "null_q025" in df.columns and not df1["null_q025"].isna().all()
        null_q025 = df1["null_q025"].values if has_null else None
        null_q975 = df1["null_q975"].values if has_null else None

        meta = next((m for m in meta_list if m.get("method") == method and m.get("dataset_id") == dataset_id), {})
        if meta:
            metric_name = meta.get("metric", "ari")
            metric_display = meta.get("metric_display_name", metric_name.upper())
        elif "metric" in df.columns:
            metric_name = str(df1["metric"].iloc[0]) if len(df1) else "ari"
            metric_display = metric_name.upper()
        cap_hit = meta.get("cap_hit", False)

        title = f"{dataset_id} / {method}"
        if cap_hit:
            title += " (stable to Kmax tested)"

        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        ax1.set_xlabel(r"$K$")
        ax1.set_ylabel(r"$\mathrm{Rep}(K)$", color="C0")
        rep_lo = np.maximum(0.0, rep_mean - rep_std)
        rep_hi = np.minimum(1.0, rep_mean + rep_std)
        ax1.plot(k, rep_mean, "o-", color="C0", label=r"$\mathrm{Rep}(K)$ (mean)", markersize=4)
        ax1.fill_between(k, rep_lo, rep_hi, color="C0", alpha=0.2,
                         label="pairwise dispersion (+/-1 SD across replicate-pairs)")

        if has_null:
            ax1.fill_between(k, null_q025, null_q975, color="gray", alpha=0.15,
                             label="null 95% band (overlap permutation)")

        ax1.axhline(args.tau, color="gray", linestyle="--", alpha=0.6, label=r"$\tau$=" + str(args.tau))
        ax1.set_ylim(0, 1.05)
        ax1.tick_params(axis="y", labelcolor="C0")
        ax1.grid(True, alpha=0.3)

        if has_gt and metric_vs_gt_mean is not None and metric_vs_gt_std is not None:
            ax2 = ax1.twinx()
            ax2.set_ylabel(f"{metric_display} vs GT", color="C1")
            gt_lo = np.maximum(0.0, metric_vs_gt_mean - metric_vs_gt_std)
            gt_hi = np.minimum(1.0, metric_vs_gt_mean + metric_vs_gt_std)
            ax2.plot(k, metric_vs_gt_mean, "s-", color="C1", label=f"{metric_display} vs GT (mean)", markersize=4)
            ax2.fill_between(k, gt_lo, gt_hi, color="C1", alpha=0.2,
                             label="replicate dispersion (+/-1 SD across replicates)")
            ax2.set_ylim(0, 1.05)
            ax2.tick_params(axis="y", labelcolor="C1")
            ax2.legend(loc="lower right", fontsize=7)

        plt.title(title)

        hat_k_bh = meta.get("hat_k_bh")
        hat_k_z = meta.get("hat_k_z")
        hat_k_tau = meta.get("hat_k_tau")
        no_significant_k = meta.get("no_significant_k", False)
        k_min_p = meta.get("k_min_p")
        alpha_val = meta.get("meta", {}).get("alpha", 0.05)
        y_bot = 0.02

        if hat_k_bh is not None and hat_k_bh in k:
            ax1.axvline(hat_k_bh, color="C2", linestyle="-", alpha=0.8, linewidth=1.5)
            ax1.annotate(r"$\hat{K}_{\mathrm{BH}}$=" + str(hat_k_bh), xy=(hat_k_bh, y_bot),
                         xytext=(5, 12), textcoords="offset points", fontsize=7.5, color="C2")
        elif no_significant_k:
            ax1.text(0.98, 0.02, f"no K significant at FDR {alpha_val}", transform=ax1.transAxes,
                     fontsize=7, ha="right", va="bottom", color="gray")
            if k_min_p is not None and k_min_p in k:
                ax1.axvline(k_min_p, color="C4", linestyle="--", alpha=0.7, linewidth=1.2)
                ax1.annotate(r"$K_{\min p}$=" + str(k_min_p) + " (best candidate, not sig.)",
                             xy=(k_min_p, y_bot), xytext=(5, 0), textcoords="offset points",
                             fontsize=6.5, color="C4")

        if hat_k_z is not None and hat_k_z in k:
            ax1.axvline(hat_k_z, color="C3", linestyle="--", alpha=0.8, linewidth=1.5)
            ax1.annotate(r"$\hat{K}_{Z}$=" + str(hat_k_z), xy=(hat_k_z, y_bot),
                         xytext=(5, 0), textcoords="offset points", fontsize=7.5, color="C3")
        if hat_k_tau is not None and hat_k_tau in k:
            ax1.axvline(hat_k_tau, color="gray", linestyle=":", alpha=0.7)
            ax1.annotate(r"$\hat{K}_{\tau}$=" + str(hat_k_tau), xy=(hat_k_tau, args.tau),
                         xytext=(5, 5), textcoords="offset points", fontsize=7, color="gray")

        handles, labels = ax1.get_legend_handles_labels()
        if hat_k_bh is not None:
            handles.append(Line2D([0], [0], color="C2", linestyle="-", linewidth=1.5))
            labels.append(r"$\hat{K}_{\mathrm{BH}}$ (FDR)")
        elif no_significant_k and k_min_p is not None:
            handles.append(Line2D([0], [0], color="C4", linestyle="--", linewidth=1.2))
            labels.append(r"$K_{\min p}$ (best candidate, not significant)")
        if hat_k_z is not None:
            handles.append(Line2D([0], [0], color="C3", linestyle="--", linewidth=1.5))
            labels.append(r"$\hat{K}_{Z}$ (effect-size)")
        ax1.legend(handles, labels, loc="upper right", fontsize=7)

        fig.tight_layout()
        out = args.out or (root / "outputs" / "replicability_benchmark" / f"{dataset_id}_{method}_curves.png")

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close()

if __name__ == "__main__":
    main()
