#!/usr/bin/env python3

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
import warnings

import pandas as pd

from _standardization_utils import (
    assert_unique_keys,
    classify_compute_tier,
    infer_requested_cap,
    normalize_nullable_bool,
    require_columns,
    resolve_n_target,
    safe_mode_or_na,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Build strict benchmark views and protocol/rerun triage tables."
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=root / "results_standardized",
    )
    p.add_argument(
        "--strict-min-b",
        type=int,
        default=200,
    )
    p.add_argument(
        "--strict-min-pairs",
        type=int,
        default=2000,
    )
    p.add_argument(
        "--strict-min-n-points",
        type=int,
        default=0,
    )
    p.add_argument("--strict-relative-n-frac", type=float, default=0.95)
    p.add_argument(
        "--dataset-inventory-csv",
        type=Path,
        default=root / "results_standardized" / "dataset_inventory.csv",
        help="Dataset inventory with true n_samples. If missing, n_samples remains NA and is flagged.",
    )
    return p.parse_args()


def _tier_severity(s: pd.Series) -> str:
    order = {"nominal": 0, "retry_relaxed": 1, "retry_low": 2, "unknown": 3}
    vals = [str(x) for x in s.dropna()]
    if not vals:
        return "unknown"
    worst = max(vals, key=lambda x: order.get(x, 3))
    return worst


def _build_group_table(
    df: pd.DataFrame,
    table_family: str,
    min_b: int,
    min_pairs: int,
    min_n: int,
    rel_n_frac: float,
    n_samples_map: dict[str, float],
) -> pd.DataFrame:
    require_columns(df, ["dataset_id", "method", "metric", "b_used", "max_pairs_used", "n_points_used"], f"{table_family} input")
    d = df.copy()
    if "protocol_source" not in d.columns:
        d["protocol_source"] = pd.NA
    if "compute_tier" not in d.columns:
        d["compute_tier"] = "unknown"
    d["b_used"] = pd.to_numeric(d["b_used"], errors="coerce")
    d["max_pairs_used"] = pd.to_numeric(d["max_pairs_used"], errors="coerce")
    d["n_points_used"] = pd.to_numeric(d["n_points_used"], errors="coerce")
    ds_meta_rows = []
    for ds, gds in d.groupby("dataset_id", sort=False):
        n_samples = n_samples_map.get(ds, np.nan)
        requested_cap_inferred, requested_cap_source = infer_requested_cap(
            gds.assign(table_family=table_family),
            n_samples=n_samples,
            explicit_cap_cols=["requested_cap", "max_points", "max_points_requested", "requested_cap_inferred"],
            b_col="b_used",
            pairs_col="max_pairs_used",
            n_points_col="n_points_used",
        )
        n_target_auto, n_target_source = resolve_n_target(
            mode="auto",
            n_samples=n_samples,
            requested_cap_inferred=requested_cap_inferred,
            force_cap=None,
        )
        n_target_full, n_target_full_source = resolve_n_target(
            mode="full",
            n_samples=n_samples,
            requested_cap_inferred=requested_cap_inferred,
            force_cap=None,
        )
        n_target_active, n_target_active_source = n_target_auto, n_target_source
        n_target_mode = "auto"

        # Reuse standardized target-size values when the upstream table already fixed them.
        if "requested_cap_inferred" in gds.columns and pd.to_numeric(gds["requested_cap_inferred"], errors="coerce").dropna().any():
            requested_cap_inferred = float(pd.to_numeric(gds["requested_cap_inferred"], errors="coerce").dropna().max())
            requested_cap_source = str(safe_mode_or_na(gds.get("requested_cap_source", pd.Series(dtype=object))))
        if "n_target_auto" in gds.columns and pd.to_numeric(gds["n_target_auto"], errors="coerce").dropna().any():
            n_target_auto = float(pd.to_numeric(gds["n_target_auto"], errors="coerce").dropna().max())
            n_target_source = str(safe_mode_or_na(gds.get("n_target_source", pd.Series(dtype=object))))
        if "n_target_full" in gds.columns and pd.to_numeric(gds["n_target_full"], errors="coerce").dropna().any():
            n_target_full = float(pd.to_numeric(gds["n_target_full"], errors="coerce").dropna().max())
            n_target_full_source = str(safe_mode_or_na(gds.get("n_target_full_source", pd.Series(dtype=object))))
        if "n_target" in gds.columns and pd.to_numeric(gds["n_target"], errors="coerce").dropna().any():
            n_target_active = float(pd.to_numeric(gds["n_target"], errors="coerce").dropna().max())
            n_target_active_source = str(safe_mode_or_na(gds.get("n_target_active_source", pd.Series(dtype=object))))
        if "n_target_mode" in gds.columns and gds["n_target_mode"].dropna().any():
            n_target_mode = str(safe_mode_or_na(gds["n_target_mode"]))
        ds_meta_rows.append(
            {
                "dataset_id": ds,
                "table_family": table_family,
                "n_samples": n_samples,
                "requested_cap_inferred": requested_cap_inferred,
                "requested_cap_source": requested_cap_source,
                "n_target_auto": n_target_auto,
                "n_target_source": n_target_source,
                "n_target_full": n_target_full,
                "n_target_full_source": n_target_full_source,
                "n_target": n_target_active,
                "n_target_mode": n_target_mode,
                "n_target_active_source": n_target_active_source,
            }
        )
    ds_meta = pd.DataFrame(ds_meta_rows)

    tuple_count = (
        d.assign(_t=d[["b_used", "max_pairs_used", "n_points_used"]].astype("string").agg("|".join, axis=1))
        .groupby(["dataset_id", "method", "metric"])["_t"]
        .nunique()
        .rename("row_tuple_candidate_count")
        .reset_index()
    )
    if "group_protocol_candidate_count" in d.columns:
        precomputed_count = (
            d.groupby(["dataset_id", "method", "metric"], as_index=False)
            .agg(precomputed_candidate_count=("group_protocol_candidate_count", lambda s: pd.to_numeric(s, errors="coerce").max()))
        )
    else:
        precomputed_count = (
            d[["dataset_id", "method", "metric"]]
            .drop_duplicates()
            .assign(precomputed_candidate_count=np.nan)
        )
    g = (
        d.groupby(["dataset_id", "method", "metric"], as_index=False)
        .agg(
            b_min=("b_used", "min"),
            b_max=("b_used", "max"),
            max_pairs_min=("max_pairs_used", "min"),
            max_pairs_max=("max_pairs_used", "max"),
            n_points_min=("n_points_used", "min"),
            n_points_max=("n_points_used", "max"),
            compute_tier=("compute_tier", _tier_severity),
            protocol_source=("protocol_source", lambda s: safe_mode_or_na(s)),
        )
        .merge(tuple_count, on=["dataset_id", "method", "metric"], how="left")
        .merge(precomputed_count, on=["dataset_id", "method", "metric"], how="left")
    )
    g["final_protocol_candidate_count"] = pd.concat(
        [
            pd.to_numeric(g["row_tuple_candidate_count"], errors="coerce"),
            pd.to_numeric(g["precomputed_candidate_count"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    g["final_protocol_candidate_count"] = g["final_protocol_candidate_count"].fillna(1).astype("Int64")
    g["n_unique_protocol_tuples"] = g["final_protocol_candidate_count"]
    g["table_family"] = table_family
    g["group_protocol_uniform"] = (g["final_protocol_candidate_count"] == 1).astype("boolean")
    g["group_protocol_mixed"] = (~g["group_protocol_uniform"].fillna(False)).astype("boolean")
    g = g.merge(ds_meta, on=["dataset_id", "table_family"], how="left")

    g["fails_b_threshold"] = ~(g["b_min"] >= min_b)
    g["fails_pairs_threshold"] = ~(g["max_pairs_min"] >= min_pairs)
    rel_ok = g["n_points_min"] >= rel_n_frac * g["n_target_auto"]
    abs_ok = True if min_n <= 0 else (g["n_points_min"] >= min_n)
    g["fails_n_threshold"] = ~(rel_ok & abs_ok)
    g["missing_protocol_metadata"] = g[["b_min", "max_pairs_min", "n_points_min"]].isna().any(axis=1)
    g["strict_auto"] = (
        g["group_protocol_uniform"].fillna(False)
        & (~g["fails_b_threshold"])
        & (~g["fails_pairs_threshold"])
        & (~g["fails_n_threshold"])
        & (~g["missing_protocol_metadata"])
    )
    full_n_ok = g["n_points_min"] >= rel_n_frac * g["n_target_full"]
    g["strict_full"] = (
        g["group_protocol_uniform"].fillna(False)
        & (~g["missing_protocol_metadata"])
        & (g["b_min"] >= min_b)
        & (g["max_pairs_min"] >= min_pairs)
        & full_n_ok
    )
    g["strict_auto"] = g["strict_auto"].astype("boolean")
    g["strict_full"] = g["strict_full"].astype("boolean")
    g["strict_ok"] = g["strict_auto"]

    def _reason(r: pd.Series) -> str:
        reasons = []
        if bool(r.get("missing_protocol_metadata", False)):
            reasons.append("missing_protocol_metadata")
        if bool(r.get("group_protocol_mixed", False)):
            reasons.append("mixed_protocol")
        if bool(r.get("fails_b_threshold", False)):
            reasons.append("low_b")
        if bool(r.get("fails_pairs_threshold", False)):
            reasons.append("low_pairs")
        if bool(r.get("fails_n_threshold", False)):
            reasons.append("low_n_points")
        return "|".join(reasons) if reasons else ""

    g["strict_failure_reason"] = g.apply(_reason, axis=1)
    g["compute_tier_group"] = [
        classify_compute_tier(b, p, n, nt) for b, p, n, nt in zip(g["b_min"], g["max_pairs_min"], g["n_points_min"], g["n_target_auto"])
    ]
    return g


def main() -> None:
    args = parse_args()
    rd = args.results_dir
    k_path = rd / "k_sweep_summary_long.csv"
    f_path = rd / "fixed_k0_summary_long.csv"
    if not k_path.exists() or not f_path.exists():
        raise FileNotFoundError(
            "Missing standardized inputs. Run standardization scripts first."
        )

    k = pd.read_csv(k_path)
    f = pd.read_csv(f_path)
    n_samples_map: dict[str, float] = {}
    if args.dataset_inventory_csv.exists():
        inv = pd.read_csv(args.dataset_inventory_csv)
        if "dataset_id" in inv.columns and "n_samples" in inv.columns:
            tmp = inv[["dataset_id", "n_samples"]].copy()
            tmp["n_samples"] = pd.to_numeric(tmp["n_samples"], errors="coerce")
            n_samples_map = {str(r["dataset_id"]): float(r["n_samples"]) for _, r in tmp.dropna().iterrows()}
    else:
        warnings.warn(
            "dataset_inventory.csv missing; n_target inference may be permissive because n_samples is unavailable."
        )

    k_group = _build_group_table(
        k, "k_sweep",
        min_b=args.strict_min_b,
        min_pairs=args.strict_min_pairs,
        min_n=args.strict_min_n_points,
        rel_n_frac=args.strict_relative_n_frac,
        n_samples_map=n_samples_map,
    )
    f_group = _build_group_table(
        f, "fixed_k0",
        min_b=args.strict_min_b,
        min_pairs=args.strict_min_pairs,
        min_n=args.strict_min_n_points,
        rel_n_frac=args.strict_relative_n_frac,
        n_samples_map=n_samples_map,
    )
    triage = pd.concat([k_group, f_group], ignore_index=True)
    triage["strict_auto"] = normalize_nullable_bool(triage["strict_auto"]).fillna(False)
    triage["strict_full"] = normalize_nullable_bool(triage["strict_full"]).fillna(False)
    triage["strict_ok"] = triage["strict_auto"]

    k_case_datasets = {"ag_news", "cifar10"}
    fixed_case_datasets = {"olivetti_faces", "20newsgroups"}
    triage["metric"] = triage["metric"].astype(str).str.lower()
    triage["dataset_id"] = triage["dataset_id"].astype(str)
    triage["affects_main_table"] = False
    triage["affects_main_figure"] = False
    k_idx = triage["table_family"].eq("k_sweep")
    f_idx = triage["table_family"].eq("fixed_k0")
    triage.loc[k_idx, "affects_main_table"] = triage.loc[k_idx, "metric"].eq("ami")
    triage.loc[k_idx, "affects_main_figure"] = triage.loc[k_idx, "metric"].eq("ami") & triage.loc[k_idx, "dataset_id"].isin(k_case_datasets)
    triage.loc[f_idx, "affects_main_table"] = triage.loc[f_idx, "metric"].isin({"ami", "ari"})
    triage.loc[f_idx, "affects_main_figure"] = triage.loc[f_idx, "metric"].isin({"ami", "ari"}) & triage.loc[f_idx, "dataset_id"].isin(fixed_case_datasets)

    def _priority(row: pd.Series) -> str:
        if row["strict_auto"]:
            return "none"
        if row["affects_main_figure"]:
            return "high"
        if row["affects_main_table"]:
            return "medium"
        if row["metric"] in {"jaccard"}:
            return "low"
        return "low"

    triage["rerun_priority"] = triage.apply(_priority, axis=1)
    triage["rerun_priority"] = pd.Categorical(
        triage["rerun_priority"],
        categories=["high", "medium", "low", "none"],
        ordered=True,
    )
    triage["recommended_b"] = triage["rerun_priority"].map({"high": 200, "medium": 200, "low": 100, "none": pd.NA})
    triage["recommended_pairs"] = triage["rerun_priority"].map({"high": 2000, "medium": 2000, "low": 1000, "none": pd.NA})
    triage["recommended_n_points"] = triage["rerun_priority"].map({"high": 5000, "medium": 3000, "low": 3000, "none": pd.NA})
    triage["recommended_n_null"] = triage["rerun_priority"].map({"high": 200, "medium": 200, "low": 100, "none": pd.NA})
    triage["recommended_rerun_params"] = triage["rerun_priority"].map(
        {
            "high": "b=200, pairs=2000, n_points=5000, n_null=200",
            "medium": "b=200, pairs=2000, n_points=3000, n_null=200",
            "low": "b=100, pairs=1000, n_points=3000, n_null=100",
            "none": "",
        }
    )
    triage["paper_usable_now"] = np.where(
        triage["affects_main_figure"],
        triage["strict_auto"],
        np.where(triage["affects_main_table"], triage["strict_auto"], True),
    )
    triage["paper_usable_now"] = normalize_nullable_bool(pd.Series(triage["paper_usable_now"], index=triage.index)).fillna(False)

    strict_groups = triage[triage["strict_auto"]][["dataset_id", "method", "metric", "table_family"]].copy()
    k_strict_groups = strict_groups[strict_groups["table_family"] == "k_sweep"][["dataset_id", "method", "metric"]]
    f_strict_groups = strict_groups[strict_groups["table_family"] == "fixed_k0"][["dataset_id", "method", "metric"]]
    k_strict = k.merge(k_strict_groups, on=["dataset_id", "method", "metric"], how="inner")
    f_strict = f.merge(f_strict_groups, on=["dataset_id", "method", "metric"], how="inner")
    k_strict.to_csv(rd / "k_sweep_summary_long_strict.csv", index=False)
    f_strict.to_csv(rd / "fixed_k0_summary_long_strict.csv", index=False)

    triage = triage.sort_values(
        ["rerun_priority", "table_family", "dataset_id", "method", "metric"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)
    assert_unique_keys(triage, ["table_family", "dataset_id", "method", "metric"], "protocol triage output uniqueness")

    out_triage = rd / "protocol_triage.csv"
    triage.to_csv(out_triage, index=False)

    print(f"Wrote {rd / 'k_sweep_summary_long_strict.csv'} ({len(k_strict)} rows)")
    print(f"Wrote {rd / 'fixed_k0_summary_long_strict.csv'} ({len(f_strict)} rows)")
    print(f"Wrote {out_triage} ({len(triage)} groups)")


if __name__ == "__main__":
    main()

