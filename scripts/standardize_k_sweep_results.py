#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
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


metric_names = ("ami", "ari", "jaccard")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Standardize old/new K-sweep outputs into one canonical long table.")
    p.add_argument(
        "--old-csv",
        type=Path,
        default=root / "outputs" / "replicability_benchmark_all_metrics" / "replicability_curves_all_metrics.csv",
    )
    p.add_argument(
        "--new-csv",
        type=Path,
        default=root / "outputs" / "replicability_benchmark_all_metrics_new_only" / "replicability_curves_all_metrics.csv",
    )
    p.add_argument(
        "--merged-csv",
        type=Path,
        default=None,
        help="Optional merged raw K-sweep source. If provided, old/new inputs are ignored.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=root / "results_standardized" / "k_sweep_summary_long.csv",
    )
    p.add_argument(
        "--old-root",
        type=Path,
        default=root / "outputs",
        help="Root output directory containing replicability_benchmark_* folders.",
    )
    p.add_argument("--strict-schema", action="store_true", help="Raise on schema-quality warnings.")
    p.add_argument("--target-cap-mode", choices=["auto", "full", "force_cap"], default="auto")
    p.add_argument("--force-cap", type=int, default=None)
    p.add_argument(
        "--dataset-inventory-csv",
        type=Path,
        default=root / "results_standardized" / "dataset_inventory.csv",
        help="Optional dataset inventory with true n_samples.",
    )
    return p.parse_args()


def _safe_float(x: Any) -> float | None:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    f = _safe_float(x)
    if f is None:
        return None
    return int(round(f))


def _load_protocol_meta(outputs_root: Path, source_block: str) -> pd.DataFrame:
    rows_out: list[dict[str, Any]] = []
    for metric in metric_names:
        if source_block == "old_baseline":
            result_path = outputs_root / f"replicability_benchmark_{metric}" / "replicability_results.json"
            if not result_path.exists():
                continue
            loaded_rows = json.loads(result_path.read_text(encoding="utf-8"))
            for row in loaded_rows:
                if "error" in row:
                    continue
                run_meta = row.get("meta", {})
                rows_out.append({
                    "dataset_id": row.get("dataset_id"),
                    "method": str(row.get("method", "")).lower(),
                    "metric": metric,
                    "source_block": source_block,
                    "b_used": _safe_int(run_meta.get("b")),
                    "max_pairs_used": _safe_int(run_meta.get("pairs_per_k")),
                    "n_points_used": _safe_int(row.get("n_points")),
                    "subsample_frac": _safe_float(run_meta.get("fraction")),
                    "null_trials": _safe_int(run_meta.get("n_null")),
                    "alpha_bh": _safe_float(run_meta.get("alpha")),
                    "protocol_source": "chunk_json_fallback",
                    "protocol_path": str(result_path),
                })
        else:
            chunk_root = outputs_root / f"replicability_benchmark_{metric}_new_only" / "chunks"
            if not chunk_root.exists():
                continue
            for dataset_chunk in sorted(chunk_root.glob("dataset_*")):
                result_path = dataset_chunk / "replicability_results.json"
                if not result_path.exists():
                    continue
                loaded_rows = json.loads(result_path.read_text(encoding="utf-8"))
                for row in loaded_rows:
                    if "error" in row:
                        continue
                    run_meta = row.get("meta", {})
                    rows_out.append({
                        "dataset_id": row.get("dataset_id"),
                        "method": str(row.get("method", "")).lower(),
                        "metric": metric,
                        "source_block": source_block,
                        "b_used": _safe_int(run_meta.get("b")),
                        "max_pairs_used": _safe_int(run_meta.get("pairs_per_k")),
                        "n_points_used": _safe_int(row.get("n_points")),
                        "subsample_frac": _safe_float(run_meta.get("fraction")),
                        "null_trials": _safe_int(run_meta.get("n_null")),
                        "alpha_bh": _safe_float(run_meta.get("alpha")),
                        "protocol_source": "chunk_json_fallback",
                        "protocol_path": str(result_path),
                    })
    return pd.DataFrame(rows_out)


def _extract_protocol_from_raw_rows(raw: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["dataset_id", "method", "metric", "source_block"]
    proto_cols = [c for c in ["b_used", "max_pairs_used", "n_points_used", "subsample_frac", "null_trials", "alpha_bh"] if c in raw.columns]
    if not proto_cols:
        return pd.DataFrame(columns=key_cols + ["protocol_source"])
    d = raw[key_cols + proto_cols].copy()
    d["method"] = d["method"].astype(str).str.lower()
    d["protocol_source"] = "merged_csv"
    return d


def _aggregate_protocol_candidates(
    df: pd.DataFrame,
    key_cols: list[str],
    source_tag: str,
) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=key_cols + ["group_protocol_candidate_count", "protocol_candidates_json", "b_val", "pairs_val", "n_val"])
    d = df.copy()
    d["b_num"] = pd.to_numeric(d.get("b_used"), errors="coerce")
    d["pairs_num"] = pd.to_numeric(d.get("max_pairs_used"), errors="coerce")
    d["n_num"] = pd.to_numeric(d.get("n_points_used"), errors="coerce")
    d["tuple_key"] = d[["b_num", "pairs_num", "n_num"]].astype("string").fillna("NA").agg("|".join, axis=1)
    agg = (
        d.groupby(key_cols, as_index=False)
        .agg(
            group_protocol_candidate_count=("tuple_key", "nunique"),
            protocol_candidates_json=("tuple_key", lambda s: json.dumps(sorted(set(str(x) for x in s.dropna())))),
            b_val=("b_num", lambda s: safe_mode_or_na(s)),
            pairs_val=("pairs_num", lambda s: safe_mode_or_na(s)),
            n_val=("n_num", lambda s: safe_mode_or_na(s)),
        )
    )
    agg["protocol_source"] = source_tag
    return agg


def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _fallback_v_mean(df: pd.DataFrame) -> pd.Series:
    if "metric_vs_gt_mean" in df.columns:
        v = pd.to_numeric(df["metric_vs_gt_mean"], errors="coerce")
    else:
        v = pd.Series(np.nan, index=df.index, dtype="float64")
    if "ari_vs_gt_mean" in df.columns:
        mask = df["metric"].eq("ari") & v.isna()
        v.loc[mask] = pd.to_numeric(df.loc[mask, "ari_vs_gt_mean"], errors="coerce")
    if "ami_vs_gt_mean" in df.columns:
        mask = df["metric"].eq("ami") & v.isna()
        v.loc[mask] = pd.to_numeric(df.loc[mask, "ami_vs_gt_mean"], errors="coerce")
    for c in ("jaccard_vs_gt_mean", "Jaccard_vs_gt_mean"):
        if c in df.columns:
            mask = df["metric"].eq("jaccard") & v.isna()
            v.loc[mask] = pd.to_numeric(df.loc[mask, c], errors="coerce")
    return v


def _apply_bh_flags(g: pd.DataFrame, alpha: float) -> pd.DataFrame:
    out = g.copy()
    assert_unique_keys(out, ["K"], f"BH group uniqueness {tuple(out[['dataset_id','method','metric']].iloc[0].tolist())}")
    pvals = out["p_value"].astype(float)
    if bool(((pvals < 0) | (pvals > 1)).fillna(False).any()):
        raise ValueError("Found p_value outside [0,1].")
    valid = pvals.notna()
    out["bh_reject"] = pd.Series(False, index=out.index, dtype="boolean")
    if valid.sum() == 0:
        out["k_bh"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out["is_k_bh"] = pd.Series(False, index=out.index, dtype="boolean")
        return out
    pv = pvals[valid]
    m = len(pv)
    ranked_idx = pv.sort_values().index
    ranks = np.arange(1, m + 1, dtype=float)
    thresh = alpha * ranks / float(m)
    passed = pv.loc[ranked_idx].to_numpy() <= thresh
    if not np.any(passed):
        out["k_bh"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out["is_k_bh"] = pd.Series(False, index=out.index, dtype="boolean")
        return out
    rejected_idx = ranked_idx[passed]
    out.loc[rejected_idx, "bh_reject"] = True
    k_bh = int(out.loc[rejected_idx, "K"].max())
    out["k_bh"] = k_bh
    out["is_k_bh"] = out["K"].eq(k_bh).astype("boolean")
    return out


def _add_group_extrema(g: pd.DataFrame) -> pd.DataFrame:
    out = g.copy()
    assert_unique_keys(out, ["K"], f"group extrema uniqueness {tuple(out[['dataset_id','method','metric']].iloc[0].tolist())}")
    out["k_at_max_T"] = pd.NA
    out["k_at_max_V"] = pd.NA
    out["k_at_max_Z"] = pd.NA
    out["max_T"] = np.nan
    out["max_V"] = np.nan
    out["max_Z"] = np.nan
    if out["T_mean"].notna().any():
        i = out["T_mean"].idxmax()
        out["k_at_max_T"] = int(out.loc[i, "K"])
        out["max_T"] = float(out.loc[i, "T_mean"])
    if out["V_mean"].notna().any():
        i = out["V_mean"].idxmax()
        out["k_at_max_V"] = int(out.loc[i, "K"])
        out["max_V"] = float(out.loc[i, "V_mean"])
    if out["z_score"].notna().any():
        i = out["z_score"].idxmax()
        out["k_at_max_Z"] = int(out.loc[i, "K"])
        out["max_Z"] = float(out.loc[i, "z_score"])
    out["k_at_max_T"] = pd.to_numeric(out["k_at_max_T"], errors="coerce").astype("Int64")
    out["k_at_max_V"] = pd.to_numeric(out["k_at_max_V"], errors="coerce").astype("Int64")
    out["k_at_max_Z"] = pd.to_numeric(out["k_at_max_Z"], errors="coerce").astype("Int64")
    return out


def _with_protocol_flags(
    df: pd.DataFrame,
    target_cap_mode: str,
    force_cap: int | None = None,
    n_samples_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    out = df.drop(
        columns=[
            "group_protocol_uniform",
            "protocol_note",
            "protocol_homogeneous",
            "compute_tier_group",
            "compute_tier",
            "n_target",
            "meets_nominal_target",
            "b_min",
            "b_max",
            "pairs_min",
            "pairs_max",
            "n_points_min",
            "n_points_max",
        ],
        errors="ignore",
    ).copy()
    for c in ("b_used", "max_pairs_used", "n_points_used"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "group_protocol_candidate_count" not in out.columns:
        out["group_protocol_candidate_count"] = pd.NA
    if "protocol_candidates_json" not in out.columns:
        out["protocol_candidates_json"] = "[]"
    cap_rows = []
    table_family = "k_sweep"
    n_samples_map = n_samples_map or {}
    for ds, gds in out.groupby("dataset_id", sort=False):
        n_samples = n_samples_map.get(str(ds), pd.NA)
        requested_cap_inferred, requested_cap_source = infer_requested_cap(
            gds.assign(table_family=table_family),
            n_samples=n_samples,
            explicit_cap_cols=["requested_cap", "max_points", "max_points_requested"],
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
        n_target_active, n_target_active_source = resolve_n_target(
            mode=target_cap_mode,
            n_samples=n_samples,
            requested_cap_inferred=requested_cap_inferred,
            force_cap=force_cap,
        )
        n_target_full, n_target_full_source = resolve_n_target(
            mode="full",
            n_samples=n_samples,
            requested_cap_inferred=requested_cap_inferred,
            force_cap=None,
        )
        cap_rows.append(
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
                "n_target_mode": target_cap_mode,
                "n_target_active_source": n_target_active_source,
            }
        )
    cap_df = pd.DataFrame(cap_rows)
    proto_count = (
        out.assign(_t=out[["b_used", "max_pairs_used", "n_points_used"]].astype("string").fillna("NA").agg("|".join, axis=1))
        .groupby(["dataset_id", "method", "metric"])["_t"]
        .nunique()
        .rename("group_protocol_candidate_count")
        .reset_index()
    )
    g = (
        out.groupby(["dataset_id", "method", "metric"], as_index=False)
        .agg(
            b_min=("b_used", "min"),
            b_max=("b_used", "max"),
            pairs_min=("max_pairs_used", "min"),
            pairs_max=("max_pairs_used", "max"),
            n_points_min=("n_points_used", "min"),
            n_points_max=("n_points_used", "max"),
            protocol_source_group=("protocol_source", lambda s: safe_mode_or_na(s)),
            protocol_candidates_json=("protocol_candidates_json", lambda s: safe_mode_or_na(s)),
            precomputed_candidate_count=("group_protocol_candidate_count", "max"),
        )
    )
    g = g.merge(proto_count.rename(columns={"group_protocol_candidate_count": "row_tuple_candidate_count"}), on=["dataset_id", "method", "metric"], how="left")
    g["group_protocol_candidate_count"] = pd.concat(
        [
            pd.to_numeric(g["precomputed_candidate_count"], errors="coerce"),
            pd.to_numeric(g["row_tuple_candidate_count"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    g["group_protocol_candidate_count"] = g["group_protocol_candidate_count"].fillna(1).astype("Int64")
    g["group_protocol_uniform"] = (g["group_protocol_candidate_count"] == 1).astype("boolean")
    g = g.merge(cap_df, on="dataset_id", how="left")
    g["meets_nominal_target"] = (g["n_points_min"] >= 0.95 * g["n_target"]).astype("boolean")
    g["compute_tier_group"] = [
        classify_compute_tier(b, p, n, nt) for b, p, n, nt in zip(g["b_min"], g["pairs_min"], g["n_points_min"], g["n_target"])
    ]
    g["compute_tier"] = g["compute_tier_group"]
    g["compute_status"] = np.where(g["group_protocol_uniform"].fillna(False), "uniform_protocol", "mixed_protocol")
    g["protocol_note"] = np.where(
        g["group_protocol_candidate_count"].isna(),
        "Missing protocol metadata",
        np.where(g["group_protocol_uniform"], "", "Mixed protocol tuples within group"),
    )
    out = out.merge(g, on=["dataset_id", "method", "metric"], how="left")
    if "protocol_source" not in out.columns:
        out["protocol_source"] = out["protocol_source_group"]
    else:
        out["protocol_source"] = out["protocol_source"].fillna(out["protocol_source_group"])
    out = out.drop(columns=["protocol_source_group"], errors="ignore")
    return out


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.merged_csv is not None:
        merged_input = pd.read_csv(args.merged_csv)
        if "source_block" not in merged_input.columns:
            merged_input["source_block"] = "merged_strict"
        frames = [("merged raw", merged_input)]
    else:
        old = pd.read_csv(args.old_csv)
        old["source_block"] = "old_baseline"
        new = pd.read_csv(args.new_csv)
        new["source_block"] = "new_only"
        frames = [("old raw", old), ("new raw", new)]

    for context, df in frames:
        require_columns(df, ["dataset_id", "method", "metric", "rep_mean", "rep_std"], f"k-sweep {context}")
        if "K" not in df.columns and "k" not in df.columns:
            raise ValueError(f"k-sweep {context}: expected either 'K' or 'k' column.")
        for c in ["null_mu0", "null_sigma0", "null_q95", "p_value", "z_score", "pairs_used"]:
            if c not in df.columns:
                warnings.warn(f"{context}: optional column missing -> filling as NaN: {c}")
                df[c] = np.nan

    raw = pd.concat([df for _, df in frames], ignore_index=True)
    raw["metric"] = raw["metric"].astype(str).str.lower()
    raw["method"] = raw["method"].astype(str).str.lower()
    assert_unique_keys(raw, ["dataset_id", "method", "metric", "source_block", "K" if "K" in raw.columns else "k"], "raw within-block duplicates")
    raw["K"] = pd.to_numeric(raw.get("K", raw.get("k")), errors="coerce").astype("Int64")
    k_missing = int(raw["K"].isna().sum())
    if k_missing > 0:
        msg = f"Rows with non-parseable K dropped: {k_missing}"
        if args.strict_schema:
            raise ValueError(msg)
        warnings.warn(msg)
    if args.merged_csv is None:
        overlap = raw.groupby(["dataset_id", "method", "metric", "K"])["source_block"].nunique()
        if bool((overlap > 1).any()):
            raise ValueError("Overlap detected across old/new source blocks for same (dataset_id,method,metric,K).")

    key_cols = ["dataset_id", "method", "metric", "source_block"]
    proto_csv = _extract_protocol_from_raw_rows(raw)
    if args.merged_csv is None:
        proto_chunk = pd.concat(
            [
                _load_protocol_meta(args.old_root, "old_baseline"),
                _load_protocol_meta(args.old_root, "new_only"),
            ],
            ignore_index=True,
        )
    else:
        proto_chunk = pd.DataFrame()
    merged = raw.copy()
    if len(proto_csv):
        proto_csv_agg = _aggregate_protocol_candidates(proto_csv, key_cols=key_cols, source_tag="merged_csv")
        assert_unique_keys(proto_csv_agg, key_cols, "aggregated proto_csv uniqueness")
        merged = merged.merge(
            proto_csv_agg.rename(
                columns={
                    "group_protocol_candidate_count": "group_protocol_candidate_count_csv",
                    "protocol_candidates_json": "protocol_candidates_json_csv",
                    "b_val": "b_used_csv",
                    "pairs_val": "max_pairs_used_csv",
                    "n_val": "n_points_used_csv",
                    "protocol_source": "protocol_source_csv",
                }
            ),
            on=key_cols,
            how="left",
        )
    if len(proto_chunk):
        chunk_agg = _aggregate_protocol_candidates(proto_chunk, key_cols=key_cols, source_tag="chunk_json_fallback")
        chunk_meta = (
            proto_chunk.groupby(key_cols, as_index=False)
            .agg(
                subsample_frac_chunk=("subsample_frac", lambda s: safe_mode_or_na(pd.to_numeric(s, errors="coerce"))),
                null_trials_chunk=("null_trials", lambda s: safe_mode_or_na(pd.to_numeric(s, errors="coerce"))),
                alpha_bh_chunk=("alpha_bh", lambda s: safe_mode_or_na(pd.to_numeric(s, errors="coerce"))),
            )
        )
        chunk_agg = chunk_agg.merge(chunk_meta, on=key_cols, how="left").rename(
            columns={
                "group_protocol_candidate_count": "group_protocol_candidate_count_chunk",
                "protocol_candidates_json": "protocol_candidates_json_chunk",
                "b_val": "b_used_chunk",
                "pairs_val": "max_pairs_used_chunk",
                "n_val": "n_points_used_chunk",
                "protocol_source": "protocol_source_chunk",
            }
        )
        assert_unique_keys(chunk_agg, key_cols, "aggregated proto_chunk uniqueness")
        merged = merged.merge(chunk_agg, on=key_cols, how="left")
    raw_b = _num_series(merged, "b_used_csv")
    chunk_b = _num_series(merged, "b_used_chunk")
    merged["b_used"] = raw_b.fillna(chunk_b)
    merged["max_pairs_used"] = _num_series(merged, "max_pairs_used_csv").fillna(_num_series(merged, "max_pairs_used_chunk"))
    merged["n_points_used"] = _num_series(merged, "n_points_used_csv").fillna(_num_series(merged, "n_points_used_chunk"))
    merged["subsample_frac"] = _num_series(merged, "subsample_frac").fillna(_num_series(merged, "subsample_frac_chunk"))
    merged["null_trials"] = _num_series(merged, "null_trials").fillna(_num_series(merged, "null_trials_chunk"))
    merged["alpha_bh"] = _num_series(merged, "alpha_bh").fillna(_num_series(merged, "alpha_bh_chunk"))
    merged["protocol_source"] = np.where(raw_b.notna(), "merged_csv", np.where(chunk_b.notna(), "chunk_json_fallback", "missing"))
    merged["group_protocol_candidate_count"] = pd.to_numeric(merged.get("group_protocol_candidate_count_csv"), errors="coerce")
    merged["group_protocol_candidate_count"] = merged["group_protocol_candidate_count"].fillna(pd.to_numeric(merged.get("group_protocol_candidate_count_chunk"), errors="coerce"))
    merged["group_protocol_candidate_count"] = merged["group_protocol_candidate_count"].fillna(1).astype("Int64")
    merged["group_protocol_uniform"] = (merged["group_protocol_candidate_count"] == 1).astype("boolean")
    merged["protocol_candidates_json"] = merged.get("protocol_candidates_json_csv", pd.Series(pd.NA, index=merged.index))
    chunk_candidates = merged.get("protocol_candidates_json_chunk")
    if chunk_candidates is not None:
        merged["protocol_candidates_json"] = merged["protocol_candidates_json"].fillna(chunk_candidates)
    merged["protocol_candidates_json"] = merged["protocol_candidates_json"].fillna("[]")

    standardized = pd.DataFrame(
        {
            "dataset_id": merged["dataset_id"],
            "method": merged["method"].astype(str).str.lower(),
            "metric": merged["metric"],
            "K": merged["K"].astype("Int64"),
            "T_mean": pd.to_numeric(merged["rep_mean"], errors="coerce"),
            "T_sd_pairs": pd.to_numeric(merged["rep_std"], errors="coerce"),
            "V_mean": _fallback_v_mean(merged),
            "null_mean": pd.to_numeric(merged.get("null_mu0"), errors="coerce"),
            "null_sd": pd.to_numeric(merged.get("null_sigma0"), errors="coerce"),
            "null_q95": pd.to_numeric(merged.get("null_q95"), errors="coerce"),
            "p_value": pd.to_numeric(merged.get("p_value"), errors="coerce"),
            "z_score": pd.to_numeric(merged.get("z_score"), errors="coerce"),
            "pairs_used": pd.to_numeric(merged.get("pairs_used"), errors="coerce").astype("Int64"),
            "source_block": merged["source_block"],
            "b_used": pd.to_numeric(merged.get("b_used"), errors="coerce").astype("Int64"),
            "max_pairs_used": pd.to_numeric(merged.get("max_pairs_used"), errors="coerce").astype("Int64"),
            "n_points_used": pd.to_numeric(merged.get("n_points_used"), errors="coerce").astype("Int64"),
            "subsample_frac": pd.to_numeric(merged.get("subsample_frac"), errors="coerce"),
            "null_trials": pd.to_numeric(merged.get("null_trials"), errors="coerce").astype("Int64"),
            "alpha_bh": pd.to_numeric(merged.get("alpha_bh"), errors="coerce"),
            "protocol_source": merged["protocol_source"],
            "group_protocol_uniform": merged["group_protocol_uniform"],
            "group_protocol_candidate_count": pd.to_numeric(merged.get("group_protocol_candidate_count"), errors="coerce").astype("Int64"),
            "protocol_candidates_json": merged.get("protocol_candidates_json"),
        }
    ).dropna(subset=["dataset_id", "method", "metric", "K"])

    standardized = standardized.sort_values(["dataset_id", "method", "metric", "K"]).reset_index(drop=True)
    standardized = standardized.groupby(["dataset_id", "method", "metric"], group_keys=False).apply(_add_group_extrema)
    def _apply_group_bh(g: pd.DataFrame) -> pd.DataFrame:
        alpha_vals = pd.to_numeric(g["alpha_bh"], errors="coerce").dropna().unique().tolist()
        if len(alpha_vals) == 0:
            alpha = 0.05
            alpha_source = "default"
        elif len(alpha_vals) == 1:
            alpha = float(alpha_vals[0])
            alpha_source = "group"
        else:
            raise ValueError(f"Multiple alpha_bh values in group {tuple(g[['dataset_id','method','metric']].iloc[0].tolist())}: {alpha_vals}")
        out = _apply_bh_flags(g, alpha=alpha)
        out["alpha_bh_used"] = alpha
        out["alpha_source"] = alpha_source
        return out
    standardized = standardized.groupby(["dataset_id", "method", "metric"], group_keys=False).apply(_apply_group_bh)
    n_samples_map: dict[str, float] = {}
    if args.dataset_inventory_csv.exists():
        inv = pd.read_csv(args.dataset_inventory_csv)
        if "dataset_id" in inv.columns and "n_samples" in inv.columns:
            inv_use = inv[["dataset_id", "n_samples"]].copy()
            inv_use["n_samples"] = pd.to_numeric(inv_use["n_samples"], errors="coerce")
            n_samples_map = {str(r["dataset_id"]): float(r["n_samples"]) for _, r in inv_use.dropna().iterrows()}
    else:
        warnings.warn(f"Missing dataset inventory at {args.dataset_inventory_csv}; n_samples unavailable for target-cap resolution.")
    standardized = _with_protocol_flags(
        standardized,
        target_cap_mode=args.target_cap_mode,
        force_cap=args.force_cap,
        n_samples_map=n_samples_map,
    )
    if "group_protocol_candidate_count_x" in standardized.columns or "group_protocol_candidate_count_y" in standardized.columns:
        standardized["group_protocol_candidate_count"] = pd.to_numeric(
            standardized.get("group_protocol_candidate_count_x"), errors="coerce"
        ).fillna(pd.to_numeric(standardized.get("group_protocol_candidate_count_y"), errors="coerce"))
        standardized["group_protocol_candidate_count"] = standardized["group_protocol_candidate_count"].astype("Int64")
    if "protocol_candidates_json_x" in standardized.columns or "protocol_candidates_json_y" in standardized.columns:
        standardized["protocol_candidates_json"] = standardized.get("protocol_candidates_json_x")
        standardized["protocol_candidates_json"] = standardized["protocol_candidates_json"].fillna(standardized.get("protocol_candidates_json_y"))
    standardized = standardized.drop(
        columns=[
            "group_protocol_candidate_count_x",
            "group_protocol_candidate_count_y",
            "protocol_candidates_json_x",
            "protocol_candidates_json_y",
        ],
        errors="ignore",
    )
    standardized["T_minus_q95"] = standardized["T_mean"] - standardized["null_q95"]
    standardized["T_minus_null_mean"] = standardized["T_mean"] - standardized["null_mean"]
    standardized["validity_present"] = standardized["V_mean"].notna().astype("boolean")

    standardized["k_bh"] = pd.to_numeric(standardized["k_bh"], errors="coerce").astype("Int64")
    standardized["k_at_max_T"] = pd.to_numeric(standardized["k_at_max_T"], errors="coerce").astype("Int64")
    standardized["k_at_max_V"] = pd.to_numeric(standardized["k_at_max_V"], errors="coerce").astype("Int64")
    standardized["k_at_max_Z"] = pd.to_numeric(standardized["k_at_max_Z"], errors="coerce").astype("Int64")
    standardized["is_k_bh"] = normalize_nullable_bool(standardized["is_k_bh"])
    standardized["bh_reject"] = normalize_nullable_bool(standardized["bh_reject"])
    standardized["validity_present"] = normalize_nullable_bool(standardized["validity_present"])

    missing_v = (
        standardized.groupby(["dataset_id", "method", "metric"])["V_mean"]
        .apply(lambda s: s.isna().all())
        .reset_index(name="all_v_missing")
    )
    bad = missing_v[missing_v["all_v_missing"] & missing_v["metric"].isin(["ami", "ari", "jaccard"])]
    if len(bad):
        msg = f"Groups with all-NaN V_mean for canonical metrics: {bad[['dataset_id','method','metric']].to_dict('records')}"
        if args.strict_schema:
            raise ValueError(msg)
        warnings.warn(msg)

    assert_unique_keys(standardized, ["dataset_id", "method", "metric", "K"], "k-sweep standardized final")

    standardized = standardized.sort_values(["dataset_id", "method", "metric", "K"]).reset_index(drop=True)
    standardized.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(standardized)} rows)")


if __name__ == "__main__":
    main()

