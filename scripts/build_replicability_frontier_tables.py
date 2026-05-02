#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

metric_order = {"ami": 0, "ari": 1, "jaccard": 2}
method_order = {"umap": 0, "tsne": 1}
metric_label = {"ami": "AMI", "ari": "ARI", "jaccard": "Jaccard"}
avg_score_cols = ["T_value", "V_value", "mu0", "sigma0", "q95", "z_value", "p_value"]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description=(
            "Build publication tables for fixed-K0 frontier analyses and "
            "cross-design comparisons against K-sweep outputs."
        )
    )
    p.add_argument(
        "--fixed-k0",
        type=Path,
        default=root / "outputs" / "fixed_k0_hyperparam_sweep" / "fixed_k0_hyperparam_sweep_results.csv",
        help="Fixed-K0 results CSV/JSON path.",
    )
    p.add_argument(
        "--k-sweep",
        type=Path,
        default=root / "outputs" / "replicability_benchmark_all_metrics" / "replicability_curves_all_metrics.csv",
        help="K-sweep results CSV/JSON path (all metrics preferred).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "fixed_k0_publication_tables",
        help="Output directory for tables and figures.",
    )
    p.add_argument(
        "--duplicate-policy",
        choices=["error", "mean"],
        default="error",
        help="How to handle duplicate rows on key identifiers.",
    )
    p.add_argument(
        "--allow-partial-table-b",
        action="store_true",
        help="Allow missing (dataset,method,metric) combinations for Table B.",
    )
    p.add_argument(
        "--export-compat-matched-alias",
        action="store_true",
        help="Export compatibility alias files with 'matched settings' name (not recommended).",
    )
    return p.parse_args()


def _format_num(x: object) -> str:
    if pd.isna(x):
        return "NA"
    x_f = float(x)
    if np.isclose(x_f, round(x_f)):
        return str(int(round(x_f)))
    return f"{x_f:g}"


def _method_label(method: str, value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"perp={_format_num(value)}" if method == "tsne" else f"nn={_format_num(value)}"


def _to_metric_key(s: object) -> str:
    v = str(s).strip().lower()
    if v in {"ami", "adjusted_mutual_info"}:
        return "ami"
    if v in {"ari", "adjusted_rand"}:
        return "ari"
    if v in {"jaccard"}:
        return "jaccard"
    return v


def _normalize_method(s: object) -> str:
    v = str(s).strip().lower()
    if v in {"t-sne", "tsne"}:
        return "tsne"
    if v in {"umap"}:
        return "umap"
    return v


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported input extension: {path.suffix}")


def _first_non_null(series: pd.Series) -> object:
    notna = series.dropna()
    return np.nan if len(notna) == 0 else notna.iloc[0]


def _unique_non_null(series: pd.Series) -> list[object]:
    vals = series.dropna().unique().tolist()
    return vals


def _resolve_duplicates(
    df: pd.DataFrame,
    key_cols: list[str],
    table_name: str,
    policy: str,
    avg_numeric_cols: list[str],
    constant_cols: list[str] | None = None,
    provenance_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    constant_cols = [c for c in (constant_cols or []) if c in df.columns and c not in key_cols]
    provenance_cols = [c for c in (provenance_cols or []) if c in df.columns and c not in key_cols]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    dup_df = df.loc[dup_mask].copy()
    duplicate_rows = int(len(dup_df))
    duplicate_groups = int(dup_df[key_cols].drop_duplicates().shape[0]) if duplicate_rows else 0

    stats = {
        "table_name": table_name,
        "raw_rows": int(len(df)),
        "rows_after_duplicate_handling": int(len(df)),
        "duplicate_rows_detected": duplicate_rows,
        "duplicate_groups_detected": duplicate_groups,
        "policy_used": policy,
        "avg_numeric_cols": [c for c in avg_numeric_cols if c in df.columns],
        "constant_cols_checked": constant_cols,
        "provenance_cols_allowed_to_vary": provenance_cols,
    }
    if duplicate_rows == 0:
        return df, stats, pd.DataFrame()

    msg = (
        f"{table_name} has duplicate key rows on {key_cols}. "
        f"Duplicate-row count: {duplicate_rows}. Duplicate groups: {duplicate_groups}."
    )

    if policy == "error":
        sample = dup_df[key_cols].drop_duplicates().head(10).to_dict("records")
        raise ValueError(f"{msg} Sample duplicate groups: {sample}")

    avg_cols = [c for c in avg_numeric_cols if c in df.columns and c not in key_cols]
    non_key_cols = [c for c in df.columns if c not in key_cols]
    inferred_constant_cols = list(constant_cols)
    for c in non_key_cols:
        if c in avg_cols or c in inferred_constant_cols or c in provenance_cols:
            continue
        inferred_constant_cols.append(c)
    other_cols = [c for c in non_key_cols if c not in avg_cols]
    agg: dict[str, object] = {}
    for c in avg_cols:
        agg[c] = "mean"
    for c in other_cols:
        agg[c] = _first_non_null

    inconsistent_rows: list[dict[str, object]] = []
    report_rows: list[dict[str, object]] = []
    for _, grp in dup_df.groupby(key_cols, dropna=False):
        key_info = {k: grp.iloc[0][k] for k in key_cols}
        inconsistent_cols = []
        for c in inferred_constant_cols:
            unique_vals = _unique_non_null(grp[c])
            if len(unique_vals) > 1:
                inconsistent_cols.append(c)
        provenance_summary = {
            f"{c}_unique_values": ";".join(str(v) for v in _unique_non_null(grp[c]))
            for c in provenance_cols
        }
        report_rows.append(
            {
                **key_info,
                "n_raw_rows": int(len(grp)),
                "policy_used": policy,
                "averaged_columns": ";".join(avg_cols),
                "constant_columns_checked": ";".join(inferred_constant_cols),
                "constant_columns_inconsistent": ";".join(inconsistent_cols),
                **provenance_summary,
            }
        )
        if inconsistent_cols:
            inconsistent_rows.append({**key_info, "inconsistent_cols": inconsistent_cols})

    if inconsistent_rows:
        sample = inconsistent_rows[:10]
        raise ValueError(
            f"{table_name} duplicate aggregation failed: constant columns vary within duplicate groups. "
            f"Sample: {sample}"
        )

    out = df.groupby(key_cols, as_index=False).agg(agg)
    if provenance_cols:
        prov_rows = []
        for _, grp in df.groupby(key_cols, dropna=False):
            item = {k: grp.iloc[0][k] for k in key_cols}
            for c in provenance_cols:
                uniq = _unique_non_null(grp[c])
                item[f"{c}_all_values"] = ";".join(str(v) for v in uniq)
                item[f"multiple_{c}"] = bool(len(uniq) > 1)
            prov_rows.append(item)
        out = out.merge(pd.DataFrame(prov_rows), on=key_cols, how="left")
    stats["rows_after_duplicate_handling"] = int(len(out))
    print(f"[warn] {msg} Aggregated using score-column whitelist only.")
    return out, stats, pd.DataFrame(report_rows)


def normalize_fixed_k0(
    df_raw: pd.DataFrame, duplicate_policy: str
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    df = df_raw.copy()
    if "dataset_id" not in df.columns or "method" not in df.columns:
        raise ValueError("Fixed-K0 input must include dataset_id and method.")

    df["method"] = df["method"].map(_normalize_method)
    if "k0" not in df.columns and "nominal_class_count" in df.columns:
        df["k0"] = df["nominal_class_count"]
    if "k0" not in df.columns:
        df["k0"] = np.nan

    if {"metric", "T_value", "V_value"}.issubset(df.columns):
        out = df.copy()
        out["metric"] = out["metric"].map(_to_metric_key)
    else:
        long_rows: list[pd.DataFrame] = []
        for m in ("ami", "ari", "jaccard"):
            suffix = metric_label[m]
            t_col = f"T_{suffix}"
            v_col = f"V_{suffix}"
            if t_col in df.columns and v_col in df.columns:
                d = df.copy()
                d["metric"] = m
                d["T_value"] = d[t_col]
                d["V_value"] = d[v_col]
                for in_name, out_name in [
                    (f"mu0_{suffix}", "mu0"),
                    (f"sigma0_{suffix}", "sigma0"),
                    (f"q95_{suffix}", "q95"),
                    (f"Z_{suffix}", "z_value"),
                    (f"p_{suffix}", "p_value"),
                ]:
                    if in_name in d.columns:
                        d[out_name] = d[in_name]
                long_rows.append(d)
        if not long_rows:
            raise ValueError(
                "Fixed-K0 input must be long format (metric/T_value/V_value) or wide format with T_*/V_*."
            )
        out = pd.concat(long_rows, ignore_index=True)

    if "hyperparam_name" not in out.columns or "hyperparam_value" not in out.columns:
        hp_name = pd.Series(index=out.index, dtype="object")
        hp_value = pd.Series(index=out.index, dtype="float64")
        if "primary_hyperparam_name" in out.columns:
            hp_name = out["primary_hyperparam_name"].astype(str)
        if "primary_hyperparam_value" in out.columns:
            hp_value = pd.to_numeric(out["primary_hyperparam_value"], errors="coerce")
        if "tsne_perplexity" in out.columns:
            m = out["method"].eq("tsne") & hp_value.isna()
            hp_value.loc[m] = pd.to_numeric(out.loc[m, "tsne_perplexity"], errors="coerce")
            hp_name.loc[m] = "perplexity"
        if "perplexity" in out.columns:
            m = out["method"].eq("tsne") & hp_value.isna()
            hp_value.loc[m] = pd.to_numeric(out.loc[m, "perplexity"], errors="coerce")
            hp_name.loc[m] = "perplexity"
        if "umap_n_neighbors" in out.columns:
            m = out["method"].eq("umap") & hp_value.isna()
            hp_value.loc[m] = pd.to_numeric(out.loc[m, "umap_n_neighbors"], errors="coerce")
            hp_name.loc[m] = "n_neighbors"
        if "n_neighbors" in out.columns:
            m = out["method"].eq("umap") & hp_value.isna()
            hp_value.loc[m] = pd.to_numeric(out.loc[m, "n_neighbors"], errors="coerce")
            hp_name.loc[m] = "n_neighbors"
        hp_name = hp_name.replace({"tsne_perplexity": "perplexity", "umap_n_neighbors": "n_neighbors"})
        out["hyperparam_name"] = hp_name
        out["hyperparam_value"] = hp_value

    for src, dst in [("null_mu0", "mu0"), ("null_sigma0", "sigma0"), ("null_q95", "q95"), ("z_score", "z_value")]:
        if dst not in out.columns and src in out.columns:
            out[dst] = out[src]

    out["metric"] = out["metric"].map(_to_metric_key)
    for c in ["T_value", "V_value", "k0", "hyperparam_value", "mu0", "sigma0", "q95", "z_value", "p_value"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["hyperparam_value", "T_value", "V_value"])

    keep_cols = [
        "dataset_id",
        "method",
        "metric",
        "k0",
        "hyperparam_name",
        "hyperparam_value",
        "T_value",
        "V_value",
        "mu0",
        "sigma0",
        "q95",
        "z_value",
        "p_value",
        "run_id",
        "fixed_run_id",
        "preprocessing",
        "fixed_preprocessing",
        "seed_policy",
        "fixed_seed_policy",
        "representation_description",
        "label_construction_note",
        "n_raw",
        "n_original",
        "n_used",
        "cap_policy",
        "number_of_classes_retained",
        "min_class_support",
        "median_class_support",
        "max_class_support",
        "number_of_classes_declared",
        "classes_dropped",
    ]
    out = out[[c for c in keep_cols if c in out.columns]].copy()
    out, stats, dup_report = _resolve_duplicates(
        out,
        key_cols=["dataset_id", "method", "metric", "hyperparam_value"],
        table_name="Fixed-K0 table",
        policy=duplicate_policy,
        avg_numeric_cols=avg_score_cols,
        constant_cols=["k0", "hyperparam_name"],
        provenance_cols=[
            "run_id",
            "fixed_run_id",
            "preprocessing",
            "fixed_preprocessing",
            "seed_policy",
            "fixed_seed_policy",
        ],
    )
    return out, stats, dup_report


def _fill_v_from_metric_specific_columns(df: pd.DataFrame) -> pd.Series:
    v = pd.Series(np.nan, index=df.index, dtype="float64")
    if "metric_vs_gt_mean" in df.columns:
        v = pd.to_numeric(df["metric_vs_gt_mean"], errors="coerce")
    if "ari_vs_gt_mean" in df.columns:
        m = df["metric"].eq("ari") & v.isna()
        v.loc[m] = pd.to_numeric(df.loc[m, "ari_vs_gt_mean"], errors="coerce")
    if "ami_vs_gt_mean" in df.columns:
        m = df["metric"].eq("ami") & v.isna()
        v.loc[m] = pd.to_numeric(df.loc[m, "ami_vs_gt_mean"], errors="coerce")
    if "jaccard_vs_gt_mean" in df.columns:
        m = df["metric"].eq("jaccard") & v.isna()
        v.loc[m] = pd.to_numeric(df.loc[m, "jaccard_vs_gt_mean"], errors="coerce")
    return v


def normalize_ksweep(
    df_raw: pd.DataFrame, duplicate_policy: str
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    df = df_raw.copy()
    if "dataset_id" not in df.columns or "method" not in df.columns:
        raise ValueError("K-sweep input must include dataset_id and method.")

    df["method"] = df["method"].map(_normalize_method)
    if "K" not in df.columns and "k" in df.columns:
        df["K"] = df["k"]
    if "metric" not in df.columns:
        raise ValueError("K-sweep input must include metric.")
    df["metric"] = df["metric"].map(_to_metric_key)

    if "T_value" not in df.columns:
        if "rep_mean" in df.columns:
            df["T_value"] = df["rep_mean"]
        else:
            raise ValueError("K-sweep input must include T_value or rep_mean.")
    if "V_value" not in df.columns:
        df["V_value"] = _fill_v_from_metric_specific_columns(df)

    for c in ["K", "T_value", "V_value", "null_mu0", "null_sigma0", "null_q95", "z_score", "p_value"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    missing_v = df["V_value"].isna()
    if missing_v.any():
        missing = df.loc[missing_v].groupby("metric", as_index=False).size().to_dict("records")
        raise ValueError(
            "K-sweep V_value missing after metric-safe fallback. "
            f"Missing rows by metric: {missing}. "
            "Provide metric-consistent GT-agreement columns."
        )

    keep_cols = [
        "dataset_id",
        "method",
        "metric",
        "K",
        "T_value",
        "V_value",
        "null_mu0",
        "null_sigma0",
        "null_q95",
        "z_score",
        "p_value",
        "run_id",
        "ksweep_run_id",
        "preprocessing",
        "ksweep_preprocessing",
        "seed_policy",
        "ksweep_seed_policy",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    out = out.dropna(subset=["K", "T_value", "V_value"])
    out, stats, dup_report = _resolve_duplicates(
        out,
        key_cols=["dataset_id", "method", "metric", "K"],
        table_name="K-sweep table",
        policy=duplicate_policy,
        avg_numeric_cols=avg_score_cols,
        # The key columns already define the K-sweep row.
        # Any other non-score metadata must therefore be constant or fail loudly.
        constant_cols=[],
        provenance_cols=[
            "run_id",
            "ksweep_run_id",
            "preprocessing",
            "ksweep_preprocessing",
            "seed_policy",
            "ksweep_seed_policy",
        ],
    )
    return out, stats, dup_report


def _argmax_info(group: pd.DataFrame, col: str) -> tuple[float, float, str, int]:
    max_val = float(group[col].max())
    winners = group[np.isclose(group[col], max_val)]["hyperparam_value"].sort_values()
    return max_val, float(winners.iloc[0]), ";".join(_format_num(v) for v in winners.tolist()), int(len(winners))


def _value_at_hyperparam(group: pd.DataFrame, value_col: str, hp_value: float) -> float:
    rows = group[np.isclose(group["hyperparam_value"], hp_value)]
    if len(rows) == 0:
        return float("nan")
    return float(rows[value_col].iloc[0])


def _winner_high(a: float, b: float) -> str:
    if np.isclose(a, b, equal_nan=True):
        return "tie"
    return "umap" if a > b else "tsne"


def _winner_low(a: float, b: float) -> str:
    if np.isclose(a, b, equal_nan=True):
        return "tie"
    return "umap" if a < b else "tsne"


def _group_value(group: pd.DataFrame, preferred: list[str]) -> object:
    for c in preferred:
        if c in group.columns:
            vals = _unique_non_null(group[c])
            if len(vals) == 1:
                return vals[0]
    return np.nan


def _group_values_all(group: pd.DataFrame, preferred: list[str]) -> object:
    for c in preferred:
        all_col = f"{c}_all_values"
        if all_col in group.columns:
            vals = _unique_non_null(group[all_col])
            if len(vals) > 0:
                return vals[0]
        if c in group.columns:
            vals = _unique_non_null(group[c])
            if len(vals) > 0:
                return ";".join(str(v) for v in vals)
    return np.nan


def _group_multiple_flag(group: pd.DataFrame, preferred: list[str]) -> bool:
    for c in preferred:
        multi_col = f"multiple_{c}"
        if multi_col in group.columns:
            vals = _unique_non_null(group[multi_col])
            if len(vals) == 1:
                return bool(vals[0])
            if len(vals) > 1:
                return True
        if c in group.columns:
            return len(_unique_non_null(group[c])) > 1
    return False


def build_table_a(fixed_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset_id, method, metric), g in fixed_long.groupby(["dataset_id", "method", "metric"], sort=False):
        g = g.sort_values("hyperparam_value").reset_index(drop=True)
        max_t, argmax_t, all_t, n_tie_t = _argmax_info(g, "T_value")
        max_v, argmax_v, all_v, n_tie_v = _argmax_info(g, "V_value")
        v_at_argmax_t = _value_at_hyperparam(g, "V_value", argmax_t)
        t_at_argmax_v = _value_at_hyperparam(g, "T_value", argmax_v)

        row = {
            "dataset_id": dataset_id,
            "method": method,
            "metric": metric,
            "is_primary_endpoint": metric == "ami",
            "metric_primary": metric == "ami",
            "k0": float(g["k0"].iloc[0]) if "k0" in g.columns else np.nan,
            "hyperparam_name": g["hyperparam_name"].iloc[0],
            "max_T": max_t,
            "argmax_T": argmax_t,
            "argmax_T_all": all_t,
            "n_argmax_T": n_tie_t,
            "V_at_argmax_T": v_at_argmax_t,
            "max_V": max_v,
            "argmax_V": argmax_v,
            "argmax_V_all": all_v,
            "n_argmax_V": n_tie_v,
            "T_at_argmax_V": t_at_argmax_v,
            "delta_opt": v_at_argmax_t - max_v,
            "delta_T_reverse": t_at_argmax_v - max_t,
            "argmax_match": bool(np.isclose(argmax_t, argmax_v)),
        }
        for in_col, out_col in [("z_value", "z_at_argmax_T"), ("p_value", "p_at_argmax_T"), ("q95", "q95_at_argmax_T")]:
            if in_col in g.columns:
                row[out_col] = _value_at_hyperparam(g, in_col, argmax_t)
        if "q95_at_argmax_T" in row:
            row["T_minus_q95_at_argmax_T"] = row["max_T"] - row["q95_at_argmax_T"]

        row["frontier_status"] = "aligned" if row["argmax_match"] else "mismatch"
        row["delta_opt_abs"] = abs(row["delta_opt"])
        rows.append(row)

    out = pd.DataFrame(rows)
    out["metric_rank"] = out["metric"].map(metric_order)
    out["method_rank"] = out["method"].map(method_order).fillna(99)
    out = out.sort_values(["dataset_id", "metric_rank", "method_rank"]).drop(columns=["metric_rank", "method_rank"])
    out["argmax_T_label"] = out.apply(lambda r: _method_label(r["method"], r["argmax_T"]), axis=1)
    out["argmax_V_label"] = out.apply(lambda r: _method_label(r["method"], r["argmax_V"]), axis=1)
    return out.reset_index(drop=True)


def build_table_method_same_numeric_values(fixed_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = fixed_long[["dataset_id", "metric", "method", "hyperparam_value", "T_value", "V_value"]].copy()
    umap = d[d["method"] == "umap"].rename(columns={"T_value": "T_umap", "V_value": "V_umap"})
    tsne = d[d["method"] == "tsne"].rename(columns={"T_value": "T_tsne", "V_value": "V_tsne"})
    merged = umap.merge(
        tsne[["dataset_id", "metric", "hyperparam_value", "T_tsne", "V_tsne"]],
        on=["dataset_id", "metric", "hyperparam_value"],
        how="inner",
    )
    merged["delta_T_umap_minus_tsne"] = merged["T_umap"] - merged["T_tsne"]
    merged["delta_V_umap_minus_tsne"] = merged["V_umap"] - merged["V_tsne"]
    merged["is_primary_endpoint"] = merged["metric"].eq("ami")
    merged["comparison_type"] = "same numeric knob value comparison"
    merged["interpretation_note"] = "Same numeric knob value comparison; not matched settings."
    merged["metric_rank"] = merged["metric"].map(metric_order)
    merged = merged.sort_values(["dataset_id", "metric_rank", "hyperparam_value"]).drop(columns=["method", "metric_rank"])

    summary = merged.groupby(["dataset_id", "metric"], as_index=False).agg(
        n_shared_numeric_values=("hyperparam_value", "size"),
        mean_delta_T=("delta_T_umap_minus_tsne", "mean"),
        median_delta_T=("delta_T_umap_minus_tsne", "median"),
        positive_delta_T_count=("delta_T_umap_minus_tsne", lambda x: int((x > 0).sum())),
        frac_delta_T_positive=("delta_T_umap_minus_tsne", lambda x: float((x > 0).mean())),
        mean_delta_V=("delta_V_umap_minus_tsne", "mean"),
        median_delta_V=("delta_V_umap_minus_tsne", "median"),
        positive_delta_V_count=("delta_V_umap_minus_tsne", lambda x: int((x > 0).sum())),
        frac_delta_V_positive=("delta_V_umap_minus_tsne", lambda x: float((x > 0).mean())),
    )
    summary["is_primary_endpoint"] = summary["metric"].eq("ami")
    summary["comparison_type"] = "same numeric knob value comparison"
    summary["interpretation_note"] = "Same numeric knob value comparison; not matched settings."
    summary["metric_rank"] = summary["metric"].map(metric_order)
    summary = summary.sort_values(["dataset_id", "metric_rank"]).drop(columns=["metric_rank"])
    return merged.reset_index(drop=True), summary.reset_index(drop=True)


def _argmax_k_info(group: pd.DataFrame, col: str) -> tuple[float, float]:
    max_val = float(group[col].max())
    winners = group[np.isclose(group[col], max_val)]["K"].sort_values()
    return max_val, float(winners.iloc[0])


def _rank_desc(values: np.ndarray, target: float) -> int:
    return int(1 + np.sum(values > target))


def _validate_table_b_coverage(
    fixed_long: pd.DataFrame, ksweep_long: pd.DataFrame, allow_partial: bool
) -> set[tuple[str, str, str]]:
    fixed_keys = set(fixed_long[["dataset_id", "method", "metric"]].drop_duplicates().itertuples(index=False, name=None))
    sweep_keys = set(ksweep_long[["dataset_id", "method", "metric"]].drop_duplicates().itertuples(index=False, name=None))
    missing = fixed_keys - sweep_keys
    if missing and not allow_partial:
        raise ValueError(
            "Table B coverage failure: some fixed-K0 groups are absent in K-sweep input. "
            f"Sample missing groups: {list(sorted(missing))[:10]}"
        )
    if missing:
        print(f"[warn] Table B will skip {len(missing)} groups missing from K-sweep input.")
    return missing


def build_table_b(
    fixed_long: pd.DataFrame,
    ksweep_long: pd.DataFrame,
    allow_partial: bool,
    fixed_source: str,
    ksweep_source: str,
) -> tuple[pd.DataFrame, set[tuple[str, str, str]]]:
    missing_groups = _validate_table_b_coverage(fixed_long, ksweep_long, allow_partial)
    rows: list[dict[str, object]] = []
    sweep_groups = ksweep_long.groupby(["dataset_id", "method", "metric"], sort=False)

    for key, gf in fixed_long.groupby(["dataset_id", "method", "metric"], sort=False):
        if key not in sweep_groups.groups:
            continue
        dataset_id, method, metric = key
        gs = sweep_groups.get_group(key).sort_values("K")

        max_t_theta, argmax_theta_t, _, _ = _argmax_info(gf, "T_value")
        max_v_theta, argmax_theta_v, _, _ = _argmax_info(gf, "V_value")
        v_at_argmax_t = _value_at_hyperparam(gf, "V_value", argmax_theta_t)
        t_at_argmax_v = _value_at_hyperparam(gf, "T_value", argmax_theta_v)
        max_t_k, argmax_k_t = _argmax_k_info(gs, "T_value")
        max_v_k, argmax_k_v = _argmax_k_info(gs, "V_value")

        k0 = float(gf["k0"].iloc[0]) if "k0" in gf.columns else np.nan
        k0_rows = gs[np.isclose(gs["K"], k0)]
        t_at_k0 = float(k0_rows["T_value"].iloc[0]) if len(k0_rows) else np.nan
        v_at_k0 = float(k0_rows["V_value"].iloc[0]) if len(k0_rows) else np.nan

        row = {
            "dataset_id": dataset_id,
            "method": method,
            "metric": metric,
            "is_primary_endpoint": metric == "ami",
            "cross_design_comparison": True,
            "comparison_note": "fixed-K0 hyperparam sweep vs original K sweep",
            "comparison_scope": "fixed optimum over theta compared against sweep over K",
            "rank_interpretation": "Rank of fixed-K0 optimum value among old K-sweep values",
            "fixed_k0_source": fixed_source,
            "ksweep_source": ksweep_source,
            "k0": k0,
            "max_T_over_K": max_t_k,
            "argmax_K_T": argmax_k_t,
            "T_at_K0_from_sweep": t_at_k0,
            "max_V_over_K": max_v_k,
            "argmax_K_V": argmax_k_v,
            "V_at_K0_from_sweep": v_at_k0,
            "max_T_over_theta_at_K0": max_t_theta,
            "argmax_theta_T": argmax_theta_t,
            "V_at_argmax_theta_T": v_at_argmax_t,
            "max_V_over_theta_at_K0": max_v_theta,
            "argmax_theta_V": argmax_theta_v,
            "T_at_argmax_theta_V": t_at_argmax_v,
            "delta_T_fixed_vs_bestK": max_t_theta - max_t_k,
            "delta_V_fixed_vs_bestK": max_v_theta - max_v_k,
            "delta_T_fixed_vs_K0sweep": max_t_theta - t_at_k0,
            "delta_V_fixed_vs_K0sweep": max_v_theta - v_at_k0,
            "rank_T_fixed_among_sweep": _rank_desc(gs["T_value"].to_numpy(dtype=float), max_t_theta),
            "rank_V_fixed_among_sweep": _rank_desc(gs["V_value"].to_numpy(dtype=float), max_v_theta),
            "fixed_run_id": _group_value(gf, ["fixed_run_id", "run_id"]),
            "ksweep_run_id": _group_value(gs, ["ksweep_run_id", "run_id"]),
            "fixed_run_id_all_values": _group_values_all(gf, ["fixed_run_id", "run_id"]),
            "ksweep_run_id_all_values": _group_values_all(gs, ["ksweep_run_id", "run_id"]),
            "multiple_fixed_run_id": _group_multiple_flag(gf, ["fixed_run_id", "run_id"]),
            "multiple_ksweep_run_id": _group_multiple_flag(gs, ["ksweep_run_id", "run_id"]),
            "fixed_preprocessing": _group_value(gf, ["fixed_preprocessing", "preprocessing"]),
            "ksweep_preprocessing": _group_value(gs, ["ksweep_preprocessing", "preprocessing"]),
            "fixed_preprocessing_all_values": _group_values_all(gf, ["fixed_preprocessing", "preprocessing"]),
            "ksweep_preprocessing_all_values": _group_values_all(gs, ["ksweep_preprocessing", "preprocessing"]),
            "fixed_seed_policy": _group_value(gf, ["fixed_seed_policy", "seed_policy"]),
            "ksweep_seed_policy": _group_value(gs, ["ksweep_seed_policy", "seed_policy"]),
            "fixed_seed_policy_all_values": _group_values_all(gf, ["fixed_seed_policy", "seed_policy"]),
            "ksweep_seed_policy_all_values": _group_values_all(gs, ["ksweep_seed_policy", "seed_policy"]),
        }
        row["fixed_k0_better_on_T_vs_bestK"] = bool(row["delta_T_fixed_vs_bestK"] > 0)
        row["fixed_k0_better_on_V_vs_bestK"] = bool(row["delta_V_fixed_vs_bestK"] > 0)
        row["fixed_k0_better_on_T_vs_K0sweep"] = bool(row["delta_T_fixed_vs_K0sweep"] > 0)
        row["fixed_k0_better_on_V_vs_K0sweep"] = bool(row["delta_V_fixed_vs_K0sweep"] > 0)
        rows.append(row)

    out = pd.DataFrame(rows)
    out["metric_rank"] = out["metric"].map(metric_order)
    out["method_rank"] = out["method"].map(method_order).fillna(99)
    out = out.sort_values(["dataset_id", "metric_rank", "method_rank"]).drop(columns=["metric_rank", "method_rank"])
    out["argmax_theta_T_label"] = out.apply(lambda r: _method_label(r["method"], r["argmax_theta_T"]), axis=1)
    out["argmax_theta_V_label"] = out.apply(lambda r: _method_label(r["method"], r["argmax_theta_V"]), axis=1)
    return out.reset_index(drop=True), missing_groups


def build_method_win_loss_summary(table_a: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (dataset_id, metric), g in table_a.groupby(["dataset_id", "metric"], sort=False):
        if set(g["method"]) < {"umap", "tsne"}:
            continue
        u = g[g["method"] == "umap"].iloc[0]
        t = g[g["method"] == "tsne"].iloc[0]
        row = {
            "dataset_id": dataset_id,
            "metric": metric,
            "is_primary_endpoint": metric == "ami",
            "max_T_umap": float(u["max_T"]),
            "max_T_tsne": float(t["max_T"]),
            "winner_max_T": _winner_high(float(u["max_T"]), float(t["max_T"])),
            "max_V_umap": float(u["max_V"]),
            "max_V_tsne": float(t["max_V"]),
            "winner_max_V": _winner_high(float(u["max_V"]), float(t["max_V"])),
            "abs_delta_opt_umap": float(abs(u["delta_opt"])),
            "abs_delta_opt_tsne": float(abs(t["delta_opt"])),
            "winner_abs_delta_opt": _winner_low(float(abs(u["delta_opt"])), float(abs(t["delta_opt"]))),
            "argmax_match_umap": bool(u["argmax_match"]),
            "argmax_match_tsne": bool(t["argmax_match"]),
            "z_at_argmax_T_umap": float(u["z_at_argmax_T"]) if "z_at_argmax_T" in u else np.nan,
            "z_at_argmax_T_tsne": float(t["z_at_argmax_T"]) if "z_at_argmax_T" in t else np.nan,
        }
        row["winner_z_at_argmax_T"] = (
            "not_available"
            if pd.isna(row["z_at_argmax_T_umap"]) and pd.isna(row["z_at_argmax_T_tsne"])
            else _winner_high(row["z_at_argmax_T_umap"], row["z_at_argmax_T_tsne"])
        )
        rows.append(row)

    row_level = pd.DataFrame(rows)
    if len(row_level) == 0:
        return row_level, pd.DataFrame()

    overall_rows: list[dict[str, object]] = []
    for metric, g in row_level.groupby("metric", sort=False):
        for criterion in ["winner_max_T", "winner_max_V", "winner_abs_delta_opt", "winner_z_at_argmax_T"]:
            counts = g[criterion].value_counts()
            row = {
                "metric": metric,
                "criterion": criterion,
                "n_datasets": int(len(g)),
            }
            for category, count in counts.items():
                row[f"count_{category}"] = int(count)
            overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)
    return row_level, overall


def export_table(df: pd.DataFrame, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base.with_suffix(".csv"), index=False)
    df.to_markdown(base.with_suffix(".md"), index=False)
    latex_df = df.copy()
    for numeric_col, label_col in [("argmax_T", "argmax_T_label"), ("argmax_V", "argmax_V_label"), ("argmax_theta_T", "argmax_theta_T_label"), ("argmax_theta_V", "argmax_theta_V_label")]:
        if numeric_col in latex_df.columns and label_col in latex_df.columns:
            latex_df[numeric_col] = latex_df[label_col]
    numeric_cols = latex_df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        latex_df[c] = latex_df[c].map(lambda x: np.nan if pd.isna(x) else round(float(x), 3))
    latex_df.to_latex(base.with_suffix(".tex"), index=False, na_rep="NA")


def build_primary_and_robustness(table_a: pd.DataFrame, output_dir: Path) -> None:
    export_table(table_a[table_a["metric"] == "ami"].copy(), output_dir / "paper_table_primary_AMI")
    export_table(table_a[table_a["metric"].isin(["ari", "jaccard"])].copy(), output_dir / "paper_table_robustness_ARI_Jaccard")


def plot_family_a(fixed_long: pd.DataFrame, table_a: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures" / "family_a"
    fig_dir.mkdir(parents=True, exist_ok=True)
    lookup = table_a.set_index(["dataset_id", "method", "metric"])
    for (dataset_id, metric), g in fixed_long.groupby(["dataset_id", "metric"], sort=False):
        methods = [m for m in ["tsne", "umap"] if m in set(g["method"])]
        if not methods:
            continue
        fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4), squeeze=False)
        for i, method in enumerate(methods):
            ax = axes[0, i]
            gm = g[g["method"] == method].sort_values("hyperparam_value")
            ax.plot(gm["hyperparam_value"], gm["T_value"], marker="o", label="T (replicability)")
            ax.plot(gm["hyperparam_value"], gm["V_value"], marker="s", label="V (truth agreement)")
            ax.set_title(f"{dataset_id} / {metric_label[metric]} / {method}")
            key = (dataset_id, method, metric)
            if key in lookup.index:
                row = lookup.loc[key]
                t_x, v_x = float(row["argmax_T"]), float(row["argmax_V"])
                if np.isclose(t_x, v_x):
                    ax.axvline(t_x, color="black", linestyle="--", alpha=0.8, label="argmax T = argmax V")
                else:
                    ax.axvline(t_x, color="C0", linestyle="--", alpha=0.7, label="argmax T")
                    ax.axvline(v_x, color="C1", linestyle=":", alpha=0.7, label="argmax V")
                ax.set_title(
                    f"{dataset_id} / {metric_label[metric]} / {method}\n"
                    f"delta_opt={float(row['delta_opt']):.3f}, {row['frontier_status']}"
                )
            xlab = "perplexity" if method == "tsne" else "n_neighbors"
            ax.set_xlabel(xlab)
            ax.set_ylabel("score")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25)
            ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"{dataset_id}_{metric}_frontier.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_family_b(table_same_numeric: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures" / "family_b"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for (dataset_id, metric), g in table_same_numeric.groupby(["dataset_id", "metric"], sort=False):
        gm = g.sort_values("hyperparam_value")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(gm["hyperparam_value"], gm["delta_T_umap_minus_tsne"], marker="o", label="UMAP - tSNE on T")
        ax.plot(gm["hyperparam_value"], gm["delta_V_umap_minus_tsne"], marker="s", label="UMAP - tSNE on V")
        ax.axhline(0.0, color="gray", linestyle="--", alpha=0.6)
        ax.set_xlabel("shared numeric knob value")
        ax.set_ylabel("delta")
        ax.set_title(f"{dataset_id} / {metric_label[metric]} same-numeric-value deltas (not matched settings)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"{dataset_id}_{metric}_same_numeric_delta.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def print_input_summary(fixed_long: pd.DataFrame, ksweep_long: pd.DataFrame) -> None:
    print("Input Summary")
    print("------------")
    print(f"Fixed-K0 rows:  {len(fixed_long):,}")
    print(f"K-sweep rows:   {len(ksweep_long):,}")
    print(f"Datasets:       {sorted(fixed_long['dataset_id'].dropna().unique().tolist())}")
    print(f"Methods:        {sorted(fixed_long['method'].dropna().unique().tolist())}")
    print(f"Fixed metrics:  {sorted(fixed_long['metric'].dropna().unique().tolist())}")
    print(f"Sweep metrics:  {sorted(ksweep_long['metric'].dropna().unique().tolist())}")


def write_output_notes(output_dir: Path, alias_exported: bool) -> None:
    note = (
        "# Output Notes\n\n"
        "- `table_fixed_k0_frontier`: Main fixed-K0 frontier table (Table A).\n"
        "- `table_method_same_numeric_knob_values`: UMAP vs t-SNE comparison on shared numeric knob values.\n"
        "- `table_method_same_numeric_knob_values_summary`: Aggregated same-numeric-value deltas.\n"
        "- `table_fixed_k0_vs_ksweep`: Secondary cross-design comparison (Table B).\n"
        "- `paper_table_primary_AMI`: AMI-only publication table (primary endpoint).\n"
        "- `paper_table_robustness_ARI_Jaccard`: ARI/Jaccard robustness appendix table.\n"
        "- `table_method_win_loss_summary`: Per-dataset, per-metric UMAP vs t-SNE win/loss table.\n\n"
        "Interpretation notes:\n"
        "- AMI is the primary endpoint.\n"
        "- ARI and Jaccard are robustness checks.\n"
        "- Same-numeric-value comparison is not a matched-settings comparison.\n"
        "- Table B is a secondary cross-design analysis.\n"
    )
    if alias_exported:
        note += "\nCompatibility alias files were exported with warning labels.\n"
    (output_dir / "OUTPUT_NOTES.md").write_text(note, encoding="utf-8")


def write_dataset_diagnostics(fixed_long: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    preferred_cols = [
        "dataset_id",
        "k0",
        "n_raw",
        "n_used",
        "representation_description",
        "label_construction_note",
        "min_class_support",
        "median_class_support",
        "max_class_support",
        "number_of_classes_retained",
        "random_state",
    ]
    keep = [c for c in preferred_cols if c in fixed_long.columns]
    if not keep:
        diag = pd.DataFrame(columns=preferred_cols)
    else:
        diag = fixed_long[keep].drop_duplicates()
        if "dataset_id" in diag.columns:
            diag = diag.sort_values(["dataset_id"]).reset_index(drop=True)
    diag.to_csv(output_dir / "dataset_diagnostics.csv", index=False)
    return diag


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_raw = _read_table(args.fixed_k0)
    ksweep_raw = _read_table(args.k_sweep)

    fixed_long, fixed_stats, fixed_dup_report = normalize_fixed_k0(fixed_raw, duplicate_policy=args.duplicate_policy)
    ksweep_long, ksweep_stats, ksweep_dup_report = normalize_ksweep(ksweep_raw, duplicate_policy=args.duplicate_policy)
    print_input_summary(fixed_long, ksweep_long)

    table_a = build_table_a(fixed_long)
    table_same_numeric, table_same_numeric_summary = build_table_method_same_numeric_values(fixed_long)
    shared_numeric_groups = set(
        table_same_numeric[["dataset_id", "metric"]].drop_duplicates().itertuples(index=False, name=None)
    )
    comparable_groups = []
    for (dataset_id, metric), g in fixed_long.groupby(["dataset_id", "metric"], sort=False):
        methods_here = set(g["method"].dropna().tolist())
        if {"umap", "tsne"}.issubset(methods_here):
            comparable_groups.append((dataset_id, metric))
    no_shared_numeric_groups = sorted(set(comparable_groups) - shared_numeric_groups)
    if no_shared_numeric_groups:
        print(
            "[warn] No shared numeric knob values for some dataset/metric groups: "
            f"{no_shared_numeric_groups[:10]}"
        )
    table_b, missing_groups = build_table_b(
        fixed_long,
        ksweep_long,
        allow_partial=args.allow_partial_table_b,
        fixed_source=args.fixed_k0.name,
        ksweep_source=args.k_sweep.name,
    )
    win_loss, win_loss_overall = build_method_win_loss_summary(table_a)

    export_table(table_a, args.output_dir / "table_fixed_k0_frontier")
    export_table(table_same_numeric, args.output_dir / "table_method_same_numeric_knob_values")
    export_table(table_same_numeric_summary, args.output_dir / "table_method_same_numeric_knob_values_summary")

    alias_exported = False
    if args.export_compat_matched_alias:
        alias_exported = True
        print(
            "[warn] Compatibility alias export enabled: these files are same numeric knob value comparisons, "
            "not scientifically matched settings."
        )
        alias_note = "Compatibility alias only: this table compares same numeric knob values, not matched settings."
        alias_table = table_same_numeric.copy()
        alias_summary = table_same_numeric_summary.copy()
        alias_table["warning_note"] = alias_note
        alias_summary["warning_note"] = alias_note
        export_table(alias_table, args.output_dir / "table_method_matched_settings")
        export_table(alias_summary, args.output_dir / "table_method_matched_settings_summary")

    export_table(table_b, args.output_dir / "table_fixed_k0_vs_ksweep")
    export_table(win_loss, args.output_dir / "table_method_win_loss_summary")
    export_table(win_loss_overall, args.output_dir / "table_method_win_loss_overall")
    build_primary_and_robustness(table_a, args.output_dir)
    plot_family_a(fixed_long, table_a, args.output_dir)
    plot_family_b(table_same_numeric, args.output_dir)

    if len(fixed_dup_report) > 0:
        fixed_dup_report.to_csv(args.output_dir / "duplicate_report_fixed_k0.csv", index=False)
    if len(ksweep_dup_report) > 0:
        ksweep_dup_report.to_csv(args.output_dir / "duplicate_report_ksweep.csv", index=False)

    dataset_diag = write_dataset_diagnostics(fixed_long, args.output_dir)
    write_output_notes(args.output_dir, alias_exported=alias_exported)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_name": Path(__file__).name,
        "fixed_k0_input_path": str(args.fixed_k0),
        "ksweep_input_path": str(args.k_sweep),
        "fixed_k0_source": args.fixed_k0.name,
        "ksweep_source": args.k_sweep.name,
        "duplicate_policy": args.duplicate_policy,
        "allow_partial_table_b": bool(args.allow_partial_table_b),
        "compat_alias_exported": alias_exported,
        "datasets_found": sorted(fixed_long["dataset_id"].dropna().unique().tolist()),
        "methods_found": sorted(fixed_long["method"].dropna().unique().tolist()),
        "fixed_metrics_found": sorted(fixed_long["metric"].dropna().unique().tolist()),
        "ksweep_metrics_found": sorted(ksweep_long["metric"].dropna().unique().tolist()),
        "fixed_k0_duplicate_stats": fixed_stats,
        "ksweep_duplicate_stats": ksweep_stats,
        "table_b_skipped_groups_count": len(missing_groups),
        "table_b_skipped_groups_sample": [list(x) for x in list(sorted(missing_groups))[:20]],
        "no_shared_numeric_groups_count": len(no_shared_numeric_groups),
        "no_shared_numeric_groups_sample": [list(x) for x in no_shared_numeric_groups[:20]],
        "table_row_counts": {
            "table_fixed_k0_frontier": int(len(table_a)),
            "table_method_same_numeric_knob_values": int(len(table_same_numeric)),
            "table_method_same_numeric_knob_values_summary": int(len(table_same_numeric_summary)),
            "table_fixed_k0_vs_ksweep": int(len(table_b)),
            "table_method_win_loss_summary": int(len(win_loss)),
            "table_method_win_loss_overall": int(len(win_loss_overall)),
            "dataset_diagnostics": int(len(dataset_diag)),
        },
        "dataset_representation_summary": (
            fixed_long[
                [
                    c
                    for c in [
                        "dataset_id",
                        "representation_description",
                        "label_construction_note",
                        "n_raw",
                        "n_original",
                        "n_used",
                        "cap_policy",
                        "number_of_classes_retained",
                        "min_class_support",
                        "median_class_support",
                        "max_class_support",
                    ]
                    if c in fixed_long.columns
                ]
            ]
            .drop_duplicates()
            .to_dict("records")
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote publication tables and figures to: {args.output_dir}")


if __name__ == "__main__":
    main()

