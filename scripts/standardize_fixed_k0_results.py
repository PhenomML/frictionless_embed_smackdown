#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
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
)

metric_suffix = {"ami": "AMI", "ari": "ARI", "jaccard": "Jaccard"}
expected_grid = [5, 10, 15, 30, 50, 100]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Standardize old/new fixed-K0 raw outputs into one canonical long table.")
    p.add_argument("--old-csv", type=Path, default=root / "outputs" / "fixed_k0_hyperparam_sweep" / "fixed_k0_hyperparam_sweep_results.csv")
    p.add_argument("--new-csv", type=Path, default=root / "outputs" / "fixed_k0_hyperparam_sweep_new_only" / "fixed_k0_hyperparam_sweep_results.csv")
    p.add_argument(
        "--merged-csv",
        type=Path,
        default=None,
        help="Optional merged raw fixed-K0 source. If provided, old/new inputs are ignored.",
    )
    p.add_argument("--output", type=Path, default=root / "results_standardized" / "fixed_k0_summary_long.csv")
    p.add_argument("--target-cap-mode", choices=["auto", "full", "force_cap"], default="auto")
    p.add_argument("--force-cap", type=int, default=None)
    p.add_argument("--strict-grid", action="store_true", help="Raise when expected hyperparameter grid is incomplete.")
    return p.parse_args()


def _fmt_numeric(x: Any) -> str:
    try:
        v = float(x)
        if np.isfinite(v) and abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:g}"
    except Exception:
        return str(x)


def _safe_label(method: str, hp_val: Any) -> str:
    if pd.isna(hp_val):
        return "NA"
    m = str(method).lower()
    if m == "tsne":
        return f"perp={_fmt_numeric(hp_val)}"
    return f"n_neighbors={_fmt_numeric(hp_val)}"


def _validate_metric_blocks(df: pd.DataFrame) -> None:
    found = []
    for _, suffix in metric_suffix.items():
        t_col, v_col = f"T_{suffix}", f"V_{suffix}"
        if t_col in df.columns and v_col in df.columns:
            found.append(suffix)
        elif t_col in df.columns or v_col in df.columns:
            raise ValueError(f"Partial metric block found: expected both {t_col} and {v_col}.")
    if not found:
        raise ValueError("No recognized metric blocks found. Expected any of T_AMI/V_AMI, T_ARI/V_ARI, T_Jaccard/V_Jaccard.")


def _resolve_k0(df: pd.DataFrame) -> pd.Series:
    k0 = pd.to_numeric(df["k0"], errors="coerce") if "k0" in df.columns else pd.Series(np.nan, index=df.index)
    nk0 = pd.to_numeric(df["nominal_class_count"], errors="coerce") if "nominal_class_count" in df.columns else pd.Series(np.nan, index=df.index)
    both = k0.notna() & nk0.notna()
    if bool((k0[both] != nk0[both]).any()):
        raise ValueError("k0 and nominal_class_count disagree on at least one row.")
    return k0.fillna(nk0).astype("Int64")


def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _dedupe_or_raise(df: pd.DataFrame) -> pd.DataFrame:
    key = ["dataset_id", "method", "metric", "hyperparam_value"]
    dup = df.duplicated(key, keep=False)
    if not bool(dup.any()):
        return df
    blocks = []
    for _, g in df.loc[dup].groupby(key, dropna=False, sort=False):
        g2 = g.copy()
        non_prov = [c for c in g2.columns if c not in {"source_block", "protocol_source"}]
        first = g2.iloc[0][non_prov]
        eq = g2[non_prov].eq(first, axis=1)
        both_na = g2[non_prov].isna() & pd.Series(first).isna()
        same = (eq | both_na).all(axis=1).all()
        if same:
            blocks.append(g2.iloc[[0]])
        else:
            raise ValueError(f"Conflicting duplicates for key={tuple(g2.iloc[0][key].tolist())}")
    out = pd.concat([df.loc[~dup]] + blocks, ignore_index=True)
    assert_unique_keys(out, key, "fixed_k0 duplicate resolution")
    return out


def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    searched = []
    base_cols = [c for c in [
        "dataset_id", "method", "k0", "nominal_class_count", "primary_hyperparam_name", "primary_hyperparam_value",
        "tsne_perplexity", "umap_n_neighbors", "b", "max_pairs", "n_points", "subsample_frac", "tau", "run_null",
        "random_state", "representation_description", "label_construction_note", "n_raw", "n_original", "n_used",
        "cap_policy", "number_of_classes_retained", "min_class_support", "median_class_support", "max_class_support",
        "source_block", "protocol_source",
    ] if c in df.columns]
    for metric, suffix in metric_suffix.items():
        t_col, v_col = f"T_{suffix}", f"V_{suffix}"
        searched.append((t_col, v_col))
        if t_col not in df.columns or v_col not in df.columns:
            continue
        d = df[base_cols].copy()
        d["metric"] = metric
        d["T_value"] = pd.to_numeric(df[t_col], errors="coerce")
        d["V_value"] = pd.to_numeric(df[v_col], errors="coerce")
        d["null_mean"] = pd.to_numeric(df.get(f"mu0_{suffix}"), errors="coerce")
        d["null_sd"] = pd.to_numeric(df.get(f"sigma0_{suffix}"), errors="coerce")
        d["null_q95"] = pd.to_numeric(df.get(f"q95_{suffix}"), errors="coerce")
        d["z_value"] = pd.to_numeric(df.get(f"Z_{suffix}"), errors="coerce")
        d["p_value"] = pd.to_numeric(df.get(f"p_{suffix}"), errors="coerce")
        rows.append(d)
    if not rows:
        raise ValueError(f"No metric blocks found during melt. Searched={searched}")
    out = pd.concat(rows, ignore_index=True)
    out["method"] = out["method"].astype(str).str.lower()
    out["k0"] = _resolve_k0(out)

    out["hyperparam_name"] = out.get("primary_hyperparam_name")
    out["hyperparam_name"] = out["hyperparam_name"].where(out["hyperparam_name"].notna(), np.where(out["method"].eq("tsne"), "perplexity", "n_neighbors"))

    hp = _num_series(out, "primary_hyperparam_value")
    hp = hp.where(hp.notna(), _num_series(out, "tsne_perplexity").where(out["method"].eq("tsne")))
    hp = hp.where(hp.notna(), _num_series(out, "umap_n_neighbors").where(out["method"].eq("umap")))
    out["hyperparam_value"] = pd.to_numeric(hp, errors="coerce")
    if not bool(out["hyperparam_value"].notna().all()):
        raise ValueError("hyperparam_value has missing values after fill logic.")
    out["hyperparam_label"] = [_safe_label(m, v) for m, v in zip(out["method"], out["hyperparam_value"])]

    cert = pd.Series(pd.NA, index=out.index, dtype="boolean")
    m = out["null_q95"].notna()
    cert.loc[m] = (out.loc[m, "T_value"] > out.loc[m, "null_q95"]).astype("boolean")
    out["certified_above_null"] = cert

    out["b_used"] = pd.to_numeric(out.get("b"), errors="coerce").astype("Int64")
    out["max_pairs_used"] = pd.to_numeric(out.get("max_pairs"), errors="coerce").astype("Int64")
    out["n_points_used"] = pd.to_numeric(out.get("n_points"), errors="coerce").astype("Int64")
    out["protocol_source"] = out["protocol_source"].fillna("merged_csv") if "protocol_source" in out.columns else "merged_csv"

    keep = [
        "dataset_id", "method", "metric", "k0", "hyperparam_name", "hyperparam_value", "hyperparam_label",
        "tsne_perplexity", "umap_n_neighbors", "T_value", "V_value", "null_mean", "null_sd", "null_q95", "z_value",
        "p_value", "certified_above_null", "source_block", "protocol_source", "b_used", "max_pairs_used",
        "n_points_used", "subsample_frac", "tau", "run_null", "random_state", "representation_description",
        "label_construction_note", "n_raw", "n_original", "n_used", "cap_policy", "number_of_classes_retained",
        "min_class_support", "median_class_support", "max_class_support",
    ]
    out = out[[c for c in keep if c in out.columns]].copy()
    out = out.dropna(subset=["dataset_id", "method", "metric", "hyperparam_value", "T_value", "V_value"])
    out = _dedupe_or_raise(out)
    out["rank_T_desc"] = out.groupby(["dataset_id", "method", "metric"])["T_value"].rank(method="dense", ascending=False).astype("Int64")
    out["rank_V_desc"] = out.groupby(["dataset_id", "method", "metric"])["V_value"].rank(method="dense", ascending=False).astype("Int64")
    return out


def _with_grid_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["dataset_id", "method", "metric"], as_index=False)
        .agg(observed=("hyperparam_value", lambda s: sorted(pd.to_numeric(s, errors="coerce").dropna().unique().tolist())))
    )
    g["n_hyperparam_settings"] = g["observed"].map(len).astype("Int64")
    g["missing_hyperparams_json"] = g["observed"].map(lambda obs: json.dumps([x for x in expected_grid if x not in set(int(round(float(v))) for v in obs)]))
    g["hyperparam_grid_complete"] = g["missing_hyperparams_json"].map(lambda s: len(json.loads(s)) == 0).astype("boolean")
    out = df.merge(g.drop(columns=["observed"]), on=["dataset_id", "method", "metric"], how="left")
    return out


def _with_protocol_flags(df: pd.DataFrame, target_cap_mode: str, force_cap: int | None = None) -> pd.DataFrame:
    out = df.drop(
        columns=[
            "group_protocol_uniform",
            "group_protocol_candidate_count",
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
    table_family = "fixed_k0"
    n_samples_map = (
        out.groupby("dataset_id", as_index=True)
        .apply(lambda g: pd.to_numeric(g.get("n_raw", pd.Series(dtype=float)), errors="coerce").dropna().max()
               if "n_raw" in g.columns and pd.to_numeric(g.get("n_raw"), errors="coerce").dropna().any()
               else (pd.to_numeric(g.get("n_original", pd.Series(dtype=float)), errors="coerce").dropna().max()
                     if "n_original" in g.columns and pd.to_numeric(g.get("n_original"), errors="coerce").dropna().any()
                     else pd.NA))
    )
    cap_rows = []
    for ds, gds in out.groupby("dataset_id", sort=False):
        n_samples = n_samples_map.get(ds, pd.NA)
        n_samples_source = "missing"
        if "n_raw" in gds.columns and pd.to_numeric(gds["n_raw"], errors="coerce").dropna().any():
            n_samples_source = "n_raw"
        elif "n_original" in gds.columns and pd.to_numeric(gds["n_original"], errors="coerce").dropna().any():
            n_samples_source = "n_original"
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
        cap_rows.append(
            {
                "dataset_id": ds,
                "table_family": table_family,
                "n_samples": n_samples,
                "n_samples_source": n_samples_source,
                "requested_cap_inferred": requested_cap_inferred,
                "requested_cap_source": requested_cap_source,
                "n_target_auto": n_target_auto,
                "n_target_source": n_target_source,
                "n_target": n_target_active,
                "n_target_mode": target_cap_mode,
                "n_target_active_source": n_target_active_source,
            }
        )
    cap_df = pd.DataFrame(cap_rows)
    g = (
        out.groupby(["dataset_id", "method", "metric"], as_index=False)
        .agg(
            b_min=("b_used", "min"),
            b_max=("b_used", "max"),
            pairs_min=("max_pairs_used", "min"),
            pairs_max=("max_pairs_used", "max"),
            n_points_min=("n_points_used", "min"),
            n_points_max=("n_points_used", "max"),
            group_protocol_candidate_count=("n_points_used", "size"),
            protocol_source_group=("protocol_source", lambda s: "|".join(sorted(set(str(x) for x in s.dropna())))),
        )
    )
    proto_tuples = (
        out.assign(_t=out[["b_used", "max_pairs_used", "n_points_used"]].astype("string").fillna("NA").agg("|".join, axis=1))
        .groupby(["dataset_id", "method", "metric"])["_t"]
        .nunique()
        .rename("group_protocol_candidate_count")
        .reset_index()
    )
    g = g.drop(columns=["group_protocol_candidate_count"]).merge(proto_tuples, on=["dataset_id", "method", "metric"], how="left")
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
        raw = pd.read_csv(args.merged_csv)
        require_columns(raw, ["dataset_id", "method"], "fixed_k0 merged raw")
        _validate_metric_blocks(raw)
        if "source_block" not in raw.columns:
            raw["source_block"] = "merged_strict"
        pre = _to_long(raw)
    else:
        old = pd.read_csv(args.old_csv)
        new = pd.read_csv(args.new_csv)
        require_columns(old, ["dataset_id", "method"], "fixed_k0 old raw")
        require_columns(new, ["dataset_id", "method"], "fixed_k0 new raw")
        _validate_metric_blocks(old)
        _validate_metric_blocks(new)

        old["source_block"] = "old_baseline"
        new["source_block"] = "new_only"
        raw = pd.concat([old, new], ignore_index=True)
        pre = _to_long(raw)
        overlap = pre.groupby(["dataset_id", "method", "metric", "hyperparam_value"])["source_block"].nunique()
        if bool((overlap > 1).any()):
            raise ValueError("Overlap detected across old/new source blocks for same canonical fixed-k0 key.")

    standardized = pre
    standardized = _with_grid_diagnostics(standardized)
    if args.strict_grid and bool((~standardized["hyperparam_grid_complete"].fillna(False)).any()):
        bad = standardized.loc[~standardized["hyperparam_grid_complete"].fillna(False), ["dataset_id", "method", "metric"]].drop_duplicates()
        raise ValueError(f"Incomplete hyperparameter grids detected: {bad.to_dict('records')}")
    standardized = _with_protocol_flags(standardized, target_cap_mode=args.target_cap_mode, force_cap=args.force_cap)
    standardized["certified_above_null"] = normalize_nullable_bool(standardized["certified_above_null"])

    key = ["dataset_id", "method", "metric", "hyperparam_value"]
    assert_unique_keys(standardized, key, "fixed_k0 standardized final")
    standardized = standardized.sort_values(key).reset_index(drop=True)
    standardized.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(standardized)} rows)")


if __name__ == "__main__":
    main()

