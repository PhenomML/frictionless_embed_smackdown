#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from _standardization_utils import (
    assert_unique_keys,
    deterministic_topk,
    normalize_nullable_bool,
    require_columns,
    safe_mode_or_na,
)
from benchmark.paper_datasets import paper_dataset_specs, paper_corpus_root


dataset_display = {dataset_id: spec.display_name for dataset_id, spec in paper_dataset_specs.items()}
dataset_modality = {dataset_id: spec.domain.lower() for dataset_id, spec in paper_dataset_specs.items()}
dataset_main_role = {dataset_id: spec.role for dataset_id, spec in paper_dataset_specs.items()}
dataset_label_type = {dataset_id: spec.label_type for dataset_id, spec in paper_dataset_specs.items()}
corpus_id_map = {
    dataset_id: spec.relative_path
    for dataset_id, spec in paper_dataset_specs.items()
    if spec.storage == "corpus"
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute fixed-K0 surrogate metrics and derived benchmark summary tables.")
    p.add_argument(
        "--fixed-summary",
        type=Path,
        default=root / "results_standardized" / "fixed_k0_summary_long.csv",
    )
    p.add_argument(
        "--k-sweep-summary",
        type=Path,
        default=root / "results_standardized" / "k_sweep_summary_long.csv",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=root / "results_standardized",
    )
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=paper_corpus_root(root),
    )
    p.add_argument(
        "--tie-tol",
        type=float,
        default=1e-12,
    )
    return p.parse_args()


def _pairwise_concordance(t: np.ndarray, v: np.ndarray, tol: float) -> float:
    """Fraction of agreeing pairwise orderings; ties under tol count as half-agreement."""
    m = len(t)
    if m < 2:
        return float("nan")
    total = m * (m - 1) / 2.0
    agree = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            dt = t[i] - t[j]
            dv = v[i] - v[j]
            prod = dt * dv
            if abs(prod) <= tol:
                agree += 0.5
            elif prod > 0:
                agree += 1.0
    return float(agree / total)


def _kendall_tau_a(t: np.ndarray, v: np.ndarray, tol: float) -> float:
    m = len(t)
    if m < 2:
        return float("nan")
    total = m * (m - 1) / 2.0
    conc = 0.0
    disc = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            dt = t[i] - t[j]
            dv = v[i] - v[j]
            prod = dt * dv
            if abs(prod) <= tol:
                continue
            if prod > 0:
                conc += 1.0
            else:
                disc += 1.0
    return float((conc - disc) / total)


def _group_alignment(g: pd.DataFrame, tol: float) -> dict[str, Any]:
    d = g.sort_values("hyperparam_value").reset_index(drop=True)
    t = d["T_value"].to_numpy(dtype=float)
    v = d["V_value"].to_numpy(dtype=float)
    if np.isnan(t).all():
        raise ValueError(f"All-NaN T_value in group {tuple(d[['dataset_id','method','metric']].iloc[0].tolist())}")
    if np.isnan(v).all():
        raise ValueError(f"All-NaN V_value in group {tuple(d[['dataset_id','method','metric']].iloc[0].tolist())}")
    labels = d["hyperparam_label"].astype(str).to_list()
    theta_vals = d["hyperparam_value"].to_numpy(dtype=float)

    t_max = float(np.nanmax(t)) if len(t) else float("nan")
    v_max = float(np.nanmax(v)) if len(v) else float("nan")
    idx_t_set = np.where(t >= (t_max - tol))[0]
    idx_v_set = np.where(v >= (v_max - tol))[0]

    t_repr_order = sorted(idx_t_set.tolist(), key=lambda i: (-float(v[i]), float(theta_vals[i]), str(labels[i])))
    v_repr_order = sorted(idx_v_set.tolist(), key=lambda i: (-float(t[i]), float(theta_vals[i]), str(labels[i])))
    idx_t = int(t_repr_order[0]) if len(t_repr_order) else int(np.argmax(t))
    idx_v = int(v_repr_order[0]) if len(v_repr_order) else int(np.argmax(v))

    theta_t = labels[idx_t]
    theta_v = labels[idx_v]

    t_const = len(np.unique(np.round(t, 12))) <= 1
    v_const = len(np.unique(np.round(v, 12))) <= 1
    if t_const and v_const:
        spearman = 1.0
        kendall = 1.0
    elif t_const or v_const:
        spearman = 0.0
        kendall = 0.0
    else:
        rank_t = pd.Series(t).rank(method="average", ascending=True)
        rank_v = pd.Series(v).rank(method="average", ascending=True)
        spearman = float(rank_t.corr(rank_v, method="pearson")) if len(d) > 1 else float("nan")
        kendall = _kendall_tau_a(t, v, tol=tol)

    top_n = min(2, len(d))
    top_t = set(deterministic_topk(labels=labels, values=t, k=top_n, tie_break_values=theta_vals))
    top_v = set(deterministic_topk(labels=labels, values=v, k=top_n, tie_break_values=theta_vals))
    top2_overlap = int(len(top_t & top_v))

    selection_regret = float(np.max(v) - np.max(v[idx_t_set])) if len(idx_t_set) else float(np.max(v) - v[idx_t])
    if selection_regret < 0 and abs(selection_regret) < 1e-9:
        selection_regret = 0.0

    z_at_theta_t = float(d["z_value"].iloc[idx_t]) if "z_value" in d.columns and pd.notna(d["z_value"].iloc[idx_t]) else np.nan
    cert_t = d["certified_above_null"].iloc[idx_t] if "certified_above_null" in d.columns else pd.NA
    cert_v = d["certified_above_null"].iloc[idx_v] if "certified_above_null" in d.columns else pd.NA

    return {
        "k0": int(d["k0"].dropna().iloc[0]) if d["k0"].dropna().any() else pd.NA,
        "theta_T_star": theta_t,
        "theta_V_star": theta_v,
        "theta_T_star_set_json": json.dumps(sorted({labels[i] for i in idx_t_set.tolist()})),
        "theta_V_star_set_json": json.dumps(sorted({labels[i] for i in idx_v_set.tolist()})),
        "n_theta_T_star": int(len(idx_t_set)),
        "n_theta_V_star": int(len(idx_v_set)),
        "argmax_match": bool(len(set(idx_t_set.tolist()) & set(idx_v_set.tolist())) > 0),
        "T_at_theta_T_star": float(t[idx_t]),
        "V_at_theta_T_star": float(v[idx_t]),
        "T_at_theta_V_star": float(t[idx_v]),
        "V_at_theta_V_star": float(v[idx_v]),
        "delta_opt_abs": float(selection_regret),
        "z_at_theta_T_star": z_at_theta_t,
        "certified_theta_T_star": cert_t,
        "certified_theta_V_star": cert_v,
        "spearman_T_V": spearman,
        "kendall_T_V": kendall,
        "selection_regret": selection_regret,
        "top2_overlap": top2_overlap,
        "pairwise_concordance": _pairwise_concordance(t, v, tol),
        "n_hyperparam_settings": int(len(d)),
    }


def _build_same_numeric(fixed: pd.DataFrame) -> pd.DataFrame:
    assert_unique_keys(
        fixed[["dataset_id", "metric", "method", "hyperparam_value"]].copy(),
        ["dataset_id", "metric", "method", "hyperparam_value"],
        "same_numeric input uniqueness",
    )
    use = fixed[["dataset_id", "metric", "method", "hyperparam_value", "T_value", "V_value"]].copy()
    u = use[use["method"] == "umap"].rename(columns={"T_value": "T_umap", "V_value": "V_umap"})
    t = use[use["method"] == "tsne"].rename(columns={"T_value": "T_tsne", "V_value": "V_tsne"})
    m = u.merge(
        t[["dataset_id", "metric", "hyperparam_value", "T_tsne", "V_tsne"]],
        on=["dataset_id", "metric", "hyperparam_value"],
        how="inner",
    )
    m["delta_T"] = m["T_umap"] - m["T_tsne"]
    m["delta_V"] = m["V_umap"] - m["V_tsne"]
    out = (
        m.groupby(["dataset_id", "metric"], as_index=False)
        .agg(
            n_shared_values=("hyperparam_value", "size"),
            mean_delta_T=("delta_T", "mean"),
            median_delta_T=("delta_T", "median"),
            frac_delta_T_positive=("delta_T", lambda x: float((x > 0).mean())),
            mean_delta_V=("delta_V", "mean"),
            median_delta_V=("delta_V", "median"),
            frac_delta_V_positive=("delta_V", lambda x: float((x > 0).mean())),
            shared_values_json=("hyperparam_value", lambda s: json.dumps(sorted(pd.to_numeric(s, errors="coerce").dropna().unique().tolist()))),
        )
        .sort_values(["dataset_id", "metric"])
        .reset_index(drop=True)
    )
    return out


def _build_cross_design(fixed: pd.DataFrame, ks: pd.DataFrame, align: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    assert_unique_keys(align, ["dataset_id", "method", "metric"], "alignment summary uniqueness before cross-design")
    align_by_key = align.set_index(["dataset_id", "method", "metric"], drop=False)
    grouped_ks = ks.groupby(["dataset_id", "method", "metric"])
    for key, gf in fixed.groupby(["dataset_id", "method", "metric"]):
        dataset_id, method, metric = key
        if key not in grouped_ks.groups:
            continue
        if key not in align_by_key.index:
            continue
        gk = grouped_ks.get_group(key).copy()
        a = align_by_key.loc[key]
        k0 = int(gf["k0"].dropna().iloc[0]) if gf["k0"].dropna().any() else None
        t_fixed = float(gf["T_value"].max())
        v_fixed = float(gf["V_value"].max())
        t_best_k = float(gk["T_mean"].max())
        v_best_k = float(gk["V_mean"].max())

        if k0 is not None and (gk["K"].astype(float) == float(k0)).any():
            k0_rows = gk[gk["K"].astype(float) == float(k0)]
            if len(k0_rows) != 1:
                raise ValueError(f"Expected one K-sweep row for k0={k0}, key={key}, got={len(k0_rows)}")
            k0_row = k0_rows.iloc[0]
            t_k0 = float(k0_row["T_mean"])
            v_k0 = float(k0_row["V_mean"])
        else:
            t_k0 = np.nan
            v_k0 = np.nan

        rows.append(
            {
                "dataset_id": dataset_id,
                "method": method,
                "metric": metric,
                "max_theta_T_fixed_k0": t_fixed,
                "max_theta_V_fixed_k0": v_fixed,
                "T_at_theta_T_star": float(a["T_at_theta_T_star"]),
                "V_at_theta_T_star": float(a["V_at_theta_T_star"]),
                "T_at_theta_V_star": float(a["T_at_theta_V_star"]),
                "V_at_theta_V_star": float(a["V_at_theta_V_star"]),
                "T_old_k0_slice": t_k0,
                "V_old_k0_slice": v_k0,
                "T_best_over_K": t_best_k,
                "V_best_over_K": v_best_k,
                "fixed_k0_better_on_T_vs_K0sweep": bool(pd.notna(t_k0) and float(a["T_at_theta_T_star"]) > t_k0),
                "fixed_k0_better_on_V_vs_K0sweep": bool(pd.notna(v_k0) and float(a["V_at_theta_T_star"]) > v_k0),
                "fixed_k0_better_on_T_vs_bestK": bool(float(a["T_at_theta_T_star"]) > t_best_k),
                "fixed_k0_better_on_V_vs_bestK": bool(float(a["V_at_theta_T_star"]) > v_best_k),
                "delta_T_vs_K0sweep": float(a["T_at_theta_T_star"] - t_k0) if pd.notna(t_k0) else np.nan,
                "delta_V_vs_K0sweep": float(a["V_at_theta_T_star"] - v_k0) if pd.notna(v_k0) else np.nan,
                "delta_T_vs_bestK": float(a["T_at_theta_T_star"] - t_best_k),
                "delta_V_vs_bestK": float(a["V_at_theta_T_star"] - v_best_k),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_id", "method", "metric"]).reset_index(drop=True)


def _read_corpus_meta(corpus_dir: Path, dataset_id: str) -> dict[str, Any]:
    cid = corpus_id_map.get(dataset_id, dataset_id)
    p = corpus_dir / cid / "meta.json"
    if not p.exists():
        print(f"[warn] Missing corpus meta: {p}")
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_dataset_inventory(fixed: pd.DataFrame, ks: pd.DataFrame, corpus_dir: Path) -> pd.DataFrame:
    ds_ids = sorted(set(fixed["dataset_id"].dropna()) | set(ks["dataset_id"].dropna()))
    rows = []
    for ds in ds_ids:
        mf = fixed[fixed["dataset_id"] == ds]
        mk = ks[ks["dataset_id"] == ds]
        meta = _read_corpus_meta(corpus_dir, ds)
        known_k0 = int(mf["k0"].dropna().iloc[0]) if len(mf) and mf["k0"].dropna().any() else pd.NA
        n_samples = pd.NA
        for k in ("n_samples", "n_total", "n"):
            if k in meta and pd.notna(meta[k]):
                n_samples = int(meta[k])
                break
        max_points_used = (
            int(
                pd.concat([mf.get("n_points_used", pd.Series(dtype=float)), mk.get("n_points_used", pd.Series(dtype=float))]).dropna().max()
            )
            if (("n_points_used" in mf.columns and mf["n_points_used"].notna().any()) or ("n_points_used" in mk.columns and mk["n_points_used"].notna().any()))
            else pd.NA
        )
        preproc = meta.get("preprocessing")
        if isinstance(preproc, list):
            preproc_s = " -> ".join(str(x) for x in preproc)
        else:
            preproc_s = meta.get("representation_description", "")
        rows.append(
            {
                "dataset_id": ds,
                "dataset_name": dataset_display.get(ds, ds),
                "modality": dataset_modality.get(ds, "unknown"),
                "n_samples": n_samples,
                "max_points_used": max_points_used,
                "input_dim": meta.get("original_dim", meta.get("representation_dim", meta.get("feature_dim", pd.NA))),
                "label_type": dataset_label_type.get(ds, "unknown"),
                "known_k0": known_k0,
                "main_role": dataset_main_role.get(ds, "benchmark_case"),
                "preprocessing_repr": preproc_s,
                "notes": "",
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_id").reset_index(drop=True)


def _build_protocol_inventory(fixed: pd.DataFrame, ks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def _row_from_block(df: pd.DataFrame, protocol_id: str, k_grid: list[int] | None = None) -> dict[str, Any]:
        is_fixed = "fixed_k0" in protocol_id
        tsne_grid = sorted(df.loc[df["method"] == "tsne", "hyperparam_value"].dropna().unique().tolist()) if is_fixed and "hyperparam_value" in df.columns else []
        umap_grid = sorted(df.loc[df["method"] == "umap", "hyperparam_value"].dropna().unique().tolist()) if is_fixed and "hyperparam_value" in df.columns else []
        group_uniform_rate = float(pd.to_numeric(df.get("group_protocol_uniform"), errors="coerce").dropna().mean()) if "group_protocol_uniform" in df.columns else np.nan
        compute_tier_mode = safe_mode_or_na(df.get("compute_tier", pd.Series(dtype=object)))
        alpha_col = "alpha_bh_used" if "alpha_bh_used" in df.columns else ("alpha_bh" if "alpha_bh" in df.columns else None)
        return {
            "protocol_id": protocol_id,
            "B": int(df["b_used"].dropna().mode().iloc[0]) if "b_used" in df.columns and df["b_used"].dropna().any() else pd.NA,
            "subsample_frac": float(df["subsample_frac"].dropna().mode().iloc[0]) if "subsample_frac" in df.columns and df["subsample_frac"].dropna().any() else 0.8,
            "clustering_algo": "kmeans",
            "clustering_restarts": 10,
            "embedding_recomputed_per_replicate": False,
            "pair_sampling_rule": "sampled_replicate_pairs_with_min_overlap",
            "pairs_target": int(df["max_pairs_used"].dropna().mode().iloc[0]) if "max_pairs_used" in df.columns and df["max_pairs_used"].dropna().any() else pd.NA,
            "null_trials": int(df["null_trials"].dropna().mode().iloc[0]) if "null_trials" in df.columns and df["null_trials"].dropna().any() else pd.NA,
            "alpha_bh": float(df[alpha_col].dropna().mode().iloc[0]) if alpha_col is not None and df[alpha_col].dropna().any() else 0.05,
            "tsne_grid": json.dumps(tsne_grid),
            "umap_grid": json.dumps(umap_grid),
            "k_grid": json.dumps(k_grid or []),
            "protocol_assumption": True,
            "group_protocol_uniform_rate": group_uniform_rate,
            "compute_tier_mode": compute_tier_mode,
            "n_groups": int(df[["dataset_id", "method", "metric"]].drop_duplicates().shape[0]) if all(c in df.columns for c in ["dataset_id", "method", "metric"]) else pd.NA,
            "remarks": "Assumed from current pipeline defaults; verify against run scripts before publication",
        }

    for source_block in sorted(fixed["source_block"].dropna().unique()):
        rows.append(_row_from_block(fixed[fixed["source_block"] == source_block], f"fixed_k0_{source_block}"))
    for source_block in sorted(ks["source_block"].dropna().unique()):
        g = ks[ks["source_block"] == source_block]
        k_grid = sorted(g["K"].dropna().astype(int).unique().tolist())
        rows.append(_row_from_block(g, f"k_sweep_{source_block}", k_grid=k_grid))
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    fixed = pd.read_csv(args.fixed_summary)
    ks = pd.read_csv(args.k_sweep_summary)
    require_columns(fixed, ["dataset_id", "method", "metric", "hyperparam_value", "T_value", "V_value"], "fixed summary input")
    require_columns(ks, ["dataset_id", "method", "metric", "K", "T_mean", "V_mean"], "k-sweep summary input")
    assert_unique_keys(fixed, ["dataset_id", "method", "metric", "hyperparam_value"], "fixed summary input uniqueness")
    fixed["certified_above_null"] = normalize_nullable_bool(fixed.get("certified_above_null", pd.Series(pd.NA, index=fixed.index)))

    align_rows = []
    for key, g in fixed.groupby(["dataset_id", "method", "metric"], sort=False):
        row = {"dataset_id": key[0], "method": key[1], "metric": key[2]}
        row.update(_group_alignment(g, tol=args.tie_tol))
        align_rows.append(row)
    align = pd.DataFrame(align_rows).sort_values(["dataset_id", "method", "metric"]).reset_index(drop=True)

    same_numeric = _build_same_numeric(fixed)
    cross_design = _build_cross_design(fixed, ks, align)
    dataset_inventory = _build_dataset_inventory(fixed, ks, args.corpus_dir)
    protocol_inventory = _build_protocol_inventory(fixed, ks)

    out_align = args.results_dir / "fixed_k0_alignment_summary.csv"
    out_same = args.results_dir / "same_numeric_comparison.csv"
    out_cross = args.results_dir / "cross_design_comparison.csv"
    out_ds = args.results_dir / "dataset_inventory.csv"
    out_proto = args.results_dir / "protocol_inventory.csv"

    assert_unique_keys(align, ["dataset_id", "method", "metric"], "alignment output uniqueness")
    assert_unique_keys(same_numeric, ["dataset_id", "metric"], "same-numeric output uniqueness")
    assert_unique_keys(cross_design, ["dataset_id", "method", "metric"], "cross-design output uniqueness")
    align.sort_values(["dataset_id", "method", "metric"]).to_csv(out_align, index=False)
    same_numeric.sort_values(["dataset_id", "metric"]).to_csv(out_same, index=False)
    cross_design.sort_values(["dataset_id", "method", "metric"]).to_csv(out_cross, index=False)
    dataset_inventory.to_csv(out_ds, index=False)
    protocol_inventory.to_csv(out_proto, index=False)

    print(f"Wrote {out_align} ({len(align)} rows)")
    print(f"Wrote {out_same} ({len(same_numeric)} rows)")
    print(f"Wrote {out_cross} ({len(cross_design)} rows)")
    print(f"Wrote {out_ds} ({len(dataset_inventory)} rows)")
    print(f"Wrote {out_proto} ({len(protocol_inventory)} rows)")


if __name__ == "__main__":
    main()

