from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.display_labels import dataset_display_name, method_display_name, metric_display_name


control_name_map = {
    "rand2d_embedding": "random 2D control",
    "rand2d_embedding_real_labels": "random 2D control",
    "rand2d_per_replicate": "random 2D control",
    "shuffled_features_embedding": "shuffled-feature control",
    "shuffled_features_per_replicate": "shuffled-feature control",
    "real_embedding_real_labels": "reference embedding",
}

weak_control_names = {"random 2D control", "shuffled-feature control"}


def first_present(columns: pd.Index | list[str], candidates: list[str]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0
    return series.astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y"])


def load_pair_subsampling_summary(
    base_csv: Path,
    extra_csvs: list[Path] | None = None,
) -> pd.DataFrame:
    frames = [pd.read_csv(base_csv)]
    for path in extra_csvs or []:
        if Path(path).exists():
            frames.append(pd.read_csv(path))
    out = pd.concat(frames, ignore_index=True)
    dedup_cols = [c for c in ["dataset_id", "method", "metric", "M_pairs"] if c in out.columns]
    if dedup_cols:
        out = out.sort_values(dedup_cols).drop_duplicates(subset=dedup_cols, keep="last")
    return out


def build_pair_subsampling_metric_stability(summary: pd.DataFrame) -> pd.DataFrame:
    required = {"metric", "M_pairs", "delta_max_abs_median", "delta_pass"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Pair-subsampling summary missing columns: {missing}")

    df = summary.copy()
    df["metric"] = df["metric"].map(metric_display_name)
    df["M_pairs"] = pd.to_numeric(df["M_pairs"], errors="coerce")
    df["delta_max_abs_median"] = pd.to_numeric(df["delta_max_abs_median"], errors="coerce")
    df["delta_pass"] = coerce_bool(df["delta_pass"])

    rows = []
    for metric, group in df.groupby("metric"):
        group = group.dropna(subset=["M_pairs", "delta_max_abs_median"]).copy()
        if group.empty:
            continue
        m_max = int(group["M_pairs"].max())
        group_max = group[group["M_pairs"] == m_max]
        pass_by_m = (
            group.groupby("M_pairs")["delta_pass"]
            .mean()
            .reset_index(name="pass_rate")
            .sort_values("M_pairs")
        )
        m_all_pass = pass_by_m.loc[pass_by_m["pass_rate"] >= 1.0, "M_pairs"]
        panel_cols = [c for c in ["dataset_id", "method"] if c in group.columns]
        n_panels = (
            int(len(group[panel_cols].drop_duplicates()))
            if len(panel_cols) == 2
            else int(len(group_max))
        )
        rows.append(
            {
                "metric": metric,
                "n_panels": n_panels,
                "M_max": m_max,
                "median_delta_at_M_max": float(group_max["delta_max_abs_median"].median()),
                "pass_rate_at_M_max": float(group_max["delta_pass"].mean()),
                "M_all_panels_pass": int(m_all_pass.iloc[0]) if len(m_all_pass) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)


def normalize_replicatewise_controls(curves: pd.DataFrame) -> pd.DataFrame:
    df = curves.rename(
        columns={
            "rep_mean": "R",
            "R_mean": "R",
            "T_mean": "R",
            "null_q95": "q95",
            "q95_null": "q95",
            "control": "control_name",
            "condition": "control_name",
            "scenario": "control_name",
        }
    ).copy()
    if "control_name" not in df.columns:
        raise ValueError("Control curves missing control/scenario column.")
    df["control_name"] = df["control_name"].replace(control_name_map)
    df = df[df["control_name"].isin(weak_control_names)].copy()
    if df.empty:
        raise ValueError("Control curves are empty after weak-control filtering.")

    needed = {"dataset_id", "control_name", "K", "R", "q95"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Control curves missing required columns: {missing}")

    df["dataset"] = df["dataset_id"].map(dataset_display_name)
    if "method" in df.columns:
        df["method"] = df["method"].map(method_display_name)
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df["R"] = pd.to_numeric(df["R"], errors="coerce")
    df["q95"] = pd.to_numeric(df["q95"], errors="coerce")
    if "p_value" in df.columns:
        df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    if "p_le_0p05" not in df.columns and "p_value" in df.columns:
        df["p_le_0p05"] = df["p_value"] <= 0.05
    elif "p_le_0p05" in df.columns:
        df["p_le_0p05"] = coerce_bool(df["p_le_0p05"])
    return df


def prepare_metric_summary(alignment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = alignment.rename(
        columns={
            "aligned": "match",
            "argmax_match": "match",
            "T_minus_q95_at_argmax_T": "gap_at_argmax_T",
        }
    ).copy()
    if "metric" not in df.columns:
        raise ValueError("Metric summary source missing metric column.")
    df["metric"] = df["metric"].map(metric_display_name)
    delta_col = first_present(
        df.columns,
        ["Delta_opt_M", "delta_opt_M", "delta_opt_abs", "selection_regret"],
    )
    if delta_col is not None:
        df["Delta_opt_M"] = pd.to_numeric(df[delta_col], errors="coerce")
    elif {"V_at_theta_V_star", "V_at_theta_T_star"}.issubset(df.columns):
        df["Delta_opt_M"] = (
            pd.to_numeric(df["V_at_theta_V_star"], errors="coerce")
            - pd.to_numeric(df["V_at_theta_T_star"], errors="coerce")
        )
    else:
        df["Delta_opt_M"] = np.nan

    missing = {"match", "Delta_opt_M"} - set(df.columns)
    if missing:
        raise ValueError(f"Metric summary source missing required columns: {missing}")
    df["match"] = coerce_bool(df["match"])
    summary = (
        df.groupby("metric")
        .agg(
            alignment_rate=("match", "mean"),
            aligned_count=("match", "sum"),
            n_total=("match", "size"),
            median_Delta_opt_M=("Delta_opt_M", "median"),
        )
        .reset_index()
    )
    summary["alignment"] = (
        summary["aligned_count"].astype(int).astype(str)
        + "/"
        + summary["n_total"].astype(int).astype(str)
    )
    return df, summary
