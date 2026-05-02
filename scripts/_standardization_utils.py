#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def require_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


def assert_unique_keys(df: pd.DataFrame, keys: list[str], context: str) -> None:
    require_columns(df, keys, context)
    dup = df.duplicated(keys, keep=False)
    if bool(dup.any()):
        sample = df.loc[dup, keys].drop_duplicates().head(10).to_dict("records")
        raise ValueError(f"{context}: duplicate rows for key {keys}. Sample={sample}")


def normalize_nullable_bool(series: pd.Series) -> pd.Series:
    def _coerce(x: Any) -> Any:
        if pd.isna(x):
            return pd.NA
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        s = str(x).strip().lower()
        if s in {"1", "true", "t", "yes", "y"}:
            return True
        if s in {"0", "false", "f", "no", "n"}:
            return False
        return pd.NA

    return series.map(_coerce).astype("boolean")


def safe_mode_or_na(series: pd.Series) -> Any:
    s = series.dropna()
    if len(s) == 0:
        return pd.NA
    m = s.mode()
    return m.iloc[0] if len(m) else pd.NA


def dataset_target_n_points(
    dataset_id: str,
    observed_n_points: Iterable[Any],
    actual_n_samples: Any = None,
    requested_cap: Any = None,
) -> float:
    vals = pd.to_numeric(pd.Series(list(observed_n_points)), errors="coerce").dropna()
    has_actual = actual_n_samples is not None and not pd.isna(actual_n_samples)
    has_cap = requested_cap is not None and not pd.isna(requested_cap)
    if has_actual and has_cap:
        try:
            a = float(actual_n_samples)
            c = float(requested_cap)
            if a > 0 and c > 0:
                return float(min(a, c))
        except Exception:
            pass
    if has_actual:
        try:
            a = float(actual_n_samples)
            if a > 0:
                return a
        except Exception:
            pass
    if has_cap:
        try:
            c = float(requested_cap)
            if c > 0:
                return c
        except Exception:
            pass
    if len(vals) == 0:
        return float("nan")
    return float(vals.max())


def classify_compute_tier(
    b: Any,
    pairs: Any,
    n_used: Any,
    n_target: Any,
    target_b: int = 200,
    target_pairs: int = 2000,
    relaxed_b: int = 100,
    relaxed_pairs: int = 1000,
) -> str:
    if pd.isna(b) or pd.isna(pairs) or pd.isna(n_used) or pd.isna(n_target):
        return "unknown"
    b_i = int(float(b))
    p_i = int(float(pairs))
    n_f = float(n_used)
    n_t = float(n_target)
    n_ok = n_f >= 0.95 * n_t
    if b_i >= target_b and p_i >= target_pairs and n_ok:
        return "nominal"
    if b_i >= relaxed_b and p_i >= relaxed_pairs and n_ok:
        return "retry_relaxed"
    return "retry_low"


def deterministic_topk(
    labels: list[Any],
    values: np.ndarray,
    k: int,
    tie_break_values: np.ndarray | None = None,
) -> list[Any]:
    if len(labels) == 0 or k <= 0:
        return []
    if tie_break_values is None:
        tie_break_values = np.array(labels, dtype=object)
    def _safe_num(x: Any) -> float:
        try:
            if pd.isna(x):
                return float("inf")
            return float(x)
        except Exception:
            return float("inf")
    order = sorted(
        range(len(labels)),
        key=lambda i: (-float(values[i]), _safe_num(tie_break_values[i]), str(labels[i])),
    )
    return [labels[i] for i in order[: min(k, len(order))]]


def infer_requested_cap(
    df: pd.DataFrame,
    n_samples: Any = None,
    explicit_cap_cols: list[str] | None = None,
    b_col: str = "b_used",
    pairs_col: str = "max_pairs_used",
    n_points_col: str = "n_points_used",
) -> tuple[float | None, str]:
    cols = explicit_cap_cols or []
    for c in cols:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals):
                return float(vals.max()), f"explicit_metadata:{c}"
    sub = df.copy()
    sub["_b"] = pd.to_numeric(sub.get(b_col), errors="coerce")
    sub["_p"] = pd.to_numeric(sub.get(pairs_col), errors="coerce")
    sub["_n"] = pd.to_numeric(sub.get(n_points_col), errors="coerce")
    sub = sub[sub[["_b", "_p", "_n"]].notna().all(axis=1)]
    if len(sub):
        b_max = sub["_b"].max()
        sub = sub[sub["_b"] == b_max]
        p_max = sub["_p"].max()
        sub = sub[sub["_p"] == p_max]
        n_max = sub["_n"].max()
        return float(n_max), "best_budget_inference"
    if n_samples is not None and not pd.isna(n_samples):
        return float(n_samples), "fallback_full_data"
    return None, "missing"


def resolve_n_target(
    mode: str,
    n_samples: Any,
    requested_cap_inferred: Any,
    force_cap: Any = None,
) -> tuple[float | None, str]:
    if mode not in {"auto", "full", "force_cap"}:
        raise ValueError(f"Unsupported target-cap mode: {mode}")
    if mode == "full":
        if n_samples is not None and not pd.isna(n_samples):
            return float(n_samples), "full:n_samples"
        return None, "full:missing_n_samples"
    if mode == "force_cap":
        if force_cap is None or pd.isna(force_cap):
            raise ValueError("force_cap mode requires force_cap value.")
        cap = float(force_cap)
        if n_samples is not None and not pd.isna(n_samples):
            return float(min(float(n_samples), cap)), "force_cap:min(n_samples,force_cap)"
        return cap, "force_cap:cap_only"
    # Infer the active target size from the observed sample count and requested cap.
    t = dataset_target_n_points(
        dataset_id="",
        observed_n_points=[],
        actual_n_samples=n_samples,
        requested_cap=requested_cap_inferred,
    )
    if pd.isna(t):
        return None, "auto:missing"
    if n_samples is not None and not pd.isna(n_samples) and requested_cap_inferred is not None and not pd.isna(requested_cap_inferred):
        return float(t), "auto:min(n_samples,requested_cap_inferred)"
    if n_samples is not None and not pd.isna(n_samples):
        return float(t), "auto:n_samples_only"
    return float(t), "auto:requested_cap_only"
