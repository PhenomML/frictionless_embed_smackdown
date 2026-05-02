#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Merge strict rerun raw outputs into canonical merged source CSVs.")
    p.add_argument("--root", type=Path, default=root, help="Project root.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "strict_reruns",
        help="Directory containing strict rerun outputs and merged artifacts.",
    )
    return p.parse_args()


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _col_or_na(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(pd.NA, index=df.index)


def _assert_unique_by_key(df: pd.DataFrame, key_cols: list[str], context: str) -> None:
    if df.empty:
        return
    dup = df.duplicated(subset=key_cols, keep=False)
    if not bool(dup.any()):
        return
    bad = df.loc[dup, key_cols].drop_duplicates().head(10).to_dict("records")
    raise ValueError(f"{context}: duplicate keys detected for {key_cols}. Examples: {bad}")


def _normalize_fixed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dataset_id" not in out.columns or "method" not in out.columns:
        raise ValueError("Fixed-K0 source missing required columns: dataset_id, method.")
    out["dataset_id"] = out["dataset_id"].astype(str)
    out["method"] = out["method"].astype(str).str.lower()

    hp_name_raw = _col_or_na(out, "primary_hyperparam_name").astype("string").str.lower()
    out["resolved_hyperparam_name"] = hp_name_raw
    out["resolved_hyperparam_name"] = out["resolved_hyperparam_name"].where(
        out["resolved_hyperparam_name"].notna(),
        pd.Series(
            ["perplexity" if m == "tsne" else "n_neighbors" if m == "umap" else pd.NA for m in out["method"]],
            index=out.index,
            dtype="object",
        ),
    )

    hp_primary_num = pd.to_numeric(_col_or_na(out, "primary_hyperparam_value"), errors="coerce")
    hp_tsne_num = pd.to_numeric(_col_or_na(out, "tsne_perplexity"), errors="coerce")
    hp_umap_num = pd.to_numeric(_col_or_na(out, "umap_n_neighbors"), errors="coerce")
    resolved_hp_num = hp_primary_num.where(hp_primary_num.notna(), hp_tsne_num.where(out["method"].eq("tsne")))
    resolved_hp_num = resolved_hp_num.where(resolved_hp_num.notna(), hp_umap_num.where(out["method"].eq("umap")))
    resolved_hp_str = resolved_hp_num.map(lambda x: f"{x:g}" if pd.notna(x) else pd.NA)
    hp_raw_str = _col_or_na(out, "primary_hyperparam_value").astype("string")
    out["resolved_hyperparam_value_key"] = resolved_hp_str.fillna(hp_raw_str).fillna("NA")

    if "k0" in out.columns:
        out["k0"] = pd.to_numeric(out["k0"], errors="coerce").astype("Int64")
    else:
        out["k0"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    if "nominal_class_count" in out.columns:
        out["nominal_class_count"] = pd.to_numeric(out["nominal_class_count"], errors="coerce").astype("Int64")
    else:
        out["nominal_class_count"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["resolved_k0"] = out["k0"].fillna(out["nominal_class_count"])
    return out


def _normalize_k_sweep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = {"dataset_id", "method", "metric"}
    if not required.issubset(set(out.columns)):
        raise ValueError("K-sweep source missing required columns: dataset_id, method, metric.")
    out["dataset_id"] = out["dataset_id"].astype(str)
    out["method"] = out["method"].astype(str).str.lower()
    out["metric"] = out["metric"].astype(str).str.lower()
    if "K" not in out.columns:
        out["K"] = pd.to_numeric(out.get("k"), errors="coerce")
    out["K"] = pd.to_numeric(out["K"], errors="coerce").round().astype("Int64")
    if "k" not in out.columns:
        out["k"] = out["K"]
    return out


def _merge_with_override(canonical: pd.DataFrame, strict: pd.DataFrame, key_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    if canonical.empty and strict.empty:
        return pd.DataFrame(), {"canonical_rows": 0, "strict_rows": 0, "overrides": 0, "appended": 0, "merged_rows": 0}

    canon_n = len(canonical)
    strict_n = len(strict)

    canonical = canonical.copy()
    strict = strict.copy()
    for c in key_cols:
        if c not in canonical.columns:
            canonical[c] = pd.NA
        if c not in strict.columns:
            strict[c] = pd.NA
    canonical["__is_strict"] = 0
    strict["__is_strict"] = 1

    if canonical.empty:
        canonical["_merge_key"] = pd.Series([], index=canonical.index, dtype="string")
    else:
        canonical["_merge_key"] = canonical[key_cols].astype("string").agg("|".join, axis=1)
    if strict.empty:
        strict["_merge_key"] = pd.Series([], index=strict.index, dtype="string")
    else:
        strict["_merge_key"] = strict[key_cols].astype("string").agg("|".join, axis=1)

    canon_keys = set(canonical["_merge_key"].dropna().tolist())
    strict_keys = set(strict["_merge_key"].dropna().tolist())
    overrides = len(canon_keys.intersection(strict_keys))
    appended = len(strict_keys.difference(canon_keys))

    if canonical.empty:
        merged = strict.copy()
    elif strict.empty:
        merged = canonical.copy()
    else:
        merged = pd.concat([canonical, strict], ignore_index=True, sort=False)
    merged = merged.sort_values("__is_strict").drop_duplicates(subset=["_merge_key"], keep="last")
    if merged["_merge_key"].duplicated().any():
        raise ValueError("Duplicate merge keys remain after override merge.")

    merged = merged.drop(columns=["__is_strict", "_merge_key"], errors="ignore")
    stats = {
        "canonical_rows": int(canon_n),
        "strict_rows": int(strict_n),
        "overrides": int(overrides),
        "appended": int(appended),
        "merged_rows": int(len(merged)),
    }
    return merged, stats


def _collect_strict_fixed_csvs(strict_root: Path) -> list[Path]:
    if not strict_root.exists():
        return []
    files = sorted(strict_root.glob("**/fixed_k0_hyperparam_sweep_results.csv"))
    merged_name = strict_root / "merged_fixed_k0_source.csv"
    return [p for p in files if p.resolve() != merged_name.resolve()]


def _collect_strict_k_sweep_csvs(strict_root: Path) -> list[Path]:
    if not strict_root.exists():
        return []
    allowed_names = {"replicability_curves_all_metrics.csv", "replicability_curves.csv"}
    files = sorted(p for p in strict_root.glob("**/*.csv") if p.name in allowed_names)
    merged_name = strict_root / "merged_k_sweep_source.csv"
    return [p for p in files if p.resolve() != merged_name.resolve()]


def _concat_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.exists():
            d = pd.read_csv(p)
            d["__source_file"] = str(p)
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    strict_root = args.output_dir.resolve()
    strict_root.mkdir(parents=True, exist_ok=True)

    fixed_old = root / "outputs" / "fixed_k0_hyperparam_sweep" / "fixed_k0_hyperparam_sweep_results.csv"
    fixed_new = root / "outputs" / "fixed_k0_hyperparam_sweep_new_only" / "fixed_k0_hyperparam_sweep_results.csv"
    ks_old = root / "outputs" / "replicability_benchmark_all_metrics" / "replicability_curves_all_metrics.csv"
    ks_new = root / "outputs" / "replicability_benchmark_all_metrics_new_only" / "replicability_curves_all_metrics.csv"

    fixed_canonical_paths = [p for p in [fixed_old, fixed_new] if p.exists()]
    ks_canonical_paths = [p for p in [ks_old, ks_new] if p.exists()]
    fixed_strict_paths = _collect_strict_fixed_csvs(strict_root)
    ks_strict_paths = _collect_strict_k_sweep_csvs(strict_root)

    fixed_canonical = _concat_csvs(fixed_canonical_paths)
    fixed_strict = _concat_csvs(fixed_strict_paths)
    ks_canonical = _concat_csvs(ks_canonical_paths)
    ks_strict = _concat_csvs(ks_strict_paths)

    if not fixed_canonical.empty:
        fixed_canonical = _normalize_fixed(fixed_canonical)
    if not fixed_strict.empty:
        fixed_strict = _normalize_fixed(fixed_strict)
    fixed_key = [
        "dataset_id",
        "method",
        "resolved_k0",
        "resolved_hyperparam_name",
        "resolved_hyperparam_value_key",
    ]
    _assert_unique_by_key(fixed_canonical, fixed_key, "fixed canonical")
    _assert_unique_by_key(fixed_strict, fixed_key, "fixed strict reruns")
    merged_fixed, fixed_stats = _merge_with_override(fixed_canonical, fixed_strict, fixed_key)
    merged_fixed = merged_fixed.drop(columns=["resolved_hyperparam_value_key"], errors="ignore")
    merged_fixed_path = strict_root / "merged_fixed_k0_source.csv"
    merged_fixed.to_csv(merged_fixed_path, index=False)

    if not ks_canonical.empty:
        ks_canonical = _normalize_k_sweep(ks_canonical)
    if not ks_strict.empty:
        ks_strict = _normalize_k_sweep(ks_strict)
    ks_key = ["dataset_id", "method", "metric", "K"]
    _assert_unique_by_key(ks_canonical, ks_key, "k-sweep canonical")
    _assert_unique_by_key(ks_strict, ks_key, "k-sweep strict reruns")
    merged_ks, ks_stats = _merge_with_override(ks_canonical, ks_strict, ks_key)
    merged_ks_path = strict_root / "merged_k_sweep_source.csv"
    merged_ks.to_csv(merged_ks_path, index=False)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_sources": {
            "fixed_k0": [str(p) for p in fixed_canonical_paths],
            "k_sweep": [str(p) for p in ks_canonical_paths],
        },
        "strict_rerun_sources": {
            "fixed_k0": [str(p) for p in fixed_strict_paths],
            "k_sweep": [str(p) for p in ks_strict_paths],
        },
        "merge_stats": {
            "fixed_k0": fixed_stats,
            "k_sweep": ks_stats,
        },
        "merge_keys": {
            "fixed_merge_key": fixed_key,
            "k_sweep_merge_key": ks_key,
        },
        "outputs": {
            "merged_fixed_k0_source_csv": str(merged_fixed_path),
            "merged_k_sweep_source_csv": str(merged_ks_path),
        },
    }
    manifest_path = strict_root / "merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {merged_fixed_path} ({len(merged_fixed)} rows)")
    print(f"Wrote {merged_ks_path} ({len(merged_ks)} rows)")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

