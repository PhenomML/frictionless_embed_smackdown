#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _rows_from_result(result: dict) -> list[dict]:
    dataset_id = result.get("dataset_id")
    method = result.get("method")
    metric = str(result.get("metric", "")).lower()
    vecs = result.get("pairwise_metric_vec_by_k")
    if not isinstance(vecs, dict):
        return []

    rows: list[dict] = []
    for k_raw, scores in vecs.items():
        try:
            k_val = int(float(k_raw))
        except Exception:
            continue
        if not isinstance(scores, list):
            continue
        for s in scores:
            try:
                score = float(s)
            except Exception:
                continue
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "method": method,
                    "metric": metric,
                    "K": k_val,
                    "score": score,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build pairwise_boxplot_long.csv from replicability_results.json files."
    )
    parser.add_argument(
        "--results-json",
        nargs="+",
        required=True,
        help="One or more replicability_results.json paths.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV path (e.g. results_standardized/controls/pairwise_boxplot_long.csv).",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for p_str in args.results_json:
        p = Path(p_str)
        if not p.exists():
            print(f"Skipping missing file: {p}")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            print(f"Skipping non-list JSON: {p}")
            continue
        for result in data:
            if isinstance(result, dict):
                rows.extend(_rows_from_result(result))

    out = args.output_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows).sort_values(
            ["metric", "dataset_id", "method", "K"]
        )
    else:
        df = pd.DataFrame(columns=["dataset_id", "method", "metric", "K", "score"])
    df.to_csv(out, index=False)
    print(f"Saved: {out}")
    print(f"Rows: {len(df)}")
    if len(df) == 0:
        print(
            "No pairwise vectors found. Re-run k-sweep with --store-pairwise-metric-vec "
            "to populate pairwise_metric_vec_by_k."
        )


if __name__ == "__main__":
    main()
