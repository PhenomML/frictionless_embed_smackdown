#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from benchmark.config import get_default_paths
from benchmark.fixed_k0_hyperparam_sweep import get_fixed_k0
from benchmark.loaders import load_dataset
from benchmark.paper_datasets import paper_dataset_ids
from benchmark.utils import subsample_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preflight audit for text replacement datasets.")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(paper_dataset_ids(domain="Text")),
        help="Candidate datasets to audit.",
    )
    p.add_argument("--max-points", type=int, default=5000, help="Cap used for fixed-K0 runs.")
    p.add_argument("--seed", type=int, default=123, help="Deterministic random seed.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "text_replacement_preflight",
        help="Output directory for audit artifacts.",
    )
    return p.parse_args()


def _safe_bool(v: object) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return bool(v)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = get_default_paths()

    rows: list[dict[str, object]] = []
    for dataset_id in args.datasets:
        x, y, meta = load_dataset(
            dataset_id=dataset_id,
            data_root=paths["data_root"],
            corpus_dir=paths["corpus_dir"],
            pca_dim=50,
            subsample=None,
            use_corpus_when_available=True,
            seed=args.seed,
        )
        if y is None:
            raise ValueError(f"{dataset_id}: loader returned no labels.")

        n_raw = int(meta.get("n_raw", x.shape[0]))
        n_original = int(x.shape[0])
        cap_policy = "none"
        if args.max_points is not None and x.shape[0] > args.max_points:
            x, y, _ = subsample_rows(x, y, args.max_points, args.seed, stratified=True)
            cap_policy = "label_stratified_subsample"
        n_used = int(x.shape[0])

        unique, counts = np.unique(y, return_counts=True)
        declared = int(meta.get("number_of_classes_declared", len(meta.get("class_names", unique.tolist()))))
        retained = int(len(unique))
        min_support = int(np.min(counts)) if len(counts) else 0
        median_support = float(np.median(counts)) if len(counts) else 0.0
        max_support = int(np.max(counts)) if len(counts) else 0
        classes_dropped = retained < declared
        k0 = int(get_fixed_k0(dataset_id))

        row = {
            "dataset_id": dataset_id,
            "declared_class_count": declared,
            "k0": k0,
            "n_raw": n_raw,
            "n_used_after_cap": n_used,
            "cap_policy": cap_policy,
            "retained_class_count": retained,
            "min_class_support": min_support,
            "median_class_support": median_support,
            "max_class_support": max_support,
            "single_label": True,
            "multiclass": retained > 2,
            "representation_description": meta.get("representation_description"),
            "label_construction_note": meta.get("label_construction_note"),
            "classes_dropped": classes_dropped,
            "k0_directly_valid": (k0 == declared and retained == declared and not classes_dropped),
        }
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["dataset_id"]).reset_index(drop=True)
    out_csv = args.output_dir / "candidate_text_dataset_preflight.csv"
    out.to_csv(out_csv, index=False)

    manifest = {
        "datasets": args.datasets,
        "max_points": args.max_points,
        "seed": args.seed,
        "output_csv": str(out_csv),
        "row_count": int(len(out)),
    }
    (args.output_dir / "preflight_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(out.to_string(index=False))
    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()

