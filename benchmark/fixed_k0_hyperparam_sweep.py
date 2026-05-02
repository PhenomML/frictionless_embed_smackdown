
from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .config import EmbedConfig, get_default_paths
from .embeddings import embed_tsne, embed_umap
from .loaders import load_dataset
from .paper_datasets import paper_dataset_aliases, paper_dataset_specs, get_paper_k0
from .replicability import k_sweep_replicability
from .utils import subsample_rows

fixed_k0_by_dataset: dict[str, int] = {
    **{dataset_id: spec.k0 for dataset_id, spec in paper_dataset_specs.items()},
    **{alias: paper_dataset_specs[target].k0 for alias, target in paper_dataset_aliases.items()},
}

metric_suffix = {
    "ari": "ARI",
    "jaccard": "Jaccard",
    "ami": "AMI",
}

def get_fixed_k0(dataset_id: str) -> int:
    try:
        return get_paper_k0(dataset_id)
    except KeyError:
        pass
    if dataset_id not in fixed_k0_by_dataset:
        known = ", ".join(sorted(fixed_k0_by_dataset.keys()))
        raise KeyError(
            f"No fixed K0 registered for dataset '{dataset_id}'. Known: {known}"
        )
    return fixed_k0_by_dataset[dataset_id]

def get_primary_hyperparameter_grid(method: str) -> list[dict[str, Any]]:
    if method == "tsne":
        return [{"tsne_perplexity": p} for p in [5, 10, 15, 30, 50, 100]]
    if method == "umap":
        return [{"umap_n_neighbors": n} for n in [5, 10, 15, 30, 50, 100]]
    raise ValueError(f"Unknown method: {method}")

def _compute_embedding(x: Any, method: str, cfg: EmbedConfig):
    if method == "tsne":
        return embed_tsne(x, cfg)
    if method == "umap":
        return embed_umap(x, cfg)
    raise ValueError(f"Unknown method: {method}")

def _primary_hyperparameter_name_and_value(
    method: str,
    embedding_params: dict[str, Any],
) -> tuple[str | None, Any]:
    if method == "tsne":
        name = "tsne_perplexity"
        return name, embedding_params.get(name)
    if method == "umap":
        name = "umap_n_neighbors"
        return name, embedding_params.get(name)
    return None, None

def _safe_first(values: Any) -> Any:
    if values is None:
        return None
    try:
        return values[0]
    except (TypeError, IndexError, KeyError):
        return None


def _write_partial_results(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_json = output_dir / "fixed_k0_hyperparam_sweep_results.partial.json"
    partial_csv = output_dir / "fixed_k0_hyperparam_sweep_results.partial.csv"

    with open(partial_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)

    fieldnames: list[str] = []
    for row in results:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(partial_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

def evaluate_fixed_k0_setting(
    dataset_id: str,
    method: str,
    embedding_params: dict[str, Any],
    k0: int,
    metrics: list[str],
    b: int,
    subsample_frac: float,
    max_pairs: int,
    run_null: bool,
    tau: int,
    random_state: int,
    max_points: int | None = 3000,
    min_pairs_m: int = 1000,
    alpha: float = 0.05,
    z_star: float = 2.0,
) -> dict[str, Any]:
    paths = get_default_paths()
    base_cfg = EmbedConfig(random_state=random_state, max_points=max_points)
    cfg = replace(base_cfg, **embedding_params)

    x, y, meta = load_dataset(
        dataset_id,
        paths["data_root"],
        paths["corpus_dir"],
        pca_dim=cfg.pca_dim,
        subsample=None,
        use_corpus_when_available=True,
        seed=random_state,
    )

    n_original = int(x.shape[0])
    cap_policy = "none"
    if max_points is not None and x.shape[0] > max_points:
        strat = (
            meta.get("label_type") in {"ground_truth", "derived"}
            and y is not None
        )
        x, y, _ = subsample_rows(x, y, max_points, random_state, stratified=strat)
        cap_policy = "label_stratified_subsample" if strat else "uniform_subsample"
    n_used = int(x.shape[0])

    if y is None:
        raise ValueError(
            f"Dataset '{dataset_id}' does not provide ground-truth labels. "
            "This fixed-K0 frontier experiment is intended for labeled datasets."
        )

    z = _compute_embedding(x, method, cfg)
    primary_name, primary_value = _primary_hyperparameter_name_and_value(
        method, embedding_params
    )

    out: dict[str, Any] = {
        "dataset_id": dataset_id,
        "method": method,
        "k0": int(k0),
        "nominal_class_count": int(k0),
        "random_state": int(random_state),
        "b": int(b),
        "subsample_frac": float(subsample_frac),
        "max_pairs": int(max_pairs),
        "run_null": bool(run_null),
        "tau": int(tau),
        "stability_mode": "fixed_embedding",
        "label_type": meta.get("label_type"),
        "representation_description": meta.get("representation_description", meta.get("analysis_rep")),
        "label_construction_note": meta.get("label_construction_note"),
        "n_raw": int(meta.get("n_raw", n_original)),
        "n_original": n_original,
        "n_used": n_used,
        "cap_policy": cap_policy,
        "number_of_classes_retained": meta.get("number_of_classes_retained"),
        "min_class_support": meta.get("min_class_support"),
        "median_class_support": meta.get("median_class_support"),
        "max_class_support": meta.get("max_class_support"),
        "number_of_classes_declared": meta.get("number_of_classes_declared"),
        "classes_dropped": meta.get("classes_dropped"),
        "n_points": int(z.shape[0]),
        "primary_hyperparam_name": primary_name,
        "primary_hyperparam_value": primary_value,
        "embedding_params": embedding_params,
        "embedding_params_json": json.dumps(embedding_params, sort_keys=True),
        "embed_config_json": json.dumps(asdict(cfg), sort_keys=True, default=str),
    }

    if y is not None:
        import numpy as np
        unique_y, counts_y = np.unique(y, return_counts=True)
        out["number_of_classes_retained"] = int(len(unique_y))
        out["min_class_support"] = int(np.min(counts_y)) if len(counts_y) else 0
        out["median_class_support"] = float(np.median(counts_y)) if len(counts_y) else 0.0
        out["max_class_support"] = int(np.max(counts_y)) if len(counts_y) else 0

    out.update({k: v for k, v in embedding_params.items()})

    metrics_use = list(dict.fromkeys(metrics))
    for metric in metrics_use:
        metric_key = metric.lower()
        if metric_key not in metric_suffix:
            raise ValueError(f"Unsupported metric: {metric}")

        suffix = metric_suffix[metric_key]

        res = k_sweep_replicability(
            z=z,
            y=y,
            meta=meta,
            k_range=[int(k0)],
            b=b,
            fraction=subsample_frac,
            base_seed=random_state,
            pairs_per_k=max_pairs,
            learn_min_overlap=True,
            min_pairs_M=min_pairs_m,
            n_null=(tau if run_null else 0),
            alpha=alpha,
            z_star=z_star,
            metric=metric_key,
        )

        out[f"T_{suffix}"] = _safe_first(res.get("rep_mean"))
        out[f"s_{suffix}"] = _safe_first(res.get("rep_std"))
        out[f"V_{suffix}"] = _safe_first(res.get("metric_vs_gt_mean"))

        out[f"pairs_used_{metric_key}"] = _safe_first(res.get("pairs_used"))
        out[f"eligible_pairs_count_{metric_key}"] = res.get("eligible_pairs_count")

        if run_null:
            out[f"mu0_{suffix}"] = _safe_first(res.get("null_mu0"))
            out[f"sigma0_{suffix}"] = _safe_first(res.get("null_sigma0"))
            out[f"q95_{suffix}"] = _safe_first(res.get("null_q95"))
            out[f"Z_{suffix}"] = _safe_first(res.get("z_score"))
            out[f"p_{suffix}"] = _safe_first(res.get("p_value"))
        else:
            out[f"mu0_{suffix}"] = None
            out[f"sigma0_{suffix}"] = None
            out[f"q95_{suffix}"] = None
            out[f"Z_{suffix}"] = None
            out[f"p_{suffix}"] = None

    return out

def run_fixed_k0_frontier_experiment(
    datasets: list[str],
    methods: list[str],
    metrics: list[str],
    b: int = 200,
    subsample_frac: float = 0.8,
    max_pairs: int = 2000,
    run_null: bool = True,
    tau: int = 200,
    random_state: int = 123,
    max_points: int | None = 3000,
    min_pairs_m: int = 1000,
    hyperparameter_grid_by_method: dict[str, list[dict[str, Any]]] | None = None,
    partial_output_dir: Path | None = None,
    partial_every_n_rows: int = 1,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    plan: list[tuple[str, str, dict[str, Any], int]] = []

    for dataset_id in datasets:
        k0 = get_fixed_k0(dataset_id)
        for method in methods:
            if hyperparameter_grid_by_method is not None:
                grid = hyperparameter_grid_by_method.get(method, [])
            else:
                grid = get_primary_hyperparameter_grid(method)
            for params in grid:
                plan.append((dataset_id, method, params, k0))

    total = len(plan)
    for idx, (dataset_id, method, params, k0) in enumerate(plan, start=1):
        print(
            f"[fixed-k0] START row={idx}/{total} dataset={dataset_id} "
            f"method={method} hyperparam={json.dumps(params, sort_keys=True)}"
        , flush=True)
        row = evaluate_fixed_k0_setting(
            dataset_id=dataset_id,
            method=method,
            embedding_params=params,
            k0=k0,
            metrics=metrics,
            b=b,
            subsample_frac=subsample_frac,
            max_pairs=max_pairs,
            run_null=run_null,
            tau=tau,
            random_state=random_state,
            max_points=max_points,
            min_pairs_m=min_pairs_m,
        )
        results.append(row)
        print(
            f"[fixed-k0] DONE  row={idx}/{total} dataset={dataset_id} "
            f"method={method} hyperparam={json.dumps(params, sort_keys=True)}"
        , flush=True)
        if partial_output_dir is not None and partial_every_n_rows > 0 and (idx % partial_every_n_rows == 0):
            _write_partial_results(results, partial_output_dir)

    return results

def save_fixed_k0_results(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "fixed_k0_hyperparam_sweep_results.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)

    csv_path = output_dir / "fixed_k0_hyperparam_sweep_results.csv"
    fieldnames: list[str] = []
    for row in results:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    meta_path = output_dir / "fixed_k0_hyperparam_sweep_runmeta.json"
    run_meta = {
        "n_rows": len(results),
        "columns": sorted(fieldnames),
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(run_meta, handle, indent=2)

