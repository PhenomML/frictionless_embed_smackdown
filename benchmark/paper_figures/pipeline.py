from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from benchmark.artifact_paths import PaperArtifactPaths
from benchmark.figure_io import first_existing_path, read_csv_required
from benchmark.paper_analysis import (
    build_pair_subsampling_metric_stability,
    load_pair_subsampling_summary,
)
from benchmark.paper_figures.appendix_rep_validity import build_appendix_rep_validity_figures
from benchmark.paper_figures.bh import plot_bh_appendix_synthetic
from benchmark.paper_figures.boxplots import plot_pairwise_boxplots
from benchmark.paper_figures.controls import plot_control_diagnostics, plot_negative_control_delta
from benchmark.paper_figures.fixed_k0 import build_fixed_k0_figures
from benchmark.paper_figures.heatmaps import plot_all6_ami_heatmaps
from benchmark.paper_figures.k_sweep import build_k_sweep_figures
from benchmark.paper_figures.metric_summary import plot_metric_summary
from benchmark.paper_figures.pair_subsampling import plot_pair_subsampling_sensitivity


def _resolve_input(
    overrides: dict[str, Path | str | None],
    key: str,
    candidates: list[Path | str | None],
) -> Path | None:
    value = overrides.get(key)
    if value is not None:
        return Path(value)
    return first_existing_path(candidates)


def default_extra_pair_subsampling_paths(paths: PaperArtifactPaths) -> list[Path]:
    return [
        paths.controls_dir / "_pair_subsampling_ari_tsne" / "pair_subsampling_summary.csv",
        paths.controls_dir / "_pair_subsampling_ari_umap" / "pair_subsampling_summary.csv",
        paths.controls_dir / "_pair_subsampling_jaccard_tsne" / "pair_subsampling_summary.csv",
        paths.controls_dir / "_pair_subsampling_jaccard_umap" / "pair_subsampling_summary.csv",
    ]


def build_paper_artifacts(
    paths: PaperArtifactPaths | None = None,
    overrides: dict[str, Path | str | None] | None = None,
    only: set[str] | None = None,
) -> dict[str, object]:
    """Build scriptable paper artifacts from standardized CSV inputs."""
    paths = paths or PaperArtifactPaths.from_root()
    paths.ensure_output_dirs()
    merged_overrides = paths.default_inputs()
    if overrides:
        merged_overrides.update(overrides)

    built: dict[str, str | None] = {}
    skipped: dict[str, str] = {}

    def wants(name: str) -> bool:
        return only is None or name in only

    fixed_path = Path(merged_overrides["fixed_k0_summary_csv"]) if merged_overrides.get("fixed_k0_summary_csv") else None
    align_path = Path(merged_overrides["fixed_k0_alignment_csv"]) if merged_overrides.get("fixed_k0_alignment_csv") else None

    fixed_df: pd.DataFrame | None = None
    align_df: pd.DataFrame | None = None
    if fixed_path is not None and fixed_path.exists():
        fixed_df = pd.read_csv(fixed_path)
    if align_path is not None and align_path.exists():
        align_df = pd.read_csv(align_path)

    if wants("embedding_gallery"):
        try:
            from benchmark.paper_figures.embedding_gallery import plot_embedding_gallery

            out = plot_embedding_gallery(paths.root, paths.figures_dir, paths.tables_dir)
            built["fig1_embedding_gallery_main"] = str(out)
        except Exception as exc:
            skipped["embedding_gallery"] = str(exc)

    if wants("heatmap"):
        heatmap_path = Path(merged_overrides["heatmap_summary_csv"]) if merged_overrides.get("heatmap_summary_csv") else None
        if heatmap_path is None or not heatmap_path.exists():
            skipped["heatmap"] = f"missing heatmap_summary_csv: {heatmap_path}"
        else:
            out = plot_all6_ami_heatmaps(pd.read_csv(heatmap_path), paths.figures_dir)
            built["fig2_all6_ami_heatmaps"] = str(out)

    if wants("k_sweep"):
        k_sweep_path = Path(merged_overrides["k_sweep_long_csv"]) if merged_overrides.get("k_sweep_long_csv") else None
        if k_sweep_path is None or not k_sweep_path.exists():
            skipped["k_sweep"] = f"missing k_sweep_long_csv: {k_sweep_path}"
        else:
            outputs = build_k_sweep_figures(pd.read_csv(k_sweep_path), paths.figures_dir)
            built.update({key: str(value) for key, value in outputs.items()})

    if wants("appendix_rep_validity"):
        k_sweep_path = Path(merged_overrides["k_sweep_long_csv"]) if merged_overrides.get("k_sweep_long_csv") else None
        if k_sweep_path is None or not k_sweep_path.exists():
            skipped["appendix_rep_validity"] = f"missing k_sweep_long_csv: {k_sweep_path}"
        else:
            outputs, summary, inventory = build_appendix_rep_validity_figures(
                pd.read_csv(k_sweep_path),
                paths.figures_dir,
            )
            built.update({key: str(value) for key, value in outputs.items()})
            summary_path = paths.tables_dir / "appendix_rep_validity_metric_overlay_summary.csv"
            inventory_path = paths.tables_dir / "appendix_rep_validity_k_inventory.csv"
            summary.to_csv(summary_path, index=False)
            inventory.to_csv(inventory_path, index=False)
            built["appendix_rep_validity_metric_overlay_summary"] = str(summary_path)
            built["appendix_rep_validity_k_inventory"] = str(inventory_path)

    if wants("boxplots"):
        boxplot_path = _resolve_input(
            merged_overrides,
            "boxplot_long_csv",
            [
                paths.controls_dir / "pairwise_boxplot_long.csv",
                paths.results_dir / "pairwise_boxplot_long.csv",
                paths.tables_dir / "pairwise_boxplot_long.csv",
            ],
        )
        if boxplot_path is None or not boxplot_path.exists():
            skipped["boxplots"] = "missing pairwise_boxplot_long.csv"
        else:
            outputs = plot_pairwise_boxplots(pd.read_csv(boxplot_path), paths.figures_dir)
            built.update({key: str(value) for key, value in outputs.items()})

    if wants("bh"):
        out = plot_bh_appendix_synthetic(paths.figures_dir)
        built["fig_bh_appendix_two_panel_synthetic"] = str(out)

    if wants("fixed_k0"):
        if fixed_df is None:
            skipped["fixed_k0"] = f"missing fixed_k0_summary_csv: {fixed_path}"
        else:
            outputs = build_fixed_k0_figures(fixed_df, align_df, paths.figures_dir)
            built.update({key: str(value) if value is not None else None for key, value in outputs.items()})

    if wants("metric_summary"):
        metric_path = Path(merged_overrides["metric_summary_source_csv"]) if merged_overrides.get("metric_summary_source_csv") else align_path
        if metric_path is None or not metric_path.exists():
            skipped["metric_summary"] = f"missing metric_summary_source_csv: {metric_path}"
        else:
            out, summary = plot_metric_summary(read_csv_required(metric_path, "metric_summary_source_csv"), paths.figures_dir)
            built["fig9_metric_summary"] = str(out)
            summary.to_csv(paths.tables_dir / "fig9_metric_summary.csv", index=False)

    if wants("pair_subsampling"):
        pair_path = _resolve_input(
            merged_overrides,
            "pair_subsampling_csv",
            [
                paths.controls_dir / "pair_subsampling_summary.csv",
                paths.results_dir / "pair_subsampling_summary.csv",
                paths.tables_dir / "pair_subsampling_summary.csv",
            ],
        )
        if pair_path is None or not pair_path.exists():
            skipped["pair_subsampling"] = "missing pair_subsampling_summary.csv"
        else:
            pair_df = pd.read_csv(pair_path)
            out = plot_pair_subsampling_sensitivity(pair_df, paths.figures_dir)
            built["pair_subsampling_sensitivity"] = str(out) if out is not None else None

    if wants("pair_subsampling_table"):
        pair_summary_path = _resolve_input(
            merged_overrides,
            "pair_subsampling_summary_csv",
            [
                paths.controls_dir / "pair_subsampling_summary.csv",
                paths.results_dir / "pair_subsampling_summary.csv",
                paths.tables_dir / "pair_subsampling_summary.csv",
            ],
        )
        if pair_summary_path is None or not pair_summary_path.exists():
            skipped["pair_subsampling_table"] = "missing pair_subsampling_summary.csv"
        else:
            pair_summary = load_pair_subsampling_summary(
                pair_summary_path,
                default_extra_pair_subsampling_paths(paths),
            )
            stability = build_pair_subsampling_metric_stability(pair_summary)
            out_csv = paths.tables_dir / "tableE_pair_subsampling_metric_stability_summary.csv"
            stability.to_csv(out_csv, index=False)
            built["tableE_pair_subsampling_metric_stability_summary"] = str(out_csv)

    if wants("controls"):
        control_path = Path(merged_overrides["control_curves_csv"]) if merged_overrides.get("control_curves_csv") else None
        if control_path is None or not control_path.exists():
            skipped["controls"] = f"missing control_curves_csv: {control_path}"
        else:
            out = plot_control_diagnostics(pd.read_csv(control_path), paths.figures_dir)
            built["figB_control_diagnostics_replicatewise"] = str(out)

    if wants("negative_controls"):
        neg_path = Path(merged_overrides["negative_control_curves_csv"]) if merged_overrides.get("negative_control_curves_csv") else None
        if neg_path is None or not neg_path.exists():
            skipped["negative_controls"] = f"missing negative_control_curves_csv: {neg_path}"
        else:
            out = plot_negative_control_delta(pd.read_csv(neg_path), paths.figures_dir)
            built["figC_replicability_under_semantically_weak_controls"] = str(out)

    manifest = {
        **{key: str(value) if value is not None else None for key, value in merged_overrides.items()},
        "built": built,
        "skipped": skipped,
    }
    manifest_path = paths.tables_dir / "final_plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return {"manifest_path": str(manifest_path), "built": built, "skipped": skipped}
