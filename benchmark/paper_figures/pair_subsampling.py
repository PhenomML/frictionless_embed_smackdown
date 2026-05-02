from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from benchmark.display_labels import dataset_display_name, method_display_name, metric_display_name
from benchmark.figure_io import save_figure_png_pdf


def normalize_pair_subsampling(summary: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset_id", "method", "metric", "M_pairs", "delta_max_abs_median", "delta_pass"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Pair-subsampling summary missing columns: {missing}")
    df = summary.copy()
    df["dataset_id"] = df["dataset_id"].map(dataset_display_name)
    df["method"] = df["method"].map(method_display_name)
    df["metric"] = df["metric"].map(metric_display_name)
    df["M_pairs"] = pd.to_numeric(df["M_pairs"], errors="coerce")
    df["delta_max_abs_median"] = pd.to_numeric(df["delta_max_abs_median"], errors="coerce")
    return df


def plot_pair_subsampling_sensitivity(
    summary: pd.DataFrame,
    figures_dir: Path,
    target_metric: str = "AMI",
    target_panels: list[tuple[str, str]] | None = None,
) -> Path | None:
    df = normalize_pair_subsampling(summary)
    target_panels = target_panels or [
        ("Olivetti Faces", "t-SNE"),
        ("Olivetti Faces", "UMAP"),
        ("20 Newsgroups", "t-SNE"),
        ("20 Newsgroups", "UMAP"),
    ]
    sub = df[df["metric"] == target_metric].copy()
    available = [
        tuple(x)
        for x in sub[["dataset_id", "method"]].dropna().drop_duplicates().itertuples(index=False, name=None)
    ]
    chosen = [panel for panel in target_panels if panel in available] or sorted(available)
    if not chosen:
        return None

    panel_colors = {
        ("Olivetti Faces", "t-SNE"): "#1f77b4",
        ("Olivetti Faces", "UMAP"): "#ff7f0e",
        ("20 Newsgroups", "t-SNE"): "#2ca02c",
        ("20 Newsgroups", "UMAP"): "#d62728",
        ("MNIST", "UMAP"): "#9467bd",
        ("AG News", "UMAP"): "#8c564b",
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    for dataset_name, method_name in chosen:
        group = sub[(sub["dataset_id"] == dataset_name) & (sub["method"] == method_name)].sort_values("M_pairs")
        if group.empty:
            continue
        ax.plot(
            group["M_pairs"],
            group["delta_max_abs_median"],
            marker="o",
            color=panel_colors.get((dataset_name, method_name)),
            label=f"{dataset_name} / {method_name}",
        )

    if "delta_threshold" in sub.columns:
        thresholds = pd.to_numeric(sub["delta_threshold"], errors="coerce").dropna().unique()
        if len(thresholds) >= 1:
            ax.axhline(
                float(thresholds[0]),
                linestyle="--",
                color="black",
                alpha=0.75,
                linewidth=1.4,
                label=rf"$\Delta_{{\mathrm{{thr}}}} = {float(thresholds[0]):.2f}$",
            )

    ax.set_title(r"Pair-subsampling sensitivity: $\Delta(M)$ vs. $M$")
    ax.set_xlabel(r"Sampled pair budget $M$")
    ax.set_ylabel(r"$\Delta(M)=\mathrm{median}_{r<s}\,\max_{K}\,|R_M^{(r)}(K)-R_M^{(s)}(K)|$")
    ax.grid(True, which="major", alpha=0.28)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best", framealpha=0.95, ncol=2)
    return save_figure_png_pdf(fig, figures_dir, "pair_subsampling_sensitivity.png")
