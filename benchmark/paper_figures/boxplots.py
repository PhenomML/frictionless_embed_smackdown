from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.display_labels import dataset_display_name, method_display_name, metric_display_name
from benchmark.figure_io import save_figure_png_pdf


panel_order = [
    ("mnist", "tsne"),
    ("mnist", "umap"),
    ("uci_har", "tsne"),
    ("uci_har", "umap"),
    ("olivetti_faces", "tsne"),
    ("olivetti_faces", "umap"),
    ("cifar10", "tsne"),
    ("cifar10", "umap"),
    ("20newsgroups", "tsne"),
    ("20newsgroups", "umap"),
    ("ag_news", "tsne"),
    ("ag_news", "umap"),
]

k_by_dataset = {
    "mnist": [2, 10],
    "uci_har": [2, 6],
    "olivetti_faces": [2, 40],
    "cifar10": [2, 10],
    "20newsgroups": [2, 20],
    "ag_news": [2, 4],
}


def _style_boxplot(bp: dict) -> None:
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C92C3")
        patch.set_alpha(0.42)
        patch.set_edgecolor("#355C7D")
        patch.set_linewidth(1.0)
    for median in bp["medians"]:
        median.set_color("#F39C12")
        median.set_linewidth(1.8)
    for mean in bp["means"]:
        mean.set_marker("o")
        mean.set_markersize(3.2)
        mean.set_markerfacecolor("darkgreen")
        mean.set_markeredgecolor("darkgreen")
    for flier in bp["fliers"]:
        flier.set_marker(".")
        flier.set_markersize(1.6)
        flier.set_alpha(0.28)


def _row_limits(df: pd.DataFrame, metric: str) -> dict[str, tuple[float, float]]:
    out = {}
    for dataset_id, ks in k_by_dataset.items():
        vals = df[
            (df["metric"] == metric)
            & (df["dataset_id"] == dataset_id)
            & (df["K"].isin(ks))
        ]["score"].dropna().to_numpy()
        if len(vals) == 0:
            continue
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        pad = 0.03 if np.isclose(lo, hi) else 0.07 * (hi - lo)
        out[dataset_id] = (max(0.0, lo - pad), min(1.02, hi + pad))
    return out


def plot_pairwise_boxplots(df: pd.DataFrame, figures_dir: Path) -> dict[str, Path]:
    required = {"dataset_id", "method", "metric", "K", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Boxplot source missing columns: {missing}")

    data = df.copy()
    data["dataset_id"] = data["dataset_id"].astype(str)
    data["method"] = data["method"].astype(str).str.lower()
    data["metric"] = data["metric"].astype(str).str.lower()
    data["K"] = pd.to_numeric(data["K"], errors="coerce")
    data["score"] = pd.to_numeric(data["score"], errors="coerce")

    outputs = {}
    for metric in ["ari", "jaccard", "ami"]:
        fig, axes = plt.subplots(6, 2, figsize=(12.4, 21.5), constrained_layout=True)
        limits = _row_limits(data, metric)
        any_plotted = False
        for ax, (dataset_id, method) in zip(axes.ravel(), panel_order):
            ks = k_by_dataset[dataset_id]
            sub = data[
                (data["metric"] == metric)
                & (data["dataset_id"] == dataset_id)
                & (data["method"] == method)
                & (data["K"].isin(ks))
            ]
            present = [k for k in ks if k in set(sub["K"].dropna())]
            if not present:
                ax.axis("off")
                continue
            arrays = [sub.loc[sub["K"] == k, "score"].dropna().to_numpy() for k in present]
            if not any(len(arr) for arr in arrays):
                ax.axis("off")
                continue
            bp = ax.boxplot(arrays, tick_labels=[str(int(k)) for k in present], patch_artist=True, showmeans=True, widths=0.16)
            _style_boxplot(bp)
            if dataset_id in limits:
                ax.set_ylim(*limits[dataset_id])
            ax.set_title(f"{dataset_display_name(dataset_id)} / {method_display_name(method)} ({metric_display_name(metric)})", fontsize=11)
            ax.set_xlabel(r"$K$", fontsize=10)
            ax.set_ylabel(f"Pairwise {metric_display_name(metric)} on overlap", fontsize=10)
            ax.grid(axis="y", alpha=0.22, linewidth=0.8)
            ax.grid(axis="x", visible=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            any_plotted = True
        if any_plotted:
            outputs[f"box_{metric}"] = save_figure_png_pdf(fig, figures_dir, f"box_{metric}.png")
        else:
            plt.close(fig)
    return outputs
