from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.display_labels import dataset_display_name, method_display_name
from benchmark.figure_io import save_figure_png_pdf


def _first_present(columns: pd.Index, candidates: list[str]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def plot_all6_ami_heatmaps(summary: pd.DataFrame, figures_dir: Path) -> Path:
    df = summary.copy()
    if "dataset" not in df.columns:
        df["dataset"] = df["dataset_name"] if "dataset_name" in df.columns else df["dataset_id"]
    rep_col = _first_present(df.columns, ["max_R_AMI", "max_T_AMI"])
    panels = [
        (rep_col, r"$\max_K R_{\mathrm{AMI}}(K)$", ".3f"),
        ("frac_sig_K_AMI", "Fraction significant K", ".2f"),
        ("auc_above_q95_AMI", r"AUC above $q_{0.95}$", ".1f"),
        ("gap_at_K_BH_AMI", r"$\Delta_{\mathrm{AMI}}(\widehat K_{\mathrm{BH}})$", ".3f"),
        ("V_at_K_BH_AMI", r"$V_{\mathrm{AMI}}(\widehat K_{\mathrm{BH}})$", ".3f"),
    ]
    missing = [col for col, _, _ in panels if col is None or col not in df.columns]
    if missing:
        raise ValueError(f"AMI heatmap source missing columns: {missing}")

    dataset_order = ["mnist", "uci_har", "olivetti_faces", "cifar10", "20newsgroups", "ag_news"]
    method_order = ["tsne", "umap"]
    df["dataset_id"] = df["dataset_id"].astype(str)
    df["method"] = df["method"].astype(str).str.lower()

    fig, axes = plt.subplots(1, len(panels), figsize=(15.5, 5.1), constrained_layout=True)
    for ax, (column, title, fmt) in zip(axes, panels):
        matrix = np.full((len(dataset_order), len(method_order)), np.nan)
        for i, dataset_id in enumerate(dataset_order):
            for j, method in enumerate(method_order):
                vals = pd.to_numeric(
                    df.loc[(df["dataset_id"] == dataset_id) & (df["method"] == method), column],
                    errors="coerce",
                ).dropna()
                if len(vals):
                    matrix[i, j] = float(vals.iloc[0])
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(method_order)), [method_display_name(m) for m in method_order], rotation=35, ha="right")
        ax.set_yticks(range(len(dataset_order)), [dataset_display_name(d) for d in dataset_order])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.82)
    return save_figure_png_pdf(fig, figures_dir, "fig2_all6_ami_heatmaps.png")
