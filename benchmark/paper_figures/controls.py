from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.display_labels import metric_display_name
from benchmark.figure_io import copy_png_pdf_alias, save_figure_png_pdf
from benchmark.paper_analysis import normalize_replicatewise_controls


control_colors = {
    "random 2D control": "#1f77b4",
    "shuffled-feature control": "#ff7f0e",
}


def _style_axis(ax) -> None:
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.4)
    ax.spines["bottom"].set_alpha(0.4)
    ax.tick_params(axis="both", labelsize=10)


def plot_control_diagnostics(curves: pd.DataFrame, figures_dir: Path) -> Path:
    df = normalize_replicatewise_controls(curves)
    if "p_value" not in df.columns:
        raise ValueError("Control diagnostics require a p_value column.")
    if "p_le_0p05" not in df.columns:
        df["p_le_0p05"] = df["p_value"] <= 0.05

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5), constrained_layout=True)
    ax_ecdf, ax_sorted, ax_curves, ax_bar = axes.ravel()
    for ax in axes.ravel():
        _style_axis(ax)

    for control_name, group in df.groupby("control_name"):
        vals = np.sort(group["p_value"].dropna().to_numpy())
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax_ecdf.step(vals, y, where="post", linewidth=2.2, color=control_colors.get(control_name), label=control_name)
        ax_sorted.plot(np.arange(1, len(vals) + 1), vals, marker="o", linewidth=2.0, markersize=5, color=control_colors.get(control_name), label=control_name)

    ax_ecdf.plot([0, 1], [0, 1], "--", color="0.35", linewidth=1.5, label="uniform")
    ax_ecdf.set_title("Control diagnostics: ECDF vs. uniform", fontsize=13)
    ax_ecdf.set_xlabel("p-value")
    ax_ecdf.set_ylabel("ECDF")
    ax_ecdf.set_xlim(-0.02, 1.02)
    ax_ecdf.set_ylim(-0.02, 1.05)
    ax_ecdf.legend(loc="upper left", frameon=True, framealpha=0.95)

    ax_sorted.set_title("Control diagnostics: sorted p-values", fontsize=13)
    ax_sorted.set_xlabel("Order statistic")
    ax_sorted.set_ylabel("p-value")
    ax_sorted.set_ylim(-0.02, 1.02)
    ax_sorted.legend(loc="upper left", frameon=True, framealpha=0.95)

    for (dataset_name, control_name), group in df.groupby(["dataset", "control_name"]):
        group = group.sort_values("K").dropna(subset=["K", "R", "q95"])
        if group.empty:
            continue
        color = control_colors.get(control_name)
        label_base = f"{dataset_name} / {control_name}"
        ax_curves.plot(group["K"], group["R"], linewidth=2.2, color=color, label=rf"{label_base}: $R$")
        ax_curves.plot(group["K"], group["q95"], linestyle="--", linewidth=2.0, color=color, alpha=0.8, label=rf"{label_base}: $q_{{0.95}}$")

    ax_curves.set_title(r"Control diagnostics: observed $R$ vs. $q_{0.95}$", fontsize=13)
    ax_curves.set_xlabel("K")
    ax_curves.set_ylabel("Score")
    ax_curves.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True, framealpha=0.95, fontsize=9)

    summary = (
        df.groupby(["dataset", "control_name"])["p_le_0p05"]
        .sum()
        .reset_index(name="uncorrected_rejections_at_alpha_0p05")
        .sort_values(["dataset", "control_name"])
    )
    summary["label"] = summary["dataset"] + "\n" + summary["control_name"]
    ax_bar.bar(summary["label"], summary["uncorrected_rejections_at_alpha_0p05"], color="#4C92C3", alpha=0.9)
    ax_bar.set_title("Uncorrected rejection counts under controls", fontsize=13)
    ax_bar.set_ylabel(r"Count of $K$ with $p < 0.05$")
    ax_bar.tick_params(axis="x", rotation=25)
    for tick in ax_bar.get_xticklabels():
        tick.set_ha("right")
    ymax = summary["uncorrected_rejections_at_alpha_0p05"].max()
    ax_bar.set_ylim(0, max(1.0, 1.1 * ymax))
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.grid(axis="x", visible=False)

    out = save_figure_png_pdf(fig, figures_dir, "figB_control_diagnostics_replicatewise.png")
    copy_png_pdf_alias(out, figures_dir / "figB_control_diagnostics.png")
    return out


def plot_negative_control_delta(
    curves: pd.DataFrame,
    figures_dir: Path,
    panel_specs: list[tuple[str, str]] | None = None,
) -> Path:
    df = normalize_replicatewise_controls(curves)
    if "method" not in df.columns:
        raise ValueError("Negative-control curves require a method column.")
    df["delta"] = pd.to_numeric(df["R"], errors="coerce") - pd.to_numeric(df["q95"], errors="coerce")
    panel_specs = panel_specs or [("AG News", "UMAP"), ("MNIST", "UMAP")]

    metric_ylabel = r"$\Delta_M(K)=R_M(K)-q_{0.95,M}(K)$"
    if "metric" in df.columns:
        metric_vals = [m for m in df["metric"].dropna().astype(str).str.lower().unique().tolist() if m]
        if len(metric_vals) == 1:
            metric = metric_display_name(metric_vals[0])
            metric_ylabel = rf"$\Delta_{{\mathrm{{{metric}}}}}(K)=R_{{\mathrm{{{metric}}}}}(K)-q_{{0.95,\mathrm{{{metric}}}}}(K)$"

    fig, axes = plt.subplots(1, len(panel_specs), figsize=(12.6, 4.9), constrained_layout=True)
    axes = np.asarray([axes]) if len(panel_specs) == 1 else np.asarray(axes)
    for ax, (dataset_name, method_name) in zip(axes, panel_specs):
        group = df[(df["dataset"] == dataset_name) & (df["method"] == method_name)].copy()
        for control_name, control_group in group.groupby("control_name"):
            control_group = control_group.sort_values("K")
            ax.plot(control_group["K"], control_group["delta"], linewidth=2.5, label=control_name)
        ax.axhline(0.0, linestyle="--", color="black", alpha=0.6)
        ax.set_title(f"{dataset_name} / {method_name}")
        ax.set_xlabel("K")
        ax.set_ylabel(metric_ylabel)
        ax.legend(framealpha=0.9)

    out = save_figure_png_pdf(fig, figures_dir, "figC_replicability_under_semantically_weak_controls.png")
    copy_png_pdf_alias(out, figures_dir / "figC_negative_control.png")
    return out
