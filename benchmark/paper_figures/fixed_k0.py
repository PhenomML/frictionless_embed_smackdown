from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.display_labels import dataset_display_name, method_display_name, metric_display_name
from benchmark.figure_io import copy_png_pdf_alias, save_figure_png_pdf


method_styles = {
    "tsne": {"linestyle": "-", "alpha": 0.95},
    "umap": {"linestyle": "--", "alpha": 0.95},
}

stat_colors = {"R": "#1f77b4", "V": "#ff7f0e"}

argmax_style = {
    ("tsne", "R"): {"marker": "D", "s": 95, "facecolor": "black", "edgecolor": "black", "linewidth": 1.0, "label": r"t-SNE $\arg\max R$"},
    ("tsne", "V"): {"marker": "s", "s": 95, "facecolor": "black", "edgecolor": "black", "linewidth": 1.0, "label": r"t-SNE $\arg\max V$"},
    ("umap", "R"): {"marker": "D", "s": 95, "facecolor": "white", "edgecolor": "black", "linewidth": 1.4, "label": r"UMAP $\arg\max R$"},
    ("umap", "V"): {"marker": "s", "s": 95, "facecolor": "white", "edgecolor": "black", "linewidth": 1.4, "label": r"UMAP $\arg\max V$"},
}


def normalize_fixed_k0_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset_id", "method", "metric", "hyperparam_value", "hyperparam_label", "T_value", "V_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fixed-K0 summary missing columns: {missing}")
    out = df.copy()
    out["dataset_id"] = out["dataset_id"].astype(str)
    out["method"] = out["method"].astype(str).str.lower()
    out["metric"] = out["metric"].astype(str).str.lower()
    return out


def _theta_candidates(row: pd.Series, stat: str) -> list[str]:
    cols = [
        f"theta_{stat}_star_label",
        f"theta_{stat}_star",
        f"hyperparam_label_argmax_{stat}",
        f"argmax_{stat}_hyperparam_label",
        f"{stat}_argmax_hyperparam_label",
    ]
    vals = []
    for col in cols:
        if col in row.index and pd.notna(row[col]):
            vals.append(str(row[col]))
    return vals


def _theta_r_candidates(row: pd.Series) -> list[str]:
    return _theta_candidates(row, "R") or _theta_candidates(row, "T")


def _alignment_lookup(alignment: pd.DataFrame | None) -> dict[tuple[str, str, str], pd.Series]:
    if alignment is None or alignment.empty:
        return {}
    required = {"dataset_id", "method", "metric"}
    if not required.issubset(alignment.columns):
        return {}
    df = alignment.copy()
    df["dataset_id"] = df["dataset_id"].astype(str)
    df["method"] = df["method"].astype(str).str.lower()
    df["metric"] = df["metric"].astype(str).str.lower()
    return {
        (row["dataset_id"], row["method"], row["metric"]): row
        for _, row in df.iterrows()
    }


def _frontier_label(raw_label: str, method: str) -> str:
    text = str(raw_label).strip()
    if method == "umap" and ("=" in text):
        val = text.split("=", 1)[1].strip()
        return rf"$\mathrm{{nn}}={val}$"
    return text


def _add_frontier_labels(ax, xvals, yvals, raw_labels, method, fontsize=7) -> None:
    offsets = [(8, 8), (8, -8), (10, 6), (10, -6), (6, 10), (6, -10)]
    for i, (x, y, raw_label) in enumerate(zip(xvals, yvals, raw_labels)):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            _frontier_label(raw_label, method),
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=fontsize,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            alpha=0.95,
            clip_on=True,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.78),
        )


def _style_axis(ax) -> None:
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.4)
    ax.spines["bottom"].set_alpha(0.4)


def plot_fixed_frontier(
    fixed_summary: pd.DataFrame,
    metric: str,
    figures_dir: Path,
    out_name: str,
    dataset_ids: list[str] | None = None,
    methods: list[str] | None = None,
) -> Path | None:
    df = normalize_fixed_k0_summary(fixed_summary)
    dataset_ids = dataset_ids or ["olivetti_faces", "20newsgroups"]
    methods = methods or ["tsne", "umap"]
    metric_key = str(metric).lower()
    metric_disp = metric_display_name(metric_key)
    fig, axes = plt.subplots(len(dataset_ids), len(methods), figsize=(12, 8), constrained_layout=True)
    axes = np.asarray(axes).reshape(len(dataset_ids), len(methods))
    any_plotted = False

    for i, dataset_id in enumerate(dataset_ids):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            group = df[
                (df["dataset_id"] == dataset_id)
                & (df["method"] == method)
                & (df["metric"] == metric_key)
            ].sort_values("hyperparam_value")
            if group.empty:
                ax.axis("off")
                continue
            xvals = group["T_value"].to_numpy(dtype=float)
            yvals = group["V_value"].to_numpy(dtype=float)
            labels = group["hyperparam_label"].astype(str).tolist()
            ax.scatter(xvals, yvals, s=44, marker="o", facecolor="#1f77b4", edgecolor="white", linewidth=0.5, zorder=3)
            ax.margins(x=0.14, y=0.14)
            _add_frontier_labels(ax, xvals, yvals, labels, method)
            ax.set_title(f"{dataset_display_name(dataset_id)} / {method_display_name(method)}")
            ax.set_xlabel(rf"$R_{{\mathrm{{{metric_disp}}}}}(K_0;\theta)$")
            ax.set_ylabel(rf"$V_{{\mathrm{{{metric_disp}}}}}(K_0;\theta)$")
            _style_axis(ax)
            any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return None
    return save_figure_png_pdf(fig, figures_dir, f"{out_name}.png")


def _prepare_profile_df(fixed_summary: pd.DataFrame, dataset_id: str, method: str, metric: str) -> pd.DataFrame:
    group = fixed_summary[
        (fixed_summary["dataset_id"] == dataset_id)
        & (fixed_summary["method"] == method)
        & (fixed_summary["metric"] == metric.lower())
    ].sort_values("hyperparam_value")
    if group.empty:
        return pd.DataFrame(columns=["theta", "R", "V", "hyperparam_label"])
    out = (
        group[["hyperparam_value", "T_value", "V_value", "hyperparam_label"]]
        .rename(columns={"hyperparam_value": "theta", "T_value": "R", "V_value": "V"})
        .dropna(subset=["theta", "R", "V"])
        .reset_index(drop=True)
    )
    out["hyperparam_label"] = out["hyperparam_label"].astype(str)
    return out


def _plot_argmax(ax, prof: pd.DataFrame, align_row: pd.Series | None, method: str, stat: str) -> None:
    if align_row is None or prof.empty:
        return
    theta_candidates = _theta_r_candidates(align_row) if stat == "R" else _theta_candidates(align_row, "V")
    ycol = stat
    if not theta_candidates:
        return
    hit = prof[prof["hyperparam_label"].isin(theta_candidates)]
    if hit.empty:
        return
    style = argmax_style[(method, stat)]
    first = hit.iloc[[0]]
    ax.scatter(
        [first["theta"].iloc[0]],
        [first[ycol].iloc[0]],
        marker=style["marker"],
        s=style["s"],
        facecolor=style["facecolor"],
        edgecolor=style["edgecolor"],
        linewidth=style["linewidth"],
        label=style["label"],
        zorder=7,
    )


def _dedup_legend(ax, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen[label] = handle
    if seen:
        ax.legend(seen.values(), seen.keys(), **kwargs)


def plot_fixed_profiles(
    fixed_summary: pd.DataFrame,
    alignment_summary: pd.DataFrame | None,
    metric: str,
    figures_dir: Path,
    out_name: str,
    dataset_ids: list[str] | None = None,
) -> Path | None:
    df = normalize_fixed_k0_summary(fixed_summary)
    lookup = _alignment_lookup(alignment_summary)
    dataset_ids = dataset_ids or ["olivetti_faces", "20newsgroups"]
    fig, axes = plt.subplots(1, len(dataset_ids), figsize=(7.2 * len(dataset_ids), 5.4), constrained_layout=True)
    axes = np.asarray([axes]) if len(dataset_ids) == 1 else np.asarray(axes)
    any_plotted = False

    for ax, dataset_id in zip(axes, dataset_ids):
        panel_plotted = False
        for method in ["tsne", "umap"]:
            prof = _prepare_profile_df(df, dataset_id, method, metric)
            if prof.empty:
                continue
            style = method_styles[method]
            ax.plot(prof["theta"], prof["R"], color=stat_colors["R"], linestyle=style["linestyle"], marker="o", label=rf"{method_display_name(method)} $R(\theta)$", zorder=3)
            ax.plot(prof["theta"], prof["V"], color=stat_colors["V"], linestyle=style["linestyle"], marker="s", label=rf"{method_display_name(method)} $V(\theta)$", zorder=3)
            align_row = lookup.get((dataset_id, method, str(metric).lower()), None)
            _plot_argmax(ax, prof, align_row, method, "R")
            _plot_argmax(ax, prof, align_row, method, "V")
            panel_plotted = True
        if panel_plotted:
            ax.set_title(dataset_display_name(dataset_id))
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel(f"{metric_display_name(metric)} score")
            _style_axis(ax)
            _dedup_legend(ax, loc="best", frameon=True, fontsize=8, ncol=2)
            any_plotted = True
        else:
            ax.axis("off")

    fig.suptitle(f"Fixed-$K_0$ {metric_display_name(metric)} profiles", y=1.03, fontsize=16)
    if not any_plotted:
        plt.close(fig)
        return None
    return save_figure_png_pdf(fig, figures_dir, f"{out_name}.png")


def build_fixed_k0_figures(
    fixed_summary: pd.DataFrame,
    alignment_summary: pd.DataFrame | None,
    figures_dir: Path,
) -> dict[str, Path | None]:
    outputs = {
        "fig5_fixedk0_ari_raw_frontier": plot_fixed_frontier(fixed_summary, "ari", figures_dir, "fig5_fixedk0_ari_raw_frontier"),
        "fig6_fixedk0_ari_profiles": plot_fixed_profiles(fixed_summary, alignment_summary, "ari", figures_dir, "fig6_fixedk0_ari_profiles"),
        "fig_fixedk0_jaccard_raw_frontier": plot_fixed_frontier(fixed_summary, "jaccard", figures_dir, "fig_fixedk0_jaccard_raw_frontier"),
        "fig6_fixedk0_jaccard_profiles": plot_fixed_profiles(fixed_summary, alignment_summary, "jaccard", figures_dir, "fig6_fixedk0_jaccard_profiles"),
        "fig7_fixedk0_ami_raw_frontier": plot_fixed_frontier(fixed_summary, "ami", figures_dir, "fig7_fixedk0_ami_raw_frontier"),
        "fig8_fixedk0_ami_profiles": plot_fixed_profiles(fixed_summary, alignment_summary, "ami", figures_dir, "fig8_fixedk0_ami_profiles"),
    }
    jaccard = outputs["fig_fixedk0_jaccard_raw_frontier"]
    if jaccard is not None:
        copy_png_pdf_alias(jaccard, figures_dir / "fig6_fixedk0_jaccard_raw_frontier.png")
    return outputs
