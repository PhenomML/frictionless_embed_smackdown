from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from benchmark.display_labels import dataset_display_name, method_display_name, metric_display_name
from benchmark.figure_io import save_figure_png_pdf


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Appendix replicability/validity source missing columns: {missing}")


def _load_curve_data(df: pd.DataFrame, source_name: str, dataset: str, method: str, metric: str) -> dict[str, object]:
    group = df[
        (df["dataset_id"] == dataset)
        & (df["method"] == method)
        & (df["metric"] == metric)
    ].sort_values("K")
    if group.empty:
        raise ValueError(f"Missing rows for {dataset}/{method}/{metric}")

    k_bh = None
    if "k_bh" in group.columns and pd.to_numeric(group["k_bh"], errors="coerce").notna().any():
        k_bh = int(pd.to_numeric(group["k_bh"], errors="coerce").dropna().iloc[0])

    valid = pd.to_numeric(group["V_mean"], errors="coerce")
    return {
        "k": pd.to_numeric(group["K"], errors="raise").astype(int).to_list(),
        "rep": pd.to_numeric(group["T_mean"], errors="raise").to_list(),
        "valid": valid.to_list() if valid.notna().any() else None,
        "validity_col": "V_mean" if valid.notna().any() else None,
        "k_bh": k_bh,
        "source": source_name,
    }


def _build_curves(
    df: pd.DataFrame,
    source_name: str,
    datasets: list[str],
    methods: list[str],
    metrics: list[str],
    min_required_k_points: int,
) -> dict[tuple[str, str, str], dict[str, object]]:
    curves = {}
    for dataset in datasets:
        for method in methods:
            k_ref = None
            for metric in metrics:
                curve = _load_curve_data(df, source_name, dataset, method, metric)
                k_values = list(curve["k"])
                rep_values = list(curve["rep"])
                valid_values = curve["valid"]

                if len(k_values) != len(rep_values):
                    raise ValueError(f"Length mismatch for replicability: {dataset}/{method}/{metric}")
                if valid_values is not None and len(valid_values) != len(k_values):
                    raise ValueError(f"Length mismatch for validity: {dataset}/{method}/{metric}")
                if len(k_values) < min_required_k_points:
                    raise ValueError(f"Too few K points for {dataset}/{method}/{metric}: {k_values}")

                if k_ref is None:
                    k_ref = k_values
                elif k_values != k_ref:
                    raise ValueError(f"K-grid mismatch for {dataset}/{method}/{metric}: {k_values} != {k_ref}")

                curves[(dataset, method, metric)] = curve
    return curves


def _validate_k_sweep_source(
    df: pd.DataFrame,
    datasets: list[str],
    methods: list[str],
    metrics: list[str],
    min_required_k_points: int,
) -> pd.DataFrame:
    _require_columns(df, {"dataset_id", "method", "metric", "K", "T_mean", "V_mean"})
    expected = pd.MultiIndex.from_product([datasets, methods, metrics], names=["dataset_id", "method", "metric"])
    observed = pd.MultiIndex.from_frame(df[["dataset_id", "method", "metric"]].drop_duplicates())
    missing_combos = expected.difference(observed)
    if len(missing_combos):
        missing = pd.DataFrame(list(missing_combos), columns=expected.names)
        raise ValueError(f"Missing required dataset/method/metric combinations:\n{missing}")

    inventory = (
        df.groupby(["dataset_id", "method", "metric"])["K"]
        .agg(n_k="nunique", k_values=lambda values: sorted(pd.unique(values)))
        .reset_index()
    )
    too_sparse = inventory[inventory["n_k"] < min_required_k_points]
    if len(too_sparse):
        raise ValueError(
            f"Some curves have fewer than {min_required_k_points} K-points:\n"
            f"{too_sparse[['dataset_id', 'method', 'metric', 'n_k', 'k_values']]}"
        )
    return inventory


def _plot_dataset_metric_overlay(
    dataset: str,
    curves: dict[tuple[str, str, str], dict[str, object]],
    methods: list[str],
    metrics: list[str],
    metric_colors: dict[str, str],
    two_panel_figsize: tuple[float, float],
    show_k_bh: bool,
    show_source: bool,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=two_panel_figsize, sharey=True)
    axes = np.asarray(axes).ravel()

    for ax, method in zip(axes, methods):
        for metric in metrics:
            curve = curves[(dataset, method, metric)]
            k_values = list(curve["k"])
            rep_values = list(curve["rep"])
            valid_values = curve["valid"]
            k_bh = curve["k_bh"]
            source_name = str(curve["source"])

            color = metric_colors[metric]
            label = metric_display_name(metric)
            ax.plot(
                k_values,
                rep_values,
                "-",
                marker="o",
                lw=2,
                ms=4,
                color=color,
                label=rf"$R_{{\mathrm{{{label}}}}}(K)$",
            )

            if valid_values is not None:
                ax.plot(
                    k_values,
                    valid_values,
                    "--",
                    lw=2,
                    color=color,
                    alpha=0.95,
                    label=rf"$V_{{\mathrm{{{label}}}}}(K)$",
                )

            if show_k_bh and k_bh is not None and k_bh in k_values:
                k_index = k_values.index(k_bh)
                ax.scatter(
                    [k_bh],
                    [rep_values[k_index]],
                    s=80,
                    marker="p",
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.6,
                    zorder=5,
                )

        title = f"{dataset_display_name(dataset)} / {method_display_name(method)}"
        if show_source:
            title += f"\nsource: {source_name}"
        ax.set_title(title)
        ax.set_xlabel(r"$K$")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Score")
    handles, labels = axes[1].get_legend_handles_labels()
    if show_k_bh:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="p",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=1.4,
                markersize=8,
            )
        )
        labels.append(r"$\widehat{K}_{\mathrm{BH}}$")
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=True)
    fig.suptitle(f"{dataset_display_name(dataset)}: replicability and validity across metrics", fontsize=14)
    fig.tight_layout(rect=[0, 0.17, 1, 0.96])
    return fig


def _build_summary_table(
    datasets: list[str],
    methods: list[str],
    metrics: list[str],
    curves: dict[tuple[str, str, str], dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        for method in methods:
            for metric in metrics:
                curve = curves[(dataset, method, metric)]
                k_values = list(curve["k"])
                rep_values = list(curve["rep"])
                valid_values = curve["valid"]
                k_bh = curve["k_bh"]

                rep_at_bh = np.nan
                val_at_bh = np.nan
                if k_bh is not None and k_bh in k_values:
                    k_index = k_values.index(k_bh)
                    rep_at_bh = rep_values[k_index]
                    if valid_values is not None:
                        val_at_bh = valid_values[k_index]

                rows.append(
                    {
                        "dataset": dataset_display_name(dataset),
                        "method": method_display_name(method),
                        "metric": metric_display_name(metric),
                        "source": curve["source"],
                        "validity_col": curve["validity_col"],
                        "K_hat_BH": k_bh,
                        "R(K_hat_BH)": rep_at_bh,
                        "Val(K_hat_BH)": val_at_bh,
                        "max R(K)": float(np.nanmax(rep_values)),
                        "max Val(K)": float(np.nanmax(valid_values)) if valid_values is not None else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_appendix_rep_validity_figures(
    k_sweep: pd.DataFrame,
    figures_dir: Path,
    *,
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
    metrics: list[str] | None = None,
    min_required_k_points: int = 4,
    source_name: str = "k_sweep_summary_long",
    two_panel_figsize: tuple[float, float] = (8.8, 5.0),
    show_k_bh: bool = True,
    show_source: bool = False,
) -> tuple[dict[str, Path], pd.DataFrame, pd.DataFrame]:
    datasets = datasets or ["mnist", "uci_har", "olivetti_faces", "cifar10", "20newsgroups", "ag_news"]
    methods = methods or ["tsne", "umap"]
    metrics = metrics or ["ami", "ari", "jaccard"]
    metric_colors = {"ami": "C0", "ari": "C1", "jaccard": "C2"}

    df = k_sweep.copy()
    df["dataset_id"] = df["dataset_id"].astype(str)
    df["method"] = df["method"].astype(str).str.lower()
    df["metric"] = df["metric"].astype(str).str.lower()
    df = df[df["dataset_id"].isin(datasets) & df["method"].isin(methods) & df["metric"].isin(metrics)].copy()
    df["K"] = pd.to_numeric(df["K"], errors="raise").astype(int)

    inventory = _validate_k_sweep_source(df, datasets, methods, metrics, min_required_k_points)
    curves = _build_curves(df, source_name, datasets, methods, metrics, min_required_k_points)

    outputs: dict[str, Path] = {}
    for dataset in datasets:
        fig = _plot_dataset_metric_overlay(
            dataset,
            curves,
            methods,
            metrics,
            metric_colors,
            two_panel_figsize,
            show_k_bh,
            show_source,
        )
        outputs[f"appendix_rep_validity_{dataset}"] = save_figure_png_pdf(
            fig,
            figures_dir,
            f"appendix_rep_validity_{dataset}.png",
        )

    summary = _build_summary_table(datasets, methods, metrics, curves)
    return outputs, summary, inventory
