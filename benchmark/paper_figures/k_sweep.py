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


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0
    return series.astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y"])


def _k_bh(group: pd.DataFrame, kbh_col: str | None, reject_col: str | None) -> float:
    if kbh_col is not None:
        vals = pd.to_numeric(group[kbh_col], errors="coerce").dropna().unique()
        if len(vals) == 1:
            return float(vals[0])
        if len(vals) > 1:
            return float(np.max(vals))
    if reject_col is not None:
        reject = _bool_series(group[reject_col])
        if reject.any():
            return float(pd.to_numeric(group.loc[reject, "K"], errors="coerce").max())
    return np.nan


def _value_at_k(group: pd.DataFrame, k: float, column: str) -> float:
    if pd.isna(k):
        return np.nan
    rows = group[pd.to_numeric(group["K"], errors="coerce") == int(k)]
    vals = pd.to_numeric(rows[column], errors="coerce").dropna().to_numpy()
    return float(vals[0]) if len(vals) else np.nan


def _plot_ami_grid(df: pd.DataFrame, datasets: list[str], figures_dir: Path, filename: str) -> Path:
    r_col = _first_present(df.columns, ["R_mean", "T_mean"])
    sd_col = _first_present(df.columns, ["R_sd_pairs", "T_sd_pairs"])
    kbh_col = _first_present(df.columns, ["K_BH_AMI", "k_bh", "K_BH", "hat_k_bh", "k_bh_plot"])
    reject_col = _first_present(df.columns, ["bh_reject", "p_le_0p05", "is_bh_reject", "BH_reject", "bh_selected", "is_k_bh"])
    required = {"dataset_id", "method", "metric", "K", "V_mean", "null_q95"}
    missing = required - set(df.columns)
    if r_col is None:
        missing.add("R_mean or T_mean")
    if missing:
        raise ValueError(f"K-sweep summary missing columns: {missing}")

    ami = df[df["metric"].astype(str).str.lower().eq("ami")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    any_plotted = False
    for i, dataset_id in enumerate(datasets):
        for j, method in enumerate(["tsne", "umap"]):
            ax = axes[i, j]
            group = ami[
                (ami["dataset_id"].astype(str) == dataset_id)
                & (ami["method"].astype(str).str.lower() == method)
            ].sort_values("K")
            if group.empty:
                ax.axis("off")
                continue
            x = pd.to_numeric(group["K"], errors="coerce")
            r = pd.to_numeric(group[r_col], errors="coerce")
            v = pd.to_numeric(group["V_mean"], errors="coerce")
            q95 = pd.to_numeric(group["null_q95"], errors="coerce")
            ax.plot(x, r, label=r"$R_{\mathrm{AMI}}$", lw=2)
            if sd_col is not None and sd_col in group.columns:
                sd = pd.to_numeric(group[sd_col], errors="coerce")
                if sd.notna().any():
                    ax.fill_between(x, r - sd, r + sd, alpha=0.2)
            ax.plot(x, v, label=r"$V_{\mathrm{AMI}}$", lw=2)
            ax.plot(x, q95, "--", label=r"$q_{0.95,\mathrm{AMI}}$", lw=1.2)
            k_bh = _k_bh(group, kbh_col, reject_col)
            y_bh = _value_at_k(group, k_bh, r_col)
            if pd.notna(y_bh):
                ax.scatter([k_bh], [y_bh], marker="p", s=95, color="black", zorder=5, label=r"$\widehat{K}_{\mathrm{BH}}$")
            ax.set_title(f"{dataset_display_name(dataset_id)} / {method_display_name(method)}")
            ax.set_xlabel("K")
            ax.set_ylabel("Score")
            ax.legend(fontsize=8)
            any_plotted = True
    if not any_plotted:
        plt.close(fig)
        raise ValueError("No AMI K-sweep panels were plotted.")
    return save_figure_png_pdf(fig, figures_dir, filename)


def build_k_sweep_figures(df: pd.DataFrame, figures_dir: Path) -> dict[str, Path]:
    return {
        "fig3_ami_hard_regime": _plot_ami_grid(df, ["ag_news", "cifar10"], figures_dir, "fig3_ami_hard_regime.png"),
        "fig4_ami_easy_regime": _plot_ami_grid(df, ["mnist", "uci_har"], figures_dir, "fig4_ami_easy_regime.png"),
    }
