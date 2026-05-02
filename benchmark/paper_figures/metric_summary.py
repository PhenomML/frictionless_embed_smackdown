from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from benchmark.figure_io import save_figure_png_pdf
from benchmark.paper_analysis import prepare_metric_summary


def plot_metric_summary(alignment_summary: pd.DataFrame, figures_dir: Path) -> tuple[Path, pd.DataFrame]:
    _, summary = prepare_metric_summary(alignment_summary)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    axes[0].bar(summary["metric"], summary["alignment_rate"])
    axes[0].set_title("Alignment rate")
    axes[1].bar(summary["metric"], summary["median_Delta_opt_M"])
    axes[1].set_title(r"Median $\Delta_{\mathrm{opt},M}$")
    for ax in axes:
        ax.yaxis.set_major_locator(MaxNLocator(5))
    out = save_figure_png_pdf(fig, figures_dir, "fig9_metric_summary.png")
    return out, summary
