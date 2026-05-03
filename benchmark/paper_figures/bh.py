from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark.figure_io import save_figure_png_pdf


def _bh(pvals_by_k: dict[int, float], alpha: float = 0.05) -> dict[str, np.ndarray | float | int | None]:
    items = sorted((int(k), float(p)) for k, p in pvals_by_k.items())
    ks = np.array([k for k, _ in items])
    pvals = np.array([p for _, p in items])
    order = np.argsort(pvals, kind="mergesort")
    sorted_ks = ks[order]
    sorted_pvals = pvals[order]
    ranks = np.arange(1, len(sorted_pvals) + 1)
    thresholds = alpha * ranks / len(sorted_pvals)
    passed = sorted_pvals <= thresholds
    if np.any(passed):
        j_star = int(np.flatnonzero(passed).max())
        rejected = np.arange(len(sorted_pvals)) <= j_star
        k_bh = float(np.max(sorted_ks[rejected]))
    else:
        j_star = None
        rejected = np.zeros(len(sorted_pvals), dtype=bool)
        k_bh = np.nan
    return {
        "ks": ks,
        "pvals": pvals,
        "sorted_ks": sorted_ks,
        "sorted_pvals": sorted_pvals,
        "ranks": ranks,
        "thresholds": thresholds,
        "rejected": rejected,
        "j_star": j_star,
        "k_bh": k_bh,
    }


def plot_bh_appendix_synthetic(figures_dir: Path) -> Path:
    pvals_by_k = {2: 0.004, 3: 0.007, 5: 0.013, 8: 0.022, 10: 0.041, 15: 0.18, 20: 0.31, 25: 0.47, 30: 0.62, 40: 0.79}
    diag = _bh(pvals_by_k)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)

    keep = ~diag["rejected"]
    axes[0].scatter(diag["ranks"][keep], diag["sorted_pvals"][keep], s=22, color="black", label="not rejected")
    axes[0].scatter(diag["ranks"][diag["rejected"]], diag["sorted_pvals"][diag["rejected"]], s=24, color="tomato", label="BH rejected")
    axes[0].plot(diag["ranks"], diag["thresholds"], color="gray", linewidth=1.8, label=r"BH line: $\alpha j/m$")
    axes[0].set_title("BH procedure")
    axes[0].set_xlabel("Rank after sorting p-values")
    axes[0].set_ylabel(r"Sorted p-value $p_{(j)}$")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=8, loc="upper left")

    order_k = np.argsort(diag["ks"])
    rejected_ks = set(diag["sorted_ks"][diag["rejected"]].tolist())
    colors = ["tomato" if int(k) in rejected_ks else "black" for k in diag["ks"][order_k]]
    axes[1].scatter(diag["ks"][order_k], diag["pvals"][order_k], c=colors, s=28)
    axes[1].plot(diag["ks"][order_k], diag["pvals"][order_k], linewidth=1.0, alpha=0.5)
    if np.isfinite(diag["k_bh"]):
        axes[1].axvline(diag["k_bh"], color="gray", linestyle="--", linewidth=1.2, label=rf"$\widehat{{K}}_{{BH}}={int(diag['k_bh'])}$")
    axes[1].set_title(r"Synthetic p-values by $K$")
    axes[1].set_xlabel(r"Cluster resolution $K$")
    axes[1].set_ylabel("Raw p-value")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(fontsize=8, loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.25)
    return save_figure_png_pdf(fig, figures_dir, "fig_bh_appendix_two_panel_synthetic.png")
