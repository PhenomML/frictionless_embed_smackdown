from __future__ import annotations

import shutil
import subprocess

import matplotlib.pyplot as plt


def latex_deps_available() -> bool:
    """Return True only when Matplotlib's TeX dependencies are available."""
    latex_bin = shutil.which("latex")
    kpsewhich_bin = shutil.which("kpsewhich")
    if latex_bin is None or kpsewhich_bin is None:
        return False
    try:
        check = subprocess.run(
            [kpsewhich_bin, "type1cm.sty"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False
    return check.returncode == 0 and bool(check.stdout.strip())


def manuscript_style(use_tex: bool | None = None) -> dict[str, object]:
    if use_tex is None:
        use_tex = latex_deps_available()
    return {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "text.usetex": use_tex,
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 14,
        "figure.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.2,
        "lines.markersize": 5.5,
    }


def apply_manuscript_style(use_tex: bool | None = None) -> bool:
    resolved_use_tex = latex_deps_available() if use_tex is None else use_tex
    plt.rcParams.update(manuscript_style(resolved_use_tex))
    return resolved_use_tex
