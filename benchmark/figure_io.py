from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def require_file(path: Path | str | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} was not set.")
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def read_csv_required(path: Path | str | None, label: str) -> pd.DataFrame:
    return pd.read_csv(require_file(path, label))


def first_existing_path(candidates: list[Path | str | None]) -> Path | None:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def save_figure_png_pdf(fig: Figure, figures_dir: Path, filename: str) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / filename
    try:
        fig.savefig(out, bbox_inches="tight")
        if out.suffix.lower() == ".png":
            fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    finally:
        plt.close(fig)
    return out


def copy_png_pdf_alias(source_png: Path, alias_png: Path) -> list[Path]:
    alias_png.parent.mkdir(parents=True, exist_ok=True)
    copied = []
    shutil.copy2(source_png, alias_png)
    copied.append(alias_png)
    source_pdf = source_png.with_suffix(".pdf")
    alias_pdf = alias_png.with_suffix(".pdf")
    if source_pdf.exists():
        shutil.copy2(source_pdf, alias_pdf)
        copied.append(alias_pdf)
    return copied
