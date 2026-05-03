from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root from the installed benchmark package."""
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PaperArtifactPaths:
    """Canonical paths used by the paper artifact generation pipeline."""

    root: Path
    results_dir: Path
    controls_dir: Path
    artifacts_dir: Path
    figures_dir: Path
    tables_dir: Path

    @classmethod
    def from_root(
        cls,
        root: Path | str | None = None,
        artifacts_dir: Path | str | None = None,
    ) -> "PaperArtifactPaths":
        resolved_root = Path(root).resolve() if root is not None else repo_root()
        resolved_artifacts = (
            Path(artifacts_dir).resolve()
            if artifacts_dir is not None
            else resolved_root / "paper_artifacts"
        )
        results_dir = resolved_root / "results_standardized"
        return cls(
            root=resolved_root,
            results_dir=results_dir,
            controls_dir=results_dir / "controls",
            artifacts_dir=resolved_artifacts,
            figures_dir=resolved_artifacts / "figures",
            tables_dir=resolved_artifacts / "tables",
        )

    def ensure_output_dirs(self) -> None:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def default_inputs(self) -> dict[str, Path | None]:
        """Default figure/table inputs matching the former plotting notebook."""
        return {
            "heatmap_summary_csv": self.tables_dir / "table3_all6_ami_summary.csv",
            "boxplot_long_csv": self.controls_dir / "pairwise_boxplot_long.csv",
            "pair_subsampling_csv": None,
            "pair_subsampling_summary_csv": None,
            "k_sweep_long_csv": self.results_dir / "k_sweep_summary_long.csv",
            "fixed_k0_summary_csv": self.results_dir / "fixed_k0_summary_long.csv",
            "fixed_k0_alignment_csv": self.results_dir / "fixed_k0_alignment_summary.csv",
            "metric_summary_source_csv": self.results_dir / "fixed_k0_alignment_summary.csv",
            "control_pvalues_csv": self.controls_dir / "replicatewise_control_curves.csv",
            "control_curves_csv": self.controls_dir / "replicatewise_control_curves.csv",
            "negative_control_curves_csv": self.controls_dir / "replicatewise_control_curves.csv",
        }
