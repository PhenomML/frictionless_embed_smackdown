from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from benchmark.artifact_paths import PaperArtifactPaths
from benchmark.manuscript_mpl import apply_manuscript_style
from benchmark.paper_figures import build_paper_artifacts


valid_targets = {
    "embedding_gallery",
    "heatmap",
    "k_sweep",
    "appendix_rep_validity",
    "boxplots",
    "bh",
    "fixed_k0",
    "metric_summary",
    "pair_subsampling",
    "pair_subsampling_table",
    "controls",
    "negative_controls",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper figures/tables from standardized benchmark outputs.",
    )
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Output artifact directory; defaults to paper_artifacts.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file of path overrides keyed like final_plot_manifest.json.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(valid_targets),
        default=None,
        help="Optional subset of artifact families to build.",
    )
    parser.add_argument(
        "--no-tex",
        action="store_true",
        help="Force Matplotlib mathtext instead of TeX.",
    )
    return parser.parse_args()


def load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return data


def _resolve_config_path(value: object, repo_root: Path) -> str | None:
    if value is None:
        return None
    path = Path(str(value))
    return str(path if path.is_absolute() else repo_root / path)


def config_overrides(config: dict[str, object], repo_root: Path) -> dict[str, str | None]:
    if "overrides" in config:
        overrides = config["overrides"]
        if not isinstance(overrides, dict):
            raise ValueError("Config field 'overrides' must be an object.")
        return {k: _resolve_config_path(v, repo_root) for k, v in overrides.items()}
    known_config_keys = {"repo_root", "artifacts_dir", "only", "use_tex", "overrides"}
    return {
        k: _resolve_config_path(v, repo_root)
        for k, v in config.items()
        if k not in known_config_keys
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config_use_tex = config.get("use_tex", None)
    if args.no_tex:
        use_tex_arg = False
    elif isinstance(config_use_tex, bool):
        use_tex_arg = config_use_tex
    else:
        use_tex_arg = None
    use_tex = apply_manuscript_style(use_tex=use_tex_arg)

    repo_root = args.repo_root
    if args.repo_root == root and config.get("repo_root") is not None:
        repo_root = Path(str(config["repo_root"]))
    artifacts_dir = args.artifacts_dir
    if artifacts_dir is None and config.get("artifacts_dir") is not None:
        artifacts_dir = Path(str(config["artifacts_dir"]))
        if not artifacts_dir.is_absolute():
            artifacts_dir = repo_root / artifacts_dir

    only = set(args.only) if args.only else None
    if only is None and config.get("only") is not None:
        config_only = config["only"]
        if not isinstance(config_only, list):
            raise ValueError("Config field 'only' must be a list.")
        only = {str(item) for item in config_only}

    paths = PaperArtifactPaths.from_root(repo_root, artifacts_dir)
    result = build_paper_artifacts(
        paths=paths,
        overrides=config_overrides(config, repo_root),
        only=only,
    )
    print(f"Manuscript style active (text.usetex={use_tex}).")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
