
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

from benchmark.paper_datasets import paper_corpus_root, paper_data_root

@dataclass(frozen=True)
class EmbedConfig:

    random_state: int = 123
    pca_dim: int = 50
    tsne_perplexity: float = 30.0
    tsne_learning_rate: float = 200.0
    tsne_n_iter: int = 1000
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    mds_metric: bool = True
    mds_n_init: int = 4
    mds_max_iter: int = 300
    max_points: int | None = 3000

@dataclass(frozen=True)
class EricaConfig:

    b: int = 200
    percent_train: float = 0.8
    ap_damping: float = 0.85
    ap_random_state: int = 5
    seed: int = 123

def get_default_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent.parent
    return {
        "data_root": paper_data_root(root),
        "corpus_dir": paper_corpus_root(root),
        "figures_dir": root / "outputs" / "figures",
        "embeddings_dir": root / "outputs" / "embeddings",
        "erica_dir": root / "outputs" / "erica_runs",
    }
