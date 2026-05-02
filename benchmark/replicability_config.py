
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class K_sweep_config:

    k_range: list[int]
    b: int = 200
    fraction: float = 0.8
    pairs_per_k: int = 2000

    learn_min_overlap: bool = True
    min_pairs_M: int = 1000
    min_overlap: int | None = None
    min_overlap_default: int = 10

    n_null: int = 200
    alpha: float = 0.05
    z_star: float = 2.0

    knife_edge_delta: float = 0.02

    enable_tau_sweep: bool = False
    tau: float = 0.8
    consec_m: int | None = None
    tau_grid: tuple[float, ...] = (0.75, 0.8, 0.85)

    metric: str = "ari"

    report_kmax_cap_hit: bool = True
    store_ari_vec_by_k: bool = False
    store_pairwise_metric_vec_by_k: bool = False
    s_curve_kmax: int = 20
    base_seed: int = 123
    n_init: int = 10
    max_iter: int = 300
