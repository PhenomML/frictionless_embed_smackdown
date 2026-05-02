
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from .partition_metrics import get_partition_metric, partition_ari_from_labels
from .replicability_config import K_sweep_config

ArrayF = NDArray[np.float32]
ArrayI = NDArray[np.int64]

def draw_subsample_indices(
    n: int,
    fraction: float,
    rng: np.random.Generator,
    stratified: bool,
    y: Optional[NDArray],
) -> ArrayI:
    n_take = max(2, int(fraction * n))
    if n_take >= n:
        return np.arange(n, dtype=np.int64)

    if stratified and y is not None:
        y_flat = np.asarray(y).ravel()
        unique, inverse = np.unique(y_flat, return_inverse=True)
        n_classes = len(unique)
        counts = np.bincount(inverse, minlength=n_classes)
        if n_take < n_classes:

            classes_chosen = rng.choice(n_classes, size=n_take, replace=False)
            idx_list = []
            for c in classes_chosen:
                class_idx = np.where(inverse == c)[0]
                chosen = rng.integers(0, len(class_idx))
                idx_list.append(class_idx[chosen])
            idx = np.sort(np.array(idx_list, dtype=np.int64))
        else:
            prop = n_take * counts / counts.sum()
            n_per_class = np.clip(np.round(prop).astype(int), 1, counts)
            total = n_per_class.sum()
            diff = n_take - total
            if diff != 0:
                c = int(np.argmax(counts))
                n_per_class[c] = np.clip(n_per_class[c] + diff, 0, counts[c])
            idx_list = []
            for c in range(n_classes):
                class_idx = np.where(inverse == c)[0]
                n_take_c = min(int(n_per_class[c]), len(class_idx))
                if n_take_c > 0:
                    chosen = rng.choice(len(class_idx), size=n_take_c, replace=False)
                    idx_list.extend(class_idx[chosen].tolist())
            idx = np.sort(np.array(idx_list, dtype=np.int64))
            if len(idx) > n_take:
                drop = rng.choice(len(idx), size=len(idx) - n_take, replace=False)
                idx = np.sort(np.delete(idx, drop))
            elif len(idx) < n_take:
                remaining = np.setdiff1d(np.arange(n), idx, assume_unique=False)
                n_extra = n_take - len(idx)
                extra = rng.choice(len(remaining), size=min(n_extra, len(remaining)), replace=False)
                idx = np.sort(np.concatenate([idx, remaining[extra]]))
    else:
        idx = rng.choice(n, size=n_take, replace=False)
        idx = np.sort(idx)
    assert len(idx) == n_take, f"expected n_take={n_take}, got {len(idx)}"
    assert len(idx) == len(np.unique(idx)), "indices must be unique for assume_unique=True"
    return idx

def run_kmeans_on_subsample(
    z: ArrayF,
    indices: ArrayI,
    k: int,
    seed_int: int,
    n_init: int = 10,
    max_iter: int = 300,
) -> tuple[ArrayI, ArrayI]:
    z_sub = z[indices].astype(np.float64)
    km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=seed_int)
    labels_sub = km.fit_predict(z_sub)
    return labels_sub.astype(np.int64), indices

def cluster_profile(labels_sub: ArrayI) -> dict[str, float | int]:
    n_sub = len(labels_sub)
    if n_sub == 0:
        return {"min_size": 0, "median_size": 0, "max_size": 0, "n_small_2": 0, "n_small_3": 0, "k_eff": 0.0}
    unique, counts = np.unique(labels_sub, return_counts=True)
    sizes = counts.astype(np.float64)
    min_s = int(np.min(sizes))
    med_s = float(np.median(sizes))
    max_s = int(np.max(sizes))
    n_small_2 = int(np.sum(sizes <= 2))
    n_small_3 = int(np.sum(sizes <= 3))
    p = sizes / n_sub
    simpson = float(np.sum(p * p))
    k_eff = 1.0 / simpson if simpson > 0 else 0.0
    return {
        "min_size": min_s,
        "median_size": med_s,
        "max_size": max_s,
        "n_small_2": n_small_2,
        "n_small_3": n_small_3,
        "k_eff": k_eff,
    }

def build_label_map(labels_sub: ArrayI, indices: ArrayI, n: int) -> ArrayI:
    label_map = np.full(n, -1, dtype=np.int64)
    label_map[indices] = labels_sub
    return label_map

def ari_on_overlap(
    label_map_a: ArrayI,
    label_map_b: ArrayI,
    overlap_indices: ArrayI,
) -> float:
    labels_a = label_map_a[overlap_indices]
    labels_b = label_map_b[overlap_indices]
    return float(adjusted_rand_score(labels_a, labels_b))

def _score_on_overlap(
    label_map_a: ArrayI,
    label_map_b: ArrayI,
    overlap_indices: ArrayI,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    labels_a = label_map_a[overlap_indices]
    labels_b = label_map_b[overlap_indices]
    return float(score_fn(labels_a, labels_b))

def precompute_subsamples(
    n: int,
    y: Optional[NDArray],
    stratified: bool,
    b: int,
    fraction: float,
    rng: np.random.Generator,
) -> list[ArrayI]:
    return [
        draw_subsample_indices(n, fraction, rng, stratified, y)
        for _ in range(b)
    ]

def precompute_kmeans_seeds(b: int, rng: np.random.Generator) -> list[int]:
    return [int(rng.integers(0, 2**31 - 1)) for _ in range(b)]

def sample_replicate_pairs(
    b: int,
    pairs_per_k: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    total_pairs = b * (b - 1) // 2
    n_sample = min(pairs_per_k, total_pairs)

    pairs = []
    for i in range(b):
        for j in range(i + 1, b):
            pairs.append((i, j))
    pairs_arr = np.array(pairs, dtype=np.int64)
    if n_sample >= total_pairs:
        return pairs_arr
    chosen = rng.choice(total_pairs, size=n_sample, replace=False)
    return pairs_arr[chosen]

def precompute_overlaps(
    indices_list: list[ArrayI],
    pairs: NDArray[np.int64],
    min_overlap: int,
) -> tuple[list[ArrayI], NDArray[np.int64]]:
    overlaps: list[ArrayI] = []
    eligible: list[int] = []
    for pi, (i, j) in enumerate(pairs):
        ov = np.intersect1d(indices_list[i], indices_list[j], assume_unique=True)
        if len(ov) >= min_overlap:
            overlaps.append(ov)
            eligible.append(pi)
    eligible_pairs = pairs[eligible] if eligible else np.empty((0, 2), dtype=np.int64)
    return overlaps, eligible_pairs

def learn_min_overlap_from_pairs(
    indices_list: list[ArrayI],
    pairs: NDArray[np.int64],
    min_pairs_M_requested: int,
) -> tuple[int, list[ArrayI], NDArray[np.int64], dict[str, Any]]:
    overlap_sizes: list[int] = []
    overlap_arrays: list[tuple[int, ArrayI]] = []
    for pi, (i, j) in enumerate(pairs):
        ov = np.intersect1d(indices_list[i], indices_list[j], assume_unique=True)
        overlap_sizes.append(len(ov))
        overlap_arrays.append((pi, ov))

    n_pairs = len(overlap_sizes)
    min_pairs_M_used = min(min_pairs_M_requested, n_pairs)

    if n_pairs == 0:
        return 0, [], np.empty((0, 2), dtype=np.int64), {
            "overlap_median": float("nan"),
            "overlap_iqr": float("nan"),
            "overlap_min": float("nan"),
            "overlap_max": float("nan"),
            "min_pairs_M_requested": min_pairs_M_requested,
            "min_pairs_M_used": 0,
        }

    sizes_arr = np.array(overlap_sizes, dtype=np.int64)
    overlap_median = float(np.median(sizes_arr))
    q25 = float(np.quantile(sizes_arr, 0.25))
    q75 = float(np.quantile(sizes_arr, 0.75))
    overlap_iqr = q75 - q25

    sorted_desc = np.sort(sizes_arr)[::-1]
    m_min = int(sorted_desc[min_pairs_M_used - 1]) if min_pairs_M_used > 0 else 0

    overlaps: list[ArrayI] = []
    eligible: list[int] = []
    for pi, ov in overlap_arrays:
        if len(ov) >= m_min:
            overlaps.append(ov)
            eligible.append(pi)
    eligible_pairs = pairs[eligible] if eligible else np.empty((0, 2), dtype=np.int64)

    if n_pairs >= min_pairs_M_used:
        assert len(eligible_pairs) >= min_pairs_M_used, (
            f"learn_min_overlap guarantee violated: requested >= {min_pairs_M_used} eligible pairs, "
            f"got {len(eligible_pairs)}"
        )

    overlap_stats = {
        "overlap_median": overlap_median,
        "overlap_iqr": overlap_iqr,
        "overlap_min": float(np.min(sizes_arr)),
        "overlap_max": float(np.max(sizes_arr)),
        "min_pairs_M_requested": min_pairs_M_requested,
        "min_pairs_M_used": min_pairs_M_used,
    }
    return m_min, overlaps, eligible_pairs, overlap_stats

def replicability_at_k(
    labels_list: list[ArrayI],
    indices_list: list[ArrayI],
    pairs: NDArray[np.int64],
    min_overlap: int,
    n: int,
    overlaps: Optional[list[ArrayI]] = None,
    eligible_pairs: Optional[NDArray[np.int64]] = None,
    score_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> tuple[float, float, int, NDArray[np.float64]]:
    if score_fn is None:
        score_fn = partition_ari_from_labels
    if overlaps is not None and eligible_pairs is not None:
        use_pairs = eligible_pairs
        use_overlaps = overlaps
    else:
        use_overlaps_list: list[ArrayI] = []
        use_eligible: list[int] = []
        for pi, (i, j) in enumerate(pairs):
            ov = np.intersect1d(indices_list[i], indices_list[j], assume_unique=True)
            if len(ov) >= min_overlap:
                use_overlaps_list.append(ov)
                use_eligible.append(pi)
        use_overlaps = use_overlaps_list
        use_pairs = pairs[use_eligible] if use_eligible else np.empty((0, 2), dtype=np.int64)

    label_maps = [
        build_label_map(labels_list[i], indices_list[i], n) for i in range(len(labels_list))
    ]
    scores = np.empty(len(use_pairs), dtype=np.float64)
    for idx, ((i, j), ov) in enumerate(zip(use_pairs, use_overlaps)):
        scores[idx] = _score_on_overlap(label_maps[i], label_maps[j], ov, score_fn)
    if len(scores) == 0:
        return float("nan"), float("nan"), 0, np.array([], dtype=np.float64)
    rep_mean = float(np.mean(scores))
    rep_std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    return rep_mean, rep_std, len(scores), scores

def precompute_labels_by_k(
    z: ArrayF,
    indices_list_base: list[ArrayI],
    k_range: list[int],
    kmeans_seed_list: list[int],
    n_init: int = 10,
    max_iter: int = 300,
) -> dict[int, list[ArrayI]]:
    labels_by_k: dict[int, list[ArrayI]] = {}
    for k in k_range:
        labels_list: list[ArrayI] = []
        for bi in range(len(indices_list_base)):
            lab, _ = run_kmeans_on_subsample(
                z, indices_list_base[bi], k, kmeans_seed_list[bi], n_init, max_iter
            )
            labels_list.append(lab)
        labels_by_k[k] = labels_list
    return labels_by_k

def compute_rep_curve_for_pairs(
    indices_list_base: list[ArrayI],
    k_range: list[int],
    pairs_subset: NDArray[np.int64],
    m_min_fixed: int,
    n: int,
    labels_by_k: dict[int, list[ArrayI]],
    score_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    if score_fn is None:
        score_fn = partition_ari_from_labels
    overlaps, eligible_pairs = precompute_overlaps(
        indices_list_base, pairs_subset, m_min_fixed
    )
    eligible_count = len(eligible_pairs)
    nk = len(k_range)
    rep_mean_arr = np.full(nk, np.nan, dtype=np.float64)
    rep_std_arr = np.full(nk, np.nan, dtype=np.float64)

    for ki, k in enumerate(k_range):
        labels_list = labels_by_k[k]
        rep_m, rep_s, _, _ = replicability_at_k(
            labels_list, indices_list_base, pairs_subset, m_min_fixed, n,
            overlaps=overlaps, eligible_pairs=eligible_pairs,
            score_fn=score_fn,
        )
        rep_mean_arr[ki] = rep_m
        rep_std_arr[ki] = rep_s

    return rep_mean_arr, rep_std_arr, eligible_count

def extract_overlap_labels(
    labels_list: list[ArrayI],
    indices_list: list[ArrayI],
    eligible_pairs: NDArray[np.int64],
    overlaps: list[ArrayI],
    n: int,
) -> tuple[list[ArrayI], list[ArrayI]]:
    label_maps = [
        build_label_map(labels_list[bi], indices_list[bi], n)
        for bi in range(len(labels_list))
    ]
    U_list: list[ArrayI] = []
    V_list: list[ArrayI] = []
    for (i, j), ov in zip(eligible_pairs, overlaps):
        U_list.append(label_maps[i][ov].copy())
        V_list.append(label_maps[j][ov].copy())
    return U_list, V_list

def simulate_null_rep_at_k(
    U_list: list[ArrayI],
    V_list: list[ArrayI],
    n_null: int,
    rng: np.random.Generator,
    score_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> NDArray[np.float64]:
    if score_fn is None:
        score_fn = partition_ari_from_labels
    n_pairs = len(U_list)
    if n_pairs == 0:
        return np.full(n_null, np.nan, dtype=np.float64)

    null_reps = np.empty(n_null, dtype=np.float64)
    for t in range(n_null):
        score_sum = 0.0
        for u, v in zip(U_list, V_list):
            v_perm = v.copy()
            rng.shuffle(v_perm)
            score_sum += score_fn(u, v_perm)
        null_reps[t] = score_sum / n_pairs
    return null_reps

def compute_null_statistics(
    rep_obs: float,
    null_reps: NDArray[np.float64],
) -> dict[str, float]:
    mu0 = float(np.mean(null_reps))
    n_null = len(null_reps)
    sigma0 = float(np.std(null_reps, ddof=1)) if n_null > 1 else 0.0
    z_score = (rep_obs - mu0) / sigma0 if sigma0 > 1e-12 else (float("inf") if rep_obs > mu0 else (float("-inf") if rep_obs < mu0 else 0.0))
    p_value = (1 + np.sum(null_reps >= rep_obs)) / (len(null_reps) + 1)
    return {
        "null_mu0": mu0,
        "null_sigma0": sigma0,
        "z_score": z_score,
        "p_value": float(p_value),
        "null_q025": float(np.quantile(null_reps, 0.025)),
        "null_q975": float(np.quantile(null_reps, 0.975)),
        "null_q95": float(np.quantile(null_reps, 0.95)),
    }

def select_hat_k_bh(
    p_values: dict[int, float],
    alpha: float = 0.05,
) -> Optional[int]:
    if not p_values:
        return None
    m = len(p_values)
    sorted_items = sorted(p_values.items(), key=lambda x: x[1])
    i_star = -1
    for rank_0, (k, p) in enumerate(sorted_items):
        rank_1 = rank_0 + 1
        if p <= alpha * rank_1 / m:
            i_star = rank_0
    if i_star < 0:
        return None
    rejected = {k for k, _ in sorted_items[:i_star + 1]}
    return max(rejected)

def select_hat_k_zscore(
    z_scores: dict[int, float],
    z_star: float = 2.0,
) -> Optional[int]:
    candidates = [k for k, z in z_scores.items() if z >= z_star and not np.isnan(z)]
    return max(candidates) if candidates else None

def select_hat_k_quant(
    rep_mean: list[float],
    k_range: list[int],
    null_q: list[float],
) -> Optional[int]:
    candidates = []
    for ki, k in enumerate(k_range):
        rep = rep_mean[ki] if ki < len(rep_mean) else None
        q = null_q[ki] if ki < len(null_q) else None
        if (
            rep is not None
            and q is not None
            and not np.isnan(rep)
            and not np.isnan(q)
            and rep >= q
        ):
            candidates.append(k)
    return max(candidates) if candidates else None

def metric_vs_ground_truth_at_k(
    labels_list: list[ArrayI],
    indices_list: list[ArrayI],
    y: ArrayI,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[float, float]:
    values = []
    for labels_b, indices_b in zip(labels_list, indices_list):
        y_b = y[indices_b]
        values.append(float(score_fn(labels_b, y_b)))
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

def label_agreement_at_k(
    labels_list: list[ArrayI],
    indices_list: list[ArrayI],
    y: ArrayI,
) -> tuple[float, float, float, float]:
    aris = []
    amis = []
    for labels_b, indices_b in zip(labels_list, indices_list):
        y_b = y[indices_b]
        aris.append(adjusted_rand_score(labels_b, y_b))
        amis.append(
            adjusted_mutual_info_score(
                labels_b,
                y_b,
                average_method="arithmetic",
            )
        )
    if not aris:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(aris)),
        float(np.std(aris, ddof=1)) if len(aris) > 1 else 0.0,
        float(np.mean(amis)),
        float(np.std(amis, ddof=1)) if len(amis) > 1 else 0.0,
    )

def k_sweep_replicability(
    z: ArrayF,
    y: Optional[NDArray],
    meta: dict[str, Any],
    k_range: list[int],
    b: int,
    fraction: float,
    base_seed: int,
    pairs_per_k: int = 2000,
    min_overlap: Optional[int] = None,
    min_overlap_default: int = 10,
    learn_min_overlap: bool = True,
    min_pairs_M: int = 1000,
    n_init: int = 10,
    max_iter: int = 300,
    n_null: int = 200,
    alpha: float = 0.05,
    z_star: float = 2.0,
    store_ari_vec_by_k: bool = False,
    store_pairwise_metric_vec_by_k: Optional[bool] = None,
    metric: str = "ari",
) -> dict[str, Any]:
    metric_spec = get_partition_metric(metric)
    score_fn = metric_spec.score_fn
    store_vec = store_pairwise_metric_vec_by_k if store_pairwise_metric_vec_by_k is not None else store_ari_vec_by_k

    n = z.shape[0]
    stratified = (
        meta.get("label_type") in {"ground_truth", "derived"}
        and y is not None
    )
    compute_metric_vs_gt = y is not None
    compute_label_agreement = (
        meta.get("label_type") == "ground_truth" and y is not None
    )
    if y is not None:
        y_gt = np.asarray(y, dtype=np.int64)

    rng_master = np.random.default_rng(base_seed)
    indices_list_base = precompute_subsamples(n, y, stratified, b, fraction, rng_master)
    kmeans_seed_list = precompute_kmeans_seeds(b, rng_master)
    pairs = sample_replicate_pairs(b, pairs_per_k, rng_master)

    pairs_total_possible = b * (b - 1) // 2
    pairs_sampled = len(pairs)
    eligible_pairs_count: int = 0
    min_pairs_M_used: Optional[int] = None

    overlap_stats: dict[str, Any] = {}
    if learn_min_overlap:
        min_ov_global, overlaps, eligible_pairs, overlap_stats = learn_min_overlap_from_pairs(
            indices_list_base, pairs, min_pairs_M
        )
        min_pairs_M_used = overlap_stats.get("min_pairs_M_used")
        eligible_pairs_count = len(eligible_pairs)
    else:
        min_ov_global = min_overlap if min_overlap is not None else min_overlap_default
        overlaps, eligible_pairs = precompute_overlaps(indices_list_base, pairs, min_ov_global)
        overlap_stats = {}
        eligible_pairs_count = len(eligible_pairs)

    gamma = fraction
    gamma2_n = (gamma ** 2) * n
    c_implied = min_ov_global / gamma2_n if gamma2_n > 0 else float("nan")

    nk = len(k_range)
    rep_mean_arr = np.full(nk, np.nan, dtype=np.float64)
    rep_std_arr = np.full(nk, np.nan, dtype=np.float64)
    pairs_used_arr = np.zeros(nk, dtype=np.int64)
    q10_arr = np.full(nk, np.nan, dtype=np.float64)
    metric_vs_gt_mean_arr: Optional[NDArray[np.float64]] = (
        np.full(nk, np.nan, dtype=np.float64) if compute_metric_vs_gt else None
    )
    metric_vs_gt_std_arr: Optional[NDArray[np.float64]] = (
        np.full(nk, np.nan, dtype=np.float64) if compute_metric_vs_gt else None
    )
    ari_mean_arr: Optional[NDArray[np.float64]] = (
        np.full(nk, np.nan, dtype=np.float64) if compute_label_agreement and metric == "ari" else None
    )
    ari_std_arr: Optional[NDArray[np.float64]] = (
        np.full(nk, np.nan, dtype=np.float64) if compute_label_agreement and metric == "ari" else None
    )
    ami_mean_arr: Optional[NDArray[np.float64]] = (
        np.full(nk, np.nan, dtype=np.float64) if compute_label_agreement and metric == "ari" else None
    )
    ami_std_arr: Optional[NDArray[np.float64]] = (
        np.full(nk, np.nan, dtype=np.float64) if compute_label_agreement and metric == "ari" else None
    )
    min_size_mean_arr = np.full(nk, np.nan, dtype=np.float64)
    min_size_std_arr = np.full(nk, np.nan, dtype=np.float64)
    median_size_mean_arr = np.full(nk, np.nan, dtype=np.float64)
    median_size_std_arr = np.full(nk, np.nan, dtype=np.float64)
    max_size_mean_arr = np.full(nk, np.nan, dtype=np.float64)
    max_size_std_arr = np.full(nk, np.nan, dtype=np.float64)
    n_small_2_mean_arr = np.full(nk, np.nan, dtype=np.float64)
    n_small_2_std_arr = np.full(nk, np.nan, dtype=np.float64)
    n_small_3_mean_arr = np.full(nk, np.nan, dtype=np.float64)
    n_small_3_std_arr = np.full(nk, np.nan, dtype=np.float64)
    k_eff_mean_arr = np.full(nk, np.nan, dtype=np.float64)
    k_eff_std_arr = np.full(nk, np.nan, dtype=np.float64)

    null_mu0_arr = np.full(nk, np.nan, dtype=np.float64)
    null_sigma0_arr = np.full(nk, np.nan, dtype=np.float64)
    z_score_arr = np.full(nk, np.nan, dtype=np.float64)
    p_value_arr = np.full(nk, np.nan, dtype=np.float64)
    null_q025_arr = np.full(nk, np.nan, dtype=np.float64)
    null_q975_arr = np.full(nk, np.nan, dtype=np.float64)
    null_q95_arr = np.full(nk, np.nan, dtype=np.float64)
    pairwise_metric_vec_by_k: dict[str, list[float]] = {}

    for ki, k in enumerate(k_range):
        labels_list: list[ArrayI] = []
        for bi in range(b):
            lab, idx_out = run_kmeans_on_subsample(
                z, indices_list_base[bi], k, kmeans_seed_list[bi], n_init, max_iter
            )
            labels_list.append(lab)

        rep_m, rep_s, n_used, score_vec = replicability_at_k(
            labels_list, indices_list_base, pairs, min_ov_global, n,
            overlaps=overlaps, eligible_pairs=eligible_pairs,
            score_fn=score_fn,
        )
        rep_mean_arr[ki] = rep_m
        rep_std_arr[ki] = rep_s
        pairs_used_arr[ki] = n_used
        q10_arr[ki] = float(np.quantile(score_vec, 0.10)) if len(score_vec) > 0 else float("nan")
        if store_vec and len(score_vec) > 0:
            pairwise_metric_vec_by_k[str(k)] = [float(x) for x in score_vec]

        if n_null > 0 and n_used > 0:

            U_list, V_list = extract_overlap_labels(
                labels_list, indices_list_base, eligible_pairs, overlaps, n,
            )

            rng_null_k = np.random.default_rng(base_seed + 7 + 104729 * k)
            null_reps = simulate_null_rep_at_k(U_list, V_list, n_null, rng_null_k, score_fn=score_fn)
            ns = compute_null_statistics(rep_m, null_reps)
            null_mu0_arr[ki] = ns["null_mu0"]
            null_sigma0_arr[ki] = ns["null_sigma0"]
            z_score_arr[ki] = ns["z_score"]
            p_value_arr[ki] = ns["p_value"]
            null_q025_arr[ki] = ns["null_q025"]
            null_q975_arr[ki] = ns["null_q975"]
            null_q95_arr[ki] = ns["null_q95"]

        if compute_metric_vs_gt:
            m_gt_mean, m_gt_std = metric_vs_ground_truth_at_k(
                labels_list, indices_list_base, y_gt, score_fn
            )
            if metric_vs_gt_mean_arr is not None:
                metric_vs_gt_mean_arr[ki] = m_gt_mean
            if metric_vs_gt_std_arr is not None:
                metric_vs_gt_std_arr[ki] = m_gt_std
        if compute_label_agreement and metric == "ari":
            ari_m, ari_s, ami_m, ami_s = label_agreement_at_k(
                labels_list, indices_list_base, y_gt
            )
            if ari_mean_arr is not None:
                ari_mean_arr[ki] = ari_m
            if ari_std_arr is not None:
                ari_std_arr[ki] = ari_s
            if ami_mean_arr is not None:
                ami_mean_arr[ki] = ami_m
            if ami_std_arr is not None:
                ami_std_arr[ki] = ami_s

        profiles = [cluster_profile(lab) for lab in labels_list]
        min_sizes = [p["min_size"] for p in profiles]
        med_sizes = [p["median_size"] for p in profiles]
        max_sizes = [p["max_size"] for p in profiles]
        n2 = [p["n_small_2"] for p in profiles]
        n3 = [p["n_small_3"] for p in profiles]
        k_effs = [p["k_eff"] for p in profiles]
        min_size_mean_arr[ki] = float(np.mean(min_sizes))
        min_size_std_arr[ki] = float(np.std(min_sizes))
        median_size_mean_arr[ki] = float(np.mean(med_sizes))
        median_size_std_arr[ki] = float(np.std(med_sizes))
        max_size_mean_arr[ki] = float(np.mean(max_sizes))
        max_size_std_arr[ki] = float(np.std(max_sizes))
        n_small_2_mean_arr[ki] = float(np.mean(n2))
        n_small_2_std_arr[ki] = float(np.std(n2))
        n_small_3_mean_arr[ki] = float(np.mean(n3))
        n_small_3_std_arr[ki] = float(np.std(n3))
        k_eff_mean_arr[ki] = float(np.mean(k_effs))
        k_eff_std_arr[ki] = float(np.std(k_effs))

    p_values_by_k = {k: p_value_arr[ki] for ki, k in enumerate(k_range) if not np.isnan(p_value_arr[ki])}
    z_scores_by_k = {k: z_score_arr[ki] for ki, k in enumerate(k_range) if not np.isnan(z_score_arr[ki])}
    hat_k_bh = select_hat_k_bh(p_values_by_k, alpha=alpha)
    hat_k_z = select_hat_k_zscore(z_scores_by_k, z_star=z_star)
    if n_null > 0 and np.any(~np.isnan(null_q95_arr)):
        hat_k_q95 = select_hat_k_quant(
            rep_mean_arr.tolist(), k_range, null_q95_arr.tolist()
        )
        hat_k_q975 = select_hat_k_quant(
            rep_mean_arr.tolist(), k_range, null_q975_arr.tolist()
        )
    else:
        hat_k_q95 = None
        hat_k_q975 = None

    bh_found = hat_k_bh is not None
    no_significant_k = hat_k_bh is None

    valid_p = [(k_range[ki], p_value_arr[ki]) for ki in range(nk)
               if not np.isnan(p_value_arr[ki]) and p_value_arr[ki] is not None]
    k_min_p, p_min_p = (min(valid_p, key=lambda x: x[1]) if valid_p else (None, None))

    valid_z = [(k_range[ki], z_score_arr[ki]) for ki in range(nk)
               if not np.isnan(z_score_arr[ki]) and z_score_arr[ki] is not None]
    k_max_z, z_max_z = (max(valid_z, key=lambda x: x[1]) if valid_z else (None, None))

    rep_at_k_min_p = None
    if k_min_p is not None and k_min_p in k_range:
        ki_mp = k_range.index(k_min_p)
        rep_at_k_min_p = rep_mean_arr[ki_mp]

    out: dict[str, Any] = {
        "k_range": k_range,
        "metric": metric_spec.name,
        "metric_display_name": metric_spec.display_name,
        "perturbations_frozen_across_k": True,
        "m_min": int(min_ov_global),
        "min_overlap_global": int(min_ov_global),
        "c_implied": float(c_implied),
        "pairs_total_possible": int(pairs_total_possible),
        "pairs_sampled": int(pairs_sampled),
        "eligible_pairs_count": eligible_pairs_count,
        "min_pairs_M_requested": int(min_pairs_M) if learn_min_overlap else None,
        "min_pairs_M_used": int(min_pairs_M_used) if min_pairs_M_used is not None else None,
        "overlap_median": overlap_stats.get("overlap_median", float("nan")),
        "overlap_iqr": overlap_stats.get("overlap_iqr", float("nan")),
        "rep_mean": rep_mean_arr.tolist(),
        "rep_std": rep_std_arr.tolist(),
        "pairs_used": pairs_used_arr.tolist(),
        "q10": q10_arr.tolist(),
        "metric_vs_gt_mean": metric_vs_gt_mean_arr.tolist() if metric_vs_gt_mean_arr is not None else None,
        "metric_vs_gt_std": metric_vs_gt_std_arr.tolist() if metric_vs_gt_std_arr is not None else None,
        "min_size_mean": min_size_mean_arr.tolist(),
        "min_size_std": min_size_std_arr.tolist(),
        "median_size_mean": median_size_mean_arr.tolist(),
        "median_size_std": median_size_std_arr.tolist(),
        "max_size_mean": max_size_mean_arr.tolist(),
        "max_size_std": max_size_std_arr.tolist(),
        "n_small_2_mean": n_small_2_mean_arr.tolist(),
        "n_small_2_std": n_small_2_std_arr.tolist(),
        "n_small_3_mean": n_small_3_mean_arr.tolist(),
        "n_small_3_std": n_small_3_std_arr.tolist(),
        "k_eff_mean": k_eff_mean_arr.tolist(),
        "k_eff_std": k_eff_std_arr.tolist(),
        "null_mu0": null_mu0_arr.tolist(),
        "null_sigma0": null_sigma0_arr.tolist(),
        "z_score": z_score_arr.tolist(),
        "p_value": p_value_arr.tolist(),
        "null_q025": null_q025_arr.tolist(),
        "null_q975": null_q975_arr.tolist(),
        "null_q95": null_q95_arr.tolist(),
        "hat_k_bh": hat_k_bh,
        "hat_k_z": hat_k_z,
        "hat_k_q95": hat_k_q95,
        "hat_k_q975": hat_k_q975,
        "bh_found": bh_found,
        "no_significant_k": no_significant_k,
        "k_min_p": k_min_p,
        "p_min_p": float(p_min_p) if p_min_p is not None else None,
        "rep_at_k_min_p": float(rep_at_k_min_p) if rep_at_k_min_p is not None else None,
        "k_max_z": k_max_z,
        "z_max_z": float(z_max_z) if z_max_z is not None else None,
        "null_params": {"n_null": n_null, "alpha": alpha, "z_star": z_star},
        "pairwise_metric_vec_by_k": pairwise_metric_vec_by_k if store_vec else None,
    }

    if metric == "ari":
        out["ari_vs_gt_mean"] = out["metric_vs_gt_mean"]
        out["ari_vs_gt_std"] = out["metric_vs_gt_std"]
        out["ami_vs_gt_mean"] = ami_mean_arr.tolist() if ami_mean_arr is not None else None
        out["ami_vs_gt_std"] = ami_std_arr.tolist() if ami_std_arr is not None else None
        out["ari_vec_by_k"] = out.get("pairwise_metric_vec_by_k")
    return out

def _is_dense_grid(k_range: list[int]) -> bool:
    if not k_range:
        return False
    return k_range == list(range(min(k_range), max(k_range) + 1))

def select_hat_k(rep_by_k: dict[int, float], tau: float) -> Optional[int]:
    candidates = [k for k, rep in rep_by_k.items() if rep >= tau and not np.isnan(rep)]
    return max(candidates) if candidates else None

def select_hat_k_consec(
    rep_by_k: dict[int, float],
    tau: float,
    m: int,
) -> Optional[int]:
    k_range = sorted(rep_by_k.keys())
    rep_mean = [rep_by_k[k] for k in k_range]
    return compute_hat_k_consec_for_tau(rep_mean, k_range, tau, m)

def compute_hat_k_consec_for_tau(
    rep_mean: list[float],
    k_range: list[int],
    tau: float,
    m: int,
) -> Optional[int]:
    rep_by_k = dict(zip(k_range, rep_mean))
    k_sorted = sorted(rep_by_k.keys(), reverse=True)
    for k in k_sorted:
        if k < m:
            continue
        run = [rep_by_k.get(k - j) for j in range(m)]
        if all(
            r is not None and not np.isnan(r) and r >= tau
            for r in run
        ):
            return k
    return None

def run_single_config(
    dataset_id: str,
    method: str,
    paths: dict[str, Any],
    embed_cfg: Any,
    ksweep_cfg: K_sweep_config,
) -> dict[str, Any]:
    from pathlib import Path

    from .config import EmbedConfig, get_default_paths
    from .loaders import load_dataset
    from .utils import set_seeds, subsample_rows, to_float32

    if isinstance(paths, dict) and "data_root" in paths:
        data_root = paths["data_root"]
        corpus_dir = paths["corpus_dir"]
    else:
        p = get_default_paths()
        data_root = p["data_root"]
        corpus_dir = p["corpus_dir"]

    cfg = embed_cfg if isinstance(embed_cfg, EmbedConfig) else EmbedConfig()
    max_points = cfg.max_points
    set_seeds(cfg.random_state)

    x, y, meta = load_dataset(
        dataset_id,
        data_root,
        corpus_dir,
        pca_dim=cfg.pca_dim,
        subsample=None,
        use_corpus_when_available=True,
        seed=cfg.random_state,
    )
    if max_points is not None and x.shape[0] > max_points:
        strat = (
            meta.get("label_type") in {"ground_truth", "derived"}
            and y is not None
        )
        x, y, _ = subsample_rows(x, y, max_points, cfg.random_state, stratified=strat)
    x = to_float32(x)

    if method == "tsne":
        from .embeddings import embed_tsne
        z = embed_tsne(x, cfg)
    elif method == "umap":
        from .embeddings import embed_umap
        z = embed_umap(x, cfg)
    elif method == "rand2d":
        rng = np.random.default_rng(cfg.random_state)
        n = x.shape[0]
        z = rng.standard_normal((n, 2)).astype(np.float32)
    else:
        raise ValueError(f"Unknown method: {method}")

    result = k_sweep_replicability(
        z,
        y,
        meta,
        ksweep_cfg.k_range,
        ksweep_cfg.b,
        ksweep_cfg.fraction,
        ksweep_cfg.base_seed,
        pairs_per_k=ksweep_cfg.pairs_per_k,
        min_overlap=ksweep_cfg.min_overlap,
        min_overlap_default=getattr(ksweep_cfg, "min_overlap_default", 10),
        learn_min_overlap=getattr(ksweep_cfg, "learn_min_overlap", True),
        min_pairs_M=getattr(ksweep_cfg, "min_pairs_M", 1000),
        n_init=ksweep_cfg.n_init,
        max_iter=ksweep_cfg.max_iter,
        n_null=ksweep_cfg.n_null,
        alpha=ksweep_cfg.alpha,
        z_star=ksweep_cfg.z_star,
        store_ari_vec_by_k=getattr(ksweep_cfg, "store_ari_vec_by_k", False),
        store_pairwise_metric_vec_by_k=getattr(ksweep_cfg, "store_pairwise_metric_vec_by_k", None),
        metric=getattr(ksweep_cfg, "metric", "ari"),
    )

    rep_by_k = dict(zip(result["k_range"], result["rep_mean"]))
    k_range = result["k_range"]

    hat_k_bh = result["hat_k_bh"]
    hat_k_z = result["hat_k_z"]
    bh_found = result.get("bh_found", hat_k_bh is not None)
    no_significant_k = result.get("no_significant_k", hat_k_bh is None)

    hat_k = hat_k_bh

    enable_tau_sweep = getattr(ksweep_cfg, "enable_tau_sweep", False)
    if enable_tau_sweep:
        hat_k_tau = select_hat_k(rep_by_k, ksweep_cfg.tau)
        hat_k_consec = None
        if ksweep_cfg.consec_m is not None and _is_dense_grid(k_range):
            hat_k_consec = select_hat_k_consec(
                rep_by_k, ksweep_cfg.tau, ksweep_cfg.consec_m
            )
        hat_k_tau075 = select_hat_k(rep_by_k, 0.75)
        hat_k_tau080 = select_hat_k(rep_by_k, 0.80)
        hat_k_tau085 = select_hat_k(rep_by_k, 0.85)
    else:
        hat_k_tau = hat_k_consec = hat_k_tau075 = hat_k_tau080 = hat_k_tau085 = None

    k_max = max(k_range) if k_range else 0
    cap_hit = hat_k == k_max if hat_k is not None else False

    knife_edge_delta = getattr(ksweep_cfg, "knife_edge_delta", 0.02)
    knife_edge = False
    if hat_k_bh is not None and hat_k_bh in k_range:
        ki_bh = k_range.index(hat_k_bh)
        rep_at_bh = result["rep_mean"][ki_bh]
        q95_at_bh = result.get("null_q95")
        q95_val = q95_at_bh[ki_bh] if q95_at_bh is not None else float("nan")
        if not (np.isnan(rep_at_bh) or np.isnan(q95_val)):
            knife_edge = (rep_at_bh - q95_val) < knife_edge_delta

    p_at_hat_bh = None
    z_at_hat_bh = None
    rep_at_hat_bh = None
    if hat_k_bh is not None and hat_k_bh in k_range:
        ki_bh = k_range.index(hat_k_bh)
        p_at_hat_bh = result["p_value"][ki_bh]
        z_at_hat_bh = result["z_score"][ki_bh]
        rep_at_hat_bh = result["rep_mean"][ki_bh]

    k_eff_at_hat = None
    min_size_at_hat = None
    n_small_2_at_hat = None
    k_hat_for_diag = hat_k
    if k_hat_for_diag is not None and k_hat_for_diag in k_range:
        ki = k_range.index(k_hat_for_diag)
        k_eff_at_hat = result.get("k_eff_mean")
        min_size_at_hat = result.get("min_size_mean")
        n_small_2_at_hat = result.get("n_small_2_mean")
        if k_eff_at_hat is not None:
            k_eff_at_hat = k_eff_at_hat[ki]
        if min_size_at_hat is not None:
            min_size_at_hat = min_size_at_hat[ki]
        if n_small_2_at_hat is not None:
            n_small_2_at_hat = n_small_2_at_hat[ki]

    degenerate_at_hat = False
    if min_size_at_hat is not None and min_size_at_hat < 5:
        degenerate_at_hat = True
    if n_small_2_at_hat is not None and n_small_2_at_hat > 0:
        degenerate_at_hat = True
    if k_eff_at_hat is not None and k_hat_for_diag is not None and k_eff_at_hat < 0.5 * k_hat_for_diag:
        degenerate_at_hat = True

    run_meta = {
        "metric": getattr(ksweep_cfg, "metric", "ari"),
        "fraction": ksweep_cfg.fraction,
        "b": ksweep_cfg.b,
        "tau": ksweep_cfg.tau,
        "pairs_per_k": ksweep_cfg.pairs_per_k,
        "min_overlap": ksweep_cfg.min_overlap,
        "learn_min_overlap": getattr(ksweep_cfg, "learn_min_overlap", True),
        "min_pairs_M_requested": result.get("min_pairs_M_requested"),
        "min_pairs_M_used": result.get("min_pairs_M_used"),
        "pairs_total_possible": result.get("pairs_total_possible"),
        "pairs_sampled": result.get("pairs_sampled"),
        "m_min": int(result.get("m_min", result.get("min_overlap_global", 10))),
        "c_implied": result.get("c_implied", float("nan")),
        "eligible_pairs_count": result.get("eligible_pairs_count", 0),
        "overlap_median": result.get("overlap_median", float("nan")),
        "overlap_iqr": result.get("overlap_iqr", float("nan")),
        "max_points": cfg.max_points,
        "perturbations_frozen_across_k": True,
        "n_null": ksweep_cfg.n_null,
        "alpha": ksweep_cfg.alpha,
        "z_star": ksweep_cfg.z_star,
        "knife_edge_delta": getattr(ksweep_cfg, "knife_edge_delta", 0.02),
        "enable_tau_sweep": getattr(ksweep_cfg, "enable_tau_sweep", False),
    }
    return {
        "dataset_id": dataset_id,
        "method": method,
        "n_points": int(z.shape[0]),
        "label_type": meta.get("label_type", "unknown"),
        "meta": run_meta,
        **result,
        "hat_k": hat_k,
        "bh_found": bh_found,
        "no_significant_k": no_significant_k,
        "hat_k_tau": hat_k_tau,
        "hat_k_consec": hat_k_consec,
        "cap_hit": cap_hit,
        "knife_edge": knife_edge,
        "hat_k_tau075": hat_k_tau075,
        "hat_k_tau080": hat_k_tau080,
        "hat_k_tau085": hat_k_tau085,
        "p_at_hat_bh": p_at_hat_bh,
        "z_at_hat_bh": z_at_hat_bh,
        "rep_at_hat_bh": rep_at_hat_bh,
        "k_min_p": result.get("k_min_p"),
        "p_min_p": result.get("p_min_p"),
        "rep_at_k_min_p": result.get("rep_at_k_min_p"),
        "k_max_z": result.get("k_max_z"),
        "z_max_z": result.get("z_max_z"),
        "k_eff_at_hat": k_eff_at_hat,
        "min_size_at_hat": min_size_at_hat,
        "n_small_2_at_hat": n_small_2_at_hat,
        "degenerate_at_hat": degenerate_at_hat,
    }

def pair_subsampling_sensitivity(
    z: ArrayF,
    indices_list_base: list[ArrayI],
    kmeans_seed_list: list[int],
    k_range: list[int],
    pairs_pool_size: int = 5000,
    M_grid: Optional[list[int]] = None,
    R: int = 5,
    base_seed: int = 123,
    n_init: int = 10,
    max_iter: int = 300,
    delta_threshold: float = 0.01,
    metric: str = "ari",
) -> dict[str, Any]:
    score_fn = get_partition_metric(metric).score_fn
    if M_grid is None:
        M_grid = [500, 1000, 2000]
    b = len(indices_list_base)
    n = z.shape[0]
    rng = np.random.default_rng(base_seed)

    pairs_pool = sample_replicate_pairs(b, pairs_pool_size, rng)
    M_for_learn = min(max(M_grid), len(pairs_pool))
    m_min_fixed, _, _, _ = learn_min_overlap_from_pairs(
        indices_list_base, pairs_pool, M_for_learn
    )

    labels_by_k = precompute_labels_by_k(
        z, indices_list_base, k_range, kmeans_seed_list, n_init, max_iter
    )

    rep_curves_by_M: dict[int, list[list[float]]] = {M: [] for M in M_grid}
    eligible_pairs_count_by_M: dict[int, list[int]] = {M: [] for M in M_grid}
    pool_size = len(pairs_pool)

    for M in M_grid:
        n_sample = min(M, pool_size)
        for r in range(R):
            rng_r = np.random.default_rng(base_seed + 1000 * M + r)
            idx = rng_r.choice(pool_size, size=n_sample, replace=False)
            pairs_subset = pairs_pool[idx]
            rep_mean, _, eligible_count = compute_rep_curve_for_pairs(
                indices_list_base, k_range, pairs_subset, m_min_fixed, n,
                labels_by_k=labels_by_k,
                score_fn=score_fn,
            )
            rep_curves_by_M[M].append(rep_mean.tolist())
            eligible_pairs_count_by_M[M].append(eligible_count)

    delta_by_M: dict[int, float] = {}
    for M in M_grid:
        curves = rep_curves_by_M[M]
        diffs = []
        for r in range(len(curves)):
            for s in range(r + 1, len(curves)):
                arr_r = np.array(curves[r])
                arr_s = np.array(curves[s])
                valid = ~(np.isnan(arr_r) | np.isnan(arr_s))
                if np.any(valid):
                    sup_norm = float(np.max(np.abs(arr_r[valid] - arr_s[valid])))
                else:
                    sup_norm = float("nan")
                diffs.append(sup_norm)
        delta_by_M[M] = float(np.nanmedian(diffs)) if diffs else float("nan")

    delta_pass = {str(M): delta_by_M[M] <= delta_threshold for M in M_grid}
    mean_eligible_count_by_M = {
        str(M): float(np.mean(eligible_pairs_count_by_M[M]))
        for M in M_grid
    }

    return {
        "k_range": k_range,
        "M_grid": M_grid,
        "R": R,
        "pairs_pool_size": pool_size,
        "m_min_fixed": int(m_min_fixed),
        "delta_threshold": delta_threshold,
        "delta_by_M": {str(M): delta_by_M[M] for M in M_grid},
        "delta_pass": delta_pass,
        "rep_curves_by_M": {
            str(M): rep_curves_by_M[M] for M in M_grid
        },
        "eligible_pairs_count_by_M": {
            str(M): eligible_pairs_count_by_M[M] for M in M_grid
        },
        "mean_eligible_count_by_M": mean_eligible_count_by_M,
    }

def run_pair_subsampling_sensitivity(
    dataset_id: str,
    method: str,
    paths: Optional[dict[str, Any]] = None,
    k_range: Optional[list[int]] = None,
    b: int = 200,
    pairs_pool_size: int = 5000,
    M_grid: Optional[list[int]] = None,
    R: int = 5,
    base_seed: int = 123,
    max_points: Optional[int] = 3000,
    delta_threshold: float = 0.01,
    metric: str = "ari",
) -> dict[str, Any]:
    from pathlib import Path

    from .config import EmbedConfig, get_default_paths
    from .loaders import load_dataset
    from .utils import set_seeds, subsample_rows, to_float32

    p = paths or get_default_paths()
    data_root = Path(p["data_root"]) if not isinstance(p["data_root"], Path) else p["data_root"]
    corpus_dir = p.get("corpus_dir") or (data_root.parent / "corpus")
    corpus_dir = Path(corpus_dir) if not isinstance(corpus_dir, Path) else corpus_dir
    set_seeds(base_seed)

    x, y, meta = load_dataset(
        dataset_id, data_root, corpus_dir,
        subsample=None, use_corpus_when_available=True, seed=base_seed,
    )
    if max_points is not None and x.shape[0] > max_points:
        strat = meta.get("label_type") in {"ground_truth", "derived"} and y is not None
        x, y, _ = subsample_rows(x, y, max_points, base_seed, stratified=strat)
    x = to_float32(x)

    cfg = EmbedConfig(max_points=max_points, random_state=base_seed)
    if method == "tsne":
        from .embeddings import embed_tsne
        z = embed_tsne(x, cfg)
    elif method == "umap":
        from .embeddings import embed_umap
        z = embed_umap(x, cfg)
    else:
        raise ValueError(f"Unknown method: {method}")

    n = z.shape[0]
    stratified = meta.get("label_type") in {"ground_truth", "derived"} and y is not None
    rng = np.random.default_rng(base_seed)
    indices_list_base = precompute_subsamples(n, y, stratified, b, 0.8, rng)
    kmeans_seed_list = precompute_kmeans_seeds(b, rng)

    if k_range is None:
        k_range = [2, 3, 5, 8, 10, 15, 20]

    out = pair_subsampling_sensitivity(
        z, indices_list_base, kmeans_seed_list, k_range,
        pairs_pool_size=pairs_pool_size, M_grid=M_grid, R=R, base_seed=base_seed,
        delta_threshold=delta_threshold,
        metric=metric,
    )
    out["dataset_id"] = dataset_id
    out["method"] = method
    out["metric"] = metric
    return out
