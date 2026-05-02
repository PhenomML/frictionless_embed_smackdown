
from __future__ import annotations

import numpy as np
import pytest

from benchmark.partition_metrics import (
    contingency_table_from_labels,
    get_partition_metric,
    partition_ami_from_labels,
    partition_jaccard_from_labels,
    partition_ari_from_labels,
    comb2,
)
from benchmark.replicability import (
    _is_dense_grid,
    cluster_profile,
    compute_hat_k_consec_for_tau,
    k_sweep_replicability,
    precompute_subsamples,
    select_hat_k_consec,
    ari_on_overlap,
    build_label_map,
    draw_subsample_indices,
    replicability_at_k,
    sample_replicate_pairs,
    select_hat_k,
    label_agreement_at_k,
    metric_vs_ground_truth_at_k,
    simulate_null_rep_at_k,
    extract_overlap_labels,
)

def test_draw_subsample_indices_size() -> None:
    n = 1000
    rng = np.random.default_rng(42)
    idx = draw_subsample_indices(n, 0.8, rng, stratified=False, y=None)
    assert len(idx) == int(0.8 * n)
    assert np.all(np.diff(idx) > 0)

def test_draw_subsample_indices_stratified() -> None:
    n = 100
    y = np.repeat(np.arange(5), 20)
    rng = np.random.default_rng(42)
    idx = draw_subsample_indices(n, 0.5, rng, stratified=True, y=y)
    assert len(idx) == 50

    y_sub = y[idx]
    for c in range(5):
        assert np.sum(y_sub == c) >= 1

def test_draw_subsample_indices_topup_no_duplicates() -> None:
    n = 50
    y = np.repeat(np.arange(10), 5)
    rng = np.random.default_rng(42)

    idx = draw_subsample_indices(n, 0.5, rng, stratified=True, y=y)
    assert len(idx) == 25
    assert len(np.unique(idx)) == 25

def test_draw_subsample_indices_stratified_float_labels() -> None:
    n = 100
    y = np.repeat(np.arange(5), 20).astype(np.float32)
    rng = np.random.default_rng(42)
    idx = draw_subsample_indices(n, 0.5, rng, stratified=True, y=y)
    assert len(idx) == 50
    y_sub = y[idx]
    for c in range(5):
        assert np.sum(y_sub == c) >= 1

def test_build_label_map() -> None:
    labels_sub = np.array([0, 1, 0])
    indices = np.array([2, 5, 7])
    n = 10
    m = build_label_map(labels_sub, indices, n)
    assert m.shape == (10,)
    assert m[2] == 0 and m[5] == 1 and m[7] == 0
    assert m[0] == -1 and m[1] == -1

def test_ari_on_overlap_identical() -> None:
    n = 20
    labels = np.array([0, 1, 0, 1] * 5)
    m = np.full(n, -1)
    m[:n] = labels
    overlap = np.arange(n)
    score = ari_on_overlap(m, m, overlap)
    assert score == 1.0

def test_ari_on_overlap_via_intersect1d() -> None:

    indices_a = np.array([0, 1, 2, 3, 4])
    indices_b = np.array([2, 3, 4, 5, 6])
    overlap = np.intersect1d(indices_a, indices_b, assume_unique=True)
    assert list(overlap) == [2, 3, 4]
    m_a = np.full(10, -1)
    m_a[indices_a] = [0, 0, 1, 1, 1]
    m_b = np.full(10, -1)
    m_b[indices_b] = [1, 1, 0, 0, 0]
    score = ari_on_overlap(m_a, m_b, overlap)
    assert -0.5 <= score <= 1.0

def test_sample_replicate_pairs_deterministic() -> None:
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    p1 = sample_replicate_pairs(50, 100, rng1)
    p2 = sample_replicate_pairs(50, 100, rng2)
    np.testing.assert_array_equal(p1, p2)

def test_sample_replicate_pairs_count() -> None:
    b = 20
    pairs_per_k = 50
    rng = np.random.default_rng(42)
    p = sample_replicate_pairs(b, pairs_per_k, rng)
    assert p.shape[1] == 2
    assert len(p) == min(pairs_per_k, b * (b - 1) // 2)
    for i, j in p:
        assert 0 <= i < j < b

def test_replicability_at_k_n_pairs_used() -> None:
    np.random.seed(42)
    n, k, b = 100, 3, 10
    labels_list = []
    indices_list = []
    rng = np.random.default_rng(42)
    for _ in range(b):
        idx = rng.choice(n, size=80, replace=False)
        idx = np.sort(idx)
        lab = np.random.randint(0, k, size=len(idx))
        labels_list.append(lab.astype(np.int64))
        indices_list.append(idx.astype(np.int64))
    pairs = sample_replicate_pairs(b, 20, rng)
    rep_m, rep_s, n_used, _ = replicability_at_k(
        labels_list, indices_list, pairs, min_overlap=10, n=n
    )
    assert n_used <= 20
    assert not np.isnan(rep_m) or n_used == 0

def test_select_hat_k_basic() -> None:
    rep = {2: 0.95, 3: 0.9, 4: 0.85, 5: 0.75}
    assert select_hat_k(rep, 0.8) == 4

def test_select_hat_k_consec() -> None:
    rep = {2: 0.9, 3: 0.85, 4: 0.82, 5: 0.75, 6: 0.81, 7: 0.83}
    assert select_hat_k_consec(rep, 0.8, 2) == 7
    assert select_hat_k_consec(rep, 0.8, 3) == 4
    rep2 = {2: 0.9, 3: 0.75, 4: 0.82, 5: 0.75, 6: 0.83}
    assert select_hat_k_consec(rep2, 0.8, 3) is None
    rep3 = {2: 0.9, 3: 0.85, 4: 0.82, 5: 0.81, 6: 0.83}
    assert select_hat_k_consec(rep3, 0.8, 3) == 6

def test_select_hat_k_none() -> None:
    rep = {2: 0.5, 3: 0.4}
    assert select_hat_k(rep, 0.8) is None

def test_cluster_profile() -> None:
    labels = np.array([0, 0, 1, 1, 1, 2])
    p = cluster_profile(labels)
    assert p["min_size"] == 1
    assert p["max_size"] == 3
    assert p["n_small_2"] == 2
    assert p["n_small_3"] == 3
    assert p["k_eff"] > 0

def test_is_dense_grid() -> None:
    assert _is_dense_grid([2, 3, 4, 5]) is True
    assert _is_dense_grid(list(range(2, 31))) is True
    assert _is_dense_grid([2, 3, 5, 8, 10, 15, 20]) is False
    assert _is_dense_grid([]) is False
    assert _is_dense_grid([5]) is True

def test_tau_sweep_pure_rule_sparse_grid() -> None:
    rep_by_k = {2: 0.9, 3: 0.85, 5: 0.82, 8: 0.81, 10: 0.75, 15: 0.72, 20: 0.7}
    assert select_hat_k(rep_by_k, 0.75) == 10
    assert select_hat_k(rep_by_k, 0.8) == 8
    assert select_hat_k(rep_by_k, 0.85) == 3

def test_compute_hat_k_consec_for_tau() -> None:
    rep_mean = [0.9, 0.85, 0.82, 0.75, 0.81, 0.83]
    k_range = [2, 3, 4, 5, 6, 7]
    assert compute_hat_k_consec_for_tau(rep_mean, k_range, 0.8, 2) == 7
    assert compute_hat_k_consec_for_tau(rep_mean, k_range, 0.8, 3) == 4

def test_draw_subsample_indices_stratified_overshoot() -> None:
    n = 50
    y = np.repeat(np.arange(5), 10)
    rng = np.random.default_rng(42)
    n_take = 25
    fraction = n_take / n
    idx = draw_subsample_indices(n, fraction, rng, stratified=True, y=y)
    assert len(idx) == n_take
    assert len(np.unique(idx)) == n_take

def test_replicability_at_k_no_pairs_used() -> None:
    n, b = 20, 5
    labels_list = [np.array([0, 1] * 10)] * b
    indices_list = [np.arange(n)] * b
    pairs = sample_replicate_pairs(b, 10, np.random.default_rng(42))
    rep_m, rep_s, n_used, _ = replicability_at_k(
        labels_list, indices_list, pairs, min_overlap=999, n=n
    )
    assert n_used == 0
    assert np.isnan(rep_m)
    assert np.isnan(rep_s)

def test_ari_on_overlap_permutation_invariance() -> None:
    n = 10
    labels_a = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    labels_b = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    m_a = np.full(n, -1)
    m_a[:] = labels_a
    m_b = np.full(n, -1)
    m_b[:] = labels_b
    overlap = np.arange(n)
    score = ari_on_overlap(m_a, m_b, overlap)
    assert score == 1.0

def test_freeze_across_k_same_subsamples() -> None:
    np.random.seed(42)
    n, b = 100, 5
    z = np.random.randn(n, 2).astype(np.float32)
    meta = {"label_type": "continuous"}
    k_range = [2, 3]
    result = k_sweep_replicability(
        z, None, meta, k_range, b=b, fraction=0.8, base_seed=123,
        pairs_per_k=10, n_init=2, max_iter=10,
    )
    assert result.get("perturbations_frozen_across_k") is True

    pairs_used = result["pairs_used"]
    assert len(pairs_used) == 2

    result2 = k_sweep_replicability(
        z, None, meta, k_range, b=b, fraction=0.8, base_seed=123,
        pairs_per_k=10, n_init=2, max_iter=10,
    )
    assert result["rep_mean"] == result2["rep_mean"]

def test_label_agreement_mean_std() -> None:
    labels_list = [
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 1, 0]),
    ]
    indices_list = [np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3])]
    y = np.array([0, 1, 0, 1])
    ari_m, ari_s, ami_m, ami_s = label_agreement_at_k(labels_list, indices_list, y)
    assert ari_m >= 0
    assert ari_s >= 0
    assert ami_m >= 0
    assert ami_s >= 0

def test_label_agreement_ami_uses_arithmetic_average_method() -> None:
    labels_list = [
        np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        np.array([2, 2, 0, 0, 1, 1], dtype=np.int64),
    ]
    indices_list = [
        np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
    ]
    y = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    _, _, ami_m, _ = label_agreement_at_k(labels_list, indices_list, y)

    expected = np.mean(
        [
            partition_ami_from_labels(labels_list[0], y),
            partition_ami_from_labels(labels_list[1], y),
        ]
    )
    assert np.isclose(ami_m, expected)

def test_jaccard_identical_partitions() -> None:
    labels1 = np.array([0, 0, 1, 1, 2, 2])
    labels2 = np.array([5, 5, 9, 9, 7, 7])
    assert partition_jaccard_from_labels(labels1, labels2) == 1.0

def test_jaccard_singleton_partitions() -> None:
    labels1 = np.array([0, 1, 2, 3])
    labels2 = np.array([10, 11, 12, 13])
    assert partition_jaccard_from_labels(labels1, labels2) == 1.0

def test_jaccard_complete_disagreement() -> None:
    labels1 = np.array([0, 0, 1, 1])
    labels2 = np.array([0, 1, 0, 1])
    assert partition_jaccard_from_labels(labels1, labels2) == 0.0

def test_jaccard_identical_labels() -> None:
    labels = np.array([0, 1, 0, 1, 1, 0])
    assert partition_jaccard_from_labels(labels, labels) == 1.0

def test_metric_registry_ari() -> None:
    assert get_partition_metric("ari").display_name == "ARI"
    assert get_partition_metric("ARI").name == "ari"

def test_metric_registry_jaccard() -> None:
    assert get_partition_metric("jaccard").display_name == "Jaccard"
    assert get_partition_metric("jaccard").value_range == (0.0, 1.0)

def test_metric_registry_ami() -> None:
    assert get_partition_metric("ami").display_name == "AMI"
    assert get_partition_metric("ami").value_range == (-1.0, 1.0)

def test_ami_identical_partitions() -> None:
    labels1 = np.array([0, 0, 1, 1, 2, 2])
    labels2 = np.array([5, 5, 9, 9, 7, 7])
    assert np.isclose(partition_ami_from_labels(labels1, labels2), 1.0)

def test_ami_label_permutation_invariance() -> None:
    labels1 = np.array([0, 0, 1, 1, 2, 2])
    labels2 = np.array([2, 2, 0, 0, 1, 1])
    assert np.isclose(partition_ami_from_labels(labels1, labels2), 1.0)

def test_metric_registry_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown metric"):
        get_partition_metric("unknown_metric")

def test_comb2() -> None:
    x = np.array([0, 1, 2, 3, 4])
    out = comb2(x)
    assert out[0] == 0
    assert out[1] == 0
    assert out[2] == 1
    assert out[3] == 3
    assert out[4] == 6

def test_contingency_table_from_labels() -> None:
    labels1 = np.array([0, 0, 1, 1])
    labels2 = np.array([0, 1, 0, 1])
    table = contingency_table_from_labels(labels1, labels2)
    assert table.shape[0] == 2
    assert table.shape[1] == 2
    assert table.sum() == 4

def test_replicability_at_k_with_jaccard() -> None:
    np.random.seed(42)
    n, k, b = 100, 3, 10
    labels_list = []
    indices_list = []
    rng = np.random.default_rng(42)
    for _ in range(b):
        idx = rng.choice(n, size=80, replace=False)
        idx = np.sort(idx)
        lab = np.random.randint(0, k, size=len(idx))
        labels_list.append(lab.astype(np.int64))
        indices_list.append(idx.astype(np.int64))
    pairs = sample_replicate_pairs(b, 20, rng)
    rep_m, rep_s, n_used, vec = replicability_at_k(
        labels_list, indices_list, pairs, min_overlap=10, n=n,
        score_fn=partition_jaccard_from_labels,
    )
    assert n_used <= 20
    assert 0 <= rep_m <= 1.0
    assert len(vec) == n_used

def test_replicability_at_k_with_ami_matches_direct() -> None:
    n = 6
    labels_a = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    labels_b = np.array([2, 2, 0, 0, 1, 1], dtype=np.int64)
    labels_list = [labels_a, labels_b]
    indices_list = [np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64)]
    pairs = np.array([[0, 1]], dtype=np.int64)
    rep_m, rep_s, n_used, vec = replicability_at_k(
        labels_list,
        indices_list,
        pairs,
        min_overlap=1,
        n=n,
        score_fn=get_partition_metric("ami").score_fn,
    )
    direct = partition_ami_from_labels(labels_a, labels_b)
    assert n_used == 1
    assert rep_s == 0.0
    assert len(vec) == 1
    assert np.isclose(rep_m, direct)

def test_null_preserves_marginals() -> None:
    u = np.array([0, 0, 1, 1, 1, 2])
    v = np.array([0, 0, 1, 1, 2, 2])
    rng = np.random.default_rng(123)
    for _ in range(10):
        v_perm = v.copy()
        rng.shuffle(v_perm)
        assert np.array_equal(np.sort(np.unique(v_perm, return_counts=True)[1]),
                              np.sort(np.unique(v, return_counts=True)[1]))

def test_null_preserves_marginals_in_ami_path() -> None:
    u = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)
    v = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    rng = np.random.default_rng(123)

    null_reps = simulate_null_rep_at_k(
        [u],
        [v],
        n_null=8,
        rng=rng,
        score_fn=get_partition_metric("ami").score_fn,
    )
    assert null_reps.shape == (8,)
    assert np.all(np.isfinite(null_reps))

    for _ in range(10):
        v_perm = v.copy()
        rng.shuffle(v_perm)
        assert np.array_equal(
            np.sort(np.unique(v_perm, return_counts=True)[1]),
            np.sort(np.unique(v, return_counts=True)[1]),
        )

def test_metric_vs_gt_with_ami() -> None:
    labels_list = [
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([1, 1, 0, 0], dtype=np.int64),
    ]
    indices_list = [
        np.array([0, 1, 2, 3], dtype=np.int64),
        np.array([0, 1, 2, 3], dtype=np.int64),
    ]
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    mean_ami, std_ami = metric_vs_ground_truth_at_k(
        labels_list,
        indices_list,
        y,
        score_fn=get_partition_metric("ami").score_fn,
    )
    assert np.isfinite(mean_ami)
    assert np.isfinite(std_ami)
    assert -1.0 <= mean_ami <= 1.0

def test_k_sweep_replicability_metric_jaccard() -> None:
    np.random.seed(42)
    n, b = 80, 5
    z = np.random.randn(n, 2).astype(np.float32)
    meta = {"label_type": "continuous"}
    k_range = [2, 3]
    result = k_sweep_replicability(
        z, None, meta, k_range, b=b, fraction=0.8, base_seed=123,
        pairs_per_k=15, n_init=2, max_iter=10,
        metric="jaccard",
    )
    assert result["metric"] == "jaccard"
    assert result["metric_display_name"] == "Jaccard"
    assert "metric_vs_gt_mean" in result
    assert "pairwise_metric_vec_by_k" in result
    assert result.get("ari_vs_gt_mean") is None

def test_k_sweep_replicability_metric_ami() -> None:
    np.random.seed(42)
    n, b = 80, 5
    z = np.random.randn(n, 2).astype(np.float32)
    meta = {"label_type": "continuous"}
    k_range = [2, 3]
    result = k_sweep_replicability(
        z, None, meta, k_range, b=b, fraction=0.8, base_seed=123,
        pairs_per_k=15, n_init=2, max_iter=10,
        metric="ami",
    )
    assert result["metric"] == "ami"
    assert result["metric_display_name"] == "AMI"
    assert "metric_vs_gt_mean" in result
    assert "pairwise_metric_vec_by_k" in result
