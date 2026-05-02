
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

ArrayI = NDArray[np.int64]

def contingency_table_from_labels(
    labels1: np.ndarray,
    labels2: np.ndarray,
) -> np.ndarray:
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)

    if labels1.ndim != 1 or labels2.ndim != 1:
        raise ValueError("labels1 and labels2 must be one-dimensional")
    if labels1.shape[0] != labels2.shape[0]:
        raise ValueError("labels1 and labels2 must have the same length")

    uniq1, inv1 = np.unique(labels1, return_inverse=True)
    uniq2, inv2 = np.unique(labels2, return_inverse=True)

    table = np.zeros((uniq1.size, uniq2.size), dtype=np.int64)
    np.add.at(table, (inv1, inv2), 1)
    return table

def comb2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    return (x * (x - 1)) // 2

def partition_jaccard_components(
    labels1: np.ndarray,
    labels2: np.ndarray,
) -> dict[str, int]:
    table = contingency_table_from_labels(labels1, labels2)
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)

    m11 = int(comb2(table).sum())
    same1 = int(comb2(row_sums).sum())
    same2 = int(comb2(col_sums).sum())
    m10 = same1 - m11
    m01 = same2 - m11
    denom = m11 + m10 + m01

    return {
        "m11": m11,
        "m10": m10,
        "m01": m01,
        "denom": denom,
    }

def partition_jaccard_from_labels(
    labels1: np.ndarray,
    labels2: np.ndarray,
    empty_union_value: float = 1.0,
) -> float:
    comp = partition_jaccard_components(labels1, labels2)
    denom = comp["denom"]
    if denom == 0:
        return float(empty_union_value)
    return float(comp["m11"] / denom)

def partition_ari_from_labels(
    labels1: np.ndarray,
    labels2: np.ndarray,
) -> float:
    return float(adjusted_rand_score(labels1, labels2))

def partition_ami_from_labels(
    labels1: np.ndarray,
    labels2: np.ndarray,
    average_method: str = "arithmetic",
) -> float:
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)

    if labels1.ndim != 1 or labels2.ndim != 1:
        raise ValueError("labels1 and labels2 must be one-dimensional")
    if labels1.shape[0] != labels2.shape[0]:
        raise ValueError("labels1 and labels2 must have the same length")

    return float(
        adjusted_mutual_info_score(
            labels1,
            labels2,
            average_method=average_method,
        )
    )

@dataclass(frozen=True)
class PartitionMetricSpec:

    name: str
    display_name: str
    score_fn: Callable[[np.ndarray, np.ndarray], float]
    value_range: tuple[float, float]
    gt_curve_label: str
    pairwise_curve_label: str

_metric_registry: dict[str, PartitionMetricSpec] = {
    "ari": PartitionMetricSpec(
        name="ari",
        display_name="ARI",
        score_fn=partition_ari_from_labels,
        value_range=(-1.0, 1.0),
        gt_curve_label="ARI vs GT",
        pairwise_curve_label="mean pairwise ARI",
    ),
    "jaccard": PartitionMetricSpec(
        name="jaccard",
        display_name="Jaccard",
        score_fn=partition_jaccard_from_labels,
        value_range=(0.0, 1.0),
        gt_curve_label="Jaccard vs GT",
        pairwise_curve_label="mean pairwise Jaccard",
    ),
    "ami": PartitionMetricSpec(
        name="ami",
        display_name="AMI",
        score_fn=partition_ami_from_labels,
        value_range=(-1.0, 1.0),
        gt_curve_label="AMI vs GT",
        pairwise_curve_label="mean pairwise AMI",
    ),
}

def get_partition_metric(metric_name: str) -> PartitionMetricSpec:
    key = str(metric_name).lower()
    if key not in _metric_registry:
        valid = ", ".join(sorted(_metric_registry))
        raise KeyError(f"Unknown metric '{metric_name}'. Valid metrics: {valid}")
    return _metric_registry[key]
