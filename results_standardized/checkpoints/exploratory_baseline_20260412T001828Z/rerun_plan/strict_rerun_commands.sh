#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1


# K-sweep reruns
python scripts/run_k_sweep_replicability.py --datasets ag_news --methods tsne umap --metric ami --b 200 --pairs-per-k 2000 --max-points 5000 --n-null 200 --alpha 0.05 --z-star 2.0 --seed 123 --output-dir outputs/strict_reruns/replicability_benchmark_ami/dataset_ag_news
python scripts/run_k_sweep_replicability.py --datasets cifar10 --methods tsne umap --metric ami --b 200 --pairs-per-k 2000 --max-points 5000 --n-null 200 --alpha 0.05 --z-star 2.0 --seed 123 --output-dir outputs/strict_reruns/replicability_benchmark_ami/dataset_cifar10


# After reruns finish:
# python scripts/standardize_k_sweep_results.py
# python scripts/standardize_fixed_k0_results.py
# python scripts/compute_fixed_k0_surrogate_metrics.py
# python scripts/build_protocol_triage.py
