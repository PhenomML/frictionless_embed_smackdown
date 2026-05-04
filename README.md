# Embed Smackdown Benchmark

This is the reproduction repo for the clustering replicability benchmark presented in the paper XXX. by Aqib Syed and David Donoho.

It contains the code, the six paper datasets, the standardized numeric results, and the final paper artifacts. 

## Active Tree

The active tree is intentionally small.

- `benchmark/`: the importable Python code for loading data, embedding, computing replicability, building summaries, and plotting paper artifacts
- `scripts/`: command-line entry points for validation, benchmark runs, standardization, and artifact generation
- `configs/`: small JSON configs for local paper reproduction
- `datasets/`: the paper datasets
- `results_standardized/`: the canonical numeric results used by the manuscript
- `paper_artifacts/`: the final figures, tables, and manifests
- `notebooks/paper_artifact_generation.ipynb`: the one thin notebook frontend
- `tests/`: tests for the core replicability code

The six paper datasets are registered in `benchmark/paper_datasets.py`.

The dataset folder has two active parts:

- `datasets/data/`: raw or local sources used by loaders
- `datasets/corpus/`: frozen representations such as `X_pca50.npy`, `y.npy`, and `meta.json`

Old non-paper datasets live in `datasets/legacy data/`. They are not part of the active paper pipeline.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Optional dependencies may be needed only if we decide to rebuild raw image/text representations from scratch.

## Minimal Reproduction

First, check that the six paper datasets are present:

```bash
python scripts/validate_paper_datasets.py
```

Then regenerate the paper-facing figures and tables from the standardized results:

```bash
python scripts/generate_paper_figures.py --config configs/paper_reproduction.json
```

The notebook version does the same thing:

```text
notebooks/paper_artifact_generation.ipynb
```

## Full Rerun Skeleton

Full reruns are expensive. The command sequence below is the intended shape of the pipeline, but the normal paper workflow should use the standardized results already in `results_standardized/`.

```bash
python scripts/validate_paper_datasets.py

for metric in ami ari jaccard; do
  python scripts/run_k_sweep_replicability.py \
    --datasets mnist uci_har cifar10 ag_news 20newsgroups olivetti_faces \
    --methods tsne umap \
    --metric "$metric" \
    --b 200 \
    --pairs-per-k 2000 \
    --max-points 5000 \
    --n-null 200
done

python scripts/run_fixed_k0_hyperparam_sweep.py \
  --datasets mnist uci_har cifar10 ag_news 20newsgroups olivetti_faces \
  --methods tsne umap \
  --metrics ami ari jaccard \
  --b 200 \
  --subsample-frac 0.8 \
  --max-pairs 2000 \
  --max-points 5000 \
  --run-null

python scripts/standardize_k_sweep_results.py
python scripts/standardize_fixed_k0_results.py
python scripts/compute_fixed_k0_surrogate_metrics.py

python scripts/run_pair_subsampling_convergence.py \
  --datasets olivetti_faces 20newsgroups \
  --method umap \
  --metric ami

python scripts/generate_paper_figures.py --config configs/paper_reproduction.json
```

## Sherlock

This repo is not currently set up for Sherlock recompute. If we need full cluster reruns later, make a separate working copy or fork and add Sherlock-specific environment files, logs, and Slurm scripts there.

## Tests

```bash
pytest -q
```

