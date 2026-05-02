# Strict Rerun Checklist

Priority filter: `high`

Target strict protocol:
- `b=200`
- `pairs=2000`
- `n_null=200`
- `max_points=5000`
- `seed=123`

Groups:
- [ ] `k_sweep` | `ag_news` | `tsne` | `ami` | tier=`retry_low` | priority=`high`
- [ ] `k_sweep` | `ag_news` | `umap` | `ami` | tier=`retry_low` | priority=`high`
- [ ] `k_sweep` | `cifar10` | `tsne` | `ami` | tier=`retry_low` | priority=`high`
- [ ] `k_sweep` | `cifar10` | `umap` | `ami` | tier=`retry_low` | priority=`high`

Next steps:
- Run `strict_rerun_commands.sh`.
- Re-run the standardization/triage refresh commands listed at the end of the script.
