#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd

from _standardization_utils import normalize_nullable_bool, require_columns


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description=(
            "Build a strict rerun plan (commands + checklist) from protocol_triage.csv "
            "without executing any benchmark runs."
        )
    )
    p.add_argument(
        "--triage-csv",
        type=Path,
        default=root / "results_standardized" / "protocol_triage.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results_standardized" / "rerun_plan",
    )
    p.add_argument(
        "--priority",
        choices=["high", "medium", "low", "all_non_strict"],
        default="high",
        help="Which protocol groups to include in the rerun plan.",
    )
    p.add_argument("--b", type=int, default=200)
    p.add_argument("--pairs", type=int, default=2000)
    p.add_argument("--n-null", type=int, default=200)
    p.add_argument("--max-points", type=int, default=5000)
    p.add_argument("--tau", type=int, default=200)
    p.add_argument("--min-pairs-m", type=int, default=1000)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def _priority_filter(df: pd.DataFrame, priority: str) -> pd.DataFrame:
    if priority == "all_non_strict":
        return df.loc[~df["strict_auto"]].copy()
    return df.loc[(~df["strict_auto"]) & (df["rerun_priority"] == priority)].copy()


def _extract_flags_from_script(script_path: Path) -> set[str]:
    text = script_path.read_text(encoding="utf-8")
    return set(re.findall(r"['\"](--[a-zA-Z0-9-]+)['\"]", text))


def _verify_cli_flags(repo_root: Path) -> None:
    k_script = repo_root / "scripts" / "run_k_sweep_replicability.py"
    f_script = repo_root / "scripts" / "run_fixed_k0_hyperparam_sweep.py"
    k_flags = _extract_flags_from_script(k_script)
    f_flags = _extract_flags_from_script(f_script)
    used_k = {"--datasets", "--methods", "--metric", "--b", "--pairs-per-k", "--max-points", "--n-null", "--alpha", "--z-star", "--seed", "--output-dir"}
    used_f = {"--datasets", "--methods", "--metrics", "--b", "--subsample-frac", "--max-pairs", "--run-null", "--tau", "--random-state", "--max-points", "--min-pairs-m", "--output-dir"}
    missing_k = sorted(used_k - k_flags)
    missing_f = sorted(used_f - f_flags)
    if missing_k or missing_f:
        raise ValueError(
            f"CLI flag verification failed. Missing in run_k_sweep_replicability.py: {missing_k}; "
            f"missing in run_fixed_k0_hyperparam_sweep.py: {missing_f}"
        )


def _build_k_sweep_commands(df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    cmds: list[str] = []
    for metric, g_metric in df.groupby("metric", sort=True):
        for dataset_id, g_ds in g_metric.groupby("dataset_id", sort=True):
            methods = sorted(g_ds["method"].dropna().astype(str).unique().tolist())
            out_dir = (
                f"outputs/strict_reruns/replicability_benchmark_{metric}"
                f"/dataset_{dataset_id}"
            )
            cmd = (
                "python scripts/run_k_sweep_replicability.py "
                f"--datasets {dataset_id} "
                f"--methods {' '.join(methods)} "
                f"--metric {metric} "
                f"--b {args.b} "
                f"--pairs-per-k {args.pairs} "
                f"--max-points {args.max_points} "
                f"--n-null {args.n_null} "
                "--alpha 0.05 "
                "--z-star 2.0 "
                f"--seed {args.seed} "
                f"--output-dir {out_dir}"
            )
            cmds.append(cmd)
    return cmds


def _build_fixed_k0_commands(df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    cmds: list[str] = []
    for (dataset_id, method), g in df.groupby(["dataset_id", "method"], sort=True):
        metrics = sorted(g["metric"].dropna().astype(str).unique().tolist())
        out_dir = (
            f"outputs/strict_reruns/fixed_k0_hyperparam_sweep"
            f"/dataset_{dataset_id}__method_{method}"
        )
        cmd = (
            "python scripts/run_fixed_k0_hyperparam_sweep.py "
            f"--datasets {dataset_id} "
            f"--methods {method} "
            f"--metrics {' '.join(metrics)} "
            f"--b {args.b} "
            "--subsample-frac 0.8 "
            f"--max-pairs {args.pairs} "
            "--run-null "
            f"--tau {args.tau} "
            f"--random-state {args.seed} "
            f"--max-points {args.max_points} "
            f"--min-pairs-m {args.min_pairs_m} "
            f"--output-dir {out_dir}"
        )
        cmds.append(cmd)
    return cmds


def _write_shell_script(path: Path, k_cmds: list[str], f_cmds: list[str]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "export OMP_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export VECLIB_MAXIMUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        "export LOKY_MAX_CPU_COUNT=1",
        "",
        "",
    ]
    if f_cmds:
        lines.append("# Fixed-K0 reruns")
        lines.extend(f_cmds)
        lines.append("")
    if k_cmds:
        lines.append("# K-sweep reruns")
        lines.extend(k_cmds)
        lines.append("")
    if not k_cmds and not f_cmds:
        lines.append("echo 'No reruns needed for selected priority.'")
    lines.extend(
        [
            "",
            "# After reruns finish:",
            "# python scripts/standardize_k_sweep_results.py",
            "# python scripts/standardize_fixed_k0_results.py",
            "# python scripts/compute_fixed_k0_surrogate_metrics.py",
            "# python scripts/build_protocol_triage.py",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_checklist(path: Path, triage_slice: pd.DataFrame, args: argparse.Namespace) -> None:
    if len(triage_slice) == 0:
        text = (
            "# Strict Rerun Checklist\n\n"
            "No groups matched the selected priority filter.\n\n"
            "Next steps:\n"
            "- Rebuild standardized outputs to confirm no-op status.\n"
        )
        path.write_text(text, encoding="utf-8")
        return

    rows = []
    for _, r in triage_slice.sort_values(
        ["table_family", "dataset_id", "method", "metric"]
    ).iterrows():
        rows.append(
            f"- [ ] `{r['table_family']}` | `{r['dataset_id']}` | `{r['method']}` | `{r['metric']}` "
            f"| tier=`{r['compute_tier']}` | priority=`{r['rerun_priority']}`"
        )
    text = (
        "# Strict Rerun Checklist\n\n"
        f"Priority filter: `{args.priority}`\n\n"
        "Target strict protocol:\n"
        f"- `b={args.b}`\n"
        f"- `pairs={args.pairs}`\n"
        f"- `n_null={args.n_null}`\n"
        f"- `max_points={args.max_points}`\n"
        f"- `seed={args.seed}`\n\n"
        "Groups:\n"
        + "\n".join(rows)
        + "\n\n"
        "Next steps:\n"
        "- Run `strict_rerun_commands.sh`.\n"
        "- Re-run the standardization/triage refresh commands listed at the end of the script.\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.triage_csv.exists():
        raise FileNotFoundError(
            f"Missing triage file: {args.triage_csv}. Run scripts/build_protocol_triage.py first."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _verify_cli_flags(Path(__file__).resolve().parent.parent)

    triage = pd.read_csv(args.triage_csv)
    require_columns(triage, ["table_family", "dataset_id", "method", "metric", "rerun_priority"], "protocol_triage input")
    if "strict_auto" not in triage.columns and "strict_ok" in triage.columns:
        triage["strict_auto"] = triage["strict_ok"]
    require_columns(triage, ["strict_auto"], "protocol_triage input")
    triage["strict_auto"] = normalize_nullable_bool(triage["strict_auto"]).fillna(False)
    triage["rerun_priority"] = triage["rerun_priority"].astype(str)
    triage_slice = _priority_filter(triage, args.priority)

    k_slice = triage_slice[triage_slice["table_family"] == "k_sweep"].copy()
    f_slice = triage_slice[triage_slice["table_family"] == "fixed_k0"].copy()
    k_cmds = _build_k_sweep_commands(k_slice, args)
    f_cmds = _build_fixed_k0_commands(f_slice, args)

    sh_path = args.output_dir / "strict_rerun_commands.sh"
    checklist_path = args.output_dir / "strict_rerun_checklist.md"
    manifest_path = args.output_dir / "strict_rerun_manifest.json"

    _write_shell_script(sh_path, k_cmds, f_cmds)
    _write_checklist(checklist_path, triage_slice, args)

    manifest = {
        "triage_csv": str(args.triage_csv),
        "priority_filter": args.priority,
        "strict_protocol": {
            "b": args.b,
            "pairs": args.pairs,
            "n_null": args.n_null,
            "max_points": args.max_points,
            "tau": args.tau,
            "min_pairs_m": args.min_pairs_m,
            "seed": args.seed,
        },
        "parameter_policy": "global_cli_override",
        "triage_recommendations_used": False,
        "group_count": int(len(triage_slice)),
        "k_sweep_command_count": int(len(k_cmds)),
        "fixed_k0_command_count": int(len(f_cmds)),
        "no_reruns_needed": bool(len(triage_slice) == 0),
        "k_sweep_commands": k_cmds,
        "fixed_k0_commands": f_cmds,
        "affected_groups_by_table_family": {
            "k_sweep": k_slice[["dataset_id", "method", "metric", "rerun_priority", "strict_failure_reason"]].to_dict("records")
            if len(k_slice)
            else [],
            "fixed_k0": f_slice[["dataset_id", "method", "metric", "rerun_priority", "strict_failure_reason"]].to_dict("records")
            if len(f_slice)
            else [],
        },
        "outputs": {
            "shell_script": str(sh_path),
            "checklist": str(checklist_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {sh_path}")
    print(f"Wrote {checklist_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

