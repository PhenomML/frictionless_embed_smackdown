#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from benchmark.paper_datasets import paper_dataset_inventory, validate_paper_dataset_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate paper dataset files under datasets/.")
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--json", action="store_true", help="Print JSON diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = validate_paper_dataset_files(args.repo_root)
    if args.json:
        print(json.dumps({"datasets": paper_dataset_inventory(), "validation": rows}, indent=2))
        return
    for row in rows:
        print(f"OK {row['dataset_id']}: {row['base_path']}")


if __name__ == "__main__":
    main()
