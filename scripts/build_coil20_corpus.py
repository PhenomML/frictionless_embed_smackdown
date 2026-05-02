#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from benchmark.preprocessing import to_float32
from benchmark.utils import ensure_dir

obj_pattern = re.compile(r"obj(\d+)_+\d+\.(?:pgm|png)", re.IGNORECASE)

def _sanity_check(corpus_dir: Path) -> tuple[bool, list[str]]:
    errs: list[str] = []
    X = np.load(corpus_dir / "X_pca50.npy")
    y = np.load(corpus_dir / "y.npy")
    if X.shape != (1440, 50):
        errs.append(f"X_pca50 shape {X.shape} != (1440, 50)")
    if len(y) != 1440:
        errs.append(f"y length {len(y)} != 1440")
    uniq = np.unique(y)
    if len(uniq) != 20:
        errs.append(f"y has {len(uniq)} unique classes, expected 20")
    if uniq.min() < 0 or uniq.max() > 19:
        errs.append(f"y range [{uniq.min()}, {uniq.max()}] outside 0-19")
    with open(corpus_dir / "meta.json") as f:
        meta = json.load(f)
    if meta.get("label_source") != "filenames":
        errs.append(f"meta label_source={meta.get('label_source')}, expected filenames")
    return (len(errs) == 0, errs)

def parse_object_id(filename: str) -> int | None:
    m = obj_pattern.match(filename)
    if not m:
        return None
    n = int(m.group(1))
    if 1 <= n <= 20:
        return n - 1
    return None

def build_coil20_corpus(
    images_dir: Path,
    corpus_dir: Path,
    resize: tuple[int, int] = (32, 32),
    pca_dim: int = 50,
    seed: int = 123,
) -> None:
    images_dir = Path(images_dir)
    corpus_dir = Path(corpus_dir)
    ensure_dir(corpus_dir)

    pairs: list[tuple[Path, int]] = []
    for p in sorted(images_dir.rglob("*.pgm")) or sorted(images_dir.rglob("*.png")):
        obj_id = parse_object_id(p.name)
        if obj_id is not None:
            pairs.append((p, obj_id))

    if len(pairs) != 1440:
        raise ValueError(
            f"Expected 1440 images (20x72), found {len(pairs)}. "
            f"Check {images_dir} contains coil-20-proc/*.pgm"
        )

    X_list = []
    y_list = []
    for p, obj_id in pairs:
        img = np.array(Image.open(p).convert("L"))
        img_resized = np.array(
            Image.fromarray(img).resize(resize, Image.BILINEAR)
        )
        X_list.append(img_resized.flatten().astype(np.float32))
        y_list.append(obj_id)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    if X_std.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        X_pca = pca.fit_transform(X_std)
    else:
        X_pca = X_std

    X_pca = to_float32(X_pca)

    np.save(corpus_dir / "X_pca50.npy", X_pca)
    np.save(corpus_dir / "y.npy", y)

    filenames_str = "\n".join(sorted(p.name for p, _ in pairs))
    filenames_hash = hashlib.sha256(filenames_str.encode()).hexdigest()[:16]

    meta = {
        "name": "COIL-20",
        "modality": "image pixels",
        "original_dim": resize[0] * resize[1],
        "preprocessing": ["resize32x32", "flatten", "standardize", "PCA->50"],
        "label_type": "ground_truth",
        "label_source": "filenames",
        "class_names": [f"object_{i}" for i in range(20)],
        "label_semantics": "Labels are object IDs (0-19) extracted from filenames like obj{N}__{angle}.png (1-indexed N in files mapped to 0-19).",
        "dataset_fingerprint": {
            "source_url": "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip",
            "build_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "n_images": 1440,
            "filenames_sha256_prefix": filenames_hash,
        },
    }
    with open(corpus_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Built {corpus_dir}")
    print(f"  X_pca50: {X_pca.shape}, y: {y.shape}")
    print(f"  label_source: filenames (ground_truth)")

    ok, errs = _sanity_check(corpus_dir)
    if ok:
        print("  Sanity check: PASSED")
    else:
        print("  Sanity check: FAILED")
        for e in errs:
            print(f"    - {e}")
        raise RuntimeError("Corpus sanity check failed")

def main() -> None:
    images_dir = Path(__file__).resolve().parent.parent / "data" / "coil20"
    if not images_dir.exists():

        alt = images_dir.parent / "coil-20-proc"
        if alt.exists():
            images_dir = alt
        else:
            print(
                f"COIL-20 images not found. Expected {images_dir} or {alt}\n"
                "Download from: http://www.cs.columbia.edu/CAVE/databases/"
                "SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
            )
            sys.exit(1)

    corpus_dir = root / "corpus" / "coil20"
    build_coil20_corpus(images_dir, corpus_dir)
    print("Done.")

if __name__ == "__main__":
    main()
