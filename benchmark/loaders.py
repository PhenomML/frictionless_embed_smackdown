
from __future__ import annotations

from pathlib import Path
import csv
from urllib.request import urlretrieve
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import (
    fetch_20newsgroups_vectorized,
    fetch_openml,
    fetch_olivetti_faces,
    load_iris as _load_iris_sklearn,
    load_wine as _load_wine_sklearn,
    make_s_curve,
    make_swiss_roll,
)
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import pca_reduce, svd_reduce_sparse, standardize
from .utils import ensure_dir, subsample_rows, to_float32

ArrayF = NDArray[np.float32]
ArrayI = NDArray[np.int64]

try:
    import torch
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as T
except ImportError:
    torch = None
    DataLoader = None
    torchvision = None
    T = None

try:
    import scanpy as sc
except Exception:

    sc = None

corpus_id_map = {
    "20newsgroups": "newsgroups20",
    "newsgroups20": "newsgroups20",
    "cifar10_embeddings": "cifar10",
    "cifar10": "cifar10",
    "mnist_subsampled": "mnist",
    "agnews": "ag_news",
    "dbpedia14": "dbpedia_14",
    "dbpedia-14": "dbpedia_14",
    "uci_har": "uci_har",
}


def _class_support_summary(y: NDArray[np.int64]) -> dict[str, Any]:
    unique, counts = np.unique(y, return_counts=True)
    if len(counts) == 0:
        return {
            "number_of_classes_retained": 0,
            "min_class_support": 0,
            "median_class_support": 0,
            "max_class_support": 0,
        }
    return {
        "number_of_classes_retained": int(len(unique)),
        "min_class_support": int(np.min(counts)),
        "median_class_support": float(np.median(counts)),
        "max_class_support": int(np.max(counts)),
    }


def _download_if_missing(url: str, out_path: Path) -> None:
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, out_path)


def _read_text_csv_rows(csv_path: Path) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                label = int(str(row[0]).strip())
            except ValueError:
                continue
            txt = " ".join([str(x).strip() for x in row[1:] if str(x).strip()])
            if not txt:
                continue
            labels.append(label - 1)
            texts.append(txt)
    return texts, labels


def _load_text_benchmark_from_csv(
    root: Path,
    dataset_subdir: str,
    train_url: str,
    test_url: str,
    expected_classes: int,
    dataset_name: str,
    class_names: list[str],
    seed: int,
) -> Tuple[ArrayF, ArrayI, Dict[str, Any]]:
    ds_dir = root / dataset_subdir
    train_csv = ds_dir / "train.csv"
    test_csv = ds_dir / "test.csv"
    _download_if_missing(train_url, train_csv)
    _download_if_missing(test_url, test_csv)

    texts_train, labels_train = _read_text_csv_rows(train_csv)
    texts_test, labels_test = _read_text_csv_rows(test_csv)
    texts = texts_train + texts_test
    y = np.array(labels_train + labels_test, dtype=np.int64)
    if len(texts) != len(y):
        raise ValueError(f"{dataset_name}: text/label length mismatch.")
    if len(texts) == 0:
        raise ValueError(f"{dataset_name}: no rows loaded from CSV files.")

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=50000,
        norm="l2",
    )
    x_tfidf = tfidf.fit_transform(texts)
    x = svd_reduce_sparse(x_tfidf, n_components=50, seed=seed)

    support = _class_support_summary(y)
    retained = int(support["number_of_classes_retained"])
    dropped = []
    if retained < expected_classes:
        present = set(np.unique(y).tolist())
        dropped = [name for i, name in enumerate(class_names) if i not in present]
    return x, y, {
        "name": dataset_name,
        "n": int(x.shape[0]),
        "d": int(x.shape[1]),
        "label_type": "ground_truth",
        "class_names": class_names,
        "representation_description": "title+description text -> TF-IDF(max_features=50000) + TruncatedSVD(50)",
        "label_construction_note": "Original benchmark labels (single-label multiclass), train+test concatenated deterministically",
        "n_raw": int(len(texts)),
        "n_used": int(len(texts)),
        "number_of_classes_declared": int(expected_classes),
        "classes_dropped": dropped,
        **support,
    }


def _load_text_benchmark_from_parquet(
    root: Path,
    dataset_subdir: str,
    train_url: str,
    test_url: str,
    expected_classes: int,
    dataset_name: str,
    class_names: list[str],
    seed: int,
) -> Tuple[ArrayF, ArrayI, Dict[str, Any]]:
    import pandas as pd

    ds_dir = root / dataset_subdir
    train_pq = ds_dir / "train.parquet"
    test_pq = ds_dir / "test.parquet"
    _download_if_missing(train_url, train_pq)
    _download_if_missing(test_url, test_pq)

    dtr = pd.read_parquet(train_pq)
    dte = pd.read_parquet(test_pq)
    df = pd.concat([dtr, dte], ignore_index=True)
    required = {"label", "title", "content"}
    if not required.issubset(df.columns):
        raise ValueError(f"{dataset_name}: parquet missing columns {sorted(required)}")
    texts = (
        df["title"].astype(str).str.strip()
        + " "
        + df["content"].astype(str).str.strip()
    ).str.strip().tolist()
    y = pd.to_numeric(df["label"], errors="raise").astype(int).to_numpy(dtype=np.int64)
    # Leave zero-indexed labels alone and shift only one-indexed sources.
    if int(np.min(y)) == 1:
        y = y - 1

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=50000,
        norm="l2",
    )
    x_tfidf = tfidf.fit_transform(texts)
    x = svd_reduce_sparse(x_tfidf, n_components=50, seed=seed)

    support = _class_support_summary(y)
    retained = int(support["number_of_classes_retained"])
    dropped = []
    if retained < expected_classes:
        present = set(np.unique(y).tolist())
        dropped = [name for i, name in enumerate(class_names) if i not in present]
    return x, y, {
        "name": dataset_name,
        "n": int(x.shape[0]),
        "d": int(x.shape[1]),
        "label_type": "ground_truth",
        "class_names": class_names,
        "representation_description": "title+content text -> TF-IDF(max_features=50000) + TruncatedSVD(50)",
        "label_construction_note": "Original benchmark labels (single-label multiclass), train+test concatenated deterministically",
        "n_raw": int(len(texts)),
        "n_used": int(len(texts)),
        "number_of_classes_declared": int(expected_classes),
        "classes_dropped": dropped,
        **support,
    }

def _derive_leiden_on_pca(x: ArrayF, resolution: float = 1.0, seed: int = 123) -> ArrayI:
    import anndata as ad
    adata = ad.AnnData(x)
    sc.pp.neighbors(adata, use_rep="X", random_state=seed)
    sc.tl.leiden(
        adata, resolution=resolution, random_state=seed,
        flavor="igraph", n_iterations=2, directed=False,
    )
    return np.array(adata.obs["leiden"].astype(int), dtype=np.int64)

def load_from_corpus(
    corpus_dir: Path,
    dataset_id: str,
    subsample: Optional[int] = None,
    seed: int = 123,
) -> Tuple[ArrayF, Optional[ArrayI], Dict[str, Any]]:
    corpus_id = corpus_id_map.get(dataset_id, dataset_id)
    path = corpus_dir / corpus_id

    if not (path / "X_pca50.npy").exists():
        raise FileNotFoundError(f"Corpus not found: {path / 'X_pca50.npy'}")

    x = to_float32(np.load(path / "X_pca50.npy"))
    y_path = path / "y.npy"
    label_source_pathway = "none"
    if y_path.exists():
        y_raw = np.load(y_path)
        if len(y_raw) == x.shape[0]:
            if np.issubdtype(y_raw.dtype, np.integer):
                y = y_raw.astype(np.int64)

                if corpus_id == "coil20" and y.size > 0 and y.min() == 1 and y.max() == 20:
                    y = y - 1
            else:
                y = y_raw.astype(np.float32)
            label_source_pathway = "corpus_y"
        else:
            y = None
    else:
        y = None

    if y is None and corpus_id == "coil20" and x.shape[0] == 1440:
        y = np.repeat(np.arange(20, dtype=np.int64), 72)
        label_source_pathway = "synthesized_position"

    if y is None and corpus_id in ("pbmc3k", "mouse_retina") and sc is not None:
        y = _derive_leiden_on_pca(x, seed=seed)
        label_source_pathway = "derived_leiden"

    if subsample is not None and x.shape[0] > subsample:
        x, y, _ = subsample_rows(x, y, subsample, seed, stratified=(y is not None and np.issubdtype(y.dtype, np.integer)))
    meta = {"name": corpus_id, "n": int(x.shape[0]), "d": int(x.shape[1])}

    meta_path = path / "meta.json"
    if meta_path.exists():
        import json
        with open(meta_path) as f:
            extra = json.load(f)
        meta.update({
            k: v for k, v in extra.items()
            if k in ("label_type", "label_source", "class_names", "leiden_params", "neighbors_params")
        })
    if y is not None and meta.get("label_type") == "derived" and meta.get("class_names") is None:
        meta["class_names"] = [f"cluster_{i}" for i in range(len(np.unique(y)))]

    if corpus_id == "newsgroups20" and meta.get("class_names") is None:
        try:
            from sklearn.datasets import fetch_20newsgroups
            _ng = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
            meta["class_names"] = list(_ng.target_names)
        except Exception:
            pass

    if corpus_id == "mouse_retina" and y is not None and label_source_pathway == "corpus_y":
        pbmc_y = corpus_dir / "pbmc3k" / "y.npy"
        if pbmc_y.exists():
            y_pbmc = np.load(pbmc_y)
            if len(y_pbmc) == len(y) and np.array_equal(y_pbmc, y):
                raise ValueError(
                    "mouse_retina corpus is contaminated with PBMC data (identical y.npy). "
                    "Remove corpus/mouse_retina/ and add real mouse retina data, or exclude from benchmark."
                )
    meta["label_source_pathway"] = label_source_pathway
    return x, y, meta

def corpus_exists(corpus_dir: Path, dataset_id: str) -> bool:
    corpus_id = corpus_id_map.get(dataset_id, dataset_id)
    return (corpus_dir / corpus_id / "X_pca50.npy").exists()

def load_mnist(root: Path) -> Tuple[ArrayF, ArrayI, Dict[str, Any]]:
    if torch is None:
        raise ImportError("Install torch+torchvision to load MNIST.")
    transform = T.Compose([T.ToTensor()])
    ds_train = torchvision.datasets.MNIST(root=str(root), train=True, download=True, transform=transform)
    ds_test = torchvision.datasets.MNIST(root=str(root), train=False, download=True, transform=transform)
    x_list, y_list = [], []
    for ds in [ds_train, ds_test]:
        for img, label in ds:
            x_list.append(img.numpy().reshape(-1).astype(np.float32))
            y_list.append(int(label))
    x = to_float32(np.stack(x_list, axis=0))
    y = np.array(y_list, dtype=np.int64)
    support = _class_support_summary(y)
    return x, y, {
        "name": "mnist", "n": int(x.shape[0]), "d": int(x.shape[1]),
        "label_type": "ground_truth", "class_names": [str(i) for i in range(10)],
        "representation_description": "Flattened grayscale pixels (28x28) -> downstream standardize + PCA(50)",
        "label_construction_note": "Original MNIST single-label digit classes",
        "n_raw": int(x.shape[0]),
        "n_used": int(x.shape[0]),
        "number_of_classes_declared": 10,
        "classes_dropped": [],
        **support,
    }


def load_ag_news(root: Path, seed: int = 123) -> Tuple[ArrayF, ArrayI, Dict[str, Any]]:
    class_names = ["World", "Sports", "Business", "Sci/Tech"]
    return _load_text_benchmark_from_csv(
        root=root,
        dataset_subdir="ag_news_csv",
        train_url="https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
        test_url="https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
        expected_classes=4,
        dataset_name="ag_news",
        class_names=class_names,
        seed=seed,
    )


def load_dbpedia_14(root: Path, seed: int = 123) -> Tuple[ArrayF, ArrayI, Dict[str, Any]]:
    class_names = [
        "Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder",
        "MeanOfTransportation", "Building", "NaturalPlace", "Village",
        "Animal", "Plant", "Album", "Film", "WrittenWork",
    ]
    return _load_text_benchmark_from_parquet(
        root=root,
        dataset_subdir="dbpedia_14_parquet",
        train_url="https://huggingface.co/datasets/fancyzhx/dbpedia_14/resolve/main/dbpedia_14/train-00000-of-00001.parquet",
        test_url="https://huggingface.co/datasets/fancyzhx/dbpedia_14/resolve/main/dbpedia_14/test-00000-of-00001.parquet",
        expected_classes=14,
        dataset_name="dbpedia_14",
        class_names=class_names,
        seed=seed,
    )

def load_fashion_mnist(root: Path) -> Tuple[ArrayF, ArrayI, Dict[str, Any]]:
    if torch is None:
        raise ImportError("Install torch+torchvision to load Fashion-MNIST.")
    transform = T.Compose([T.ToTensor()])
    ds_train = torchvision.datasets.FashionMNIST(root=str(root), train=True, download=True, transform=transform)
    ds_test = torchvision.datasets.FashionMNIST(root=str(root), train=False, download=True, transform=transform)
    x_list, y_list = [], []
    for ds in [ds_train, ds_test]:
        for img, label in ds:
            x_list.append(img.numpy().reshape(-1).astype(np.float32))
            y_list.append(int(label))
    x = to_float32(np.stack(x_list, axis=0))
    y = np.array(y_list, dtype=np.int64)
    return x, y, {
        "name": "fashion_mnist", "n": int(x.shape[0]), "d": int(x.shape[1]),
        "label_type": "ground_truth",
        "class_names": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    }

def load_emnist(root: Path, split: str = "balanced") -> Tuple[ArrayF, ArrayI, Dict[str, Any]]:
    if torch is None:
        raise ImportError("Install torch+torchvision to load EMNIST.")
    transform = T.Compose([T.ToTensor()])
    ds_train = torchvision.datasets.EMNIST(root=str(root), split=split, train=True, download=True, transform=transform)
    ds_test = torchvision.datasets.EMNIST(root=str(root), split=split, train=False, download=True, transform=transform)
    class_names = list(ds_train.classes)
    x_list, y_list = [], []
    for ds in [ds_train, ds_test]:
        for img, label in ds:
            x_list.append(img.numpy().reshape(-1).astype(np.float32))
            y_list.append(int(label))
    x = to_float32(np.stack(x_list, axis=0))
    y = np.array(y_list, dtype=np.int64)
    return x, y, {
        "name": f"emnist_{split}", "n": int(x.shape[0]), "d": int(x.shape[1]),
        "label_type": "ground_truth", "class_names": class_names,
    }

def load_olivetti(root: Path) -> Tuple[ArrayF, ArrayI, Dict[str, Any]]:
    data = fetch_olivetti_faces(data_home=str(root), shuffle=False)
    x = to_float32(data.data)
    y = data.target.astype(np.int64)
    return x, y, {
        "name": "olivetti_faces", "n": int(x.shape[0]), "d": int(x.shape[1]),
        "label_type": "ground_truth", "class_names": [f"person_{i}" for i in range(40)],
    }

def load_20newsgroups(root: Path) -> Tuple[ArrayF, Optional[ArrayI], Dict[str, Any]]:
    data = fetch_20newsgroups_vectorized(data_home=str(root))
    x_svd = svd_reduce_sparse(data.data, n_components=2000, seed=123)
    y = data.target.astype(np.int64)
    class_names = list(data.target_names)
    return x_svd, y, {
        "name": "20newsgroups_vectorized", "n": int(x_svd.shape[0]), "d": int(x_svd.shape[1]),
        "label_type": "ground_truth", "class_names": class_names,
    }

def load_swiss_roll(_root: Path, n_samples: int = 1500) -> Tuple[ArrayF, ArrayF, Dict[str, Any]]:
    x, t = make_swiss_roll(n_samples=n_samples, noise=0.0, random_state=123)
    return to_float32(x), to_float32(t.astype(np.float32)), {
        "name": "swiss_roll", "n": int(x.shape[0]), "d": int(x.shape[1]),
        "label_type": "continuous", "class_names": None,
    }

def load_s_curve(_root: Path, n_samples: int = 1500) -> Tuple[ArrayF, ArrayF, Dict[str, Any]]:
    x, t = make_s_curve(n_samples=n_samples, noise=0.0, random_state=123)
    return to_float32(x), to_float32(t.astype(np.float32)), {
        "name": "s_curve", "n": int(x.shape[0]), "d": int(x.shape[1]),
        "label_type": "continuous", "class_names": None,
    }

def load_iris(_root: Path) -> Tuple[ArrayF, ArrayI, Dict[str, Any]]:
    d = _load_iris_sklearn()
    return to_float32(d.data), d.target.astype(np.int64), {
        "name": "iris", "n": int(d.data.shape[0]), "d": int(d.data.shape[1]),
        "label_type": "ground_truth", "class_names": list(d.target_names),
    }

def load_wine(_root: Path) -> Tuple[ArrayF, ArrayI, Dict[str, Any]]:
    d = _load_wine_sklearn()
    return to_float32(d.data), d.target.astype(np.int64), {
        "name": "wine", "n": int(d.data.shape[0]), "d": int(d.data.shape[1]),
        "label_type": "ground_truth", "class_names": [f"class_{i}" for i in range(3)],
    }

def load_pbmc3k(root: Path, leiden_resolution: float = 1.0, seed: int = 123, **kwargs: Any) -> Tuple[ArrayF, ArrayI, Dict[str, Any]]:
    if sc is None:
        raise ImportError("Install scanpy/anndata to load PBMC3k.")
    ensure_dir(root)
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack", n_comps=50)
    sc.pp.neighbors(adata, use_rep="X_pca", random_state=seed)
    sc.tl.leiden(adata, resolution=leiden_resolution, random_state=seed, flavor="igraph", n_iterations=2, directed=False)
    y = np.array(adata.obs["leiden"].astype(int), dtype=np.int64)
    x = to_float32(adata.obsm["X_pca"])
    n_clusters = len(np.unique(y))
    return x, y, {
        "name": "pbmc3k", "n": int(x.shape[0]), "d": int(x.shape[1]),
        "label_type": "derived", "class_names": [f"cluster_{i}" for i in range(n_clusters)],
    }

def load_mouse_retina(root: Path, leiden_resolution: float = 1.0, seed: int = 123, **kwargs: Any) -> Tuple[ArrayF, ArrayI, Dict[str, Any]]:
    if sc is None:
        raise ImportError(
            "Install scanpy/anndata to load mouse_retina: pip install scanpy anndata. "
            "Or add corpus/mouse_retina/X_pca50.npy (and optionally y.npy) to avoid the dependency."
        )
    ensure_dir(root)
    try:
        adata = sc.datasets.muris()
    except (AttributeError, ModuleNotFoundError) as e:
        import warnings
        warnings.warn(
            f"mouse_retina: muris() unavailable ({e}). Falling back to pbmc3k-labels will be identical to PBMC. "
            "Install scanpy with muris support or use corpus/mouse_retina/ with pre-built data.",
            UserWarning,
            stacklevel=2,
        )
        adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack", n_comps=50)
    if adata.n_obs > 5000:
        sc.pp.subsample(adata, n_obs=5000, copy=False, random_state=seed)
    sc.pp.neighbors(adata, use_rep="X_pca", random_state=seed)
    sc.tl.leiden(adata, resolution=leiden_resolution, random_state=seed, flavor="igraph", n_iterations=2, directed=False)
    y = np.array(adata.obs["leiden"].astype(int), dtype=np.int64)
    x = to_float32(adata.obsm["X_pca"])
    n_clusters = len(np.unique(y))
    return x, y, {
        "name": "mouse_retina", "n": int(x.shape[0]), "d": int(x.shape[1]),
        "label_type": "derived", "class_names": [f"cluster_{i}" for i in range(n_clusters)],
    }

def load_cifar10_embeddings(root: Path, batch_size: int = 256) -> Tuple[ArrayF, ArrayI, Dict[str, Any]]:
    if torch is None:
        raise ImportError("Install torch+torchvision to load CIFAR-10 embeddings.")
    class ResNetFeatureExtractor(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.backbone.fc = torch.nn.Identity()
        def forward(self, x):
            return self.backbone(x)
    ensure_dir(root)
    transform = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    ds_train = torchvision.datasets.CIFAR10(root=str(root), train=True, download=True, transform=transform)
    ds_test = torchvision.datasets.CIFAR10(root=str(root), train=False, download=True, transform=transform)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False, num_workers=0)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)
    backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model = ResNetFeatureExtractor(backbone).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    feats, labels = [], []
    with torch.no_grad():
        for loader in [loader_train, loader_test]:
            for xb, yb in loader:
                feats.append(model(xb.to(device)).cpu().numpy().astype(np.float32))
                labels.extend([int(v) for v in yb.numpy().tolist()])
    x = to_float32(np.vstack(feats))
    y = np.array(labels, dtype=np.int64)
    support = _class_support_summary(y)
    return x, y, {
        "name": "cifar10_resnet18_embeddings", "n": int(x.shape[0]), "d": int(x.shape[1]),
        "label_type": "ground_truth",
        "class_names": ["airplane", "automobile", "bird", "cat", "dog", "deer", "frog", "horse", "ship", "truck"],
        "representation_description": "ResNet18 penultimate embedding (frozen pretrained backbone)",
        "label_construction_note": "Original CIFAR-10 single-label classes",
        "n_raw": int(x.shape[0]),
        "n_used": int(x.shape[0]),
        "number_of_classes_declared": 10,
        "classes_dropped": [],
        **support,
    }


def _load_uci_har_from_local(root: Path) -> Optional[Tuple[ArrayF, ArrayI, Dict[str, Any]]]:
    base = root / "uci_har" / "UCI HAR Dataset"
    x_train = base / "train" / "X_train.txt"
    y_train = base / "train" / "y_train.txt"
    x_test = base / "test" / "X_test.txt"
    y_test = base / "test" / "y_test.txt"
    if not all(p.exists() for p in [x_train, y_train, x_test, y_test]):
        return None
    xtr = np.loadtxt(x_train, dtype=np.float32)
    ytr = np.loadtxt(y_train, dtype=np.int64)
    xte = np.loadtxt(x_test, dtype=np.float32)
    yte = np.loadtxt(y_test, dtype=np.int64)
    x = to_float32(np.vstack([xtr, xte]))
    y = np.concatenate([ytr, yte]).astype(np.int64) - 1
    class_names = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ]
    support = _class_support_summary(y)
    return x, y, {
        "name": "uci_har",
        "n": int(x.shape[0]),
        "d": int(x.shape[1]),
        "label_type": "ground_truth",
        "class_names": class_names,
        "representation_description": "Raw HAR features (561) -> downstream standardize + PCA(50)",
        "label_construction_note": "Original UCI HAR single-label activity classes",
        "n_raw": int(x.shape[0]),
        "n_used": int(x.shape[0]),
        "number_of_classes_declared": 6,
        "classes_dropped": [],
        **support,
    }


def _load_uci_har_from_openml() -> Optional[Tuple[ArrayF, ArrayI, Dict[str, Any]]]:
    candidates = [
        {"name": "har", "version": 1},
        {"name": "UCI_HAR_Dataset", "version": 1},
    ]
    for c in candidates:
        try:
            d = fetch_openml(name=c["name"], version=c["version"], as_frame=False, parser="auto")
            x = to_float32(np.asarray(d.data, dtype=np.float32))
            y_raw = np.asarray(d.target)
            classes, y_idx = np.unique(y_raw.astype(str), return_inverse=True)
            y = y_idx.astype(np.int64)
            if len(classes) != 6:
                continue
            support = _class_support_summary(y)
            return x, y, {
                "name": "uci_har_openml",
                "n": int(x.shape[0]),
                "d": int(x.shape[1]),
                "label_type": "ground_truth",
                "class_names": classes.tolist(),
                "representation_description": "OpenML HAR features -> downstream standardize + PCA(50)",
                "label_construction_note": "OpenML single-label HAR mapping (deterministic class-index encoding)",
                "n_raw": int(x.shape[0]),
                "n_used": int(x.shape[0]),
                "number_of_classes_declared": 6,
                "classes_dropped": [],
                **support,
            }
        except Exception:
            continue
    return None


def load_uci_har(root: Path, seed: int = 123) -> Tuple[ArrayF, ArrayI, Dict[str, Any]]:
    local = _load_uci_har_from_local(root)
    if local is not None:
        return local
    from_openml = _load_uci_har_from_openml()
    if from_openml is not None:
        return from_openml
    raise FileNotFoundError(
        "UCI HAR not found. Provide local files under data/uci_har/UCI HAR Dataset/"
        " or ensure OpenML access for HAR."
    )

def load_coil20(root: Path) -> Tuple[ArrayF, Optional[ArrayI], Dict[str, Any]]:
    project_root = root.parent
    corpus_coil = project_root / "corpus" / "coil20"
    coil_path = corpus_coil if corpus_coil.exists() else root / "coil20"
    for x_name, y_name in [("X_pca50.npy", "y.npy"), ("X.npy", "y.npy")]:
        x_path, y_path = coil_path / x_name, coil_path / y_name
        if x_path.exists():
            x = to_float32(np.load(x_path))
            n = x.shape[0]
            if y_path.exists():
                y_raw = np.load(y_path)
                if len(y_raw) == n and np.issubdtype(y_raw.dtype, np.integer):
                    y = y_raw.astype(np.int64)

                    if y.min() == 1 and y.max() == 20:
                        y = y - 1
                else:
                    y = None
            else:
                y = None

            if y is None and n == 1440:
                y = np.repeat(np.arange(20, dtype=np.int64), 72)
            return x, y, {
                "name": "coil20", "n": int(n), "d": int(x.shape[1]),
                "label_type": "ground_truth", "class_names": [f"object_{i}" for i in range(20)],
            }
    raise FileNotFoundError(f"COIL-20 not found. Put X_pca50.npy (and y.npy) in corpus/coil20/.")

loader_registry: Dict[str, Callable] = {
    "mnist": load_mnist, "fashion_mnist": load_fashion_mnist, "coil20": load_coil20,
    "ag_news": load_ag_news, "agnews": load_ag_news,
    "dbpedia_14": load_dbpedia_14, "dbpedia14": load_dbpedia_14, "dbpedia-14": load_dbpedia_14,
    "olivetti": load_olivetti, "olivetti_faces": load_olivetti,
    "newsgroups20": load_20newsgroups, "20newsgroups": load_20newsgroups,
    "swiss_roll": load_swiss_roll, "s_curve": load_s_curve,
    "iris": load_iris, "wine": load_wine, "pbmc3k": load_pbmc3k,
    "cifar10": load_cifar10_embeddings, "cifar10_embeddings": load_cifar10_embeddings,
    "emnist": lambda r: load_emnist(r, "balanced"), "emnist_balanced": lambda r: load_emnist(r, "balanced"),
    "mouse_retina": load_mouse_retina,
    "uci_har": load_uci_har,
}

def load_dataset(
    dataset_id: str,
    data_root: Path,
    corpus_dir: Path,
    pca_dim: int = 50,
    subsample: Optional[int] = None,
    use_corpus_when_available: bool = True,
    seed: int = 123,
    use_standardize: bool = True,
) -> Tuple[ArrayF, Optional[ArrayI], Dict[str, Any]]:
    corpus_id = corpus_id_map.get(dataset_id, dataset_id)
    if use_corpus_when_available and corpus_exists(corpus_dir, dataset_id):
        x, y, meta = load_from_corpus(corpus_dir, dataset_id, subsample=subsample, seed=seed)

        if y is None and corpus_id in ("pbmc3k", "mouse_retina"):
            pass
        else:
            return x, y, meta

    loader = loader_registry.get(dataset_id)
    if loader is None:
        raise KeyError(f"No loader for dataset: {dataset_id}")
    try:
        x_raw, y, meta = loader(data_root, seed=seed)
    except TypeError:
        x_raw, y, meta = loader(data_root)
    if subsample is not None and x_raw.shape[0] > subsample:
        strat = y is not None and hasattr(y, "dtype") and np.issubdtype(y.dtype, np.integer)
        x_raw, y, _ = subsample_rows(x_raw, y, subsample, seed, stratified=strat)
    if use_standardize:
        x_raw = standardize(x_raw)
    if x_raw.shape[1] > pca_dim:
        x_raw = pca_reduce(x_raw, n_components=pca_dim, seed=seed)
    elif x_raw.shape[1] < pca_dim and x_raw.shape[1] >= 3:
        pad = np.zeros((x_raw.shape[0], pca_dim - x_raw.shape[1]), dtype=np.float32)
        x_raw = np.hstack([x_raw, pad])
    meta["analysis_rep"] = f"standardize+pca{pca_dim}" if use_standardize else f"pca{pca_dim}"
    meta["label_source_pathway"] = "raw_derived_leiden" if meta.get("label_type") == "derived" else "raw_loader"
    return x_raw, y, meta
