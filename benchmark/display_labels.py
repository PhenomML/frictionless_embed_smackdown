from __future__ import annotations


dataset_display_names = {
    "20newsgroups": "20 Newsgroups",
    "20 newsgroups": "20 Newsgroups",
    "ag_news": "AG News",
    "ag news": "AG News",
    "cifar10": "CIFAR-10",
    "cifar-10": "CIFAR-10",
    "mnist": "MNIST",
    "uci_har": "UCI HAR",
    "uci har": "UCI HAR",
    "olivetti_faces": "Olivetti Faces",
    "olivetti faces": "Olivetti Faces",
    "s_curve": "S-curve",
    "s-curve": "S-curve",
}

method_display_names = {
    "tsne": "t-SNE",
    "t-sne": "t-SNE",
    "umap": "UMAP",
    "mds": "MDS",
}

metric_display_names = {
    "ami": "AMI",
    "ari": "ARI",
    "jaccard": "Jaccard",
}


def _clean_key(value: object) -> str:
    return str(value).strip().lower()


def dataset_display_name(value: object) -> str:
    key = _clean_key(value)
    return dataset_display_names.get(key, str(value))


def method_display_name(value: object) -> str:
    key = _clean_key(value)
    return method_display_names.get(key, str(value))


def metric_display_name(value: object) -> str:
    key = _clean_key(value)
    return metric_display_names.get(key, str(value))
