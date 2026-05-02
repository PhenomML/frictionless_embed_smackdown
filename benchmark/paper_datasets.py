from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PaperDatasetSpec:
    display_name: str
    domain: str
    k0: int
    n_target: int
    storage: str
    relative_path: str
    loader_id: str
    role: str
    label_type: str = "ground_truth"

    @property
    def corpus_id(self) -> str | None:
        return self.relative_path if self.storage == "corpus" else None

    @property
    def data_path(self) -> str | None:
        return self.relative_path if self.storage == "data" else None

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["corpus_id"] = self.corpus_id
        out["data_path"] = self.data_path
        return out


paper_dataset_specs: dict[str, PaperDatasetSpec] = {
    "mnist": PaperDatasetSpec(
        display_name="MNIST",
        domain="Image",
        k0=10,
        n_target=5000,
        storage="corpus",
        relative_path="mnist",
        loader_id="mnist",
        role="easy_regime",
    ),
    "uci_har": PaperDatasetSpec(
        display_name="UCI HAR",
        domain="Sensor",
        k0=6,
        n_target=5000,
        storage="data",
        relative_path="uci_har",
        loader_id="uci_har",
        role="easy_regime",
    ),
    "cifar10": PaperDatasetSpec(
        display_name="CIFAR-10",
        domain="Image",
        k0=10,
        n_target=5000,
        storage="corpus",
        relative_path="cifar10",
        loader_id="cifar10",
        role="stable_not_truthful",
    ),
    "ag_news": PaperDatasetSpec(
        display_name="AG News",
        domain="Text",
        k0=4,
        n_target=5000,
        storage="data",
        relative_path="ag_news_csv",
        loader_id="ag_news",
        role="text_regime",
    ),
    "20newsgroups": PaperDatasetSpec(
        display_name="20 Newsgroups",
        domain="Text",
        k0=20,
        n_target=18846,
        storage="corpus",
        relative_path="newsgroups20",
        loader_id="20newsgroups",
        role="tuning_anchor",
    ),
    "olivetti_faces": PaperDatasetSpec(
        display_name="Olivetti Faces",
        domain="Image",
        k0=40,
        n_target=400,
        storage="data",
        relative_path="olivetti_py3.pkz",
        loader_id="olivetti_faces",
        role="tuning_anchor",
    ),
}

paper_dataset_id_order: tuple[str, ...] = tuple(paper_dataset_specs)

paper_dataset_aliases: dict[str, str] = {
    "newsgroups20": "20newsgroups",
    "agnews": "ag_news",
    "olivetti": "olivetti_faces",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def datasets_root(root: Path | str | None = None) -> Path:
    base = Path(root).resolve() if root is not None else repo_root()
    return base / "datasets"


def paper_data_root(root: Path | str | None = None) -> Path:
    return datasets_root(root) / "data"


def paper_corpus_root(root: Path | str | None = None) -> Path:
    return datasets_root(root) / "corpus"


def paper_dataset_paths(root: Path | str | None = None) -> dict[str, Path]:
    return {
        "datasets_root": datasets_root(root),
        "data_root": paper_data_root(root),
        "corpus_dir": paper_corpus_root(root),
    }


def canonical_paper_dataset_id(dataset_id: str) -> str:
    key = str(dataset_id)
    return paper_dataset_aliases.get(key, key)


def get_paper_dataset_spec(dataset_id: str) -> PaperDatasetSpec:
    canonical = canonical_paper_dataset_id(dataset_id)
    if canonical not in paper_dataset_specs:
        known = ", ".join(paper_dataset_id_order)
        raise KeyError(f"Unknown paper dataset '{dataset_id}'. Known paper datasets: {known}")
    return paper_dataset_specs[canonical]


def get_paper_k0(dataset_id: str) -> int:
    return get_paper_dataset_spec(dataset_id).k0


def paper_dataset_registry_as_dict() -> dict[str, dict[str, Any]]:
    return {dataset_id: spec.as_dict() for dataset_id, spec in paper_dataset_specs.items()}


def paper_dataset_inventory() -> list[dict[str, Any]]:
    return [
        {"dataset_id": dataset_id, **spec.as_dict()}
        for dataset_id, spec in paper_dataset_specs.items()
    ]


def paper_dataset_ids(
    *,
    domain: str | None = None,
    role: str | None = None,
) -> tuple[str, ...]:
    ids = []
    for dataset_id, spec in paper_dataset_specs.items():
        if domain is not None and spec.domain.lower() != domain.lower():
            continue
        if role is not None and spec.role != role:
            continue
        ids.append(dataset_id)
    return tuple(ids)


def validate_paper_dataset_files(root: Path | str | None = None) -> list[dict[str, Any]]:
    """Return per-dataset file availability diagnostics; raises if anything is missing."""
    paths = paper_dataset_paths(root)
    rows: list[dict[str, Any]] = []
    missing_messages: list[str] = []
    for dataset_id, spec in paper_dataset_specs.items():
        if spec.storage == "corpus":
            base = paths["corpus_dir"] / spec.relative_path
            expected = [base / "X_pca50.npy", base / "y.npy", base / "meta.json"]
        elif spec.storage == "data":
            base = paths["data_root"] / spec.relative_path
            expected = [base]
        else:
            raise ValueError(f"Unsupported storage for {dataset_id}: {spec.storage}")

        missing = [str(path) for path in expected if not path.exists()]
        rows.append(
            {
                "dataset_id": dataset_id,
                "display_name": spec.display_name,
                "storage": spec.storage,
                "base_path": str(base),
                "ok": not missing,
                "missing": missing,
            }
        )
        if missing:
            missing_messages.append(f"{dataset_id}: {missing}")

    if missing_messages:
        raise FileNotFoundError(
            "Missing paper dataset files:\n" + "\n".join(missing_messages)
        )
    return rows


def load_paper_dataset(dataset_id: str, root: Path | str | None = None, **kwargs):
    """Load one of the six paper datasets through the existing loader stack."""
    from benchmark.loaders import load_dataset

    spec = get_paper_dataset_spec(dataset_id)
    paths = paper_dataset_paths(root)
    return load_dataset(
        dataset_id=spec.loader_id,
        data_root=paths["data_root"],
        corpus_dir=paths["corpus_dir"],
        use_corpus_when_available=True,
        **kwargs,
    )
